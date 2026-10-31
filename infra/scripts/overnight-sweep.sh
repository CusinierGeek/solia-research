#!/usr/bin/env bash
# Unattended overnight sweep (#181 / #172): checks the preconditions, then runs the #160 overfitting
# search — stage 1 (stride x row budget) then stage 2 (regularization on stage 1's choice) — detached
# inside the running ai-service container, so closing this terminal doesn't stop it.
#
#   infra/scripts/overnight-sweep.sh [--dry-run] [run options]
#
#   --dry-run      check the preconditions only
#   run options    passed to `python -m solia_ai.overnight run` (e.g. --stage2-trials 12,
#                  --stride 30 --max-rows 8000000 to force stage 2's data, --restart to start over)
#
# Resumable: every finished trial is checkpointed in the models volume, so after a reboot (or a crash)
# run the same command again: finished trials are skipped, only the one in progress is lost.
# Output, all in the models volume (/app/models/reports):
#   overnight.log            python -u log of the whole run (appended across reruns)
#   overfitting-*.json/.diff the two stages' reports and proposed config diffs
#   overnight-summary.txt    best trials, baseline, elapsed, failures
# Containers: AI_CONTAINER (default solia-ai-service-1), BACKEND_CONTAINER (default solia-backend-1).
set -euo pipefail

# Git Bash would rewrite in-container paths passed to docker into Windows paths.
export MSYS_NO_PATHCONV=1

AI="${AI_CONTAINER:-solia-ai-service-1}"
BACKEND="${BACKEND_CONTAINER:-solia-backend-1}"
REPORTS=/app/models/reports

dry_run=false
if [ "${1:-}" = "--dry-run" ]; then
  dry_run=true
  shift
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$AI" 2>/dev/null)" != "true" ]; then
  echo "REFUSED: the ai-service container $AI is not running (start the stack with the GPU override)" >&2
  exit 1
fi

# The bot runs in the backend: its env decides. Not running = no bot.
paper_bot=""
if [ "$(docker inspect -f '{{.State.Running}}' "$BACKEND" 2>/dev/null)" = "true" ]; then
  paper_bot="$(docker exec "$BACKEND" printenv PAPER_BOT_ENABLED || true)"
fi

if ! docker exec "$AI" python -m solia_ai.overnight preflight \
    --reports-dir "$REPORTS" --paper-bot-enabled "$paper_bot"; then
  echo "overnight sweep NOT started: fix the problems above and rerun" >&2
  exit 1
fi

cat <<'EOF'

  Before leaving it overnight: PAUSE WINDOWS UPDATE (Settings > Windows Update > Pause updates).
  An update reboot killed the 2026-10-30 ablation. If the PC reboots anyway, rerun this command:
  the sweep resumes from its checkpoints.

EOF

if [ "$dry_run" = true ]; then
  echo "dry run: preconditions OK, nothing started"
  exit 0
fi

# Detached exec: survives this terminal / the docker client going away. `exec` so the python process
# is the one the preflight of a later launch sees as "already running".
docker exec -d "$AI" sh -c \
  'mkdir -p "$0" && exec python -u -m solia_ai.overnight run --reports-dir "$0" "$@" >> "$0/overnight.log" 2>&1' \
  "$REPORTS" "$@"

echo "overnight sweep started in $AI"
echo "  follow:  docker exec $AI tail -f $REPORTS/overnight.log"
echo "  summary: docker exec $AI cat $REPORTS/overnight-summary.txt   (when it ends)"
