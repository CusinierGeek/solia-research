#!/usr/bin/env bash
# Replay a long UTC window as N concurrent data-pipeline replays, one per contiguous slice.
#
#   infra/scripts/parallel-replay.sh --from 2026-10-27T00:00Z --to 2026-11-02T20:00Z --chunks 4 [--warmup-hours 24]
#                                    [--dry-run] [-- <extra replay args, e.g. --rebuild>]
#
# Each slice is a normal `replay` (detached container `replay-<k>`) with its own warm-up, so its first
# rows see the same wallet / market state as live ingestion; slices are half-open and disjoint, so no
# features row is written twice. One replay is CPU-bound on about one core: N slices run ~N times faster
# until Postgres or the disk saturates (measure the first one). The sink takes its row locks in one fixed
# order, so slices sharing long-lived mints wait for each other instead of deadlocking.
#
# Compose flags come from SOLIA_COMPOSE_ARGS (e.g. "-p solia --env-file ../Solia/.env"); run it from the
# checkout whose image you want. Exit code: the highest exit code among the slices (see the replay's own
# 0 / 1 / 2 / 3 codes in docs/DATA-PIPELINE.md); 2 = invalid arguments, nothing launched.
set -euo pipefail

usage() { sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

from="" to="" chunks="" warmup=24 dry_run=0 extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --from) from="${2:-}"; shift 2 ;;
    --to) to="${2:-}"; shift 2 ;;
    --chunks) chunks="${2:-}"; shift 2 ;;
    --warmup-hours) warmup="${2:-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --) shift; extra=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[ -n "$from" ] && [ -n "$to" ] && [ -n "$chunks" ] || usage
[[ "$chunks" =~ ^[1-9][0-9]*$ ]] || { echo "--chunks must be a positive integer" >&2; exit 2; }
[[ "$warmup" =~ ^[0-9]+$ ]] || { echo "--warmup-hours must be a non-negative integer" >&2; exit 2; }

to_epoch() { date -u -d "${1%Z}Z" +%s 2>/dev/null || { echo "invalid time: $1" >&2; exit 2; }; }
from_s=$(to_epoch "$from")
to_s=$(to_epoch "$to")
(( from_s % 3600 == 0 && to_s % 3600 == 0 )) || { echo "--from and --to must be whole UTC hours" >&2; exit 2; }
hours=$(( (to_s - from_s) / 3600 ))
(( hours > 0 )) || { echo "--from must be before --to" >&2; exit 2; }
(( chunks <= hours )) || { echo "--chunks ($chunks) exceeds the window's $hours hours" >&2; exit 2; }

iso() { date -u -d "@$1" +%Y-%m-%dT%H:%MZ; }

# Whole-hour slices, the first (hours % chunks) one hour longer.
read -r -a compose <<< "${SOLIA_COMPOSE_ARGS:-}"
names=()
start=$from_s
for (( k = 1; k <= chunks; k++ )); do
  len=$(( hours / chunks + (k <= hours % chunks ? 1 : 0) ))
  end=$(( start + len * 3600 ))
  name="replay-$k"
  cmd=(docker compose "${compose[@]}" run -d --name "$name" data-pipeline replay
       --from "$(iso "$start")" --to "$(iso "$end")" --warmup-hours "$warmup" "${extra[@]}")
  if (( dry_run )); then
    printf '%s\n' "${cmd[*]}"
  else
    MSYS_NO_PATHCONV=1 "${cmd[@]}" > /dev/null
    echo "started $name: $(iso "$start") .. $(iso "$end") ($len h + ${warmup} h warm-up)"
  fi
  names+=("$name")
  start=$end
done
(( dry_run )) && exit 0

worst=0
for name in "${names[@]}"; do
  code=$(docker wait "$name")
  echo "== $name exit $code"
  docker logs "$name" 2>&1 | grep -E "featureRowsWritten|missing: |failed: " || true
  (( code > worst )) && worst=$code
  docker rm "$name" > /dev/null
done
exit "$worst"
