#!/usr/bin/env bash
# Shortcuts for the compose stack (docker-compose.yml). The Makefile targets call this script, so the
# same commands work without make (Git Bash on Windows): `infra/scripts/solia.sh <command> [args]`.
#
#   up       build and start the whole stack, wait until it's healthy (dashboard: http://localhost:8081)
#   down     stop and remove the containers; volumes (DB, models) are kept
#   logs     follow the logs (all services, or the ones named: `logs backend dashboard`)
#   ps       list the containers, including the exited db-migrate job
#   reset    down + delete the volumes (DB data, trained models). Asks first; --yes skips the prompt.
#   smoke    end-to-end smoke test in its own compose project (infra/scripts/smoke.sh)
#   dev-db   Postgres alone, published on 127.0.0.1:5432 for `dotnet run` / pytest
set -euo pipefail

# Git Bash would rewrite in-container paths passed to docker into Windows paths.
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/../.."

usage() {
  sed -n '2,11s/^# \{0,1\}//p' "$0"
}

confirm_reset() {
  if [ "${1:-}" = "--yes" ] || [ "${1:-}" = "-y" ]; then return 0; fi
  if [ ! -t 0 ]; then
    echo "reset deletes the DB and models volumes: pass --yes to confirm (no terminal to ask)" >&2
    exit 1
  fi
  local answer
  read -r -p "Delete the containers AND the volumes (DB data, trained models)? [y/N] " answer
  case "$answer" in
    y | Y | yes | YES) ;;
    *) echo "aborted" >&2; exit 1 ;;
  esac
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift

case "$cmd" in
  up)
    [ -f .env ] || echo "note: no .env, using the defaults (cp .env.example .env to customize)" >&2
    exec docker compose up -d --build --wait "$@" ;;
  down)
    exec docker compose down "$@" ;;
  logs)
    exec docker compose logs -f --tail=100 "$@" ;;
  ps)
    exec docker compose ps -a "$@" ;;
  reset)
    confirm_reset "${1:-}"
    exec docker compose down -v --remove-orphans ;;
  smoke)
    exec bash infra/scripts/smoke.sh "$@" ;;
  dev-db)
    exec docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres "$@" ;;
  help | -h | --help)
    usage ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 2 ;;
esac
