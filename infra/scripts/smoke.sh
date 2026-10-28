#!/usr/bin/env bash
# End-to-end smoke test of the containerized stack (docker-compose.yml): health, topology, non-root,
# persistence across `down`/`up`, and an opt-in data-flow check. Runs in GitHub Actions and in Git Bash
# on Windows; needs only docker (compose 2.24+), curl and coreutils (no jq).
#
#   infra/scripts/smoke.sh                     # build, run every check, then tear the stack down
#   SMOKE_DATA_FLOW=1 infra/scripts/smoke.sh   # + pumpapi ingest → DB → backend (needs the internet)
#
# Options (env):
#   COMPOSE_PROJECT_NAME  default solia-smoke: its own containers and volumes, so the default teardown
#                         (down -v) never touches a dev stack's data. Ports 8080/8081 must be free.
#   SMOKE_BUILD=0         reuse the images already built instead of `up --build`.
#   SMOKE_KEEP=1          leave the stack running at the end (e.g. for more CI steps).
#   SMOKE_TIMEOUT         seconds for the stack to turn healthy (default 600).
#   SMOKE_DATA_FLOW=1     wait until ingested rows reach the DB and /api/v1/tokens/live (needs INGEST on,
#                         the compose default, and the pumpapi live stream reachable).
#   SMOKE_DATA_FLOW_TIMEOUT  seconds for the data-flow check (default 300).
set -euo pipefail

# Git Bash would rewrite in-container paths (/proc/1/status, /app/models) into Windows paths.
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")/../.."

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-solia-smoke}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-600}"
SMOKE_DATA_FLOW_TIMEOUT="${SMOKE_DATA_FLOW_TIMEOUT:-300}"
SERVICES="postgres data-pipeline ai-service backend dashboard"   # long-running, with a healthcheck
INTERNAL="postgres ai-service data-pipeline"                    # must not be reachable from the host

log() { echo "smoke: $*"; }
# fail <message> [service…]: report, print the logs of the failing service(s), exit 1.
fail() {
  local msg="$1"; shift
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::smoke: $msg"; else echo "smoke: FAIL: $msg" >&2; fi
  docker compose ps -a >&2 || true
  for s in "$@"; do
    echo "----- logs: $s -----" >&2
    docker compose logs --no-color --tail 200 "$s" >&2 || true
  done
  exit 1
}
cid() { docker compose ps -aq "$1"; }
state() { local id; id=$(cid "$1"); [ -n "$id" ] && docker inspect -f "$2" "$id" || echo missing; }
psql_q() { docker compose exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -qAt -U "$POSTGRES_USER" -d "$POSTGRES_DB"'; }
http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" || true; }

teardown() {
  if [ "${SMOKE_KEEP:-0}" = 1 ]; then log "stack left running (SMOKE_KEEP=1)"; return; fi
  docker compose down -v --remove-orphans > /dev/null 2>&1 || true
}
trap teardown EXIT

# up: `up -d --wait`, then the explicit checks below are the verdict (they name the failing service;
# --wait alone only says that one didn't make it).
up() {
  local build=(); [ "${SMOKE_BUILD:-1}" = 1 ] && build=(--build)
  docker compose up -d --wait --wait-timeout "$SMOKE_TIMEOUT" "${build[@]}" \
    || log "compose up --wait reported a failure; checking each service"
  assert_healthy
}

# The migrator exited 0 and every long-running service is healthy (short grace period after --wait).
assert_healthy() {
  local deadline=$((SECONDS + 60)) migrate h s pending
  while :; do
    migrate=$(state db-migrate '{{.State.Status}} {{.State.ExitCode}}')
    case "$migrate" in
      "exited 0") ;;
      exited* | dead*) fail "db-migrate failed ($migrate)" db-migrate ;;
    esac
    pending=""
    for s in $SERVICES; do
      h=$(state "$s" '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}')
      [ "$h" = unhealthy ] && fail "$s is unhealthy" "$s"
      [ "$h" = healthy ] || pending="$pending $s"
    done
    if [ "$migrate" = "exited 0" ] && [ -z "$pending" ]; then log "ok: db-migrate exited 0, all services healthy"; return; fi
    if [ $SECONDS -ge $deadline ]; then
      # shellcheck disable=SC2086 # word-split on purpose: one service per word
      fail "not healthy:$pending (db-migrate: $migrate)" $pending
    fi
    sleep 3
  done
}

# backend:8080/health (DB + ai-service checks) and dashboard:8081/health/live answer 200 from the host.
assert_published() {
  local deadline=$((SECONDS + 60)) b d
  until b=$(http_code http://localhost:8080/health) && d=$(http_code http://localhost:8081/health/live) \
        && [ "$b" = 200 ] && [ "$d" = 200 ]; do
    [ $SECONDS -lt $deadline ] || fail "host probes: backend /health=$b, dashboard /health/live=$d" backend dashboard
    sleep 3
  done
  log "ok: backend :8080/health and dashboard :8081/health/live answer 200"
}

# postgres / ai-service / data-pipeline publish nothing, and their ports don't answer on the host.
assert_internal() {
  local s ports rc
  for s in $INTERNAL; do
    ports=$(docker port "$(cid "$s")")
    [ -z "$ports" ] || fail "$s publishes a host port: $ports"
  done
  # 8080 is the backend's on the host (the pipeline's in-container port), so only 5432 and 8000 are probed.
  for p in 5432 8000; do
    # A container outside this project (e.g. the dev override's Postgres) may hold the port: not ours.
    if docker ps --filter "publish=$p" --format '{{.Label "com.docker.compose.project"}}' \
        | grep -qvx "$COMPOSE_PROJECT_NAME"; then
      log "warn: localhost:$p is published by another container; probe skipped"; continue
    fi
    rc=0; curl -s -o /dev/null --max-time 3 "http://localhost:$p/" || rc=$?
    # 7 = connection refused, 28 = timeout: nothing listens. Anything else means something answered.
    case "$rc" in 7 | 28) ;; *) fail "localhost:$p is reachable from the host (curl exit $rc)" ;; esac
  done
  log "ok: postgres, ai-service and data-pipeline are not reachable from the host"
}

# Every running container's main process (PID 1) runs as a non-root user (real and effective UID).
assert_non_root() {
  local id name uids n=0
  for id in $(docker compose ps -q); do
    name=$(docker inspect -f '{{.Name}}' "$id")
    uids=$(docker exec "$id" cat /proc/1/status | awk '/^Uid:/ { print $2, $3 }')
    [ -n "$uids" ] || fail "$name: could not read the PID 1 UID"
    case " $uids " in *" 0 "*) fail "$name runs as root (Uid: $uids)" ;; esac
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || fail "no running container"
  log "ok: $n running containers, none as root"
}

# Opt-in: the pipeline ingests pumpapi events into the DB and the backend serves them.
assert_data_flow() {
  local deadline=$((SECONDS + SMOKE_DATA_FLOW_TIMEOUT)) before now live
  before=$(echo "SELECT count(*) FROM features;" | psql_q)
  log "data flow: waiting for ingested rows (features: $before)"
  while :; do
    now=$(echo "SELECT count(*) FROM features;" | psql_q)
    live=$(curl -fsS --max-time 5 "http://localhost:8080/api/v1/tokens/live?limit=1" || true)
    if [ "$now" -gt "$before" ] && case "$live" in *'"mint"'*) true ;; *) false ;; esac; then
      log "ok: features $before → $now, /api/v1/tokens/live non-empty"; return
    fi
    [ $SECONDS -lt $deadline ] \
      || fail "no data flow in ${SMOKE_DATA_FLOW_TIMEOUT}s (features $before → $now, tokens/live: ${live:-error})" data-pipeline backend
    sleep 5
  done
}

# `down` (no -v) then `up` keeps the DB rows and the models volume content.
assert_persistence() {
  local marker counts after
  marker="smoke-$(date +%s)-$RANDOM"
  # Row counts of the migrator's and the pipeline's tables, plus a marker row and a marker file in the
  # models volume.
  local count_sql="SELECT (SELECT count(*) FROM schema_migrations) || '/' || (SELECT count(*) FROM features)
    || '/' || (SELECT count(*) FROM tokens_history) || '/' || (SELECT count(*) FROM logs);"
  echo "INSERT INTO logs (service, level, message) VALUES ('smoke', 'info', '$marker');" | psql_q
  docker compose exec -T ai-service sh -c "echo $marker > /app/models/.smoke-marker" \
    || fail "cannot write to the models volume" ai-service
  # The pipeline stops first, so ingestion can't add rows after the counts are read.
  docker compose stop data-pipeline > /dev/null 2>&1
  counts=$(echo "$count_sql" | psql_q)
  log "persistence: down (volumes kept) then up, with rows $counts (migrations/features/tokens/logs)"
  docker compose down
  up
  after=$(echo "$count_sql" | psql_q)
  # Counts may only grow (the pipeline ingests again after `up`); migrations and the marker must match.
  [ "${after%%/*}" = "${counts%%/*}" ] || fail "schema_migrations changed across down/up: $counts → $after"
  local i b a
  for i in 2 3 4; do
    b=$(echo "$counts" | cut -d/ -f$i); a=$(echo "$after" | cut -d/ -f$i)
    [ "$a" -ge "$b" ] || fail "rows lost across down/up: $counts → $after"
  done
  [ "$(echo "SELECT count(*) FROM logs WHERE message = '$marker';" | psql_q)" = 1 ] \
    || fail "the marker row did not survive down/up"
  [ "$(docker compose exec -T ai-service cat /app/models/.smoke-marker | tr -d '\r')" = "$marker" ] \
    || fail "the models volume content did not survive down/up" ai-service
  echo "DELETE FROM logs WHERE message = '$marker';" | psql_q
  docker compose exec -T ai-service rm -f /app/models/.smoke-marker
  log "ok: DB rows ($after) and the models volume survived down/up"
}

# Created by us, not by the bind mount (Docker would make it root-owned).
mkdir -p secrets
log "project $COMPOSE_PROJECT_NAME: starting the stack"
up
assert_published
assert_internal
assert_non_root
if [ "${SMOKE_DATA_FLOW:-0}" = 1 ]; then assert_data_flow; else log "skip: data flow (set SMOKE_DATA_FLOW=1)"; fi
assert_persistence
# After the restart too: the stack came back healthy (up), reachable, and still non-root.
assert_published
assert_non_root
log "all checks passed"
