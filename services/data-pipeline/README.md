# data-pipeline (.NET 10 worker)

Ingests Pump.fun activity from **pumpapi.io** — live WebSocket **and** historical replay — normalizes events, computes features across the full token lifecycle (pre/post-bonding), and persists them to PostgreSQL.

- **Design:** [DATA-SOURCES.md](../../docs/DATA-SOURCES.md) · [DATA-PIPELINE.md](../../docs/DATA-PIPELINE.md)
- **Sprint:** [Sprint 1](../../docs/sprints/SPRINT-01-data-pipeline.md)

## Status

Runnable host (T2) with config, logging + a `logs`-table sink, `/health`, graceful shutdown.
Canonical event model + pumpapi parser (T3). Live WebSocket source with auto-reconnect (T4) —
observable via the opt-in probe below. Features and persistence land in T6/T7.

## Run locally

```bash
# 1. Start PostgreSQL + apply migrations (see services/db/README.md)
docker compose up -d postgres
POSTGRES_HOST=localhost dotnet run --project services/db/Solia.DbMigrator

# 2. Run the worker (the launch profile listens on http://localhost:8090)
POSTGRES_HOST=localhost dotnet run --project services/data-pipeline/Solia.DataPipeline
```

- `GET http://localhost:8090/health` → `Healthy` when the database is reachable, `Unhealthy`
  (HTTP 503) otherwise. The database check times out after 3 s.
- A startup row appears in `logs` (`service = 'data-pipeline'`).

### Watch real events (opt-in probe)

```bash
STREAM_PROBE=true POSTGRES_HOST=localhost dotnet run --project services/data-pipeline/Solia.DataPipeline
```

Connects to the pumpapi live stream and logs every event (kind, mint, venue, price) at **Debug**,
plus a running count at Information every 30 s (set `Logging__LogLevel__Solia.DataPipeline.Hosting=Debug`
to see each event) — handy to see real Pump.fun activity and to verify reconnection (drop/restore the
network and watch it back off and resume). This probe does **not** persist anything; continuous
ingestion into the DB is wired in T7. When enabled, the `/health` `stream` check reflects the
connection (`Degraded` while reconnecting).

### Replay historical data

pumpapi archives its stream as hourly `.jsonl.zst` files (since 2026-04-18). Replay feeds those
through the **same parser** as live, so historical and live events are identical downstream.

- `PumpApiReplaySource` streams a `ReplayWindow` via an archive opener
  (`HttpReplayArchiveOpener` = `{ReplayBaseUrl}/yyyy/MM/dd/HH.jsonl.zst`). Events are trimmed to
  the window's exact bounds; the window must start on/after 2026-04-18 and before its end, or the
  source throws. Missing hours (404) are skipped (`MissingHours`). Other failures (HTTP errors,
  dropped downloads, corrupt/truncated archives) are retried 3 times with backoff, resuming after
  the last line read so nothing is emitted twice; an hour that still fails is skipped and listed in
  `FailedHours`. An event stamped just before an hour boundary can sit in the next hour's archive,
  so a window ending exactly on the hour may miss a few such boundary events.
- `FileReplaySource` replays a local `.jsonl` / `.jsonl.zst` file (tests, or a mirrored archive).
- Cancelling either source throws `OperationCanceledException` (never a silent partial end).
- `Fixtures/sample.jsonl` is 20 real lines captured from `2026/04/20/12.jsonl.zst`. Note the real
  schema: the event type is `txType`, and SOL-quoted pools report `sol*` fields (`solAmount`,
  `solInPool`, `vSolInBondingCurve`, `marketCapSol`) while other pools report `quote*` + `quoteMint`.
  `RawEvent.QuoteMint` records which currency prices are in.

**Mirror a window locally** (full archive is ~1.7 TB — always pick a window), then replay the files:

```bash
# PowerShell — mirror 2026-04-18 hours 00..03
0..3 | ForEach-Object { $h = '{0:D2}' -f $_
  Invoke-WebRequest "https://replay.pumpapi.io/2026/04/18/$h.jsonl.zst" -OutFile "data/2026-04-18-$h.jsonl.zst" }
```

Choosing the actual training window and wiring replay into ingestion are Sprint 2 / T7. See
[DATA-SOURCES.md](../../docs/DATA-SOURCES.md).

## Configuration

Bound from `appsettings.json` (or any configuration source, e.g. `Pipeline__WindowSeconds`), with the
flat env vars from `.env.example` mapped onto their section keys and taking precedence
(`Config/FlatEnvAliases.cs`). The database settings are resolved by the shared
`Solia.Shared` library, the same code `Solia.DbMigrator` uses. Values are trimmed; a value that is set
but invalid (bad port, `ENRICH_HOLDERS=maybe`) fails startup with the variable's name.

| Env var | Default | Meaning |
|---------|---------|---------|
| `PUMPAPI_WS_URL` | `wss://stream.pumpapi.io/` | live stream |
| `PUMPAPI_REPLAY_BASE_URL` | `https://replay.pumpapi.io` | historical replay base |
| `ENRICH_HOLDERS` | `false` | paid holder enrichment (off = approximate from stream); accepts `true/false`, `1/0`, `yes/no`, `on/off` |
| `STREAM_PROBE` | `false` | opt-in: stream live events and log them (observability), no persistence |
| `SOLIA_DB_CONNECTION` | — | full Npgsql connection string (wins over `POSTGRES_*`) |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | `localhost/5432/solia/solia/change_me` | DB parts |

`Pipeline` section knobs: `SnapshotCadenceSeconds` (1–3600), `WindowSeconds` (1–86400, and
≥ the cadence), `MaxTrackedTokens` (1–1,000,000), `EnrichHolders`, `StreamProbe`. `PumpApi` URLs
must be absolute (`ws(s)://` for the stream, `http(s)://` for replay); reconnect backoff is
`ReconnectBaseDelaySeconds` (1–300) doubling up to `ReconnectMaxDelaySeconds` (1–3600), reset only
after a connection that delivered data. `ReceiveIdleTimeoutSeconds` (5–3600, default 60) drops a
connection that has gone silent; `MaxMessageBytes` (1 KiB–64 MiB, default 1 MiB) drops one that
sends an oversized message. Invalid values stop the worker at startup.

## Tests

```bash
dotnet test Solia.slnx
```

`Solia.DataPipeline.Tests` covers flat-key mapping, options validation, log levels and the host's
`/health` behaviour (against unreachable databases, so no Postgres is needed).

> The autonomous `SolanaRpcSource` fallback (future) reuses the Pump.fun IDL decoder from the old SolGemFinder project.
