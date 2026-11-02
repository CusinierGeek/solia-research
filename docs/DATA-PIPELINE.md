# Data pipeline

## Purpose

Turn raw Pump.fun activity (live **and** historical) into clean, time-aligned **feature snapshots** stored in PostgreSQL, ready for training and inference — covering the **full token lifecycle, pre- and post-bonding**.

## Implementation: .NET worker

The pipeline is a **.NET worker service** (`services/data-pipeline`), not Python. Rationale:
- the on-chain decoding know-how already exists in C# (SolGemFinder — the fallback `SolanaRpcSource`);
- unified stack with the .NET backend (shared models, EF Core, tooling);
- Python is reserved exclusively for the ML service.

## Source: pumpapi.io (primary)

Provider selection and cost model are documented in [`DATA-SOURCES.md`](DATA-SOURCES.md). Summary:
- **Live:** free WebSocket `wss://stream.pumpapi.io/` — token creations, buys, sells, migrations, pool events, across Pump.fun + AMMs (PumpSwap/Raydium/Meteora/Orca). Pre- and post-bonding in one stream.
- **Historical:** free hourly replay archives (`.jsonl.zst`) since 2026-04-18, **same event format** as live.
- **Cost:** streaming + replay are free; only action/enrichment REST calls cost 10 000 lamports; Solia derives from the stream and avoids these (see *derive, don't fetch*).

### Swappable `DataSource` abstraction

```csharp
public interface IDataSource
{
    IAsyncEnumerable<RawEvent> StreamAsync(CancellationToken ct);
}

// adapters (see DATA-SOURCES.md):
//   PumpApiStreamSource   — live WS                       [primary]
//   PumpApiReplaySource   — historical .jsonl.zst         [training backfill]
//   PumpPortalSource      — alternative live              [optional]
//   SolanaRpcSource       — logsSubscribe + Pump.fun IDL  [fallback, from SolGemFinder]
//   FileReplaySource      — recorded fixtures             [tests]
```

Downstream code depends only on `RawEvent` → nothing else knows about the provider.

## Stages

```
RawEvent ─▶ normalize ─▶ per-token rolling window ─▶ feature compute ─▶ persist
                                                                         │
                     tokens_live / tokens_history ◀───────────────────────┘
                                                      features ◀──────────┘
```

1. **Normalize**: map provider payload to a canonical event (`mint`, `ts`, `type`, `venue`, `price`, `amount`, `is_buy`, pool reserves, `bonding_progress`, `is_migrated`, ...). Live and replay normalize through the **same** path.
2. **Window**: maintain in-memory rolling windows per active token.
3. **Feature compute**: at a fixed cadence (or per event), compute the feature vector.
4. **Persist**: upsert `tokens_live`, append `tokens_history` (first seen), insert `features`.

## Feature set

Base: `price, volume, market_cap, holders, liquidity, buy_events, sell_events, token_age`.

Derived: `buy_sell_ratio, holders_growth_rate, volume_growth_rate, tx_creation_rate, volatility, ema, rsi, price_derivative, trending_score, social_mentions`, plus `onchain_signals` and `behavioral_signals` (JSONB).

**Derived directly from stream events (no paid fetch):** price (pool reserves ratio), market cap, volume, buy/sell ratio, bonding progress.

**Holders** is the one genuinely hard metric from the stream alone:
- *Free (default):* approximate by counting **unique traders** seen in the stream per token.
- *Paid (optional):* periodic `Get Token Info` (10 000 lamports), cached. Off by default.

Feature definitions must stay **in sync** with `ai-service/features.py` — single documented spec, mirrored, with a parity test (Sprint 2).

### Market-regime context (FE-T11, #195)

Every other feature describes one token in isolation. `MarketContext` (fed every event by
`TokenWindowStore.Apply`, tracked token or not) maintains **market-wide rolling aggregates** over
the whole stream, so the model can tell "this token is strong" from "the whole market is up":

| Column | Window | Meaning |
|---|---|---|
| `mkt_launch_rate_5m` / `_1h` | 5 min / 1 h | `Create` events per minute |
| `mkt_graduations_1h` | 1 h | `Migrate` events (graduations) |
| `mkt_graduation_ratio_1h` | 1 h | graduations / launches (NULL without a launch) |
| `mkt_buy_sol_5m` / `_1h` | 5 min / 1 h | SOL buy volume across all tokens (SOL-quoted trades) |
| `mkt_net_sol_flow_5m` / `_1h` | 5 min / 1 h | SOL buys − sells across all tokens |
| `mkt_active_tokens_5m` / `_1h` | 5 min / 1 h | distinct tokens with a trade |
| `mkt_coverage_sec` | — | seconds of stream history behind them, capped at 3600 (always written) |

- **Computed once per tick.** `SnapshotActive` aggregates once and attaches the same
  `MarketAggregates` instance to every snapshot of that flush; nothing is recomputed per token.
- **Bounded memory.** Counters live in one-second buckets of a ring buffer spanning the long window
  (plus the clock skew), not raw events. The active-token counts keep each token's last two distinct
  trade seconds, and forget tokens that last traded before the long window.
- **Point-in-time.** The aggregates attached at `ts` sum the whole seconds strictly before the second
  containing `ts`: `[floor(ts) − W, floor(ts))`. Nothing at or after the snapshot leaks in, and the
  result depends only on event timestamps, not on arrival order or flush cadence, so live and replay
  agree (`MarketContextTests`). Like every feature, an event arriving after a tick that already read
  its second is not applied to that tick retroactively. Events are bucketed against their own
  `EventClock` (same skew rules as the engine): one far-future timestamp is dropped instead of
  overwriting a bucket of the past.
- **Warm-up.** An aggregate is **NULL until its whole window has been observed** since the process (or
  replay) started: 5 min for the `_5m` columns, **1 h** for the `_1h` ones. A replay meant for training
  must therefore start **at least one hour before the first snapshot used for training** (e.g. replay
  from 13:00 to train on rows from 14:00), or those rows carry NULL long-window aggregates. The same
  holds for an hour after every live restart; `mkt_coverage_sec` tells such rows apart (the model's
  NULL imputation, see docs/ML-MODEL.md, keeps them from looking like a dead market). A replay with a
  missing archive hour counts that hour as quiet.
- **SOL/USD is left out.** The stream only carries Pump.fun-ecosystem token pools, almost all quoted in
  SOL; it has no SOL-vs-stable pool, and a SOL price inferred from the few tokens quoted in both SOL and
  a stable would be sparse and dominated by those tokens' own moves. Deriving it would need an external
  price feed (out of scope: "derive, don't fetch"). Everything here is in SOL, so a SOL/USD move does
  not distort the aggregates themselves.

### Granularity: aggregated snapshots, not raw trades

The pipeline persists **one aggregated `FeatureSnapshot` per token per cadence** — it does **not** store individual trade events. `TokenState.Apply` streams every `RawEvent` into an in-memory rolling window (carrying `Ts`, `Kind`, `QuoteAmount` in SOL, `QuoteMint`, and the `Trader`/txSigner), and `Snapshot` reduces that window to the feature row. The `features.raw` JSONB column exists but is currently **unused**.

Consequence: **per-trade granularity is not recoverable from the database** — a feature that needs it (net SOL flow, holder concentration from `txSigner`, large-buy distribution) cannot be backfilled from stored rows.

This is **not** a problem, because the real source of truth is **re-playable**:

- The raw material already flows through `RawEvent` (SOL amounts, reserves, txSigner), so new per-trade features are computed in `TokenState`/`Snapshot` with **no new ingestion path** — just a wider feature set. Adding them is low-impact; see the enrichment epic.
- To apply a new feature to historical data, **re-run the free pumpapi replay** (`PumpApiReplaySource`) through the same path — it regenerates the whole `features` table enriched. See *Historical backfill* below.

Persisting raw per-trade rows (a dedicated `trade_events` table) is deliberately **avoided**: it adds write volume and retention burden for no gain, since re-replay already covers backfill.

## Post-bonding coverage

Events carry a `venue` and migration markers, so a token is tracked **continuously through graduation** (bonding curve → AMM). This is required for correct `pump`/`rug` labels over the full lifecycle (see [`ML-MODEL.md`](ML-MODEL.md)).

## Reliability

- Reconnect with exponential backoff + jitter; resume cleanly.
- Idempotent writes (dedupe by `(mint, ts, type, signature)`).
- Bounded memory: cap tracked tokens; evict stale/inactive (`is_active=false`).
- Backpressure: batch DB writes; if lagging, drop low-value snapshots (logged).

## Historical replay (training data)

A one-shot **replay mode** of the data-pipeline replays a UTC window of pumpapi's hourly archives
(`https://replay.pumpapi.io/YYYY/MM/DD/HH.jsonl.zst`, free, since 2026-04-18) through
`PumpApiReplaySource` → the **same** `PipelineEngine` / `TokenWindowStore` / `PostgresTokenSink` as live
ingestion → Postgres, then exits. There is no parallel feature code: a replayed snapshot is computed
exactly like a live one. It runs in its own process (never alongside live ingestion's token state), with
the same configuration as the live host (`Pipeline:*`, `PUMPAPI_REPLAY_BASE_URL`, DB connection, and the
same environment: `ASPNETCORE_ENVIRONMENT` or `DOTNET_ENVIRONMENT` picks `appsettings.{env}.json`), and
never resets `tokens_live` (its newer-wins guards keep an old window from rolling live rows back). At the
end it marks the tokens it still tracks inactive, as of the last replayed event (rows live ingestion has
updated since are untouched), so historical tokens never show as live.
"Derive, don't fetch": no enrichment call is made during a replay.

```bash
# local (Postgres from `make dev-db`)
dotnet run --project services/data-pipeline/Solia.DataPipeline -- replay --from <utc> --to <utc> [options]
# container, against the compose stack's database
docker compose run --rm data-pipeline replay --from <utc> --to <utc> [options]
```

| Argument | Meaning |
|----------|---------|
| `--from`, `--to` | Half-open UTC window `[from, to)`, ISO 8601 (`2026-05-01T03:00Z`; a bare date = midnight UTC; no offset = UTC). Rejected: `from >= to`, a start before 2026-04-18, an end after the start of the current hour (that hour's archive isn't published until it is over). |
| `--mirror <dir>` | **Opt-in** local mirror (off by default). Hours are read from `<dir>/YYYY/MM/DD/HH.jsonl.zst` first and downloaded into it only when absent. The directory must already exist and be writable (it is never created for you). In compose: `--mirror /replay-archives` (the `replay-archives` volume). |
| `--mirror-only` | With `--mirror`: only download the window into the mirror; no replay, no database. |
| `--rebuild` | Refresh `features` rows the replay recomputes (see *Rebuild* below). |
| `--allow-expired` | Replay a window older than the `features` retention TTL anyway (see *Replay vs retention* below). |
| `--Section:Key=value`, `--environment <name>` | Configuration overrides, as on the live host's command line (e.g. `--Pipeline:SnapshotCadenceSeconds=5`; `--Key value` works too). |

**Default = direct streaming.** Each hour is downloaded and zstd-decompressed on the fly (~2 GB/hour
decompressed, never held in memory); nothing but the database is written. A dropped download, a download
that stalls (no data for 60 s) or a corrupt hour is retried with backoff, resuming after the last line already read; an hour that keeps failing is
skipped and reported. Hours absent at the source (404) are reported as missing.

**Mirror.** Downloads go to `HH.jsonl.zst.part` and are renamed into place only once complete, so the
mirror never holds a half-written hour (an interrupted run leaves at most a `.part`, overwritten next
time). A mirror file that fails to read is downloaded again on the retry and replaced only once the new
copy is complete; if the source no longer has that hour, the local copy is kept and the hour is reported
**failed** (exit `3`), not missing. Paths are culture-invariant, so every host maps an hour to the same file.
Budget **~400 MB per hour** compressed (~9.4 GiB/day): before it starts, the run logs the estimate for the
hours **not yet mirrored** and the free space. The
full archive is ~1.7 TB — mirror chosen windows, never all of it.

**Idempotent / resumable.** `features` rows are keyed by `(mint, ts)` and inserted
`ON CONFLICT DO NOTHING`, so re-running a window (after an interruption or failed hours) inserts no
duplicates; with a mirror, the re-run doesn't download again either.

**Rebuild** (for feature enrichment). By default a re-replay **keeps** existing rows, so a column added
since they were written would stay NULL. `--rebuild` switches the insert to `ON CONFLICT (mint, ts) DO
UPDATE` of every column, refreshing those rows in place (ids kept) — but only for tokens whose **creation
the replay observed**: a token born before the window starts is tracked from an empty state, so its holders,
unique traders, growth rates and momentum would come from a truncated history; its stored rows are kept
(missing ones are still inserted). Snapshot times are derived from the
event clock, so a row is refreshed only when the replay recomputes the same `(mint, ts)`: use the **same
window start and `Pipeline:SnapshotCadenceSeconds`** as the run that wrote it (a different start writes
new rows next to the old ones). `tokens_live` / `tokens_history` follow their usual merge rules.

**Summary and exit codes.** The run ends with a summary line (hours, processed, `missing`
(`MissingHours`), `failed` (`FailedHours`), `featureRowsWritten` (rows inserted, or refreshed with
`--rebuild`), `failedBatches`, mirror hits/downloads) followed by each missing/failed hour. The summary is
also written to the `logs` table (`service = 'data-pipeline'`, best-effort; not for `--mirror-only`).

| Exit code | Meaning |
|-----------|---------|
| `0` | Every hour read (or missing at the source) and every batch written. |
| `1` | Fatal: unreachable database, interruption (SIGTERM / Ctrl+C), unexpected error. Re-run the window. |
| `2` | Invalid arguments or configuration (bad window, missing/unwritable mirror dir, window older than the retention TTL); nothing ran. |
| `3` | Ran to the end, but some hours or write batches were given up on: re-run the same window to fill them. |

### Replay vs retention

With retention on (`RETENTION_ENABLED=true`, the compose default — see *Data retention* below), the next purge
deletes every `features` row older than `RETENTION_FEATURES_TTL_DAYS` (14 d), replayed or not. So a replay
whose window **starts before that cutoff is refused** (exit `2`, nothing written) instead of being silently
wiped within the hour; one whose oldest rows expire within a day is replayed with a warning. The check uses the
replay process's own retention settings — under compose, the same `.env` as the live pipeline.

To **train on history older than the TTL**:
1. raise `RETENTION_FEATURES_TTL_DAYS` in `.env` to cover the window (+ the days you need it for) and recreate
   the pipeline (`docker compose up -d data-pipeline`) — the live purge then keeps it too; **or**
2. set `RETENTION_ENABLED=false` and recreate the pipeline for the time you train on it (the table then grows
   unbounded: ~17 GB/day of live ingestion).

`--allow-expired` replays regardless (with a warning) — only useful when the live pipeline's purge is off
or uses a longer TTL than the replay's configuration.

### Recipes

Replay **recent** windows first (the most recent days are closest to what the live model will see).
Cold start: a token's rolling window begins at the replay's first event, so the first
`Pipeline:WindowSeconds` (5 min) of a window carry partial-history features — negligible over hours.

- **Test window (3 h)**: smoke-tests the path and the DB in a few minutes.
  `replay --from 2026-05-01T00:00Z --to 2026-05-01T03:00Z`
- **Baseline (3–7 days)**: the first real training set. With a mirror, so later rebuilds and ablations
  don't download it again (3 days ≈ 28 GiB, 7 days ≈ 66 GiB):
  `replay --from 2026-05-01 --to 2026-05-08 --mirror /replay-archives`
  (or download first with `--mirror-only`, then replay from disk).
- **Learning curve (1 / 2 / 4 / 8 days, most recent window held out)**: tells whether more data still
  helps before replaying more. Fix a held-out window at the **end** (e.g. the most recent day, never
  trained on), then train on growing windows that end where it starts: `D-1..D`, `D-2..D`, `D-4..D`,
  `D-8..D`. Each larger window contains the smaller ones, so replay the largest once (idempotent) and
  select the training range at train time (`solia_ai.train` and the ablation harness split by time).
  Plot the held-out metric against days: still climbing at 8 days → replay more; flat → spend the effort
  on features/model instead.

## Config

- `PUMPAPI_WS_URL`, `PUMPAPI_REPLAY_BASE_URL`, DB connection, window length, compute cadence, max tracked tokens, `ENRICH_HOLDERS` (bool) — via env / worker config.

## Testing

- Unit: feature math on synthetic windows.
- Integration: `FileReplaySource` feeds a recorded event file → assert rows in an ephemeral Postgres (Testcontainers).
- Replay mode: window/argument validation, direct mode keeps nothing on disk, mirror hit / miss / corrupt
  file (kept when the source lost the hour), atomic downloads, stalled downloads, end-of-replay deactivation,
  idempotent re-run and `--rebuild` refresh limited to tokens born in the window (Postgres parts in the
  Integration category).

## Data retention

Nothing else ever deletes data, and live ingestion writes ~1 M `features` rows/hour (~17 GB/day), so the
data-pipeline runs a periodic purge (`Retention/RetentionService`). It is the **single owner** of retention: it
owns `features` (the bulk of the volume) and the replay that must not write rows the purge would remove.

| Table | Kept | Env var |
|-------|------|---------|
| `features` | 14 days | `RETENTION_FEATURES_TTL_DAYS` |
| `logs` (every service) | 14 days | `RETENTION_LOGS_TTL_DAYS` |
| `predictions` | 30 days | `RETENTION_PREDICTIONS_TTL_DAYS` |

`trades`, `models`, `training_events`, `tokens_history` and `tokens_live` are never purged: P&L and model
history stay complete. A `trades.prediction_id` / `predictions.feature_id` pointing at a purged row becomes
`NULL` (`ON DELETE SET NULL`).

- **When**: at startup, then every `RETENTION_INTERVAL_MINUTES` (60). `RETENTION_ENABLED` (compose: `true`;
  a bare `dotnet run`: `false`, like `INGEST`) turns it off — then nothing is ever deleted.
- **How**: rows with `ts` strictly older than `now − TTL` are deleted oldest first in batches of
  `RETENTION_BATCH_SIZE` (10 000) rows, each its own short transaction (predictions before features), then
  the tables that shrank get a plain `VACUUM (ANALYZE)` so the space is **reused** by new rows (the files don't
  shrink; the table stays at ~TTL worth of data instead of growing). No lock blocks ingestion, the API or a
  retrain reading `features` (it keeps its MVCC snapshot; those rows are reclaimed once it ends).
  Measured: ~15 k rows/s, so an hour of live ingestion is purged in about a minute, worst batch ~3 s
  (see [DATA-MODEL.md § Retention](DATA-MODEL.md#retention)).
- **Training**: the retrain reads the whole `features` table, so the TTL bounds it. Keep it above what training
  needs (held-out windows + label horizons, max 1 h).
- **Logging**: one line per table (rows, time span removed, duration) on stdout and one `logs` row per run
  (`service = 'data-pipeline'`, `message = 'retention run'`, per-table `context`). A table that fails (e.g.
  database down) is logged as a warning and retried on the next tick; never fatal.
- **On demand**:

  ```bash
  # report what would be deleted, change nothing
  docker compose run --rm data-pipeline retention --dry-run
  # purge once now (whether or not RETENTION_ENABLED is set), e.g. with a shorter TTL
  docker compose run --rm data-pipeline retention --Retention:FeaturesTtlDays=7
  ```

  Exit `0` = done, `1` = a table failed (re-run), `2` = invalid arguments / configuration.
- **Config** is validated at startup (TTLs 1–3650 days, interval 1–1440 min, batch 100–1 000 000); an invalid
  value stops the pipeline (or the command) with the option's name.
