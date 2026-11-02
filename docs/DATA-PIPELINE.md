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

### Cross-token wallet memory (FE-T9, #189)

Every other feature comes from one token's own trades; on Pump.fun *who* trades a token is often a stronger
signal (a launch bought by wallets created minutes ago, a creator who already launched ten tokens today).
`WalletStore` (owned by `TokenWindowStore`, fed every event) keeps **per-wallet state across tokens**, derived
for free from the signers of the stream's trades and creates (`RawEvent.Trader` = pumpapi `txSigner`): first /
last seen, trade count, tokens traded, tokens created and the times of the latest 16 launches. The model never
sees an address, only statistics built from them, so an unknown wallet simply counts as fresh and nothing needs
a retrain when wallets rotate.

| Column | Meaning |
|---|---|
| `wal_fresh_1h_share` / `_24h_share` | share of the window's signers first seen < 1 h / < 24 h before `ts` (unknown = fresh) |
| `wal_median_trader_age_sec` | median wallet age of the window's signers |
| `wal_creator_age_sec` | age of the creator's wallet (signer of the token's `Create`) |
| `wal_creator_prev_launches` | the creator's launches before this one within the age cap |
| `wal_creator_buy_sol` / `_buy_supply_share` | the creator's dev buy (carried by the `Create` event: `solAmount`, `initialBuy`), in SOL and / supply |
| `wal_creator_sold_share` | tokens the creator sold / bought on this launch |
| `wal_creator_first_sell_sec` | seconds since the creator's first sell (NULL while it hasn't sold) |
| `wal_coverage_sec` | seconds of wallet history behind the lookups, capped at the age cap (always written) |

- **Point-in-time.** `WalletStore.Apply` only queues an event; each tick advances the store to the snapshot
  time, applying the queued events dated **strictly before** it, in timestamp order, before any lookup. A trade
  at or after a snapshot's `ts` never changes it (`WalletStoreTests`). The creator-behaviour columns come from
  the token's own events, like every per-token feature (events up to `ts`).
- **Live = replay.** The result depends only on event timestamps and arrival order, so the same events give the
  same values live and in a replay that reads the archive hours in order (`ReplayWalletTests` replays a fixture
  both ways).
- **Bounded.** Ages are capped at `Pipeline:WalletAgeCapHours` (default **24 h**) and launches counted within
  it, so any history at least that long gives the same features. A wallet inactive for longer than
  `Pipeline:WalletTtlHours` (default **336 h = 14 d**, the `features` retention) is forgotten — logically at
  once (it counts as fresh again, whenever the hourly sweep runs), physically by the sweep, which also deletes it
  from `wallet_state`.
- **Warm-up.** The history-dependent columns (freshness, ages, launches) are **NULL until their horizon has been
  observed**: 1 h for `wal_fresh_1h_share`, 24 h for the others. A replay meant for training therefore uses
  **`--warmup-hours 24`** (at least `WalletAgeCapHours`): the 24 hours before `--from` are replayed through the
  same engine, nothing dated before `--from` is written, and the window starts with a warm wallet memory (and
  market context). With 24 h of warm-up the replay reproduces live values except for wallets dormant between
  24 h and the 14-day TTL, which live still remembers (their age is capped at 24 h either way, so only their
  "fresh" flag can differ); a warm-up of 336 h reproduces live exactly.
- **Restart persistence (live only).** The live host loads `wallet_state` before the first event and saves the
  changed and evicted wallets every `Pipeline:WalletPersistSeconds` (default 300 s of event time) and at
  shutdown — in batches, never per trade (binary `COPY` + one upsert). A failed load starts from an empty memory
  (only the coverage is lost), a failed save is retried with the next one. If the first event after a restart
  comes more than 1 h after the last saved one, the coverage restarts (the outage left a hole: wallets that
  traded during it would look fresher than they are), so the history-dependent columns are NULL for 24 h again.
  A replay never reads or writes `wallet_state`: it may run alongside live ingestion.
- **Size.** Each wallet costs roughly 150–250 bytes in memory (address string + record + dictionary entry).
  Every save logs the wallets held and the managed heap (`saved wallet state: … wallets in memory, managed heap
  … MB`): the bound on one day of live traffic (memory, `pg_total_relation_size('wallet_state')`) is recorded
  here once measured on the deployed pipeline.
- **Out of scope here:** funding links between wallets (need Solana RPC, which is paid). Outcome-based
  reputation is built on this state: see *Wallet reputation* below.

### Wallet reputation (FE-T10, #190)

Wallet age says a buyer is new, not whether it is *good*. The same `WalletStore` derives behaviour statistics
from **outcomes** (never identities, so they stay valid as wallets rotate), and `TokenState` adds the launch's
bundling from its own buys:

| Column | Meaning |
|---|---|
| `rep_creator_settled_launches` | the creator's earlier launches (within the age cap) whose outcome is recorded |
| `rep_creator_rug_share` / `_pump_share` | share of those that rugged / pumped in their first hour |
| `rep_early_buyers` | the token's first distinct buyers seen (creator excepted, at most `Pipeline:ReputationEarlyBuyers` = 20; always written) |
| `rep_early_known_share` | share of them with ≥ `Pipeline:ReputationMinTrips` (3) closed round trips ("known"; the rest are "unknown") |
| `rep_early_win_rate` / `rep_early_pnl_sol` | the known early buyers' mean win rate and mean realized P&L per round trip (SOL) |
| `rep_smart_buy_share` | share of the window's SOL buy volume from smart wallets (known, win rate > `Pipeline:ReputationSmartWinRate` = 0.6) |
| `rep_smart_net_flow_sol` / `rep_smart_traders` | smart wallets' net SOL flow over the window, and how many traded in it — "good wallets are buying now", all along the token's life |
| `rep_bundle_create_block_share` / `_top_block_share` | share of the first 20 buys (with a `RawEvent.Block`) landing in the creation block / in one single block |

- **Round trips (realized P&L).** Per (wallet, token), SOL-quoted buys and sells only (as `buy_pressure`; the
  create's dev buy counts as a buy): buys add tokens and SOL cost, a sell realizes `SOL received − average cost
  of the tokens sold`, and the position closes — one round trip, a win when its P&L is > 0 — once ≥ 99 % of the
  tokens held are sold. A sell without a known buy (bought before the history, or not in SOL) is ignored. **A
  position that isn't closed counts as open, never as a loss.** Positions live in fixed UTC buckets of
  `Pipeline:ReputationPositionMaxHours` (6 h, aligned on the Unix epoch): one still open at a boundary is dropped
  (still never a loss). A wallet's record counts its trips **opened within the age cap minus one bucket (18 h)**,
  at most its latest 32.
- **Launch outcomes, label-safe.** A create starts tracking the token's SOL price over its first
  `Pipeline:ReputationOutcomeHorizonSeconds` (3 600 s, the longest label horizon). *Pumped*: highest price ≥
  (1 + `ReputationPumpThreshold` = 2.0) × first price (the 1 h pump label's threshold). *Rugged*: the price fell by
  ≥ `ReputationRugThreshold` = 0.7 **from its running peak** (the 1 h rug threshold; measured from the peak since a
  curve launch can't fall far below its launch price). A launch with no price in its hour is recorded as neither.
  The outcome is recorded on the creator only once **the horizon plus the label embargo**
  (`ReputationOutcomeEmbargoSeconds` = 3 600 s) is over, strictly before the snapshot: a snapshot never sees an
  outcome that wasn't known at its `ts`, and the outcome can't leak a label. Unsigned trades still feed the price
  track.
- **Point-in-time and live = replay.** Everything goes through the store's queue (events strictly before the
  snapshot, in timestamp order), so the snapshot cadence doesn't matter and a later event never changes an earlier
  snapshot. The reputation columns are NULL until the age cap (24 h) is covered, like the wallet ages; with
  `--warmup-hours 24` a replay reproduces the live values **exactly**: the outcomes counted are of launches created
  within the age cap, and the trips counted all opened after a bucket boundary the warm-up covers, so nothing open
  before the replay's start can change them (`WalletReputationTests` checks a 40 h stream against its last 24 h).
- **Restarts.** Closed round trips and recorded outcomes are saved with the wallet in `wallet_state` (0015). Open
  positions and launches still inside their hour are not: after a restart they start over (a few trips / outcomes
  of the restart's hours are lost).
- **Size.** Bounded like the rest of the memory: per wallet at most 32 trips (16 B each) and 16 outcomes; open
  positions live at most 6 h; launches are tracked for 2 h.
- **Overfitting watch.** A reputation feature can encode a handful of very active wallets, letting the model
  memorize them. The combined ablation (#198) reports, for the `wallet_reputation` group, the number of distinct
  creators and wallets behind the top-1 % entry selection before the group may be enabled.

### Token metadata (FE-T12, #196)

At launch a token's only information is its metadata. The `metadata` group reads it two ways:

| Column | Source | Meaning |
|---|---|---|
| `meta_coverage_sec` | stream | seconds of stream history behind the narrative counts, capped at 6 h (always written) |
| `meta_uri_content_addressed` | stream | 1 when the create's `uri` is content-addressed (IPFS CID behind any gateway, `ipfs://`, Arweave — the pump.fun UI default), 0 for a launch tool's own host (j7tracker, uxento, rapidlaunch...); NULL without a create / URI |
| `meta_name_matches_1h` | stream | other launches in the last hour sharing a normalized name or ticker (NULL before 1 h of history) |
| `meta_narrative_grads_3h` | stream | other tokens sharing one that graduated (`Migrate`) in the last 3 h, counted only when created at most 3 h before graduating (NULL before 6 h of history) |
| `meta_has_website` / `_twitter` / `_telegram` | fetched | the metadata JSON lists a non-blank link (top level, or Metaplex `extensions`); NULL unless known by the row's ts |

- **Stream part (free, live = replay).** pumpapi's `create` carries `name`, `symbol` and `uri` (verified on the
  2026-11-01 archives: every create of the hour). Names and tickers are normalized (`NarrativeKeys`: NFKD,
  diacritics / punctuation / spaces / emoji dropped, lower-cased, at least 2 chars: `$PEPE`, `Pépé`, `pepe 🐸` →
  `pepe`); a launch matches when any of its keys equals any of the token's, and the token itself never counts.
  `NarrativeTracker` counts only whole seconds strictly before the row's second, and bounds a graduation's
  creation look-back on timestamps, so any 6 h of history (a replay's `--warmup-hours 24` included) gives the live
  values exactly. The counts are dynamic: copycats launched after the token count once they happened.
- **Fetched part (live only, once per token).** On a `create`, live ingestion hands the token to
  `MetadataFetcher` with a non-blocking enqueue (a full queue of `Metadata:QueueCapacity` = 10 000 skips the
  token; ingestion never waits). The fetcher works in the background, at most `Metadata:MaxRequestsPerSecond` (4)
  starts and `MaxConcurrency` (4) in flight, each request bounded by `TimeoutSeconds` (8, body included) and
  `MaxResponseBytes` (64 KiB). Every outcome — `ok`, `failed` (unreachable, timeout, non-2xx, too large,
  malformed JSON, blocked address) or `unsupported` (no HTTPS URL) — is recorded **once, never retried**, with its
  completion time, in memory (`MetadataCache`) and in `token_metadata` (0016). Only the three flags are kept:
  never the URI, the links, the description or the image; failures are logged with the mint and a category only.
- **Gateways.** About 70 % of URIs point at `ipfs.io`, which now answers scripted reads with 429 ("switching to a
  service worker gateway"): any `/ipfs/<cid>` URI (and `ipfs://`) is rewritten to `Metadata:IpfsGateway`
  (`https://4everland.io/ipfs/`, ~0.8 s), Arweave ids to `ArweaveGateway`. Other hosts are fetched as given,
  HTTPS on the default port only.
- **SSRF guard.** The URI is chosen by whoever launched the token. The client (`PublicAddress.CreateHandler`)
  resolves the host itself and connects only to a public address (no loopback, private, link-local / cloud
  metadata, CGNAT, multicast, or IPv6 forms embedding one), never follows a redirect, never uses a proxy.
- **Point-in-time.** A row sees the flags only from `fetched_at` (the fetch's completion; wall clock, close to
  event time live) and for `Metadata:LinkHorizonHours` (48 h) after it. Live loads the last 48 h of results at
  startup; a replay **never fetches**: it loads the results whose `fetched_at` falls in its span (from 48 h before
  it) and so reproduces what live ingestion saw. **Consequence:** every window replayed from before the fetcher's
  deployment has the three link flags NULL; the link features are only measurable on windows ingested live after
  it (see docs/ML-MODEL.md, `metadata` results).
- **Cost.** About 1 650 creates / h, so ~40 k fetches / day on free public hosts (no pumpapi call, no SOL);
  `token_metadata` grows ~40 k small rows / day and is not in the retention purge yet (a few MB / day).
- **Config.** `Metadata:FetchEnabled` (default true; off: link flags stay NULL), `IpfsGateway`, `ArweaveGateway`,
  `MaxRequestsPerSecond`, `MaxConcurrency`, `TimeoutSeconds`, `MaxResponseBytes`, `QueueCapacity`,
  `LinkHorizonHours` (env `Metadata__FetchEnabled=false`, ...). A summary is logged every 1 000 fetches (ok /
  failed / unsupported / skipped / not stored).

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
| `--warmup-hours <n>` | Replay the `n` hours (0–336) before `--from` first, writing nothing dated before `--from`, so the state that needs history (wallet memory, market context, growth baselines) is warm when the window starts. The wallet features need **24** (see *Cross-token wallet memory*). The warm-up hours count in the summary and are mirrored by `--mirror-only`; the retention check applies to the window. |
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
- Wallet memory: state updates, TTL eviction, point-in-time ordering, restart persistence (engine + Postgres
  round trip), live/replay parity and the warm-up (`WalletStoreTests`, `WalletPersistenceTests`,
  `ReplayWalletTests`, `PostgresWalletStateStoreTests`).
- Wallet reputation: round trips (P&L, open positions, buckets), label-safe launch outcomes, early buyers, smart
  flow, bundling, unknown wallets, cadence independence and warm-up parity, persistence (`WalletReputationTests`,
  `PostgresWalletStateStoreTests`, `PostgresTokenSinkTests`).

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
