# Data model (PostgreSQL 16)

> Reference schema. Authoritative DDL lives in `services/db/migrations` (implemented in Sprint 1). Types are indicative and may be refined during implementation.

## Conventions

- All timestamps are `TIMESTAMPTZ` (UTC).
- Primary keys are `BIGSERIAL` unless a natural key fits.
- A token is identified by its **mint address** (`mint TEXT`).
- Monetary/price values use `NUMERIC` to avoid float drift; feature vectors may use `DOUBLE PRECISION`.
  Precision is bounded to 28 digits (migration `0002`) so every value fits a .NET `decimal`: prices
  `NUMERIC(28,24)`, amounts/P&L `NUMERIC(28,12)`. Feature prices and amounts are in **SOL**: a token's
  trades on pools quoted in another currency (USDC, other tokens) count as buys/sells but never set its
  price, cap, liquidity, pool or volume (see `features` below).
- Soft relationships via `mint`; hard FKs where lifecycle is coupled.
- Enumerated `TEXT` columns (status, strategy, horizon, level, …) and probability/confidence ranges are enforced with `CHECK` constraints; the values listed below are the only accepted ones.

## Entity relationships

```
tokens_history (mint) 1──┬──* features
                         ├──* predictions
                         └──* trades
models (id) 1──* predictions
models (id) 1──* training_events
models (id) 1──* bot_sessions 1──* trades
tokens_live  = hot view / table of currently tracked tokens
logs         = standalone
```

## Tables

### `tokens_live`
Currently tracked (hot) tokens. Small, frequently updated.

| Column | Type | Notes |
|--------|------|-------|
| mint | TEXT PK | token mint address |
| symbol | TEXT | |
| name | TEXT | |
| created_at | TIMESTAMPTZ | token creation time |
| last_price | NUMERIC(28,24) | latest observed price |
| last_market_cap | NUMERIC(28,12) | |
| last_liquidity | NUMERIC(28,12) | |
| holders | INT | latest holder count |
| trending_score | DOUBLE PRECISION | |
| venue | TEXT | current venue (`pumpfun` / `pumpswap` / `raydium` / ...) |
| is_migrated | BOOLEAN | graduated off the bonding curve |
| is_active | BOOLEAN | still being tracked |
| updated_at | TIMESTAMPTZ | |

### `tokens_history`
Immutable registry of every token we have ever seen (one row per mint).

| Column | Type | Notes |
|--------|------|-------|
| mint | TEXT PK | |
| symbol | TEXT | |
| name | TEXT | |
| created_at | TIMESTAMPTZ | |
| first_seen_at | TIMESTAMPTZ | when pipeline first ingested it |
| last_seen_at | TIMESTAMPTZ | |
| final_outcome | TEXT | nullable: `pump` / `rug` / `neutral` (labeled later) |
| metadata | JSONB | raw source metadata |

### `features`
Time-series feature snapshots per token. The model's input unit.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| mint | TEXT | FK → tokens_history |
| ts | TIMESTAMPTZ | snapshot time |
| venue | TEXT | `pumpfun` / `pumpswap` / `raydium` / `meteora` / ... — one-hot encoded for the model at load time (ai-service `VENUE_COLUMNS`), not stored |
| is_migrated | BOOLEAN | token has graduated off the bonding curve |
| pool_address | TEXT | nullable; the SOL-quoted pool of the latest SOL trade (AMM pool once migrated) |
| price | NUMERIC(28,24) | in SOL, from SOL-quoted trades only (a migrated token trades on several pools, each in its own quote currency); NULL for a token never traded against SOL |
| volume | NUMERIC(28,12) | SOL volume of the window |
| market_cap | NUMERIC(28,12) | in SOL, from SOL-quoted trades |
| holders | INT | |
| liquidity | NUMERIC(28,12) | SOL in the SOL-quoted pool |
| buy_events | INT | in window |
| sell_events | INT | in window |
| buy_sell_ratio | DOUBLE PRECISION | |
| token_age_sec | INT | |
| trending_score | DOUBLE PRECISION | |
| social_mentions | INT | nullable |
| holders_growth_rate | DOUBLE PRECISION | d(holders)/dt |
| volume_growth_rate | DOUBLE PRECISION | |
| tx_creation_rate | DOUBLE PRECISION | tx/sec |
| volatility | DOUBLE PRECISION | |
| ema | DOUBLE PRECISION | |
| rsi | DOUBLE PRECISION | |
| price_derivative | DOUBLE PRECISION | first time-derivative |
| net_sol_flow | NUMERIC(28,12) | buy − sell SOL volume in window (SOL-quoted trades only) |
| buy_volume_ratio | DOUBLE PRECISION | buy SOL volume / total SOL volume; NULL without SOL trades |
| avg_buy_sol | NUMERIC(28,12) | mean buy size (SOL); NULL without SOL buys |
| max_buy_sol | NUMERIC(28,12) | largest buy (SOL); NULL without SOL buys |
| large_buy_count | INT | buys ≥ `Pipeline:LargeBuySol` (default 1 SOL) |
| return_short | DOUBLE PRECISION | p_now / p_(now−short) − 1 (`Pipeline:MomentumShortSeconds`, default 10 s); NULL if the price history doesn't reach back (see `price_history_sec`) |
| return_medium | DOUBLE PRECISION | same over `MomentumMediumSeconds` (default 30 s) |
| return_long | DOUBLE PRECISION | same over `MomentumLongSeconds` (default 60 s) |
| price_acceleration | DOUBLE PRECISION | short return − the short return one short window earlier; NULL without 2 short windows of prices |
| ema_slope | DOUBLE PRECISION | per-second change of the EMA as of the snapshot time / previous snapshot's EMA; NULL on the first snapshot or after a quote change |
| volume_acceleration | DOUBLE PRECISION | (volume rate of the last short window − of the one before) / short, quote/s²; NULL until 2 short windows observed |
| volume_burst_ratio | DOUBLE PRECISION | short / long per-second volume rate (rated over the observed span when shorter); 1 when nothing traded over the long window. Always written |
| price_history_sec | DOUBLE PRECISION | how far back (s) the momentum price history reaches, capped at `MomentumLongSeconds`; 0 without a price. A return over a longer look-back is NULL. Always written |
| unique_traders | INT | distinct signers (`txSigner`) of the window's SOL-quoted trades; trades without a signer are skipped. Always written |
| top_trader_share | DOUBLE PRECISION | largest single wallet's share of the window's signed SOL volume (non-positive amounts carry none), (0, 1]; NULL without signed volume |
| trader_hhi | DOUBLE PRECISION | Herfindahl-Hirschman index of per-wallet volume shares (Σ share²), (0, 1]: 1 = one wallet, 1/n = volume split evenly over n wallets; NULL without signed volume |
| new_trader_ratio | DOUBLE PRECISION | share of the traders of the last `MomentumShortSeconds` first seen on the token in that span (the rest are returning, even from before the window; past `HolderCounter.ExactLimit` wallets, first seen in the window), [0, 1]; NULL without a signed trade in that span |
| burned_liquidity | DOUBLE PRECISION | % [0, 100] of the AMM pool's LP burned, from the current pool's latest event reporting it (reset when the token moves to another pool); NULL when unknown (none seen, e.g. still on the curve, or malformed). Kept equal to `onchain_signals.burnedLiquidityPercent`: 0008's insert trigger fills it when a writer omits it, 0009 backfills earlier rows in batches |
| mkt_coverage_sec | INT | market context (FE-T11, 0013): seconds of stream history behind the `mkt_*` aggregates, capped at 3600; always written (NULL = row predates 0013) |
| mkt_launch_rate_5m, mkt_launch_rate_1h | DOUBLE PRECISION | market-wide `Create` events per minute over the last 5 min / 1 h |
| mkt_graduations_1h | INT | market-wide graduations (`Migrate` events) over the last hour |
| mkt_graduation_ratio_1h | DOUBLE PRECISION | graduations / launches over the last hour; NULL without a launch |
| mkt_buy_sol_5m, mkt_buy_sol_1h | DOUBLE PRECISION | market-wide SOL buy volume (SOL-quoted trades) |
| mkt_net_sol_flow_5m, mkt_net_sol_flow_1h | DOUBLE PRECISION | market-wide SOL buys minus sells |
| mkt_active_tokens_5m, mkt_active_tokens_1h | INT | distinct tokens with a trade in the window |
| wal_coverage_sec | INT | wallet features (FE-T9, 0014): seconds of wallet history behind the lookups, capped at `Pipeline:WalletAgeCapHours` (24 h); always written (NULL = row predates 0014) |
| wal_fresh_1h_share, wal_fresh_24h_share | DOUBLE PRECISION | share of the window's signers first seen less than 1 h / 24 h before `ts` (an unknown wallet is fresh), [0, 1]; NULL without a signer or before 1 h / 24 h of history |
| wal_median_trader_age_sec | DOUBLE PRECISION | median wallet age of the window's signers, capped at the age cap; NULL without a signer or before the age cap is covered |
| wal_creator_age_sec | INT | creator wallet age (signer of the token's `Create`), capped; NULL when no create was seen or before the age cap is covered |
| wal_creator_prev_launches | INT | the creator's launches before this one within the age cap (at most 16); NULL like `wal_creator_age_sec` |
| wal_creator_buy_sol | DOUBLE PRECISION | the creator's dev buy (the create's own buy), SOL; 0 without one; NULL when the creator is unknown |
| wal_creator_buy_supply_share | DOUBLE PRECISION | dev buy tokens / supply; 0 without a dev buy; NULL when the creator or supply is unknown |
| wal_creator_sold_share | DOUBLE PRECISION | tokens the creator sold / bought on this launch, [0, 1]; NULL when it bought none |
| wal_creator_first_sell_sec | INT | seconds since the creator's first sell on this launch; NULL while it hasn't sold |
| rep_creator_settled_launches | INT | wallet reputation (FE-T10, 0015): the creator's launches before this one (within the age cap) whose outcome is recorded (at most 16); 0 for an unknown creator; NULL without a create or before the age cap is covered |
| rep_creator_rug_share, rep_creator_pump_share | DOUBLE PRECISION | share of those launches that rugged / pumped in their first hour, [0, 1]; NULL without one |
| rep_early_buyers | INT | distinct first buyers seen (the creator excepted), at most `Pipeline:ReputationEarlyBuyers` (20); always written (NULL = row predates 0015) |
| rep_early_known_share | DOUBLE PRECISION | share of them with at least `Pipeline:ReputationMinTrips` (3) closed round trips; NULL without a buyer or before the age cap is covered |
| rep_early_win_rate, rep_early_pnl_sol | DOUBLE PRECISION | the rated early buyers' mean win rate, [0, 1], and mean realized P&L per round trip, SOL; NULL when none is rated |
| rep_smart_buy_share | DOUBLE PRECISION | share of the window's SOL buy volume from smart wallets (rated, win rate > `Pipeline:ReputationSmartWinRate`, 0.6), [0, 1]; NULL without a SOL buy or before the age cap is covered |
| rep_smart_net_flow_sol, rep_smart_traders | DOUBLE PRECISION, INT | smart wallets' net SOL flow (buys − sells) over the window, and how many traded in it; NULL before the age cap is covered |
| rep_bundle_create_block_share, rep_bundle_top_block_share | DOUBLE PRECISION | share of the first buys (those reporting a block) landing in the creation block / in the busiest single block, [0, 1]; NULL without such a buy (the first also without a creation block) |
| onchain_signals | JSONB | Pump.fun-specific signals |
| behavioral_signals | JSONB | pump/rug pattern flags |
| raw | JSONB | nullable debug payload |

Constraints: `UNIQUE (mint, ts)` (idempotent live + replay ingestion; also serves `(mint, ts)` lookups). Index: `ts`.

The `mkt_*` columns are the same on every row of a pipeline tick (computed once per tick from whole
seconds strictly before the second of `ts`) and NULL until their window has been fully observed
(replay warm-up, see docs/DATA-PIPELINE.md "Market-regime context").

The `wal_*` columns come from the pipeline's cross-token wallet memory: the wallet lookups of a row use only
events strictly before its `ts`; the history-dependent ones stay NULL until `Pipeline:WalletAgeCapHours` of
history has been observed (replay warm-up, see docs/DATA-PIPELINE.md "Cross-token wallet memory"). The
`rep_*` columns come from the same memory (see docs/DATA-PIPELINE.md "Wallet reputation"); a launch outcome is
visible only once its first hour plus the label embargo is over.

### `wallet_state`, `wallet_state_meta`
The live pipeline's cross-token wallet memory (FE-T9, 0014), so a restart doesn't reset every wallet to fresh.
Written with raw SQL by the data-pipeline only, in batches (binary `COPY` into a staging table + one upsert,
every `Pipeline:WalletPersistSeconds`); replays keep theirs in memory and never write here. No EF mapping
(nothing reads it but the pipeline).

| Column | Type | Notes |
|--------|------|-------|
| wallet | TEXT PK | signer address (pumpapi `txSigner`); never a model input |
| first_seen_at, last_seen_at | TIMESTAMPTZ | first / latest trade or create seen (second precision) |
| trade_count | INT | trades signed |
| tokens_traded | INT | distinct tokens traded (approximate: a token re-entered after 4 others counts again) |
| tokens_created | INT | `Create` events signed |
| recent_launches | TIMESTAMPTZ[] | latest creations, ascending, at most 16 |
| trip_opened, trip_pnl_sol | TIMESTAMPTZ[], DOUBLE PRECISION[] | latest closed round trips (FE-T10, 0015), parallel arrays ascending by first buy, at most 32: when each opened, its realized SOL P&L |
| outcome_launches, outcome_flags | TIMESTAMPTZ[], SMALLINT[] | latest recorded launch outcomes (FE-T10, 0015), parallel arrays ascending by launch, at most 16: launch time, flags (1 = rugged, 2 = pumped) |

`wallet_state_meta` is one row (`id` = TRUE): `history_start` (start of the history behind the state — the
features' coverage), `last_event_at` (newest event applied), `saved_at`. Bounded by the pipeline: a wallet
inactive for longer than `Pipeline:WalletTtlHours` (14 d, the `features` retention) is deleted at the next save.

### `predictions`
Model outputs. One row per inference call.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| mint | TEXT | FK → tokens_history |
| model_id | BIGINT | FK → models |
| ts | TIMESTAMPTZ | inference time |
| horizon | TEXT | `30s` / `1m` / `5m` / `15m` / `1h` |
| pump_prob | DOUBLE PRECISION | 0–1 |
| rug_prob | DOUBLE PRECISION | 0–1 |
| strategy | TEXT | `buy` / `sell` / `hold` |
| position_size | NUMERIC(28,12) | suggested sizing |
| take_profit | NUMERIC(28,24) | dynamic TP |
| stop_loss | NUMERIC(28,24) | dynamic SL |
| confidence | DOUBLE PRECISION | 0–100 |
| feature_id | BIGINT | FK → features (input snapshot), `ON DELETE SET NULL` |

Indexes: `(mint, ts)`, `(model_id)`, `(feature_id)`.

### `trades`
Paper-trading bot activity. One row per closed (or open) position. Amounts are SOL; trades from before #165
(2026-10-30) were filled at the mark with no fees (see docs/ML-MODEL.md "Realistic fills").

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| mint | TEXT | FK → tokens_history |
| prediction_id | BIGINT | FK → predictions (the triggering signal), `ON DELETE SET NULL` |
| opened_at | TIMESTAMPTZ | |
| closed_at | TIMESTAMPTZ | nullable while open |
| entry_price | NUMERIC(28,24) | SOL per token, all-in fill price (with realistic fills: mark + impact + entry fee) |
| exit_price | NUMERIC(28,24) | nullable; SOL per token the exit filled at (after impact + exit fee) |
| size | NUMERIC(28,12) | SOL committed (the bot's `StartingBalance` is SOL too) |
| duration_sec | INT | nullable |
| pnl | NUMERIC(28,12) | realized P&L in SOL, fees and slippage included |
| pnl_pct | DOUBLE PRECISION | |
| strategy | TEXT | strategy used |
| confidence | DOUBLE PRECISION | confidence at entry |
| exit_reason | TEXT | `tp` / `sl` / `signal` / `timeout` |
| features_snapshot | JSONB | features at entry (for training) |
| status | TEXT | `open` / `closed` |
| session_id | BIGINT | FK → bot_sessions: the session the trade was opened in (nullable: rows written outside the bot) |

Indexes: `(mint)`, `(status)`, `(closed_at)`, `(prediction_id)`, `(session_id, closed_at)`.

### `bot_sessions`
A run of the paper bot under one configuration and one champion (migration 0012, #169), so trades are read per
regime. Written only by the bot (`PostgresBotSessionStore`, raw SQL). On every start the bot fingerprints its
effective options (normalized JSON of `BotOptions` minus `SessionLabel`, plus the traded horizon) and the champion
version: same fingerprint as the current session → it continues (across restarts); different → the current one
ends and a new one starts. The bot re-checks every 30 s before opening a trade, so a promotion while it runs also
opens a new session; a trade keeps the session it was opened in. Backfill: trades opened before 2026-10-30
20:26:42 UTC → `naive fills (pre-#165)` (ended), later ones → `realistic fills (#165)` (current, adopted by the
bot's first start if its champion is unchanged).

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| started_at | TIMESTAMPTZ | default `now()` |
| ended_at | TIMESTAMPTZ | NULL for the current session (at most one: partial unique index) |
| config_hash | TEXT | SHA-256 of normalized options + champion version; NULL when backfilled |
| options | JSONB | effective bot options at the start (entry rule, exits, fill model, sizing, horizon) |
| model_id | BIGINT | FK → models, `ON DELETE SET NULL`: the champion at the start |
| model_version | TEXT | |
| label | TEXT | optional human label (`Bot:SessionLabel`, applied when a new session starts) |
| resumed_after_trade_id | BIGINT | open trades with `id ≤` the current session's value were left by an earlier bot run: orphans |

### `training_events`
Audit trail of training/retrain runs.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| started_at | TIMESTAMPTZ | |
| finished_at | TIMESTAMPTZ | nullable |
| trigger | TEXT | `initial` / `scheduled` / `n_trades` / `manual` (`POST /model/retrain`; added by 0010) |
| n_samples | INT | training set size |
| candidate_model_id | BIGINT | FK → models |
| champion_model_id | BIGINT | FK → models (nullable) |
| metrics | JSONB | loss, AUC, precision/recall, etc. |
| decision | TEXT | `promoted` / `rejected` / `rollback` / `failed` (added by 0010); NULL while in flight — set together with `finished_at` (CHECK, 0010) |
| notes | TEXT | |

Every run is recorded, including the ones that crash or error: those get `finished_at` + decision
`failed`. `decision` and `finished_at` are both NULL while a run is in flight and both set once it ends
(`training_events_finished_decision_check`; 0010 backfilled older rows: finished without a decision →
`failed`, decided without `finished_at` → closed at `started_at`). Only a run that trained a model on the
new trades resets the trades-since-last-train count: `promoted` or `rejected` (a rejected candidate still
consumed them). A `failed` run produced nothing usable and a `rollback` only restores the previous
champion, so neither does. The backend (`TrainingDecisions.ResetTradeCount`) and the ai-service
(`registry.RESET_TRADE_COUNT_DECISIONS`, used by `registry.last_train_cutoff`) derive this set from the
decisions minus the no-model ones; parity tests compare both to the CHECKs the migrated database enforces.

### `models`
Model registry / versioning.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| version | TEXT UNIQUE | semantic or incremental version |
| created_at | TIMESTAMPTZ | |
| path | TEXT | checkpoint location (`MODEL_DIR/...`) |
| is_champion | BOOLEAN | the currently served model |
| metrics | JSONB | validation metrics at promotion |
| hyperparams | JSONB | config snapshot |
| n_params | BIGINT | |
| pruned_at | TIMESTAMPTZ | when the retention policy deleted the checkpoint file (added by 0011); NULL = never pruned. The row is kept for its history; never set on the champion (`models_champion_not_pruned_check`), and a rollback to a pruned model is refused |

Only one row should have `is_champion = true`.

### `logs`
Operational logs surfaced on the dashboard.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| ts | TIMESTAMPTZ | |
| service | TEXT | `data-pipeline` / `ai-service` / `backend` / `bot` |
| level | TEXT | `debug` / `info` / `warn` / `error` |
| message | TEXT | |
| context | JSONB | structured fields |

Index: `(ts)`, `(service, level)`.

## Retention
TTL purge by the data-pipeline (details and knobs: [DATA-PIPELINE.md § Data retention](DATA-PIPELINE.md#data-retention)):
`features` 14 days, `logs` 14 days, `predictions` 30 days; `trades`, `models`, `training_events`,
`tokens_history`, `tokens_live` are never purged. `predictions.feature_id` and `trades.prediction_id` are
`ON DELETE SET NULL`, so a purge never blocks on, nor deletes, the rows that reference purged ones.

**Decision: batched `DELETE` + `VACUUM`, not day partitions** (measured 2026-10-30, PostgreSQL 16, a 2 M-row
`features` table shaped like the real one, 10 % of its rows referenced by `predictions`):
- deleting the oldest 1 M rows (≈ 1 h of live ingestion) in 10 000-row batches: 67 s (~15 k rows/s, incl. the
  `SET NULL` of 100 k predictions), worst batch 3.2 s; `VACUUM (ANALYZE)` afterwards: 7 s;
- re-inserting 1 M rows after the vacuum: the heap stayed at 868 MB (space reused), the indexes grew 2 %.

An hourly purge therefore costs ~1 min/h of background work, with no lock that blocks writers. Day partitions
would make it an instant `DROP`, but need a full rewrite of the existing table (35 GB on the live database) and
dropping the `predictions.feature_id` FK, for a cost that isn't a bottleneck. Revisit if the ingest rate grows
by an order of magnitude. Retention needs no schema change: `features_ts_idx`, `logs_ts_idx` and
`predictions_ts_idx` already serve the oldest-first batches. The `VACUUM` runs with `PARALLEL 0`: parallel
index vacuuming needs dynamic shared memory, which Docker's default 64 MB `/dev/shm` can't provide on a large table.
