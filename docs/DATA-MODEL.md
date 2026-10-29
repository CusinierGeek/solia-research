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
| onchain_signals | JSONB | Pump.fun-specific signals |
| behavioral_signals | JSONB | pump/rug pattern flags |
| raw | JSONB | nullable debug payload |

Constraints: `UNIQUE (mint, ts)` (idempotent live + replay ingestion; also serves `(mint, ts)` lookups). Index: `ts`.

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
Paper-trading bot activity. One row per closed (or open) position.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| mint | TEXT | FK → tokens_history |
| prediction_id | BIGINT | FK → predictions (the triggering signal), `ON DELETE SET NULL` |
| opened_at | TIMESTAMPTZ | |
| closed_at | TIMESTAMPTZ | nullable while open |
| entry_price | NUMERIC(28,24) | |
| exit_price | NUMERIC(28,24) | nullable |
| size | NUMERIC(28,12) | |
| duration_sec | INT | nullable |
| pnl | NUMERIC(28,12) | realized P&L |
| pnl_pct | DOUBLE PRECISION | |
| strategy | TEXT | strategy used |
| confidence | DOUBLE PRECISION | confidence at entry |
| exit_reason | TEXT | `tp` / `sl` / `signal` / `timeout` |
| features_snapshot | JSONB | features at entry (for training) |
| status | TEXT | `open` / `closed` |

Indexes: `(mint)`, `(status)`, `(closed_at)`, `(prediction_id)`.

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

## Retention (reserved for Sprint 8)
- `features` and `logs` are high-volume → partition by day / apply TTL.
- `predictions` keep ~30 days hot, archive beyond.
- `predictions.feature_id` and `trades.prediction_id` are `ON DELETE SET NULL`, so deleting old rows is not blocked. Day-partitioning `features` would additionally require dropping the `predictions.feature_id` FK (a partitioned table can't be an FK target on `id` alone).
