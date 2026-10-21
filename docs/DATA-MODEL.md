# Data model (PostgreSQL 16)

> Reference schema. Authoritative DDL lives in `services/db/migrations` (implemented in Sprint 1). Types are indicative and may be refined during implementation.

## Conventions

- All timestamps are `TIMESTAMPTZ` (UTC).
- Primary keys are `BIGSERIAL` unless a natural key fits.
- A token is identified by its **mint address** (`mint TEXT`).
- Monetary/price values use `NUMERIC` to avoid float drift; feature vectors may use `DOUBLE PRECISION`.
- Soft relationships via `mint`; hard FKs where lifecycle is coupled.

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
| last_price | NUMERIC | latest observed price |
| last_market_cap | NUMERIC | |
| last_liquidity | NUMERIC | |
| holders | INT | latest holder count |
| trending_score | DOUBLE PRECISION | |
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
| price | NUMERIC | |
| volume | NUMERIC | |
| market_cap | NUMERIC | |
| holders | INT | |
| liquidity | NUMERIC | |
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
| onchain_signals | JSONB | Pump.fun-specific signals |
| behavioral_signals | JSONB | pump/rug pattern flags |
| raw | JSONB | nullable debug payload |

Indexes: `(mint, ts)`, `ts`.

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
| position_size | NUMERIC | suggested sizing |
| take_profit | NUMERIC | dynamic TP |
| stop_loss | NUMERIC | dynamic SL |
| confidence | DOUBLE PRECISION | 0–100 |
| feature_id | BIGINT | FK → features (input snapshot) |

Indexes: `(mint, ts)`, `(model_id)`.

### `trades`
Paper-trading bot activity. One row per closed (or open) position.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| mint | TEXT | FK → tokens_history |
| prediction_id | BIGINT | FK → predictions (the triggering signal) |
| opened_at | TIMESTAMPTZ | |
| closed_at | TIMESTAMPTZ | nullable while open |
| entry_price | NUMERIC | |
| exit_price | NUMERIC | nullable |
| size | NUMERIC | |
| duration_sec | INT | nullable |
| pnl | NUMERIC | realized P&L |
| pnl_pct | DOUBLE PRECISION | |
| strategy | TEXT | strategy used |
| confidence | DOUBLE PRECISION | confidence at entry |
| exit_reason | TEXT | `tp` / `sl` / `signal` / `timeout` |
| features_snapshot | JSONB | features at entry (for training) |
| status | TEXT | `open` / `closed` |

Indexes: `(mint)`, `(status)`, `(closed_at)`.

### `training_events`
Audit trail of training/retrain runs.

| Column | Type | Notes |
|--------|------|-------|
| id | BIGSERIAL PK | |
| started_at | TIMESTAMPTZ | |
| finished_at | TIMESTAMPTZ | nullable |
| trigger | TEXT | `initial` / `scheduled` / `n_trades` |
| n_samples | INT | training set size |
| candidate_model_id | BIGINT | FK → models |
| champion_model_id | BIGINT | FK → models (nullable) |
| metrics | JSONB | loss, AUC, precision/recall, etc. |
| decision | TEXT | `promoted` / `rejected` / `rollback` |
| notes | TEXT | |

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
