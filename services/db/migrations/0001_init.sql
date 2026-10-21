-- Solia — initial schema (Sprint 1 / T1, issue #3)
-- Source of truth for the Solia database. Authoritative reference: docs/DATA-MODEL.md.
-- Conventions: TIMESTAMPTZ (UTC), NUMERIC for price/money, DOUBLE PRECISION for ratios/
-- indicators, JSONB for signals/metadata/context. Applied by Solia.DbMigrator inside a
-- single transaction; the runner guarantees it runs at most once (idempotent re-runs).

-- ---------------------------------------------------------------------------
-- models — model registry / versioning (written by ai-service, Sprint 2/7)
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    id           BIGSERIAL   PRIMARY KEY,
    version      TEXT        NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    path         TEXT        NOT NULL,
    is_champion  BOOLEAN     NOT NULL DEFAULT FALSE,
    metrics      JSONB,
    hyperparams  JSONB,
    n_params     BIGINT
);
-- Only one champion at a time.
CREATE UNIQUE INDEX models_single_champion_idx ON models (is_champion) WHERE is_champion;

-- ---------------------------------------------------------------------------
-- tokens_history — immutable registry of every token ever seen (one per mint)
-- ---------------------------------------------------------------------------
CREATE TABLE tokens_history (
    mint          TEXT        PRIMARY KEY,
    symbol        TEXT,
    name          TEXT,
    created_at    TIMESTAMPTZ,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at  TIMESTAMPTZ,
    final_outcome TEXT             -- labeled later, nullable
                  CHECK (final_outcome IN ('pump', 'rug', 'neutral')),
    metadata      JSONB
);

-- ---------------------------------------------------------------------------
-- tokens_live — hot set of currently tracked tokens (frequently updated)
-- ---------------------------------------------------------------------------
CREATE TABLE tokens_live (
    mint             TEXT        PRIMARY KEY REFERENCES tokens_history (mint),
    symbol           TEXT,
    name             TEXT,
    created_at       TIMESTAMPTZ,
    last_price       NUMERIC,
    last_market_cap  NUMERIC,
    last_liquidity   NUMERIC,
    holders          INTEGER,
    trending_score   DOUBLE PRECISION,
    venue            TEXT,             -- 'pumpfun' / 'pumpswap' / 'raydium' / 'meteora' / ...
    is_migrated      BOOLEAN     NOT NULL DEFAULT FALSE,
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- features — time-series feature snapshots per token (the model's input unit)
-- ---------------------------------------------------------------------------
CREATE TABLE features (
    id                  BIGSERIAL   PRIMARY KEY,
    mint                TEXT        NOT NULL REFERENCES tokens_history (mint),
    ts                  TIMESTAMPTZ NOT NULL,
    venue               TEXT,                 -- pumpfun / pumpswap / raydium / meteora / ...
    is_migrated         BOOLEAN     NOT NULL DEFAULT FALSE,
    pool_address        TEXT,                 -- nullable; AMM pool once migrated
    price               NUMERIC,
    volume              NUMERIC,
    market_cap          NUMERIC,
    holders             INTEGER,
    liquidity           NUMERIC,
    buy_events          INTEGER,
    sell_events         INTEGER,
    buy_sell_ratio      DOUBLE PRECISION,
    token_age_sec       INTEGER,
    trending_score      DOUBLE PRECISION,
    social_mentions     INTEGER,
    holders_growth_rate DOUBLE PRECISION,
    volume_growth_rate  DOUBLE PRECISION,
    tx_creation_rate    DOUBLE PRECISION,
    volatility          DOUBLE PRECISION,
    ema                 DOUBLE PRECISION,
    rsi                 DOUBLE PRECISION,
    price_derivative    DOUBLE PRECISION,
    onchain_signals     JSONB,
    behavioral_signals  JSONB,
    raw                 JSONB,
    -- One snapshot per token per instant: live + replay ingestion can upsert idempotently.
    -- Also serves (mint, ts) range lookups.
    CONSTRAINT features_mint_ts_key UNIQUE (mint, ts)
);
CREATE INDEX features_ts_idx ON features (ts);

-- ---------------------------------------------------------------------------
-- predictions — model outputs, one row per inference call
-- ---------------------------------------------------------------------------
CREATE TABLE predictions (
    id            BIGSERIAL   PRIMARY KEY,
    mint          TEXT        NOT NULL REFERENCES tokens_history (mint),
    model_id      BIGINT      REFERENCES models (id),
    ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
    horizon       TEXT        NOT NULL
                  CHECK (horizon IN ('30s', '1m', '5m', '15m', '1h')),
    pump_prob     DOUBLE PRECISION CHECK (pump_prob BETWEEN 0 AND 1),
    rug_prob      DOUBLE PRECISION CHECK (rug_prob BETWEEN 0 AND 1),
    strategy      TEXT        CHECK (strategy IN ('buy', 'sell', 'hold')),
    position_size NUMERIC,
    take_profit   NUMERIC,
    stop_loss     NUMERIC,
    confidence    DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 100),
    -- SET NULL so features retention (Sprint 8) can delete old snapshots.
    feature_id    BIGINT      REFERENCES features (id) ON DELETE SET NULL
);
CREATE INDEX predictions_mint_ts_idx ON predictions (mint, ts);
CREATE INDEX predictions_model_idx   ON predictions (model_id);
CREATE INDEX predictions_feature_idx ON predictions (feature_id);

-- ---------------------------------------------------------------------------
-- trades — paper-trading bot activity, one row per position
-- ---------------------------------------------------------------------------
CREATE TABLE trades (
    id                BIGSERIAL   PRIMARY KEY,
    mint              TEXT        NOT NULL REFERENCES tokens_history (mint),
    -- SET NULL so predictions retention (Sprint 8) can delete old signals.
    prediction_id     BIGINT      REFERENCES predictions (id) ON DELETE SET NULL,
    opened_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at         TIMESTAMPTZ,
    entry_price       NUMERIC,
    exit_price        NUMERIC,
    size              NUMERIC,
    duration_sec      INTEGER,
    pnl               NUMERIC,
    pnl_pct           DOUBLE PRECISION,
    strategy          TEXT        CHECK (strategy IN ('buy', 'sell', 'hold')),
    confidence        DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 100),
    exit_reason       TEXT        CHECK (exit_reason IN ('tp', 'sl', 'signal', 'timeout')),
    features_snapshot JSONB,
    status            TEXT        NOT NULL DEFAULT 'open'
                      CHECK (status IN ('open', 'closed'))
);
CREATE INDEX trades_mint_idx      ON trades (mint);
CREATE INDEX trades_prediction_idx ON trades (prediction_id);
CREATE INDEX trades_status_idx    ON trades (status);
CREATE INDEX trades_closed_at_idx ON trades (closed_at);

-- ---------------------------------------------------------------------------
-- training_events — audit trail of training / retrain runs
-- ---------------------------------------------------------------------------
CREATE TABLE training_events (
    id                 BIGSERIAL   PRIMARY KEY,
    started_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at        TIMESTAMPTZ,
    trigger            TEXT        CHECK (trigger IN ('initial', 'scheduled', 'n_trades')),
    n_samples          INTEGER,
    candidate_model_id BIGINT      REFERENCES models (id),
    champion_model_id  BIGINT      REFERENCES models (id),
    metrics            JSONB,
    decision           TEXT        CHECK (decision IN ('promoted', 'rejected', 'rollback')),
    notes              TEXT
);

-- ---------------------------------------------------------------------------
-- logs — operational logs surfaced on the dashboard
-- ---------------------------------------------------------------------------
CREATE TABLE logs (
    id      BIGSERIAL   PRIMARY KEY,
    ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    service TEXT        NOT NULL,                -- 'data-pipeline' / 'ai-service' / 'backend' / 'bot'
    level   TEXT        NOT NULL CHECK (level IN ('debug', 'info', 'warn', 'error')),
    message TEXT        NOT NULL,
    context JSONB
);
CREATE INDEX logs_ts_idx            ON logs (ts);
CREATE INDEX logs_service_level_idx ON logs (service, level);
