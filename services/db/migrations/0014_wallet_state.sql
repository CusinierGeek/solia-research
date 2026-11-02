-- Solia — cross-token wallet memory + wallet-age features (feature enrichment FE-T9, issue #189).
-- Every other feature is computed from one token's own trades; these say *who* trades it. The pipeline keeps a
-- per-wallet memory across tokens, derived for free from the stream's signers (pumpapi txSigner; no RPC, no paid
-- API). The model never sees an address, only behavioural statistics, so an unknown wallet simply counts as fresh.
-- Point-in-time: the wallet lookups of a row use only events strictly before its ts. Ages are capped at
-- Pipeline:WalletAgeCapHours (24 h) and the history-dependent columns stay NULL until that much history has been
-- observed (replay warm-up, see docs/DATA-PIPELINE.md); wal_coverage_sec is always written, so a NULL there marks a
-- row written before this migration.

ALTER TABLE features
    ADD COLUMN wal_coverage_sec              INT,              -- seconds of wallet history behind the lookups, capped at the age cap
    ADD COLUMN wal_fresh_1h_share            DOUBLE PRECISION, -- window signers first seen < 1 h ago (unknown = fresh), [0, 1]
    ADD COLUMN wal_fresh_24h_share           DOUBLE PRECISION, -- window signers first seen < 24 h ago, [0, 1]
    ADD COLUMN wal_median_trader_age_sec     DOUBLE PRECISION, -- median wallet age of the window signers, capped at the age cap
    ADD COLUMN wal_creator_age_sec           INT,              -- creator wallet age, capped (NULL: creator unknown or warm-up)
    ADD COLUMN wal_creator_prev_launches     INT,              -- creator's launches before this one within the age cap (max 16)
    ADD COLUMN wal_creator_buy_sol           DOUBLE PRECISION, -- creator's dev buy (the create's own buy), SOL
    ADD COLUMN wal_creator_buy_supply_share  DOUBLE PRECISION, -- dev buy tokens / supply
    ADD COLUMN wal_creator_sold_share        DOUBLE PRECISION, -- tokens the creator sold / bought on this launch, [0, 1]
    ADD COLUMN wal_creator_first_sell_sec    INT;              -- seconds since the creator's first sell (NULL: not sold)

-- The live pipeline's wallet memory, persisted in batches (never per trade) so a restart doesn't reset every
-- wallet to fresh. Written with raw SQL by the data-pipeline only (replays keep theirs in memory). Bounded by the
-- pipeline: a wallet inactive for longer than Pipeline:WalletTtlHours (14 days) is deleted.
CREATE TABLE wallet_state (
    wallet          TEXT        PRIMARY KEY,           -- signer address (never a model input)
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL,
    trade_count     INT         NOT NULL,
    tokens_traded   INT         NOT NULL,              -- distinct tokens traded (approximate: a re-entry after 4 other tokens recounts)
    tokens_created  INT         NOT NULL,
    recent_launches TIMESTAMPTZ[] NOT NULL DEFAULT '{}' -- latest creations, ascending, at most 16
);

-- Bounds of the history behind wallet_state (the features' coverage), one row.
CREATE TABLE wallet_state_meta (
    id            BOOLEAN     PRIMARY KEY DEFAULT TRUE CHECK (id),
    history_start TIMESTAMPTZ NOT NULL,
    last_event_at TIMESTAMPTZ NOT NULL,
    saved_at      TIMESTAMPTZ NOT NULL
);
