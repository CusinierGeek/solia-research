-- Solia — holder-side supply concentration (feature enrichment FE-T13, issue #197).
-- The `concentration` group measures who TRADED in the window; this one approximates who HOLDS the supply, free,
-- from the stream: per token, each wallet's net token position = the create's dev buy + buys − sells + liquidity
-- removed − added + token transfers in − out (pumpapi `transfer` events). Pool accounts are excluded, so the
-- circulating supply is the sum of the positive positions. Point-in-time: only events dated at or before the row's ts.
-- Approximation limits (docs/DATA-PIPELINE.md): positions are complete only if the token's create is in the ingested
-- range (hs_incomplete = 1 otherwise); transfers pumpapi doesn't emit, trades credited to a router signer and the
-- smallest holders evicted past Pipeline:HolderMaxWallets are not reflected.
-- Flags are SMALLINT 0/1 so every feature column casts to float8. hs_holders and hs_incomplete are always written,
-- so a NULL there marks a row written before this migration. ADD COLUMN without a default is metadata-only.

ALTER TABLE features
    ADD COLUMN IF NOT EXISTS hs_holders        INT,              -- wallets holding a positive position
    ADD COLUMN IF NOT EXISTS hs_top1_share     DOUBLE PRECISION, -- largest holder's share of the circulating supply, [0, 1]; NULL while nothing circulates
    ADD COLUMN IF NOT EXISTS hs_top10_share    DOUBLE PRECISION, -- 10 largest holders' share of the circulating supply, [0, 1]; NULL while nothing circulates
    ADD COLUMN IF NOT EXISTS hs_top10_net_flow DOUBLE PRECISION, -- their net token flow over the window / circulating supply (negative = distributing)
    ADD COLUMN IF NOT EXISTS hs_large_holders  INT,              -- wallets holding > Pipeline:HolderLargeShare (2 %) of the total supply; NULL when unknown
    ADD COLUMN IF NOT EXISTS hs_incomplete     SMALLINT;         -- 1 = the token's create wasn't seen: positions incomplete
