-- Solia — SOL-volume buy pressure features (feature enrichment FE-T2, issue #111).
-- Count-based buy pressure can't tell one whale buy from ten dust buys; these SOL-weighted columns can.
-- All derived from the free stream over the snapshot window, counting SOL-quoted trades only.
-- Amounts in SOL, bounded like every amount since 0002 (NUMERIC(28,12)). NULL where undefined.
-- Earlier rows stay NULL until a re-replay regenerates them (the model imputes NULL to 0).

ALTER TABLE features
    ADD COLUMN net_sol_flow     NUMERIC(28,12),   -- buy - sell SOL volume (negative under net selling)
    ADD COLUMN buy_volume_ratio DOUBLE PRECISION, -- buy SOL volume / total SOL volume, in [0, 1]
    ADD COLUMN avg_buy_sol      NUMERIC(28,12),   -- mean buy size
    ADD COLUMN max_buy_sol      NUMERIC(28,12),   -- largest buy
    ADD COLUMN large_buy_count  INTEGER;          -- buys >= the pipeline's LargeBuySol threshold
