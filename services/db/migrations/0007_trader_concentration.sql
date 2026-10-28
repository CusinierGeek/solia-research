-- Solia — holder-concentration proxy from trade signers (feature enrichment FE-T4, issue #113).
-- Rugs and fake pumps are often driven by a few wallets. Derived for free from the stream's txSigner
-- over the pipeline's rolling window (no on-chain holder snapshot): per-wallet volume is rebuilt each
-- snapshot from the window's trades, so it never grows with the token's lifetime. Shares are of the
-- signed volume in the current quote currency; trades without a signer are skipped. Earlier rows stay
-- NULL until a re-replay regenerates them; unique_traders is always written, so a NULL there marks
-- such a row.

ALTER TABLE features
    ADD COLUMN unique_traders   INT,              -- distinct signers in the window
    ADD COLUMN top_trader_share DOUBLE PRECISION, -- largest wallet's share of the window volume, (0, 1]
    ADD COLUMN trader_hhi       DOUBLE PRECISION, -- sum of squared per-wallet volume shares, (0, 1]
    ADD COLUMN new_trader_ratio DOUBLE PRECISION; -- share of the short window's traders not seen earlier in the window
