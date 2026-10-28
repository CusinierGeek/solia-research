-- Solia — momentum & acceleration features (feature enrichment FE-T3, issue #112).
-- Pumps are characterised by price and volume *speeding up*, which level features capture poorly.
-- All derived from the free stream in the pipeline's rolling window, over the configurable
-- Pipeline:Momentum{Short,Medium,Long}Seconds look-backs (defaults 10/30/60 s). NULL where the
-- window lacks the history (see docs/DATA-MODEL.md); price_history_sec says how far back it reaches,
-- so the model can tell "no history yet" from a flat price. Earlier rows stay NULL until a re-replay
-- regenerates them; volume_burst_ratio and price_history_sec are always written, so a NULL there
-- marks such a row.

ALTER TABLE features
    ADD COLUMN return_short        DOUBLE PRECISION, -- p_now / p_(now-short) - 1
    ADD COLUMN return_medium       DOUBLE PRECISION, -- p_now / p_(now-medium) - 1
    ADD COLUMN return_long         DOUBLE PRECISION, -- p_now / p_(now-long) - 1
    ADD COLUMN price_acceleration  DOUBLE PRECISION, -- short return minus the previous short window's
    ADD COLUMN ema_slope           DOUBLE PRECISION, -- per-second EMA change / previous EMA
    ADD COLUMN volume_acceleration DOUBLE PRECISION, -- change of per-second volume rate between short windows, per s
    ADD COLUMN volume_burst_ratio  DOUBLE PRECISION, -- short / long per-second volume rate (1 = steady)
    ADD COLUMN price_history_sec   DOUBLE PRECISION; -- how far back the price history reaches, capped at long
