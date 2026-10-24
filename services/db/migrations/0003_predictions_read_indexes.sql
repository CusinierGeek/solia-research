-- Solia — indexes for the prediction reads (Sprint 5, PR #82 review).
-- predictions grows by one row per inference call; the API reads it three ways, each now an ordered
-- index scan (read backward for the DESC orders) instead of a scan + sort of the whole table:
--   /predictions/live            latest per (mint, horizon)  → (mint, horizon, ts, id)
--   /predictions/history?mint=   one token, newest first     → (mint, ts, id)
--   /predictions/history         everything, newest first    → (ts, id)
-- id is the tiebreak of every order, so it closes each key.

DROP INDEX predictions_mint_ts_idx;

CREATE INDEX predictions_mint_ts_idx         ON predictions (mint, ts, id);
CREATE INDEX predictions_mint_horizon_ts_idx ON predictions (mint, horizon, ts, id);
CREATE INDEX predictions_ts_idx              ON predictions (ts, id);
