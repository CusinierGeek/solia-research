-- solia:no-transaction
-- Solia — index for the retention of established tokens (#258, docs/DATA-PIPELINE.md § Data retention).
-- The purge keeps the rows of a token that reaches RETENTION_ESTABLISHED_AGE_HOURS (token_age_sec, >= 1 h)
-- beyond the general features TTL. Deciding that for one token is an index-only probe here: "a row of this
-- mint in [t, t + age] with token_age_sec >= age", instead of reading that token's rows from the heap.
-- Partial (only rows aged >= 1 h, a minority) so it stays small; the threshold is spliced as a literal by the
-- purge so the planner can prove the predicate. CONCURRENTLY: no lock blocking ingestion while it builds.
-- If a build fails it leaves an INVALID index that IF NOT EXISTS skips: DROP INDEX CONCURRENTLY it, re-run.
CREATE INDEX CONCURRENTLY IF NOT EXISTS features_established_idx
    ON features (mint, ts) INCLUDE (token_age_sec) WHERE token_age_sec >= 3600;
