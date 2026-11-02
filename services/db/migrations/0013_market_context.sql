-- Solia — market-regime context (feature enrichment FE-T11, issue #195).
-- Every other feature describes one token in isolation; these describe the market around it, so the
-- model can tell "this token is strong" from "the whole market is up". Derived for free from the
-- stream (no RPC, no paid API): market-wide rolling aggregates over every token, computed once per
-- snapshot tick and written on every row of that tick. Point-in-time: only whole seconds strictly
-- before the second of the row's ts count. Short window = 5 min, long = 1 h; rates are per minute,
-- SOL amounts count SOL-quoted trades only. An aggregate stays NULL until its whole window has been
-- observed (replay warm-up, see docs/DATA-PIPELINE.md); mkt_coverage_sec is always written, so a NULL
-- there marks a row written before this migration.

ALTER TABLE features
    ADD COLUMN mkt_coverage_sec        INT,              -- seconds of stream history behind the aggregates, capped at 3600
    ADD COLUMN mkt_launch_rate_5m      DOUBLE PRECISION, -- Create events per minute, last 5 min
    ADD COLUMN mkt_launch_rate_1h      DOUBLE PRECISION, -- Create events per minute, last hour
    ADD COLUMN mkt_graduations_1h      INT,              -- Migrate events (graduations), last hour
    ADD COLUMN mkt_graduation_ratio_1h DOUBLE PRECISION, -- graduations / launches, last hour (NULL without launches)
    ADD COLUMN mkt_buy_sol_5m          DOUBLE PRECISION, -- SOL buy volume across all tokens, last 5 min
    ADD COLUMN mkt_buy_sol_1h          DOUBLE PRECISION, -- SOL buy volume across all tokens, last hour
    ADD COLUMN mkt_net_sol_flow_5m     DOUBLE PRECISION, -- SOL buys minus sells across all tokens, last 5 min
    ADD COLUMN mkt_net_sol_flow_1h     DOUBLE PRECISION, -- SOL buys minus sells across all tokens, last hour
    ADD COLUMN mkt_active_tokens_5m    INT,              -- distinct tokens with a trade, last 5 min
    ADD COLUMN mkt_active_tokens_1h    INT;              -- distinct tokens with a trade, last hour
