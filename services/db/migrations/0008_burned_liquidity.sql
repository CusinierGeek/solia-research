-- Solia — burned liquidity as a first-class feature (feature enrichment FE-T5, issue #114).
-- The share of an AMM pool's LP tokens burned at pool creation is a strong rug signal (0% burned
-- = the deployer can pull the liquidity), but it was buried in onchain_signals JSONB, invisible to
-- the model. NULL = unknown (no pool-creation event seen, e.g. still on the bonding curve).
-- The pipeline still mirrors it in onchain_signals.burnedLiquidityPercent, so the column is kept
-- equal to that key: a trigger fills it on insert when the writer left it NULL (a pipeline binary
-- older than this migration, during a rolling deploy), and 0009 backfills the earlier rows in
-- batches, outside this transaction. The venue one-hot is encoded at load time (ai-service
-- features.py), not stored.

ALTER TABLE features
    ADD COLUMN burned_liquidity DOUBLE PRECISION; -- % of the pool's LP burned, [0, 100]; NULL = unknown

CREATE FUNCTION features_fill_burned_liquidity() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.burned_liquidity IS NULL
        AND jsonb_typeof(NEW.onchain_signals -> 'burnedLiquidityPercent') = 'number' THEN
        NEW.burned_liquidity := (NEW.onchain_signals ->> 'burnedLiquidityPercent')::DOUBLE PRECISION;
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER features_fill_burned_liquidity
    BEFORE INSERT ON features
    FOR EACH ROW EXECUTE FUNCTION features_fill_burned_liquidity();
