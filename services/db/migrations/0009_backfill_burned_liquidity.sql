-- solia:no-transaction
-- Solia — backfill features.burned_liquidity (0008) from onchain_signals.burnedLiquidityPercent for
-- the rows written before it. Batched by id range, one commit per batch, so a large table isn't
-- locked and rewritten in a single transaction while ingestion is live. Rows inserted meanwhile are
-- filled by 0008's trigger. Re-runnable: it only touches rows still NULL with a numeric JSONB value.

DO $$
DECLARE
    batch CONSTANT BIGINT := 50000;
    lo BIGINT;
    hi BIGINT;
BEGIN
    SELECT min(id), max(id) INTO lo, hi FROM features;
    WHILE lo <= hi LOOP
        UPDATE features
        SET burned_liquidity = (onchain_signals ->> 'burnedLiquidityPercent')::DOUBLE PRECISION
        WHERE id >= lo AND id < lo + batch
            AND burned_liquidity IS NULL
            AND jsonb_typeof(onchain_signals -> 'burnedLiquidityPercent') = 'number';
        COMMIT;
        lo := lo + batch;
    END LOOP;
END
$$;
