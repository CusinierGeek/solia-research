-- Solia — #165 before/after: the live paper trades (filled by the idealized simulator) recomputed under the
-- realistic fills (Solia.Bot Trading/Fills.cs, BotOptions defaults). Read-only.
--
--   docker exec -i solia-postgres-1 psql -U solia -d solia < docs/reports/paper-fills-165.sql
--
-- Each closed trade is joined with its token's last `features` row at opened_at (the pool it entered) and at
-- closed_at (the pool it exited into). "before" = the stored trade (fills at the mark). "after" = the same
-- entry/exit marks and times, but: the liquidity gate (venue floor), the size capped at 2 % of the pool's SOL
-- reserve (real + curve virtual SOL; dust < 0.001 skipped), x·y=k impact and the venue fee on both sides.
-- Approximation: entries/exits/sizes are otherwise unchanged — the capital and slots a skipped trade would
-- have freed are not re-traded.

WITH venue_cfg (venue, floor_sol, virtual_sol, fee) AS (
    VALUES ('pumpfun', 0.0, 30.0, 0.0125),
           ('raydium-launchpad', 0.0, 30.0, 0.01),
           ('meteora-launchpad', 0.0, 30.0, 0.01),
           ('bonk', 0.0, 30.0, 0.01),
           ('pumpswap', 1.0, 0.0, 0.003)
),
dflt AS (SELECT 1.0 AS floor_sol, 0.0 AS virtual_sol, 0.0025 AS fee),
t AS (
    SELECT tr.id, tr.entry_price AS entry_mark, tr.exit_price AS exit_mark, tr.size, tr.pnl,
           fo.venue AS venue_in, fo.liquidity AS liq_in, fc.venue AS venue_out, fc.liquidity AS liq_out
    FROM trades tr
    LEFT JOIN LATERAL (SELECT venue, liquidity FROM features f
                       WHERE f.mint = tr.mint AND f.ts <= tr.opened_at ORDER BY f.ts DESC LIMIT 1) fo ON TRUE
    LEFT JOIN LATERAL (SELECT venue, liquidity FROM features f
                       WHERE f.mint = tr.mint AND f.ts <= tr.closed_at ORDER BY f.ts DESC LIMIT 1) fc ON TRUE
    WHERE tr.status = 'closed'
),
c AS (
    SELECT t.*,
           COALESCE(i.floor_sol, d.floor_sol) AS floor_in,
           COALESCE(i.virtual_sol, d.virtual_sol) AS virt_in,
           COALESCE(i.fee, d.fee) AS fee_in,
           COALESCE(o.virtual_sol, d.virtual_sol) AS virt_out,
           COALESCE(o.fee, d.fee) AS fee_out,
           CASE WHEN i.virtual_sol > 0 THEN 'curve' ELSE 'amm' END
             || CASE WHEN t.liq_in IS NULL THEN ' unknown' WHEN t.liq_in < 1 THEN ' <1' WHEN t.liq_in < 10 THEN ' 1-10'
                     ELSE ' >=10' END AS bucket
    FROM t CROSS JOIN dflt d
    LEFT JOIN venue_cfg i ON i.venue = lower(trim(t.venue_in))
    LEFT JOIN venue_cfg o ON o.venue = lower(trim(t.venue_out))
),
sized AS (
    SELECT c.*, c.liq_in + c.virt_in AS reserve_in,
           trunc(LEAST(c.size, 0.02 * (c.liq_in + c.virt_in)), 9) AS new_size,
           (c.liq_in IS NOT NULL AND c.liq_in >= c.floor_in AND c.liq_in + c.virt_in > 0) AS tradeable
    FROM c
),
filled AS (
    SELECT s.*, (s.tradeable AND s.new_size >= 0.001) AS enters,
           COALESCE(s.liq_out, 0) + s.virt_out AS reserve_out,
           -- value at the exit mark of the tokens bought: n/P · R/(R+n) · P_exit
           s.new_size * (1 - s.fee_in) * s.reserve_in / (s.reserve_in + s.new_size * (1 - s.fee_in))
             * s.exit_mark / s.entry_mark AS value_out
    FROM sized s
),
after AS (
    SELECT f.*,
           CASE WHEN NOT f.enters THEN NULL
                WHEN f.reserve_out <= 0 THEN -f.new_size
                ELSE f.value_out * f.reserve_out / (f.reserve_out + f.value_out) * (1 - f.fee_out) - f.new_size
           END AS new_pnl
    FROM filled f
),
rows AS (
    SELECT bucket, size, pnl, enters, new_size, new_pnl FROM after
    UNION ALL
    SELECT 'all', size, pnl, enters, new_size, new_pnl FROM after
)
SELECT bucket,
       count(*)                                                        AS trades_before,
       round(sum(pnl), 3)                                              AS pnl_before,
       round(avg((pnl > 0)::int), 3)                                   AS win_before,
       round(sum(size), 2)                                             AS sol_in_before,
       count(*) FILTER (WHERE enters)                                  AS trades_after,
       round(COALESCE(sum(new_pnl), 0), 3)                             AS pnl_after,
       round(avg((new_pnl > 0)::int) FILTER (WHERE enters), 3)         AS win_after,
       round(COALESCE(sum(new_size) FILTER (WHERE enters), 0), 2)      AS sol_in_after
FROM rows
GROUP BY bucket
ORDER BY bucket = 'all', bucket;
