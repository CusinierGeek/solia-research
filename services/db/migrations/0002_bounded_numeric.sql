-- Solia — bound NUMERIC precision so every value fits a .NET System.Decimal (Sprint 4, PR #63 review).
-- Unconstrained NUMERIC accepts values (e.g. a 1e-9 price with 30+ fractional digits from Python or a
-- SQL division) that Npgsql cannot read into a decimal: the backend read throws OverflowException.
-- Precision 28 always fits a decimal; Postgres rounds extra fractional digits on write instead.
-- Units are quote-currency (SOL / USDC):
--   prices        NUMERIC(28,24): < 10 000 per token, 24 fractional digits (a 1e-12 price keeps 12 significant)
--   amounts / P&L NUMERIC(28,12): < 1e16 (market cap, liquidity, volume, sizes, pnl)

ALTER TABLE tokens_live
    ALTER COLUMN last_price      TYPE NUMERIC(28,24),
    ALTER COLUMN last_market_cap TYPE NUMERIC(28,12),
    ALTER COLUMN last_liquidity  TYPE NUMERIC(28,12);

ALTER TABLE features
    ALTER COLUMN price      TYPE NUMERIC(28,24),
    ALTER COLUMN volume     TYPE NUMERIC(28,12),
    ALTER COLUMN market_cap TYPE NUMERIC(28,12),
    ALTER COLUMN liquidity  TYPE NUMERIC(28,12);

ALTER TABLE predictions
    ALTER COLUMN position_size TYPE NUMERIC(28,12),
    ALTER COLUMN take_profit   TYPE NUMERIC(28,24),
    ALTER COLUMN stop_loss     TYPE NUMERIC(28,24);

ALTER TABLE trades
    ALTER COLUMN entry_price TYPE NUMERIC(28,24),
    ALTER COLUMN exit_price  TYPE NUMERIC(28,24),
    ALTER COLUMN size        TYPE NUMERIC(28,12),
    ALTER COLUMN pnl         TYPE NUMERIC(28,12);
