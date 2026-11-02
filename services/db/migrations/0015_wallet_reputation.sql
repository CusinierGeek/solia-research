-- Solia — wallet reputation features (feature enrichment FE-T10, issue #190).
-- Wallet age (#189) says a buyer is new, not whether it is *good*. These come from outcomes, in the pipeline's
-- cross-token wallet memory (stream-derived, free): the creator's past launches that rugged / pumped, the realized
-- round-trip record (SOL-quoted buy -> sell) of the token's first buyers and of the wallets buying it now
-- ("smart money"), and whether the launch's first buys landed in one block (a bundle). Behaviour statistics, never
-- identities. Point-in-time: wallet lookups use only events strictly before the row's ts, and a launch's outcome
-- (its first hour, the longest label horizon) is recorded only after that hour plus the label embargo, so it can't
-- leak a label. The history-dependent columns stay NULL until Pipeline:WalletAgeCapHours (24 h) of history has been
-- observed (replay warm-up); rep_early_buyers is always written, so a NULL there marks a row written before this
-- migration.

ALTER TABLE features
    ADD COLUMN rep_creator_settled_launches   INT,              -- creator's earlier launches (age cap) with a recorded outcome
    ADD COLUMN rep_creator_rug_share          DOUBLE PRECISION, -- share of them that rugged, [0, 1]
    ADD COLUMN rep_creator_pump_share         DOUBLE PRECISION, -- share of them that pumped, [0, 1]
    ADD COLUMN rep_early_buyers               INT,              -- distinct first buyers seen (creator excepted), <= Pipeline:ReputationEarlyBuyers
    ADD COLUMN rep_early_known_share          DOUBLE PRECISION, -- share of them with >= Pipeline:ReputationMinTrips round trips, [0, 1]
    ADD COLUMN rep_early_win_rate             DOUBLE PRECISION, -- mean win rate of the rated early buyers, [0, 1]
    ADD COLUMN rep_early_pnl_sol              DOUBLE PRECISION, -- mean realized P&L per round trip of the rated early buyers, SOL
    ADD COLUMN rep_smart_buy_share            DOUBLE PRECISION, -- window SOL buy volume from smart wallets, [0, 1]
    ADD COLUMN rep_smart_net_flow_sol         DOUBLE PRECISION, -- window net SOL flow of smart wallets (buys - sells)
    ADD COLUMN rep_smart_traders              INT,              -- distinct smart wallets trading in the window
    ADD COLUMN rep_bundle_create_block_share  DOUBLE PRECISION, -- first buys landing in the creation block, [0, 1]
    ADD COLUMN rep_bundle_top_block_share     DOUBLE PRECISION; -- first buys landing in the busiest single block, [0, 1]

-- The reputation behind those columns, persisted with the rest of the live wallet memory (0014). Parallel arrays,
-- ascending by time: the latest closed round trips (first-buy time, realized SOL P&L; at most 32) and the latest
-- recorded launch outcomes (launch time, flags: 1 = rugged, 2 = pumped; at most 16). Open positions and launches
-- still inside their outcome horizon are not persisted (they start over after a restart).
ALTER TABLE wallet_state
    ADD COLUMN trip_opened      TIMESTAMPTZ[]      NOT NULL DEFAULT '{}',
    ADD COLUMN trip_pnl_sol     DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    ADD COLUMN outcome_launches TIMESTAMPTZ[]      NOT NULL DEFAULT '{}',
    ADD COLUMN outcome_flags    SMALLINT[]         NOT NULL DEFAULT '{}';
