"""Canonical model feature spec.

The single source of truth for *which* columns of the ``features`` table (written by the .NET
pipeline) feed the model, and in what order. Parity tests check this list against the schema the
migrations produce, so the two can never drift silently. NULLs are imputed to 0 (absence of
signal) unless :data:`NULL_FILL` gives the column a neutral value; a missing column or a
non-numeric value is an error, never silently zeroed. Rows written before a group's migration
(NULL where the pipeline now always writes a value) are refused for that group, see
:func:`check_group_coverage`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

# FE-T5 (#114): the `venue` column, one-hot encoded at load time rather than in the database (no
# schema churn when the list changes; the raw venue stays the single stored fact). Fixed category
# list: the canonical venues the pipeline emits (PumpApiVenueMap); anything else — a new venue,
# orca/sunrise, or NULL — falls in the `venue_other` bucket. Appending a venue appends a column.
VENUES: tuple[str, ...] = (
    "pumpfun",
    "pumpswap",
    "raydium-launchpad",
    "raydium",
    "meteora-launchpad",
    "meteora",
    "bonk",
)
VENUE_OTHER = "venue_other"
VENUE_COLUMNS: tuple[str, ...] = (
    *(f"venue_{v.replace('-', '_')}" for v in VENUES),
    VENUE_OTHER,
)

# FE-T11 (#195): market-regime context — market-wide rolling aggregates over every token of the
# stream (5 min / 1 h), the same on every row of a pipeline tick. Point-in-time: only whole seconds
# strictly before the row's ts count. NULL while the pipeline hasn't observed the whole window yet
# (replay warm-up); mkt_coverage_sec, always written, tells how much it has.
MARKET_COLUMNS: tuple[str, ...] = (
    "mkt_coverage_sec",
    "mkt_launch_rate_5m",
    "mkt_launch_rate_1h",
    "mkt_graduations_1h",
    "mkt_graduation_ratio_1h",
    "mkt_buy_sol_5m",
    "mkt_buy_sol_1h",
    "mkt_net_sol_flow_5m",
    "mkt_net_sol_flow_1h",
    "mkt_active_tokens_5m",
    "mkt_active_tokens_1h",
)

# FE-T9 (#189): wallet features — who trades the token, from the pipeline's cross-token wallet
# memory (the model never sees an address): freshness of the window's signers and the creator's
# age, past launches and behaviour on this launch. Point-in-time: wallet lookups use only events
# strictly before the row's ts. The history-dependent ones (freshness, ages, launches) are NULL
# until the pipeline has observed Pipeline:WalletAgeCapHours (24 h) of history (replay warm-up);
# wal_coverage_sec, always written, tells how much it has.
WALLET_COLUMNS: tuple[str, ...] = (
    "wal_coverage_sec",
    "wal_fresh_1h_share",
    "wal_fresh_24h_share",
    "wal_median_trader_age_sec",
    "wal_creator_age_sec",
    "wal_creator_prev_launches",
    "wal_creator_buy_sol",
    "wal_creator_buy_supply_share",
    "wal_creator_sold_share",
    "wal_creator_first_sell_sec",
)

# FE-T10 (#190): wallet reputation — whether the wallets behind a token are *good*, from outcomes in
# the same wallet memory (behaviour statistics, never identities): the creator's earlier launches
# that rugged / pumped (an outcome is recorded only after its 1 h horizon plus the label embargo, so
# it can't leak a label), the realized round-trip record (SOL-quoted buy -> sell) of the token's
# first buyers and of the "smart" wallets trading it in the window, and whether its first buys
# landed in one block (a bundle). The history-dependent ones are NULL until 24 h of history has
# been observed; rep_early_buyers, always written, marks the rows computed since migration 0015.
REPUTATION_COLUMNS: tuple[str, ...] = (
    "rep_creator_settled_launches",
    "rep_creator_rug_share",
    "rep_creator_pump_share",
    "rep_early_buyers",
    "rep_early_known_share",
    "rep_early_win_rate",
    "rep_early_pnl_sol",
    "rep_smart_buy_share",
    "rep_smart_net_flow_sol",
    "rep_smart_traders",
    "rep_bundle_create_block_share",
    "rep_bundle_top_block_share",
)

# FE-T12 (#196): token metadata — what the token says about itself at launch. Stream-derived (free,
# live = replay): whether its metadata URI is content-addressed (IPFS / Arweave, the pump.fun UI
# default) or a launch tool's own host, and the narrative heat of its normalized name / ticker
# (other launches sharing it in the last hour; tokens sharing it that graduated in the last 3 h,
# NULL until 1 h / 6 h of history). Fetched once per token by live ingestion only: whether the
# metadata lists a website / X / Telegram link, NULL unless the fetch completed by the row's ts — so
# every replay of a window before the fetcher's deployment has them NULL. Flags are 0/1;
# meta_coverage_sec, always written, marks the rows computed since migration 0016.
METADATA_COLUMNS: tuple[str, ...] = (
    "meta_coverage_sec",
    "meta_uri_content_addressed",
    "meta_has_website",
    "meta_has_twitter",
    "meta_has_telegram",
    "meta_name_matches_1h",
    "meta_narrative_grads_3h",
)

# FE-T13 (#197): holder-side supply concentration, from each wallet's net token position on the
# token as the stream shows it (create dev buy, buys - sells, liquidity, token transfers; pools
# excluded). Shares are of the circulating supply (sum of the positive positions), NULL while
# nothing circulates; the top-10 net flow is over the pipeline window (negative = distributing);
# large holders hold > Pipeline:HolderLargeShare (2 %) of the total supply. hs_incomplete = 1 when
# the token's create wasn't seen (positions miss its earlier history: replay start, eviction).
# hs_holders and hs_incomplete, always written, mark the rows computed since migration 0017.
HOLDERS_SUPPLY_COLUMNS: tuple[str, ...] = (
    "hs_holders",
    "hs_top1_share",
    "hs_top10_share",
    "hs_top10_net_flow",
    "hs_large_holders",
    "hs_incomplete",
)

# Ordered model inputs: the numeric features plus the migrated flag, then the venue one-hot. Keep
# the stored ones (DB_FEATURE_COLUMNS) and the `features` table (services/db/migrations) in
# lockstep — see tests/test_features.py.
FEATURE_COLUMNS: list[str] = [
    "price",
    "volume",
    "market_cap",
    "holders",
    "liquidity",
    "buy_events",
    "sell_events",
    "buy_sell_ratio",
    "token_age_sec",
    "trending_score",
    "social_mentions",
    "holders_growth_rate",
    "volume_growth_rate",
    "tx_creation_rate",
    "volatility",
    "ema",
    "rsi",
    "price_derivative",
    "is_migrated",
    "net_sol_flow",
    "buy_volume_ratio",
    "avg_buy_sol",
    "max_buy_sol",
    "large_buy_count",
    "return_short",
    "return_medium",
    "return_long",
    "price_acceleration",
    "ema_slope",
    "volume_acceleration",
    "volume_burst_ratio",
    "price_history_sec",
    "unique_traders",
    "top_trader_share",
    "trader_hhi",
    "new_trader_ratio",
    "burned_liquidity",
    *MARKET_COLUMNS,
    *WALLET_COLUMNS,
    *REPUTATION_COLUMNS,
    *METADATA_COLUMNS,
    *HOLDERS_SUPPLY_COLUMNS,
    *VENUE_COLUMNS,
]

# The model inputs stored as `features` columns (the venue one-hot is derived from `venue`).
DB_FEATURE_COLUMNS: list[str] = [col for col in FEATURE_COLUMNS if col not in VENUE_COLUMNS]

# Feature groups for ablation (solia_ai.ablation): every model input belongs to exactly one group
# — see tests/test_features.py. An enrichment task adds its columns to FEATURE_COLUMNS *and* a new
# group here, so its lift can be measured against the baseline before it is accepted.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price": ("price", "market_cap", "ema", "rsi", "price_derivative", "volatility"),
    "liquidity": ("liquidity", "is_migrated"),
    "flow": (
        "volume",
        "buy_events",
        "sell_events",
        "buy_sell_ratio",
        "volume_growth_rate",
        "tx_creation_rate",
    ),
    "holders": ("holders", "holders_growth_rate"),
    "context": ("token_age_sec", "trending_score", "social_mentions"),
    # FE-T2 (#111): SOL-weighted buy pressure, SOL-quoted trades only.
    "buy_pressure": (
        "net_sol_flow",
        "buy_volume_ratio",
        "avg_buy_sol",
        "max_buy_sol",
        "large_buy_count",
    ),
    # FE-T3 (#112): momentum & acceleration over the pipeline's short/medium/long windows.
    "momentum": (
        "return_short",
        "return_medium",
        "return_long",
        "price_acceleration",
        "ema_slope",
        "volume_acceleration",
        "volume_burst_ratio",
        "price_history_sec",  # how far back the returns can look: "no history" vs a flat price
    ),
    # FE-T4 (#113): holder-concentration proxy from trade signers (txSigner) over the window.
    "concentration": (
        "unique_traders",
        "top_trader_share",
        "trader_hhi",
        "new_trader_ratio",
    ),
    # FE-T5 (#114): % of the AMM pool's LP burned (rug signal), from the pool-creation event.
    "burn": ("burned_liquidity",),
    # FE-T11 (#195): market-regime context (launch / graduation rates, market-wide SOL flow,
    # active tokens), stream-derived and point-in-time.
    "market": MARKET_COLUMNS,
    # FE-T9 (#189): wallet freshness and creator behaviour, from the cross-token wallet memory.
    "wallet": WALLET_COLUMNS,
    # FE-T10 (#190): creator track record, early-buyer quality, smart-money flow, launch bundling.
    "wallet_reputation": REPUTATION_COLUMNS,
    # FE-T12 (#196): metadata host kind, social links (fetched, point-in-time), narrative heat.
    "metadata": METADATA_COLUMNS,
    # FE-T13 (#197): holder-side supply concentration from stream-derived positions.
    "holders_supply": HOLDERS_SUPPLY_COLUMNS,
    # FE-T5 (#114): trading venue (Pump.fun curve vs launchpads vs AMMs), one-hot.
    "venue": VENUE_COLUMNS,
}

# The groups making up the baseline model: the original 19 columns. Groups outside it are
# candidates, disabled (zeroed) by default until their ablation lift clears the margin — the
# model trains on ``features.groups`` from the config, which defaults to this baseline.
BASELINE_GROUPS: tuple[str, ...] = ("price", "liquidity", "flow", "holders", "context")


def group_columns(groups: Iterable[str]) -> list[str]:
    """The columns of ``groups``, in :data:`FEATURE_COLUMNS` order.

    Raises:
        ValueError: on a group name not declared in :data:`FEATURE_GROUPS`.
    """
    groups = list(groups)
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature group(s): {', '.join(unknown)} "
            f"(known: {', '.join(FEATURE_GROUPS)})"
        )
    wanted = {col for g in groups for col in FEATURE_GROUPS[g]}
    return [col for col in FEATURE_COLUMNS if col in wanted]


def is_ordered_subset(columns: Sequence[str], spec: Sequence[str] = FEATURE_COLUMNS) -> bool:
    """Whether ``columns`` appear in ``spec`` in the same order (gaps allowed): a checkpoint trained
    on such a spec still loads, on its own columns, after columns were appended or inserted."""
    remaining = iter(spec)
    return all(any(c == s for s in remaining) for c in columns)


def column_mask(columns: Sequence[str], enabled: Iterable[str]) -> np.ndarray:
    """A ``[len(columns)]`` boolean mask: ``True`` where the column is in ``enabled``.

    Raises:
        ValueError: on an enabled column that isn't one of ``columns``.
    """
    enabled = set(enabled)
    if unknown := sorted(enabled - set(columns)):
        raise ValueError(f"enabled column(s) not in the feature spec: {', '.join(unknown)}")
    return np.array([col in enabled for col in columns], dtype=bool)


def group_mask(groups: Iterable[str]) -> np.ndarray:
    """A ``[F]`` boolean mask over :data:`FEATURE_COLUMNS`: ``True`` where the column is in an
    enabled group. The model zeroes disabled columns (no signal)."""
    return column_mask(FEATURE_COLUMNS, group_columns(groups))


# NULL imputation where 0 would mean something: no SOL trade in the window is "no pressure either
# way" (0.5), not "only sells" (0.0); a burst ratio of 1 is steady volume, 0 would be a collapse.
# No recent signed trader is -1 for new_trader_ratio: 0 is a real reading ("all returning"). The
# concentration shares live in (0, 1], so their default 0 already marks "no signed volume".
# Unknown burned liquidity (no pool-creation event seen, e.g. still on the curve) is -1: 0% burned
# is a real reading, and a strong rug signal.
# A market aggregate is NULL while its window isn't fully observed yet (warm-up) — and the 1 h
# graduation ratio also without any launch: -1 for the ones that are never negative, so a warm-up
# row can't pass for a dead market (0 launches, 0 active tokens). The net flows are signed: they
# keep 0 (no net flow either way), and mkt_coverage_sec marks the warm-up for them.
# Every wallet column but the coverage is never negative, and NULL means "not computable" (warm-up,
# unknown creator, no signer, creator hasn't sold or bought nothing): -1, so it can't pass for a
# real 0 (no fresh trader, a creator born this second, no dev buy, nothing sold yet).
NULL_FILL: dict[str, float] = {
    "buy_volume_ratio": 0.5,
    "volume_burst_ratio": 1.0,
    "new_trader_ratio": -1.0,
    "burned_liquidity": -1.0,
    **{
        col: -1.0
        for col in MARKET_COLUMNS
        if col not in ("mkt_coverage_sec", "mkt_net_sol_flow_5m", "mkt_net_sol_flow_1h")
    },
    **{col: -1.0 for col in WALLET_COLUMNS if col != "wal_coverage_sec"},
    # Reputation columns are NULL when not computable (warm-up, no creator, no rated buyer, no SOL
    # buy, no block): -1 for the never-negative ones, so it can't pass for a real 0 (a creator with
    # no recorded launch, no smart volume, an unbundled launch). The signed P&L / net flow keep 0.
    **{
        col: -1.0
        for col in REPUTATION_COLUMNS
        if col not in ("rep_early_buyers", "rep_early_pnl_sol", "rep_smart_net_flow_sol")
    },
    # Metadata columns are NULL when unknown (no create seen, links not fetched by ts, narrative
    # warm-up): -1, so it can't pass for a real 0 (no link, no copycat, no graduated namesake).
    **{col: -1.0 for col in METADATA_COLUMNS if col != "meta_coverage_sec"},
    # Holder-supply shares / large-holder count are NULL while nothing circulates or the supply is
    # unknown: -1, so it can't pass for a real 0. The signed net flow keeps 0 (nothing moved).
    "hs_top1_share": -1.0,
    "hs_top10_share": -1.0,
    "hs_large_holders": -1.0,
}

# Per group, the columns the pipeline always writes since the group's migration: a NULL there
# means the row predates it (never computed), not "no signal" — such rows must be re-replayed
# before the group is trained or ablated on, or they'd teach the model a fake zero.
ALWAYS_WRITTEN: dict[str, tuple[str, ...]] = {
    "buy_pressure": ("net_sol_flow", "large_buy_count"),  # migration 0005
    "momentum": ("volume_burst_ratio", "price_history_sec"),  # migration 0006
    "concentration": ("unique_traders",),  # migration 0007
    "market": ("mkt_coverage_sec",),  # migration 0013 (the aggregates are NULL during warm-up)
    "wallet": ("wal_coverage_sec",),  # migration 0014 (history-dependent ones NULL in warm-up)
    "wallet_reputation": (
        "rep_early_buyers",
    ),  # migration 0015 (history-dependent ones NULL in warm-up)
    "metadata": ("meta_coverage_sec",),  # migration 0016 (links / counts NULL when unknown)
    "holders_supply": ("hs_holders", "hs_incomplete"),  # migration 0017
    # `burn` has none: burned_liquidity is legitimately NULL (unknown), and the database keeps it
    # equal to onchain_signals.burnedLiquidityPercent (0008's insert trigger covers writers that
    # omit it, 0009 backfills earlier rows), so a NULL there is never a stale row. `venue` reads a
    # column every row has had since 0001.
}


def check_group_coverage(df: pd.DataFrame, groups: Iterable[str]) -> None:
    """Refuse ``df`` if an enabled group has rows written before its migration.

    Raises:
        ValueError: naming the group and how many rows lack it (re-replay them first).
    """
    for group in dict.fromkeys(groups):
        cols = [c for c in ALWAYS_WRITTEN.get(group, ()) if c in df.columns]
        if not cols:
            continue
        uncomputed = int(df[cols].isna().any(axis=1).sum())
        if uncomputed:
            raise ValueError(
                f"{uncomputed} of {len(df)} feature rows predate the '{group}' group "
                f"(NULL {', '.join(cols)}): re-replay them before enabling it"
            )


# Columns of the `features` table that are deliberately NOT model inputs as stored (identity,
# bookkeeping, categorical/JSONB). `venue` feeds the model only through its one-hot
# (VENUE_COLUMNS); the JSONB signals are future work.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "mint",
        "ts",
        "venue",
        "pool_address",
        "onchain_signals",
        "behavioral_signals",
        "raw",
    }
)


def encode_venue(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot :data:`VENUE_COLUMNS` from ``df["venue"]``, as 0/1 floats on ``df``'s index.

    Matching is case- and whitespace-insensitive; a venue outside :data:`VENUES`, a non-string
    value or NULL sets :data:`VENUE_OTHER`, so every row has exactly one 1.
    """
    # Vectorized (this runs per mint over the whole training set): the string dtype keeps NULLs as
    # <NA>, which match no venue; a non-string becomes its repr, which matches none either.
    venue = df["venue"].astype("string").str.strip().str.lower()
    hot = {
        col: (venue == name).fillna(False).astype("float64")
        for col, name in zip(VENUE_COLUMNS[:-1], VENUES, strict=True)
    }
    hot[VENUE_OTHER] = (~venue.isin(VENUES)).astype("float64")
    return pd.DataFrame(hot, index=df.index, columns=list(VENUE_COLUMNS))


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    rows: np.ndarray | None = None,
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Project ``features`` rows (``df``'s ``rows`` positions in that order, default all) onto
    ``columns`` (default :data:`FEATURE_COLUMNS`; else an ordered subset of it, e.g. the enabled
    groups' columns: a narrow matrix, see :class:`solia_ai.dataset.Samples`) as a ``float32``
    matrix. Only the columns projected need to be in ``df``.

    Columns are taken in canonical order; booleans and ``NUMERIC``/``Decimal`` values are coerced
    to numbers. NULLs become their :data:`NULL_FILL` value (default ``0.0``); non-finite values
    (±inf, or magnitudes beyond float32) become ``0.0``. The venue one-hot is encoded from the
    ``venue`` column (:func:`encode_venue`); a frame without it must carry :data:`VENUE_COLUMNS`
    already encoded.

    Raises:
        ValueError: if a feature column is missing (e.g. a frame from another table, or a renamed
            column) or holds a value that isn't numeric, or if the frame carries both ``venue`` and
            some of its one-hot columns (ambiguous: which one is the venue?).
    """
    columns = FEATURE_COLUMNS if columns is None else list(columns)
    if not columns or not is_ordered_subset(columns):
        raise ValueError(
            "feature matrix columns must be a non-empty ordered subset of FEATURE_COLUMNS, "
            f"got {columns!r}"
        )
    has_venue = "venue" in df.columns
    if has_venue and (encoded := [col for col in VENUE_COLUMNS if col in df.columns]):
        raise ValueError(
            f"feature frame carries both 'venue' and its one-hot column(s): {', '.join(encoded)}"
        )
    wants_venue = any(col in VENUE_COLUMNS for col in columns)
    required = [col for col in columns if not (has_venue and col in VENUE_COLUMNS)]
    missing = [col for col in required if col not in df.columns]
    if not has_venue and any(col in missing for col in VENUE_COLUMNS):
        missing = [col for col in missing if col not in VENUE_COLUMNS] + ["venue"]
    if missing:
        raise ValueError(f"feature frame is missing column(s): {', '.join(missing)}")

    # Filled one column at a time: only one column's float64 transient exists at once, and the
    # frame (as large as the matrix several times over) is never copied.
    def column(source: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return source if rows is None else source.iloc[rows]

    venue = encode_venue(column(df[["venue"]])) if has_venue and wants_venue else None
    matrix = np.empty((len(df) if rows is None else len(rows), len(columns)), np.float32)
    for j, col in enumerate(columns):
        values = venue[col] if venue is not None and col in VENUE_COLUMNS else column(df[col])
        coerced = pd.to_numeric(values, errors="coerce")
        invalid = coerced.isna() & values.notna()
        if invalid.any():
            sample = values[invalid].iloc[0]
            raise ValueError(f"feature column '{col}' holds a non-numeric value: {sample!r}")
        filled = coerced.astype("float64").fillna(NULL_FILL.get(col, 0.0)).to_numpy()
        with np.errstate(over="ignore"):  # out-of-range values become inf here, imputed below
            matrix[:, j] = filled
    return np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
