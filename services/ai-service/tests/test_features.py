import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import strip_sql_comments

from solia_ai.features import (
    BASELINE_GROUPS,
    DB_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    MARKET_COLUMNS,
    NON_FEATURE_COLUMNS,
    VENUE_COLUMNS,
    VENUE_OTHER,
    VENUES,
    build_feature_matrix,
    check_group_coverage,
    column_mask,
    encode_venue,
    group_columns,
    group_mask,
)

# Lines inside CREATE TABLE that define constraints, not columns.
_CONSTRAINT_WORDS = {"constraint", "primary", "unique", "check", "foreign", "exclude", "like"}


def table_columns(table: str, migration_sql: list[str]) -> list[str]:
    """Replay CREATE TABLE / ALTER TABLE ... ADD|DROP|RENAME COLUMN for ``table`` across the
    migrations in order. Any column type is accepted (only the name matters)."""
    columns: list[str] = []
    ident = r'"?([A-Za-z_][A-Za-z0-9_]*)"?'
    for sql in migration_sql:
        sql = strip_sql_comments(sql)
        create = re.search(
            rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{table}\s*\((.*?)\n\);", sql, re.I | re.S
        )
        if create:
            columns = []
            depth = 0
            for line in create.group(1).splitlines():
                stripped = line.strip()
                # Continuation lines of a multi-line definition sit inside parentheses or start
                # with a constraint keyword; a column line starts with an identifier at depth 0.
                match = re.match(ident + r"\s+\S", stripped)
                if depth == 0 and match and match.group(1).lower() not in _CONSTRAINT_WORDS:
                    columns.append(match.group(1).lower())
                depth += stripped.count("(") - stripped.count(")")

        for alter in re.finditer(
            rf"ALTER\s+TABLE\s+(?:ONLY\s+)?{table}\s+(.*?);", sql, re.I | re.S
        ):
            for action in re.split(r",(?![^()]*\))", alter.group(1)):
                action = action.strip()
                if m := re.match(
                    rf"ADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?{ident}\s+\S", action, re.I
                ):
                    if (
                        m.group(1).lower() not in _CONSTRAINT_WORDS
                        and m.group(1).lower() not in columns
                    ):
                        columns.append(m.group(1).lower())
                elif m := re.match(
                    rf"DROP\s+(?:COLUMN\s+)?(?:IF\s+EXISTS\s+)?{ident}", action, re.I
                ):
                    if m.group(1).lower() != "constraint" and m.group(1).lower() in columns:
                        columns.remove(m.group(1).lower())
                elif m := re.match(rf"RENAME\s+(?:COLUMN\s+)?{ident}\s+TO\s+{ident}", action, re.I):
                    columns[columns.index(m.group(1).lower())] = m.group(2).lower()
    return columns


def assert_spec_covers(db_columns: set[str]) -> None:
    # Model columns and excluded columns are disjoint and together cover the whole table —
    # so adding/removing a DB column (or a FEATURE_COLUMNS entry) without updating both fails.
    # The venue one-hot is derived from `venue` at load time, not stored.
    assert set(FEATURE_COLUMNS).isdisjoint(NON_FEATURE_COLUMNS)
    assert "venue" in NON_FEATURE_COLUMNS
    assert db_columns - NON_FEATURE_COLUMNS == set(DB_FEATURE_COLUMNS), "spec vs DB drift"
    assert NON_FEATURE_COLUMNS <= db_columns, "NON_FEATURE_COLUMNS lists columns the DB lacks"


def test_feature_columns_match_every_migration(migration_files: list[Path]):
    columns = table_columns("features", [f.read_text("utf-8") for f in migration_files])

    assert len(columns) == len(set(columns))
    assert_spec_covers(set(columns))


# --- the parser itself must notice the drift it guards against ---

BASE = """
CREATE TABLE features (
    id      BIGSERIAL PRIMARY KEY,
    mint    TEXT NOT NULL,
    score   REAL,
    lamports BIGINT,
    "Quoted" VARCHAR(10),
    status  TEXT NOT NULL DEFAULT 'x'
            CHECK (status IN ('a', 'b')),
    CONSTRAINT features_key UNIQUE (mint)
);
"""


def test_parser_reads_any_column_type_and_skips_constraints():
    assert table_columns("features", [BASE]) == [
        "id",
        "mint",
        "score",
        "lamports",
        "quoted",
        "status",
    ]


def test_parser_replays_later_alter_table_migrations():
    later = """
    ALTER TABLE features ADD COLUMN whale_ratio DOUBLE PRECISION;
    ALTER TABLE features DROP COLUMN IF EXISTS score, ADD COLUMN IF NOT EXISTS n SMALLINT;
    ALTER TABLE features RENAME COLUMN lamports TO lamports_total;
    ALTER TABLE features ADD CONSTRAINT c CHECK (n > 0);
    ALTER TABLE other ADD COLUMN ignored INT;
    """

    assert table_columns("features", [BASE, later]) == [
        "id",
        "mint",
        "lamports_total",
        "quoted",
        "status",
        "whale_ratio",
        "n",
    ]


def test_a_column_added_by_a_later_migration_breaks_parity(migration_files: list[Path]):
    sql = [f.read_text("utf-8") for f in migration_files]
    sql.append("ALTER TABLE features ADD COLUMN whale_ratio DOUBLE PRECISION;")

    with pytest.raises(AssertionError, match="drift"):
        assert_spec_covers(set(table_columns("features", sql)))


def test_a_column_dropped_by_a_later_migration_breaks_parity(migration_files: list[Path]):
    sql = [f.read_text("utf-8") for f in migration_files]
    sql.append("ALTER TABLE features DROP COLUMN rsi;")

    with pytest.raises(AssertionError, match="drift"):
        assert_spec_covers(set(table_columns("features", sql)))


# --- build_feature_matrix ---


def full_row(**overrides):
    row = dict.fromkeys(FEATURE_COLUMNS, 1)
    row.update(overrides)
    return row


def column(name: str) -> int:
    return FEATURE_COLUMNS.index(name)


def test_build_feature_matrix_order_and_dtype():
    frame = pd.DataFrame([{col: i for i, col in enumerate(FEATURE_COLUMNS)}])

    matrix = build_feature_matrix(frame)

    assert matrix.shape == (1, len(FEATURE_COLUMNS))
    assert matrix.dtype == np.float32
    assert matrix[0].tolist() == list(range(len(FEATURE_COLUMNS)))  # canonical order


def test_extra_columns_are_ignored():
    # A DB row: identity/bookkeeping columns beside the stored features (venue is encoded).
    row = {**dict.fromkeys(DB_FEATURE_COLUMNS, 1), "mint": "M", "venue": "pumpfun", "extra": "x"}
    frame = pd.DataFrame([row])

    assert build_feature_matrix(frame).shape == (1, len(FEATURE_COLUMNS))


def test_build_feature_matrix_imputes_nulls_and_encodes_bool():
    row = dict.fromkeys(FEATURE_COLUMNS, None)
    row["price"] = 1.5
    row["is_migrated"] = True

    matrix = build_feature_matrix(pd.DataFrame([row]))

    assert matrix[0, column("price")] == np.float32(1.5)
    assert matrix[0, column("is_migrated")] == np.float32(1.0)  # bool -> 1.0
    assert matrix[0, column("volume")] == np.float32(0.0)  # None -> 0.0


def test_decimals_from_numeric_columns_are_converted():
    matrix = build_feature_matrix(pd.DataFrame([full_row(price=Decimal("0.000123"))]))

    assert matrix[0, column("price")] == pytest.approx(0.000123)


@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan"), Decimal("1e60"), Decimal("Infinity")]
)
def test_non_finite_and_out_of_float32_range_values_become_zero(value):
    matrix = build_feature_matrix(pd.DataFrame([full_row(market_cap=value)]))

    assert matrix[0, column("market_cap")] == 0.0
    assert np.isfinite(matrix).all()


def test_missing_columns_are_an_error_not_zeros():
    frame = pd.DataFrame([full_row()]).rename(columns={"market_cap": "mc"}).drop(columns=["rsi"])

    with pytest.raises(ValueError, match="missing column\\(s\\): market_cap, rsi"):
        build_feature_matrix(frame)


def test_non_numeric_values_are_an_error_not_zeros():
    with pytest.raises(ValueError, match="'volume' holds a non-numeric value: 'lots'"):
        build_feature_matrix(pd.DataFrame([full_row(volume="lots")]))


def test_empty_frame_gives_an_empty_matrix():
    matrix = build_feature_matrix(pd.DataFrame(columns=FEATURE_COLUMNS))

    assert matrix.shape == (0, len(FEATURE_COLUMNS))


# --- feature groups (ablation) ---


def test_every_feature_column_is_in_exactly_one_group():
    grouped = [col for cols in FEATURE_GROUPS.values() for col in cols]

    assert sorted(grouped) == sorted(FEATURE_COLUMNS)
    assert len(grouped) == len(set(grouped))


def test_baseline_groups_are_the_original_19_columns():
    assert set(BASELINE_GROUPS) <= set(FEATURE_GROUPS)
    assert group_columns(BASELINE_GROUPS) == FEATURE_COLUMNS[:19]
    assert len(group_columns(BASELINE_GROUPS)) == 19


def test_buy_pressure_is_a_candidate_group_outside_the_baseline():
    # FE-T2 (#111): stays disabled by default until its ablation lift clears the margin.
    assert "buy_pressure" not in BASELINE_GROUPS
    assert FEATURE_GROUPS["buy_pressure"] == (
        "net_sol_flow",
        "buy_volume_ratio",
        "avg_buy_sol",
        "max_buy_sol",
        "large_buy_count",
    )
    assert FEATURE_COLUMNS[19:24] == list(FEATURE_GROUPS["buy_pressure"])


def test_momentum_is_a_candidate_group_outside_the_baseline():
    # FE-T3 (#112): stays disabled by default until its ablation lift clears the margin.
    assert "momentum" not in BASELINE_GROUPS
    assert FEATURE_GROUPS["momentum"] == (
        "return_short",
        "return_medium",
        "return_long",
        "price_acceleration",
        "ema_slope",
        "volume_acceleration",
        "volume_burst_ratio",
        "price_history_sec",
    )
    assert FEATURE_COLUMNS[24:32] == list(FEATURE_GROUPS["momentum"])


def test_concentration_is_a_candidate_group_outside_the_baseline():
    # FE-T4 (#113): stays disabled by default until its ablation lift clears the margin.
    assert "concentration" not in BASELINE_GROUPS
    assert FEATURE_GROUPS["concentration"] == (
        "unique_traders",
        "top_trader_share",
        "trader_hhi",
        "new_trader_ratio",
    )
    assert FEATURE_COLUMNS[32:36] == list(FEATURE_GROUPS["concentration"])


def test_group_columns_keep_canonical_order():
    columns = group_columns(["holders", "price"])

    assert set(columns) == set(FEATURE_GROUPS["holders"]) | set(FEATURE_GROUPS["price"])
    assert columns == [col for col in FEATURE_COLUMNS if col in columns]


def test_group_mask_selects_exactly_the_group_columns():
    mask = group_mask(["liquidity"])

    selected = [col for col, keep in zip(FEATURE_COLUMNS, mask, strict=True) if keep]
    assert mask.dtype == bool and mask.shape == (len(FEATURE_COLUMNS),)
    assert selected == ["liquidity", "is_migrated"]


def test_no_group_masks_everything():
    assert not group_mask([]).any()


def test_unknown_group_is_an_error():
    with pytest.raises(ValueError, match="unknown feature group.*nope"):
        group_columns(["price", "nope"])
    with pytest.raises(ValueError, match="unknown feature group"):
        group_mask(["nope"])


# --- NULL semantics of the buy-pressure columns ---


def _frame(**overrides) -> pd.DataFrame:
    row = {col: 1.0 for col in FEATURE_COLUMNS}
    row.update(overrides)
    return pd.DataFrame([row])


def test_null_buy_volume_ratio_is_neutral_not_all_sells():
    no_sol_trades = build_feature_matrix(_frame(buy_volume_ratio=None))
    all_sells = build_feature_matrix(_frame(buy_volume_ratio=0.0))

    col = FEATURE_COLUMNS.index("buy_volume_ratio")
    assert no_sol_trades[0, col] == 0.5
    assert all_sells[0, col] == 0.0


def test_null_volume_burst_ratio_is_steady_not_a_collapse():
    matrix = build_feature_matrix(_frame(volume_burst_ratio=None))

    assert matrix[0, FEATURE_COLUMNS.index("volume_burst_ratio")] == 1.0


def test_null_momentum_look_backs_impute_to_zero():
    # Too little history for a return/acceleration: no momentum signal.
    matrix = build_feature_matrix(_frame(return_long=None, price_acceleration=None, ema_slope=None))

    for col in ("return_long", "price_acceleration", "ema_slope"):
        assert matrix[0, FEATURE_COLUMNS.index(col)] == 0.0


def test_other_nulls_still_impute_to_zero():
    matrix = build_feature_matrix(_frame(avg_buy_sol=None, rsi=None))

    assert matrix[0, FEATURE_COLUMNS.index("avg_buy_sol")] == 0.0
    assert matrix[0, FEATURE_COLUMNS.index("rsi")] == 0.0


@pytest.mark.parametrize("column", ["net_sol_flow", "large_buy_count"])
def test_rows_predating_buy_pressure_are_refused_when_it_is_enabled(column):
    df = pd.concat([_frame(), _frame(**{column: None})], ignore_index=True)

    with pytest.raises(ValueError, match="1 of 2 feature rows predate the 'buy_pressure' group"):
        check_group_coverage(df, [*BASELINE_GROUPS, "buy_pressure"])


def test_rows_predating_buy_pressure_are_fine_while_it_is_disabled():
    df = _frame(net_sol_flow=None, large_buy_count=None)

    check_group_coverage(df, BASELINE_GROUPS)  # no error


def test_null_buy_sizes_alone_are_legitimate_not_uncomputed():
    # No SOL buy in the window: the pipeline writes NULL avg/max but always writes the flow.
    df = _frame(avg_buy_sol=None, max_buy_sol=None, buy_volume_ratio=None)

    check_group_coverage(df, ["buy_pressure"])  # no error


def test_rows_predating_momentum_are_refused_when_it_is_enabled():
    df = pd.concat([_frame(), _frame(volume_burst_ratio=None)], ignore_index=True)

    with pytest.raises(ValueError, match="1 of 2 feature rows predate the 'momentum' group"):
        check_group_coverage(df, [*BASELINE_GROUPS, "momentum"])
    check_group_coverage(df, BASELINE_GROUPS)  # disabled: no error


def test_rows_without_price_history_sec_predate_momentum():
    with pytest.raises(ValueError, match="predate the 'momentum' group.*price_history_sec"):
        check_group_coverage(_frame(price_history_sec=None), ["momentum"])


def test_null_momentum_look_backs_alone_are_legitimate_not_uncomputed():
    # A young token: the pipeline writes NULL returns/accelerations but always the burst ratio.
    df = _frame(
        return_short=None,
        return_medium=None,
        return_long=None,
        price_acceleration=None,
        ema_slope=None,
        volume_acceleration=None,
    )

    check_group_coverage(df, ["momentum"])  # no error


def test_config_samples_refuses_uncomputed_rows_for_an_enabled_group():
    from test_train import make_features_df, small_config

    from solia_ai.dataset import config_samples

    df = make_features_df()
    df.loc[df.index[0], "net_sol_flow"] = None
    cfg = {**small_config(), "features": {"groups": [*BASELINE_GROUPS, "buy_pressure"]}}

    with pytest.raises(ValueError, match="re-replay"):
        config_samples(df, cfg)
    config_samples(df, small_config())  # the baseline doesn't need them


def test_column_mask_rejects_unknown_columns():
    assert column_mask(["a", "b"], ["b"]).tolist() == [False, True]
    with pytest.raises(ValueError, match="not in the feature spec: c"):
        column_mask(["a", "b"], ["c"])


def test_rows_predating_concentration_are_refused_when_it_is_enabled():
    df = pd.concat([_frame(), _frame(unique_traders=None)], ignore_index=True)

    with pytest.raises(ValueError, match="1 of 2 feature rows predate the 'concentration' group"):
        check_group_coverage(df, [*BASELINE_GROUPS, "concentration"])
    check_group_coverage(df, BASELINE_GROUPS)  # disabled: no error


def test_null_concentration_shares_alone_are_legitimate_not_uncomputed():
    # No signed volume / no recent trader: NULL shares, but unique_traders is always written.
    df = _frame(top_trader_share=None, trader_hhi=None, new_trader_ratio=None)

    check_group_coverage(df, ["concentration"])  # no error
    row = build_feature_matrix(df)[0]
    # The shares live in (0, 1]: 0 is out of range, so it can't pass for a real reading.
    assert row[FEATURE_COLUMNS.index("top_trader_share")] == 0.0
    assert row[FEATURE_COLUMNS.index("trader_hhi")] == 0.0
    # 0 is a real ratio ("every recent trader returns"): no recent trader must not look like it.
    assert row[FEATURE_COLUMNS.index("new_trader_ratio")] == -1.0


def test_no_recent_trader_is_told_apart_from_all_returning():
    df = pd.concat([_frame(new_trader_ratio=None), _frame(new_trader_ratio=0.0)], ignore_index=True)

    col = build_feature_matrix(df)[:, FEATURE_COLUMNS.index("new_trader_ratio")]

    assert col[0] != col[1]


# --- FE-T5 (#114): burned liquidity + venue one-hot ---


def test_burn_and_venue_are_candidate_groups_outside_the_baseline():
    # Both stay disabled by default until their ablation lift clears the margin.
    assert "burn" not in BASELINE_GROUPS and "venue" not in BASELINE_GROUPS
    assert FEATURE_GROUPS["burn"] == ("burned_liquidity",)
    assert FEATURE_GROUPS["venue"] == VENUE_COLUMNS
    assert FEATURE_COLUMNS[36:] == ["burned_liquidity", *MARKET_COLUMNS, *VENUE_COLUMNS]


def test_venue_categories_are_a_fixed_list_plus_an_other_bucket():
    assert VENUE_COLUMNS == (
        "venue_pumpfun",
        "venue_pumpswap",
        "venue_raydium_launchpad",
        "venue_raydium",
        "venue_meteora_launchpad",
        "venue_meteora",
        "venue_bonk",
        "venue_other",
    )
    assert VENUE_OTHER == VENUE_COLUMNS[-1]
    assert len(VENUES) == len(VENUE_COLUMNS) - 1


def test_the_venue_one_hot_is_derived_not_stored():
    assert not set(VENUE_COLUMNS) & set(DB_FEATURE_COLUMNS)
    assert DB_FEATURE_COLUMNS == [c for c in FEATURE_COLUMNS if c not in VENUE_COLUMNS]


def _venue_rows(*venues) -> np.ndarray:
    df = pd.DataFrame([{**dict.fromkeys(DB_FEATURE_COLUMNS, 1.0), "venue": v} for v in venues])
    matrix = build_feature_matrix(df)
    return matrix[:, [FEATURE_COLUMNS.index(c) for c in VENUE_COLUMNS]]


@pytest.mark.parametrize("venue", VENUES)
def test_each_known_venue_sets_exactly_its_column(venue):
    hot = _venue_rows(venue)[0]

    assert hot.sum() == 1.0
    assert VENUE_COLUMNS[int(hot.argmax())] == f"venue_{venue.replace('-', '_')}"


@pytest.mark.parametrize("venue", ["orca", "sunrise", "brand-new-amm", None, 7])
def test_unknown_missing_or_non_string_venue_falls_in_other(venue):
    hot = _venue_rows(venue)[0]

    assert hot.tolist() == [0.0] * (len(VENUE_COLUMNS) - 1) + [1.0]


def test_venue_matching_ignores_case_and_whitespace():
    assert _venue_rows(" PumpFun ")[0][0] == 1.0


def test_encode_venue_keeps_the_frame_index():
    df = pd.DataFrame({"venue": ["pumpswap", "x"]}, index=[10, 20])

    hot = encode_venue(df)

    assert list(hot.index) == [10, 20]
    assert list(hot.columns) == list(VENUE_COLUMNS)
    assert hot.loc[10, "venue_pumpswap"] == 1.0 and hot.loc[20, VENUE_OTHER] == 1.0


def test_venue_beside_its_encoded_columns_is_ambiguous_and_rejected():
    row = {**dict.fromkeys(FEATURE_COLUMNS, 1.0), "venue": "meteora"}

    with pytest.raises(ValueError, match="both 'venue' and its one-hot column"):
        build_feature_matrix(pd.DataFrame([row]))


def test_an_all_null_venue_column_encodes_as_other():
    # A frame whose venue column pandas typed as float (every value NULL) still encodes.
    df = pd.DataFrame({"venue": [np.nan, None]})

    hot = encode_venue(df)

    assert hot[VENUE_OTHER].tolist() == [1.0, 1.0]
    assert hot.drop(columns=[VENUE_OTHER]).to_numpy().sum() == 0.0


def test_an_already_encoded_frame_needs_no_venue():
    row = {**dict.fromkeys(DB_FEATURE_COLUMNS, 1.0), **dict.fromkeys(VENUE_COLUMNS, 0.0)}
    row["venue_bonk"] = 1.0

    matrix = build_feature_matrix(pd.DataFrame([row]))

    assert matrix[0, FEATURE_COLUMNS.index("venue_bonk")] == 1.0


def test_a_frame_without_venue_nor_its_encoding_names_venue_missing():
    with pytest.raises(ValueError, match=r"missing column\(s\): venue$"):
        build_feature_matrix(pd.DataFrame([dict.fromkeys(DB_FEATURE_COLUMNS, 1.0)]))


def test_unknown_burned_liquidity_is_told_apart_from_none_burned():
    df = pd.concat(
        [
            _frame(burned_liquidity=None),
            _frame(burned_liquidity=0.0),
            _frame(burned_liquidity=100.0),
        ],
        ignore_index=True,
    )

    col = build_feature_matrix(df)[:, FEATURE_COLUMNS.index("burned_liquidity")]

    assert col.tolist() == [-1.0, 0.0, 100.0]  # unknown, nothing burned (rug risk), all burned


def test_null_burned_liquidity_is_legitimate_not_uncomputed():
    # No ALWAYS_WRITTEN marker: 0009 backfills earlier rows, NULL is a real "unknown".
    df = _frame(burned_liquidity=None)

    check_group_coverage(df, ["burn", "venue"])  # no error


def db_frame(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Rows as the bounded read returns them: venue text, NULLs, huge values, one bool column."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {col: rng.normal(size=n) * 10.0 ** rng.integers(-8, 8) for col in DB_FEATURE_COLUMNS}
    )
    frame.loc[rng.random(n) < 0.2, "volume"] = np.nan
    frame.loc[0, "market_cap"] = 1e300  # beyond float32: imputed to 0
    frame["is_migrated"] = rng.random(n) < 0.5
    frame["venue"] = rng.choice(["pumpfun", " PumpSwap ", "unknown", None], size=n)
    frame["mint"] = rng.choice(["A", "B"], size=n)
    return frame


def test_selected_rows_project_like_the_sliced_frame():
    frame = db_frame()
    rows = np.random.default_rng(1).permutation(len(frame))[:30]

    np.testing.assert_array_equal(
        build_feature_matrix(frame, rows=rows), build_feature_matrix(frame.iloc[rows])
    )


def test_the_read_s_float32_columns_give_the_same_matrix():
    from solia_ai.db import _FLOAT64_COLUMNS, _compact

    frame = db_frame()
    compact = _compact(frame.copy())

    np.testing.assert_array_equal(build_feature_matrix(compact), build_feature_matrix(frame))
    for col in DB_FEATURE_COLUMNS:
        if col == "is_migrated":
            continue
        expected = np.float64 if col in _FLOAT64_COLUMNS else np.float32
        assert compact[col].dtype == expected, col
    assert {"price", "liquidity", "unique_traders", "volume", "token_age_sec"} <= _FLOAT64_COLUMNS


# --- FE-T11 (#195): market-regime context ---


def test_market_is_a_candidate_group_outside_the_defaults():
    # Disabled by default until its ablation lift clears the margin.
    assert "market" not in BASELINE_GROUPS
    assert FEATURE_GROUPS["market"] == MARKET_COLUMNS
    assert len(MARKET_COLUMNS) == 11 and all(c.startswith("mkt_") for c in MARKET_COLUMNS)
    assert group_columns(["market"]) == list(MARKET_COLUMNS)


def test_rows_predating_the_market_group_are_refused_when_it_is_enabled():
    df = pd.concat([_frame(), _frame(mkt_coverage_sec=None)], ignore_index=True)

    with pytest.raises(ValueError, match="1 of 2 feature rows predate the 'market' group"):
        check_group_coverage(df, [*BASELINE_GROUPS, "market"])
    check_group_coverage(df, BASELINE_GROUPS)  # disabled: no error


def test_market_warm_up_is_told_apart_from_a_dead_market():
    # Warm-up: coverage written, aggregates NULL. A dead market: real zeros.
    warm_up = {col: None for col in MARKET_COLUMNS if col != "mkt_coverage_sec"}
    dead = {col: 0.0 for col in MARKET_COLUMNS if col != "mkt_coverage_sec"}
    df = pd.concat(
        [_frame(mkt_coverage_sec=120.0, **warm_up), _frame(mkt_coverage_sec=3600.0, **dead)],
        ignore_index=True,
    )

    check_group_coverage(df, ["market"])  # warm-up rows are computed rows, not stale ones
    matrix = build_feature_matrix(df)
    for col in ("mkt_launch_rate_1h", "mkt_graduation_ratio_1h", "mkt_active_tokens_5m"):
        j = FEATURE_COLUMNS.index(col)
        assert matrix[0, j] == -1.0 and matrix[1, j] == 0.0
    # Signed flows can't take a sentinel: neutral 0, the coverage column marks the warm-up.
    j = FEATURE_COLUMNS.index("mkt_net_sol_flow_1h")
    assert matrix[0, j] == 0.0
    assert matrix[:, FEATURE_COLUMNS.index("mkt_coverage_sec")].tolist() == [120.0, 3600.0]
