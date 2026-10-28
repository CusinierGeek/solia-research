import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import strip_sql_comments

from solia_ai.features import (
    BASELINE_GROUPS,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    NON_FEATURE_COLUMNS,
    build_feature_matrix,
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
    assert set(FEATURE_COLUMNS).isdisjoint(NON_FEATURE_COLUMNS)
    assert db_columns - NON_FEATURE_COLUMNS == set(FEATURE_COLUMNS), "spec vs DB drift"
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
    frame = pd.DataFrame([full_row(mint="M", venue="pumpfun", extra="x")])

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


def test_baseline_groups_are_the_current_19_columns():
    assert set(BASELINE_GROUPS) <= set(FEATURE_GROUPS)
    assert group_columns(BASELINE_GROUPS) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 19


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
