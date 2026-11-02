"""The bounded training read (solia_ai.db.read_training_features): lookback window, whole tokens,
deterministic row cap. Runs against the shared Testcontainers PostgreSQL (skipped without Docker).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from solia_ai.db import REQUIRED_COLUMNS, DataSettings, read_training_features
from solia_ai.features import DB_FEATURE_COLUMNS

NEWEST = datetime(2026, 1, 10, tzinfo=UTC)
CONFIG = {"horizons": ["1m", "1h"], "promotion": {"windows": 3}}  # lookback floor: 4 h


@pytest.fixture
def engine(pg_url):
    eng = create_engine(pg_url)
    with eng.begin() as conn:
        conn.exec_driver_sql("TRUNCATE features RESTART IDENTITY CASCADE")
    yield eng
    with eng.begin() as conn:
        conn.exec_driver_sql("TRUNCATE features RESTART IDENTITY CASCADE")
    eng.dispose()


def insert(engine, rows: list[tuple[str, datetime]]) -> None:
    with engine.begin() as conn:
        for mint in {m for m, _ in rows}:
            conn.execute(
                text("INSERT INTO tokens_history (mint) VALUES (:m) ON CONFLICT DO NOTHING"),
                {"m": mint},
            )
        for i, (mint, ts) in enumerate(rows):
            conn.execute(
                text(
                    "INSERT INTO features (mint, ts, is_migrated, price, volume, liquidity, "
                    "holders, buy_events, sell_events, raw) VALUES (:m, :ts, false, :p, 5, 10, 3, "
                    '2, 1, \'{"big": "payload"}\')'
                ),
                {"m": mint, "ts": ts, "p": 0.001 * (i + 1)},
            )


def history(mint: str, first: datetime, n: int, step: timedelta = timedelta(minutes=10)):
    return [(mint, first + i * step) for i in range(n)]


def settings(lookback_days: float = 1.0, max_rows: int = 1_000_000) -> DataSettings:
    return DataSettings.from_config(CONFIG, lookback_days=lookback_days, max_rows=max_rows)


def md5_order(mints) -> list[str]:
    return sorted(mints, key=lambda m: hashlib.md5(m.encode()).hexdigest())


@pytest.fixture
def seeded(engine):
    """Tokens inside the 1-day window ending at NEWEST, one before it, one straddling its start."""
    start = NEWEST - timedelta(days=1)
    whole = {f"W{i}": 3 + i for i in range(8)}  # 3..10 rows each, spread over the window
    rows = []
    for i, (mint, n) in enumerate(whole.items()):
        rows += history(mint, start + timedelta(hours=2 * i + 1), n)
    rows += history("OLD", NEWEST - timedelta(days=3), 4)  # entirely before the window
    rows += history("PART", start - timedelta(minutes=30), 6)  # straddles the window start
    rows += [("EDGE", start), ("EDGE", start + timedelta(minutes=1))]  # starts exactly at it
    rows += [("LAST", NEWEST - timedelta(minutes=10)), ("LAST", NEWEST)]  # the newest row
    rows += history("LATE", NEWEST - timedelta(hours=6, minutes=30), 6)  # straddles NEWEST - 6 h
    insert(engine, rows)
    return {**whole, "EDGE": 2, "LAST": 2, "LATE": 6}


def test_only_the_window_of_whole_tokens_is_read(engine, seeded):
    df, bounds = read_training_features(engine, settings())

    assert set(df["mint"]) == set(seeded)  # OLD (before) and PART (straddling) are out
    assert df.groupby("mint").size().to_dict() == seeded
    assert df["ts"].min() == NEWEST - timedelta(days=1)  # the start is inclusive
    assert df["ts"].max() == NEWEST
    assert bounds.window_start == NEWEST - timedelta(days=1) and bounds.window_end == NEWEST
    assert bounds.tokens_in_window == len(seeded) + 1  # + PART's in-window rows
    assert bounds.tokens_partial == 1 and bounds.tokens_capped == 0
    assert bounds.tokens_kept == len(seeded) and bounds.rows_read == len(df) == sum(seeded.values())
    assert bounds.rows_in_window == len(df) + 3  # PART's rows inside the window


def test_rows_are_ordered_by_mint_then_time_with_only_the_model_columns(engine, seeded):
    df, _ = read_training_features(engine, settings(), chunksize=3)  # several chunks

    assert list(df.columns) == ["mint", "ts", "venue", *DB_FEATURE_COLUMNS]  # no JSONB payload
    ordered = df.sort_values(["mint", "ts"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(df, ordered)
    assert df["price"].dtype == "float64" and df["volume"].dtype == "float64"  # not Decimal
    assert df["is_migrated"].dtype == bool
    assert df["mint"].dtype == object and df["venue"].dtype == object  # not left categorical


def test_a_pruned_read_holds_only_its_columns_of_the_same_rows(engine, seeded):
    whole, a = read_training_features(engine, settings())
    slim, b = read_training_features(
        engine, settings(), columns=["volume_growth_rate"], chunksize=3
    )

    wanted = {"volume_growth_rate", *REQUIRED_COLUMNS}
    assert list(slim.columns) == ["mint", "ts", "venue",
                                  *[c for c in DB_FEATURE_COLUMNS if c in wanted]]  # fmt: skip
    pd.testing.assert_frame_equal(slim, whole[list(slim.columns)])
    assert a == b  # same window, tokens and rows


def test_a_read_of_an_unknown_column_is_refused(engine, seeded):
    with pytest.raises(ValueError, match="not stored feature column"):
        read_training_features(engine, settings(), columns=["raw"])


def test_the_window_follows_the_newest_row_not_the_clock(engine):
    insert(engine, history("A", datetime(2020, 5, 1, tzinfo=UTC), 5))  # an old replay only
    df, bounds = read_training_features(engine, settings())
    assert len(df) == 5 and bounds.window_end == datetime(2020, 5, 1, 0, 40, tzinfo=UTC)


def test_a_shorter_lookback_cuts_older_tokens(engine, seeded):
    df, bounds = read_training_features(engine, settings(lookback_days=0.25))  # last 6 h
    # LATE's history starts before the 6 h start: dropped whole, never cut; the W* are older.
    assert set(df["mint"]) == {"LAST"} and len(df) == seeded["LAST"]
    assert bounds.window_start == NEWEST - timedelta(hours=6)
    assert bounds.tokens_in_window == 2 and bounds.tokens_partial == 1


def test_the_cap_keeps_whole_tokens_in_md5_order(engine, seeded):
    cap = 25
    df, bounds = read_training_features(engine, settings(max_rows=cap))

    expected, total = [], 0
    for mint in md5_order(seeded):
        if total + seeded[mint] > cap:
            break
        expected.append(mint)
        total += seeded[mint]
    assert sorted(set(df["mint"])) == sorted(expected)
    assert len(df) == total <= cap
    assert df.groupby("mint").size().to_dict() == {m: seeded[m] for m in expected}  # no cut token
    assert bounds.tokens_kept == len(expected)
    assert bounds.tokens_capped == len(seeded) - len(expected)
    assert bounds.rows_read == total and bounds.max_rows == cap


def test_the_cap_is_deterministic_across_runs(engine, seeded):
    first, a = read_training_features(engine, settings(max_rows=20))
    second, b = read_training_features(engine, settings(max_rows=20), chunksize=2)
    pd.testing.assert_frame_equal(first, second)
    assert a == b


def test_the_capped_sample_is_not_a_time_slice(engine):
    # 40 tokens of 5 rows spread evenly over the day: a cap of a quarter of the rows keeps tokens
    # from both halves of the window, unlike a "first N rows" or "latest N rows" cut.
    start = NEWEST - timedelta(days=1)
    rows = []
    for i in range(40):
        rows += history(f"T{i:02d}", start + timedelta(minutes=35 * i + 1), 5)
    insert(engine, rows)
    df, _ = read_training_features(engine, settings(max_rows=50))
    first_seen = df.groupby("mint")["ts"].min()
    midday = start + timedelta(hours=12)
    assert (first_seen < midday).any() and (first_seen >= midday).any()


def test_an_empty_table_is_an_error(engine):
    with pytest.raises(ValueError, match="features table is empty"):
        read_training_features(engine, settings())


def test_a_cap_no_whole_token_fits_is_an_error(engine, seeded):
    with pytest.raises(ValueError, match="no whole token .* fits"):
        read_training_features(engine, settings(max_rows=1))


def test_a_window_of_only_partial_tokens_is_an_error(engine):
    insert(engine, history("PART", NEWEST - timedelta(days=2), 3, step=timedelta(days=1)))
    with pytest.raises(ValueError, match="no whole token"):
        read_training_features(engine, settings())


def test_a_non_positive_chunksize_is_rejected(engine, seeded):
    with pytest.raises(ValueError, match="chunksize"):
        read_training_features(engine, settings(), chunksize=0)


def test_a_pinned_end_rereads_the_same_window_after_new_rows_arrive(engine, seeded):
    first, a = read_training_features(engine, settings())
    insert(engine, history("NEW", NEWEST + timedelta(hours=1), 3))  # live rows since the start

    again, b = read_training_features(engine, settings(), end=a.window_end)

    pd.testing.assert_frame_equal(first, again)
    assert a == b and b.window_end == NEWEST
    _, moved = read_training_features(engine, settings())  # unpinned: the window moved on
    assert moved.window_end > NEWEST
