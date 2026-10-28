"""db.py against a real PostgreSQL (Testcontainers) with every migration applied."""

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from solia_ai.config import ConfigError
from solia_ai.db import make_engine, read_features, read_trades
from solia_ai.features import (
    DB_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NON_FEATURE_COLUMNS,
    build_feature_matrix,
)

T0 = datetime(2026, 4, 20, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def engine(pg_url):
    eng = create_engine(pg_url)
    with eng.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO tokens_history (mint, first_seen_at, last_seen_at) "
            "VALUES ('M1', now(), now()), ('M2', now(), now()), ('M3', now(), now())"
        )
        conn.exec_driver_sql(
            "INSERT INTO features (mint, ts, is_migrated, price, volume, holders, "
            "buy_events, sell_events) VALUES "
            "('M1','2026-04-20T12:00:00Z', false, 0.001, 5, 3, 2, 1), "
            "('M1','2026-04-20T12:00:10Z', false, 0.0012, 7, 4, 3, 1), "
            "('M2','2026-04-20T12:00:05Z', true, 0.5, 100, 50, 40, 10)"
        )
        conn.exec_driver_sql(
            "INSERT INTO trades (mint, opened_at, entry_price, size, strategy, status) VALUES "
            "('M3','2026-04-20T12:01:00Z', 0.01, 1, 'buy', 'open'), "
            "('M1','2026-04-20T12:02:00Z', 0.02, 2, 'buy', 'closed')"
        )
    yield eng
    eng.dispose()


def test_feature_spec_matches_the_migrated_schema(engine):
    with engine.connect() as conn:
        db_columns = set(
            conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'features'"
                )
            ).scalars()
        )

    assert db_columns - NON_FEATURE_COLUMNS == set(DB_FEATURE_COLUMNS)  # venue one-hot: derived
    assert NON_FEATURE_COLUMNS <= db_columns


def test_read_features_returns_rows_time_ordered(engine):
    df = read_features(engine)

    assert list(df["mint"]) == ["M1", "M1", "M2"]  # ordered by (mint, ts)
    m1_ts = df.loc[df["mint"] == "M1", "ts"].tolist()
    assert m1_ts == sorted(m1_ts)  # ascending within a token


def test_build_matrix_from_db_rows(engine):
    matrix = build_feature_matrix(read_features(engine))

    assert matrix.shape == (3, len(FEATURE_COLUMNS))
    assert matrix[2, FEATURE_COLUMNS.index("is_migrated")] == 1.0  # M2 is migrated
    assert matrix[0, FEATURE_COLUMNS.index("market_cap")] == 0.0  # NULL -> 0
    assert matrix[0, FEATURE_COLUMNS.index("price")] == pytest.approx(0.001)  # NUMERIC -> float


def test_read_features_filters_by_mint(engine):
    assert list(read_features(engine, mints=["M2"])["mint"]) == ["M2"]
    assert read_features(engine, mints=[]).empty


def test_time_window_is_half_open(engine):
    df = read_features(engine, start=T0, end=T0 + timedelta(seconds=10))

    assert list(df["ts"]) == [
        pd.Timestamp(T0),
        pd.Timestamp(T0 + timedelta(seconds=5)),
    ]  # 12:00:10 excluded


def test_aware_bounds_in_another_zone_are_normalized(engine):
    montreal = timezone(timedelta(hours=-4))
    start = datetime(2026, 4, 20, 8, 0, 5, tzinfo=montreal)  # == 12:00:05 UTC

    df = read_features(engine, start=start)

    assert len(df) == 2  # 12:00:05 and 12:00:10


def test_naive_bounds_are_rejected(engine):
    with pytest.raises(ValueError, match="start must be timezone-aware"):
        read_features(engine, start=datetime(2026, 4, 20, 12))
    with pytest.raises(ValueError, match="end must be timezone-aware"):
        read_features(engine, end=datetime(2026, 4, 20, 12))


def test_inverted_window_is_rejected(engine):
    with pytest.raises(ValueError, match="must be before end"):
        read_features(engine, start=T0 + timedelta(hours=1), end=T0)


def test_limit_caps_rows_in_order(engine):
    assert list(read_features(engine, limit=2)["mint"]) == ["M1", "M1"]


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": -1}, {"chunksize": 0}, {"limit": True}])
def test_non_positive_limit_or_chunksize_is_rejected(engine, kwargs):
    with pytest.raises(ValueError, match="must be a positive integer"):
        read_features(engine, **kwargs)


def test_single_string_mints_is_rejected(engine):
    with pytest.raises(TypeError, match="not a single string"):
        read_features(engine, mints="M1")


def test_chunksize_streams_frames_in_order(engine):
    chunks = list(read_features(engine, chunksize=2))

    assert [len(c) for c in chunks] == [2, 1]
    assert list(pd.concat(chunks)["mint"]) == ["M1", "M1", "M2"]


def test_chunksize_uses_a_server_side_cursor(engine, monkeypatch):
    seen = {}
    real_read_sql = pd.read_sql

    def spy(query, con, **kwargs):
        seen["stream_results"] = con.get_execution_options().get("stream_results")
        return real_read_sql(query, con, **kwargs)

    monkeypatch.setattr(pd, "read_sql", spy)
    list(read_features(engine, chunksize=1))

    assert seen["stream_results"] is True


def test_read_trades_returns_rows_ordered_and_filtered(engine):
    df = read_trades(engine)

    assert list(df["mint"]) == ["M1", "M3"]  # ordered by (mint, opened_at)
    assert list(read_trades(engine, start=T0 + timedelta(minutes=2))["mint"]) == ["M1"]


def test_make_engine_reads_the_environment(pg_url, monkeypatch):
    url = create_engine(pg_url).url
    monkeypatch.setenv(
        "SOLIA_DB_CONNECTION",
        f"Host={url.host};Port={url.port};Database={url.database};Username={url.username};"
        f"Password='{url.password}'",
    )

    assert len(read_features(make_engine())) == 3


def test_make_engine_fails_fast_on_invalid_environment(monkeypatch):
    monkeypatch.delenv("SOLIA_DB_CONNECTION", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "db.example")

    with pytest.raises(ConfigError, match="POSTGRES_PASSWORD is not set"):
        make_engine()


def test_unreachable_database_raises_promptly(monkeypatch):
    monkeypatch.setenv(
        "SOLIA_DB_CONNECTION", "Host=127.0.0.1;Port=1;Username=u;Password=p;Timeout=2"
    )

    with pytest.raises(OperationalError):
        read_features(make_engine())
