"""density.py's query against a real PostgreSQL (Testcontainers) with every migration applied."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text

from solia_ai.density import read_density

T0 = datetime(2026, 4, 21, 11, 50, tzinfo=UTC)
INSERT = text("INSERT INTO features (mint, ts, is_migrated, price) VALUES (:m, :ts, false, 1)")


def _series(mint, start, minutes, offset_seconds=0.0):
    """One row per 10 s for ``minutes``, the way one writer's snapshots land."""
    return [
        {"m": mint, "ts": start + timedelta(seconds=i * 10 + offset_seconds)}
        for i in range(minutes * 6)
    ]


@pytest.fixture(scope="module")
def engine(pg_url):
    eng = create_engine(pg_url)
    with eng.begin() as conn:
        conn.exec_driver_sql("TRUNCATE tokens_history RESTART IDENTITY CASCADE")
        conn.exec_driver_sql(
            "INSERT INTO tokens_history (mint, first_seen_at, last_seen_at) "
            "VALUES ('A', now(), now()), ('B', now(), now())"
        )
        # Slot 11:50: two tokens, one series each.
        conn.execute(INSERT, _series("A", T0, 5))
        conn.execute(INSERT, _series("B", T0, 5))
        # Slot 11:55: A carries a second series, ticking 5 s off the first; B doesn't.
        doubled = T0 + timedelta(minutes=5)
        conn.execute(INSERT, _series("A", doubled, 5))
        conn.execute(INSERT, _series("A", doubled, 5, offset_seconds=5))
        conn.execute(INSERT, _series("B", doubled, 5))
        # Slot 12:00: A alone, doubled.
        after = T0 + timedelta(minutes=10)
        conn.execute(INSERT, _series("A", after, 5))
        conn.execute(INSERT, _series("A", after, 5, offset_seconds=5))
    yield eng
    eng.dispose()


def test_density_separates_a_single_series_from_interleaved_ones(engine):
    slots = read_density(engine, T0, T0 + timedelta(minutes=15))
    assert [slot.start for slot in slots] == [T0 + timedelta(minutes=m) for m in (0, 5, 10)]
    assert slots[0].density == 1.0
    assert (slots[1].n_rows, slots[1].n_buckets) == (90, 60)  # A: 30 + 30 rows, B: 30
    assert slots[1].density == 1.5  # one token of two is doubled
    assert slots[2].density == 2.0


def test_density_window_is_half_open(engine):
    slots = read_density(engine, T0, T0 + timedelta(minutes=5))
    assert [slot.n_rows for slot in slots] == [60]


def test_density_merges_a_slot_split_across_query_steps(engine):
    """Hourly steps from 11:50 cut the 12:00-13:00 slot's rows at 12:50, none lost or repeated."""
    start = T0 + timedelta(minutes=10)
    extra = start + timedelta(minutes=55)  # 12:55, in the step after the one holding 12:00-12:05
    with engine.begin() as conn:
        conn.execute(INSERT, _series("B", extra, 1))
    try:
        slots = read_density(engine, T0, extra + timedelta(minutes=1), slot_minutes=60)
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM features WHERE mint = 'B' AND ts >= :ts"), {"ts": extra})
    assert [slot.start for slot in slots] == [T0.replace(minute=0), T0.replace(hour=12, minute=0)]
    assert slots[1].n_rows == 60 + 6  # A doubled over 12:00-12:05, then B's minute at 12:55
