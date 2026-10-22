"""Read-only access to the pipeline's PostgreSQL tables, into pandas.

Connection details come from the environment via :func:`solia_ai.config.database_url`. Queries
are parameterized; table/order identifiers are internal literals, never user input.

Time bounds are half-open (``start <= t < end``) and must be timezone-aware: the columns are
``TIMESTAMPTZ``, and a naive datetime would be read in the database session's time zone, silently
shifting the window. Aware values are normalized to UTC.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.sql.elements import TextClause

from .config import database_url


def make_engine() -> Engine:
    """A SQLAlchemy engine built from the environment (see :func:`database_url`)."""
    return _sa_create_engine(database_url(), pool_pre_ping=True)


def read_features(
    engine: Engine,
    *,
    mints: Sequence[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read `features` ordered by `(mint, ts)`.

    With ``chunksize``, returns an iterator of frames streamed through a server-side cursor, so
    memory stays bounded by the chunk size rather than the result size.
    """
    return _read(
        engine,
        "features",
        time_col="ts",
        order_by="mint, ts",
        mints=mints,
        start=start,
        end=end,
        limit=limit,
        chunksize=chunksize,
    )


def read_trades(
    engine: Engine,
    *,
    mints: Sequence[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read `trades` ordered by `(mint, opened_at)` (same options as :func:`read_features`)."""
    return _read(
        engine,
        "trades",
        time_col="opened_at",
        order_by="mint, opened_at",
        mints=mints,
        start=start,
        end=end,
        limit=limit,
        chunksize=chunksize,
    )


def _read(
    engine: Engine,
    table: str,
    *,
    time_col: str,
    order_by: str,
    mints: Sequence[str] | None,
    start: datetime | None,
    end: datetime | None,
    limit: int | None,
    chunksize: int | None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    if isinstance(mints, str):
        raise TypeError("mints must be a sequence of mint addresses, not a single string")
    _check_positive("limit", limit)
    _check_positive("chunksize", chunksize)

    clauses: list[str] = []
    params: dict[str, object] = {}
    if mints is not None:
        clauses.append("mint = ANY(:mints)")
        params["mints"] = list(mints)
    if start is not None:
        clauses.append(f"{time_col} >= :start")
        params["start"] = _utc("start", start)
    if end is not None:
        clauses.append(f"{time_col} < :end")
        params["end"] = _utc("end", end)
    if start is not None and end is not None and params["start"] >= params["end"]:
        raise ValueError(f"start ({start}) must be before end ({end})")

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = ""
    if limit is not None:
        limit_sql = " LIMIT :limit"
        params["limit"] = limit

    query = text(f"SELECT * FROM {table}{where} ORDER BY {order_by}{limit_sql}")
    if chunksize is None:
        return pd.read_sql(query, engine, params=params)
    return _stream(engine, query, params, chunksize)


def _stream(
    engine: Engine, query: TextClause, params: dict[str, object], chunksize: int
) -> Iterator[pd.DataFrame]:
    # A plain cursor would make psycopg fetch the whole result before the first chunk;
    # stream_results uses a server-side cursor, kept open for the iterator's lifetime.
    with engine.connect().execution_options(stream_results=True, max_row_buffer=chunksize) as conn:
        yield from pd.read_sql(query, conn, params=params, chunksize=chunksize)


def _utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{name} must be timezone-aware (e.g. datetime(..., tzinfo=UTC)), got {value!r}"
        )
    return value.astimezone(UTC)


def _check_positive(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
