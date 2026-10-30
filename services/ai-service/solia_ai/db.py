"""Read-only access to the pipeline's PostgreSQL tables, into pandas.

Connection details come from the environment via :func:`solia_ai.config.database_url`. Queries
are parameterized; table/order identifiers are internal literals, never user input.

Time bounds are half-open (``start <= t < end``) and must be timezone-aware: the columns are
``TIMESTAMPTZ``, and a naive datetime would be read in the database session's time zone, silently
shifting the window. Aware values are normalized to UTC.

Training runs read through :func:`read_training_features`, which bounds the read (see
:class:`DataSettings`) so a retrain's memory never grows with the table.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from sqlalchemy import create_engine as _sa_create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.sql.elements import TextClause

from .config import DATA_DEFAULTS, ConfigError, data_bounds_error, database_url
from .eligibility import INPUT_COLUMNS as ELIGIBILITY_INPUTS
from .features import DB_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Rows streamed per fetch by the bounded read: the transient buffer next to the frame being built.
TRAINING_CHUNKSIZE = 200_000


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


# --------------------------------------------------------------------------- bounded training read


@dataclass(frozen=True)
class DataSettings:
    """How much of ``features`` a training run reads (``config.yaml`` ``data``).

    ``lookback_days``: only rows of the most recent N days, counted back from the newest row (so a
    database holding only replayed history works too). ``max_rows``: when the window holds more
    rows, a subset of whole tokens is kept (see :func:`read_training_features`).
    """

    lookback_days: float
    max_rows: int

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        lookback_days: float | None = None,
        max_rows: int | None = None,
    ) -> DataSettings:
        """The config's bounds, each overridden when given (the CLI flags); validated.

        Raises:
            ConfigError: a non-positive bound, or a window too short for the promotion gate.
        """
        data = {**DATA_DEFAULTS, **config.get("data", {})}
        if lookback_days is not None:
            data["lookback_days"] = lookback_days
        if max_rows is not None:
            data["max_rows"] = max_rows
        problem = data_bounds_error(config, data["lookback_days"], data["max_rows"])
        if problem:
            raise ConfigError(problem)
        return cls(float(data["lookback_days"]), int(data["max_rows"]))


@dataclass(frozen=True)
class DataBounds:
    """What a bounded read actually used; recorded with the run and the candidate."""

    lookback_days: float
    max_rows: int
    window_start: datetime  # inclusive
    window_end: datetime  # the newest row, inclusive
    rows_in_window: int
    tokens_in_window: int
    tokens_partial: int  # dropped: their history starts before the window
    tokens_capped: int  # dropped: over max_rows
    tokens_kept: int
    rows_read: int

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["window_start"] = self.window_start.isoformat()
        out["window_end"] = self.window_end.isoformat()
        return out

    def describe(self) -> str:
        return (
            f"features window {self.window_start.isoformat()} .. {self.window_end.isoformat()} "
            f"({self.lookback_days:g} d): {self.rows_read} rows of {self.tokens_kept} tokens read "
            f"(window held {self.rows_in_window} rows of {self.tokens_in_window} tokens; dropped "
            f"{self.tokens_partial} starting before it, {self.tokens_capped} over max_rows "
            f"{self.max_rows})"
        )


def add_data_bounds_arguments(parser: argparse.ArgumentParser) -> None:
    """``--lookback-days`` / ``--max-rows``: override the config's ``data`` bounds (shared by the
    retrain, candidate and train CLIs)."""
    parser.add_argument(
        "--lookback-days", type=float,
        help="read only the most recent N days of features (default: data.lookback_days)",
    )  # fmt: skip
    parser.add_argument(
        "--max-rows", type=int,
        help="cap the feature rows read, keeping whole tokens (default: data.max_rows)",
    )  # fmt: skip


# NUMERIC columns arrive as Decimal objects (object dtype, ~4x the memory of a float64); the model
# reads float32 anyway and the 0002 bounds keep every value within float8, so they're cast in SQL.
_TRAINING_COLUMNS = ", ".join(
    ["mint", "ts", "venue"]
    + [col if col == "is_migrated" else f"{col}::float8 AS {col}" for col in DB_FEATURE_COLUMNS]
)


# Text columns held as categories while the chunks are collected: a str object per row, kept
# alive among each chunk's transient row objects, fragments the heap (measured: RSS ~4x the frame).
_TEXT_COLUMNS = ("mint", "venue")


# Held as float64: the columns labels (price, liquidity) and the eligibility rule are computed
# from, compared to thresholds. Every other feature only feeds the model, which casts it to float32
# anyway (the same rounding), so it is held as float32 from the read: ~30 % less memory per row.
_FLOAT64_COLUMNS = frozenset({"price", "liquidity", *ELIGIBILITY_INPUTS})


def _compact(chunk: pd.DataFrame) -> pd.DataFrame:
    for col in _TEXT_COLUMNS:
        chunk[col] = chunk[col].astype("category")
    for col in DB_FEATURE_COLUMNS:
        if col not in _FLOAT64_COLUMNS and chunk[col].dtype == np.float64:
            with np.errstate(over="ignore"):  # beyond float32 -> inf, imputed to 0 like the model
                chunk[col] = chunk[col].astype(np.float32)
    return chunk


def read_training_features(
    engine: Engine, settings: DataSettings, *, chunksize: int = TRAINING_CHUNKSIZE
) -> tuple[pd.DataFrame, DataBounds]:
    """The ``features`` a training run uses, bounded by ``settings``, ordered by ``(mint, ts)``.

    1. The window is ``[newest - lookback_days, newest]``, ``newest`` being the newest row.
    2. Only **whole tokens** are kept: one with a row before the window is dropped, never cut.
    3. When those hold more than ``max_rows`` rows, tokens are kept in ``md5(mint)`` order while
       the running total fits: deterministic across runs and unrelated to time, so the sample
       spans the whole window and no token is truncated. As data grows the kept set is mostly
       stable (a prefix of the same order).

    Only the model's columns are read (no JSONB), streamed in ``chunksize`` rows, so peak memory
    follows ``max_rows``, never the table (docs/ML-MODEL.md "Bounded training read").

    Raises:
        ValueError: the table is empty, or no whole token fits the window / cap.
    """
    _check_positive("chunksize", chunksize)
    with engine.connect() as conn:
        newest = conn.execute(text("SELECT max(ts) FROM features")).scalar()
        if newest is None:
            raise ValueError("the features table is empty: nothing to train on")
        end = newest.astimezone(UTC)
        start = end - timedelta(days=settings.lookback_days)
        bounds = {"start": start, "end": end}
        tokens = pd.read_sql(
            text(
                "SELECT f.mint, count(*) AS n, md5(f.mint) AS h, EXISTS (SELECT 1 FROM features p "
                "WHERE p.mint = f.mint AND p.ts < :start) AS partial FROM features f "
                "WHERE f.ts >= :start AND f.ts <= :end GROUP BY f.mint"
            ),
            conn,
            params=bounds,
        )
    whole = tokens[~tokens["partial"].astype(bool)].sort_values("h", kind="stable")
    kept = whole[whole["n"].cumsum() <= settings.max_rows]
    if kept.empty:
        raise ValueError(
            f"no whole token in the features window {start.isoformat()} .. {end.isoformat()} "
            f"fits (data.max_rows {settings.max_rows}; {len(whole)} whole of {len(tokens)} "
            "tokens): widen data.lookback_days or raise data.max_rows"
        )

    query = text(
        f"SELECT {_TRAINING_COLUMNS} FROM features WHERE mint = ANY(:mints) "
        "AND ts >= :start AND ts <= :end ORDER BY mint, ts"
    )
    params = {**bounds, "mints": kept["mint"].tolist()}
    chunks = [_compact(chunk) for chunk in _stream(engine, query, params, chunksize)]
    for col in _TEXT_COLUMNS:  # one category set, so the concat stays categorical
        categories = union_categoricals([chunk[col] for chunk in chunks]).categories
        for chunk in chunks:
            chunk[col] = chunk[col].cat.set_categories(categories)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    for col in _TEXT_COLUMNS:  # back to plain object columns, sharing one str per distinct value
        df[col] = df[col].astype(object)
    result = DataBounds(
        lookback_days=settings.lookback_days,
        max_rows=settings.max_rows,
        window_start=start,
        window_end=end,
        rows_in_window=int(tokens["n"].sum()),
        tokens_in_window=len(tokens),
        tokens_partial=len(tokens) - len(whole),
        tokens_capped=len(whole) - len(kept),
        tokens_kept=len(kept),
        rows_read=len(df),
    )
    logger.info(result.describe())
    return df, result


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
