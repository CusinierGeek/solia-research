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
from .features import (
    BASELINE_GROUPS,
    DB_FEATURE_COLUMNS,
    VENUE_COLUMNS,
    build_feature_matrix,
    group_columns,
)
from .rows import FRAME_FLOAT_COLUMNS, TrainingRows, always_written_columns, epoch_ns

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


def add_prune_columns_argument(parser: argparse.ArgumentParser) -> None:
    """``--prune-columns``: read and hold only the enabled groups' columns (``data.prune_columns``,
    :func:`training_columns`). Research runs only (search, throughput)."""
    parser.add_argument(
        "--prune-columns", action="store_true",
        help="read only the enabled feature groups' columns and keep a narrow feature matrix "
        "(less memory per row; same windows, same model inputs)",
    )  # fmt: skip


def with_prune_columns(config: Mapping[str, Any], prune: bool) -> dict[str, Any]:
    """``config`` with ``data.prune_columns`` set to ``prune`` (a copy; ``config`` unchanged)."""
    return {**config, "data": {**config.get("data", {}), "prune_columns": bool(prune)}}


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
def _select_list(columns: Sequence[str]) -> str:
    return ", ".join(
        ["mint", "ts", "venue"]
        + [col if col == "is_migrated" else f"{col}::float8 AS {col}" for col in columns]
    )


# What every training read needs whatever the enabled groups: the labels' price and pool liquidity,
# and the eligibility rule's inputs.
REQUIRED_COLUMNS: tuple[str, ...] = ("price", "liquidity", *ELIGIBILITY_INPUTS)


def training_columns(config: Mapping[str, Any]) -> list[str] | None:
    """The ``features`` columns a training run under ``config`` reads: ``None`` (every model
    column) unless ``data.prune_columns``, else the enabled groups' (``features.groups``) stored
    columns plus :data:`REQUIRED_COLUMNS`, in table order.

    Disabled groups are zeroed by the model before anything else (``SoliaTransformer
    .mask_features``), so not reading them changes nothing the model sees; the windows' matrix then
    holds only the enabled columns (:class:`solia_ai.dataset.Samples` ``columns``).
    """
    data = {**DATA_DEFAULTS, **config.get("data", {})}
    if not data["prune_columns"]:
        return None
    groups = config.get("features", {}).get("groups", BASELINE_GROUPS)
    wanted = set(group_columns(groups)) | set(REQUIRED_COLUMNS)
    return [col for col in DB_FEATURE_COLUMNS if col in wanted]


def _check_columns(columns: Sequence[str]) -> list[str]:
    if isinstance(columns, str):
        raise TypeError("columns must be a sequence of column names, not a single string")
    unknown = sorted(set(columns) - set(DB_FEATURE_COLUMNS))
    if unknown:
        hint = (
            " (the venue one-hot is derived from 'venue')"
            if set(unknown) & set(VENUE_COLUMNS)
            else ""
        )
        raise ValueError(f"not stored feature column(s): {', '.join(unknown)}{hint}")
    wanted = set(columns) | set(REQUIRED_COLUMNS)
    return [col for col in DB_FEATURE_COLUMNS if col in wanted]


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
        if col in chunk.columns and col not in _FLOAT64_COLUMNS and chunk[col].dtype == np.float64:
            with np.errstate(over="ignore"):  # beyond float32 -> inf, imputed to 0 like the model
                chunk[col] = chunk[col].astype(np.float32)
    return chunk


def read_training_features(
    engine: Engine,
    settings: DataSettings,
    *,
    chunksize: int = TRAINING_CHUNKSIZE,
    end: datetime | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, DataBounds]:
    """The ``features`` a training run uses, bounded by ``settings``, ordered by ``(mint, ts)``.

    1. The window is ``[newest - lookback_days, newest]``, ``newest`` being the newest row, or
       ``end`` when given (a resumed search re-reads the window it started on, not today's).
    2. Only **whole tokens** are kept: one with a row before the window is dropped, never cut.
    3. When those hold more than ``max_rows`` rows, tokens are kept in ``md5(mint)`` order while
       the running total fits: deterministic across runs and unrelated to time, so the sample
       spans the whole window and no token is truncated. As data grows the kept set is mostly
       stable (a prefix of the same order).

    Only the model's columns are read (no JSONB), streamed in ``chunksize`` rows, so peak memory
    follows ``max_rows``, never the table (docs/ML-MODEL.md "Bounded training read"). With
    ``columns`` (see :func:`training_columns`), only those stored feature columns are read, plus
    :data:`REQUIRED_COLUMNS` (and ``mint`` / ``ts`` / ``venue``); rows and tokens are the same.

    Raises:
        ValueError: the table is empty, no whole token fits the window / cap, or ``columns`` names
            a column that isn't a stored feature.
    """
    _check_positive("chunksize", chunksize)
    selected = list(DB_FEATURE_COLUMNS) if columns is None else _check_columns(columns)
    tokens, kept, bounds = _select_tokens(engine, settings, end)

    query = text(
        f"SELECT {_select_list(selected)} FROM features WHERE mint = ANY(:mints) "
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
    result = _data_bounds(settings, bounds, tokens, kept, len(df))
    logger.info(result.describe())
    return df, result


def _select_tokens(
    engine: Engine, settings: DataSettings, end: datetime | None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, datetime]]:
    """The bounded read's token selection (:func:`read_training_features` steps 1-3): every token
    of the window with its row count, the kept ones (whole, ``md5(mint)`` order, within
    ``max_rows``), and the window ``{"start", "end"}``."""
    with engine.connect() as conn:
        newest = end or conn.execute(text("SELECT max(ts) FROM features")).scalar()
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
    return tokens, kept, bounds


def _data_bounds(
    settings: DataSettings,
    bounds: Mapping[str, datetime],
    tokens: pd.DataFrame,
    kept: pd.DataFrame,
    rows_read: int,
) -> DataBounds:
    n_whole = int((~tokens["partial"].astype(bool)).sum())
    return DataBounds(
        lookback_days=settings.lookback_days,
        max_rows=settings.max_rows,
        window_start=bounds["start"],
        window_end=bounds["end"],
        rows_in_window=int(tokens["n"].sum()),
        tokens_in_window=len(tokens),
        tokens_partial=len(tokens) - n_whole,
        tokens_capped=n_whole - len(kept),
        tokens_kept=len(kept),
        rows_read=rows_read,
    )


# Hours of features per query of :func:`read_training_rows`: short enough that PostgreSQL walks
# the ts index in order (a whole multi-day window gets a parallel scan + an on-disk sort).
ROWS_SLICE_HOURS = 1.0


def read_training_rows(
    engine: Engine,
    settings: DataSettings,
    *,
    read_columns: Sequence[str],
    matrix_columns: Sequence[str],
    chunksize: int = TRAINING_CHUNKSIZE,
    end: datetime | None = None,
    slice_hours: float = ROWS_SLICE_HOURS,
) -> tuple[TrainingRows, DataBounds]:
    """The rows :func:`read_training_features` reads (same window, tokens and values), held as
    :class:`solia_ai.rows.TrainingRows` and never as a frame of every row: each streamed chunk goes
    straight to its final place in a preallocated narrow ``float32`` matrix (``matrix_columns``,
    an ordered subset of FEATURE_COLUMNS) and a slim frame. Research reads only (#202).

    The window is read in ``slice_hours`` slices, each in ``ts`` order, so every token's rows
    arrive in time order and its row count (from the token selection) gives each row its slot.

    Raises:
        ValueError: as :func:`read_training_features`; or the kept tokens gained or lost rows inside
            the window during the read (retry, or pin ``end`` to an older time).
    """
    _check_positive("chunksize", chunksize)
    if not slice_hours > 0:
        raise ValueError(f"slice_hours must be > 0, got {slice_hours}")
    selected = _check_columns(read_columns)
    matrix_columns = tuple(matrix_columns)
    tokens, kept, bounds = _select_tokens(engine, settings, end)

    mints = np.sort(kept["mint"].to_numpy(dtype=object))  # the build's mint order
    dtype = pd.CategoricalDtype(pd.Index(mints, dtype=object))
    counts = kept.set_index("mint")["n"].reindex(mints).to_numpy(np.int64)
    offset = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    filled = np.zeros(len(mints), np.int64)
    n_rows = int(counts.sum())
    features = np.empty((n_rows, len(matrix_columns)), np.float32)
    ts = np.empty(n_rows, np.int64)
    venue = np.full(n_rows, np.nan, object)  # a NULL venue reads as NaN, as in the frame read
    floats = {c: np.empty(n_rows, np.float64) for c in FRAME_FLOAT_COLUMNS if c in selected}
    nulls = dict.fromkeys(always_written_columns(selected), 0)

    query = text(
        f"SELECT {_select_list(selected)} FROM features "
        "WHERE ts >= :lo AND (ts < :hi OR (:last AND ts = :hi)) ORDER BY ts"
    )
    step = timedelta(hours=slice_hours)
    lo = bounds["start"]
    while True:
        hi = min(lo + step, bounds["end"])
        last = hi == bounds["end"]
        for chunk in _stream(engine, query, {"lo": lo, "hi": hi, "last": last}, chunksize):
            codes = chunk["mint"].astype(dtype).cat.codes.to_numpy(np.int64)
            ours = codes >= 0
            if not ours.any():
                continue
            chunk, codes = chunk[ours], codes[ours]
            rank = pd.Series(codes).groupby(codes, sort=False).cumcount().to_numpy(np.int64)
            rank = rank + filled[codes]
            if (rank >= counts[codes]).any():
                raise ValueError(
                    "the kept tokens gained rows in the window during the read: retry, or pin "
                    "the window's end to an older time"
                )
            slots = offset[codes] + rank
            filled += np.bincount(codes, minlength=len(mints))
            features[slots] = build_feature_matrix(chunk, columns=matrix_columns)
            ts[slots] = epoch_ns(chunk["ts"])
            venues = chunk["venue"].to_numpy(dtype=object)
            venue[slots] = np.where(pd.isna(venues), np.nan, venues)
            for col, values in floats.items():
                values[slots] = pd.to_numeric(chunk[col], errors="coerce").to_numpy(np.float64)
            for col in nulls:
                nulls[col] += int(chunk[col].isna().sum())
        if last:
            break
        lo = hi
    if not np.array_equal(filled, counts):
        raise ValueError(
            f"read {int(filled.sum())} of the {n_rows} rows counted for the kept tokens: the "
            "window changed during the read; retry"
        )
    codes = np.repeat(np.arange(len(mints)), counts)
    frame = pd.DataFrame(
        {
            "mint": pd.Categorical.from_codes(codes, dtype=dtype),
            "ts": pd.to_datetime(ts, utc=True),
            "venue": venue,
            **floats,
        }
    )
    rows = TrainingRows(frame, features, matrix_columns, nulls)
    result = _data_bounds(settings, bounds, tokens, kept, n_rows)
    logger.info(result.describe())
    return rows, result


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


def read_training_data(
    engine: Engine,
    settings: DataSettings,
    config: Mapping[str, Any],
    *,
    end: datetime | None = None,
) -> tuple[pd.DataFrame | TrainingRows, DataBounds]:
    """A research CLI's read under ``config``: with ``data.prune_columns``, the compact
    :func:`read_training_rows` (the enabled groups' matrix only); else the default
    :func:`read_training_features` frame, unchanged."""
    columns = training_columns(config)
    if columns is None:
        return read_training_features(engine, settings, end=end)
    groups = config.get("features", {}).get("groups", BASELINE_GROUPS)
    return read_training_rows(
        engine, settings, read_columns=columns, matrix_columns=group_columns(groups), end=end
    )
