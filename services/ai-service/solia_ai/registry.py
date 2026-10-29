"""Write the ``models`` registry row for a trained checkpoint.

The registry (see docs/DATA-MODEL.md) is the source of truth for which checkpoint inference serves
(``is_champion``) and what each version scored (``metrics``) with which knobs (``hyperparams``). A
partial unique index enforces a single champion, so promoting a new one demotes the old champion in
the **same transaction** — never leaving the table with two or zero champions mid-flip.

Kept torch-free (only SQLAlchemy + json) so it stays importable from light contexts, like the DB
access layer it sits beside.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError


@dataclass(frozen=True)
class ModelRecord:
    """The persisted registry row (``id`` and ``created_at`` assigned by the database)."""

    id: int
    version: str
    path: str
    is_champion: bool
    n_params: int | None
    created_at: datetime


# The ``models`` columns a :class:`ModelRecord` is read from (:func:`record_columns`).
RECORD_FIELDS = ("id", "version", "path", "is_champion", "n_params", "created_at")
CHAMPION_INDEX = "models_single_champion_idx"
VERSION_CONSTRAINT = "models_version_key"

# ``training_events`` audit values, mirroring the SQL CHECKs (migration 0010). A run that crashed
# or errored is recorded with ``finished_at`` and decision ``failed``; a ``rollback`` only restores
# the previous champion. Neither trained a model on the new trades, so only the remaining decisions
# (RESET_TRADE_COUNT_DECISIONS) restart the trades-since-last-train count, as in the backend.
TRAINING_TRIGGERS = ("initial", "scheduled", "n_trades", "manual")
TRAINING_DECISIONS = ("promoted", "rejected", "rollback", "failed")
NO_MODEL_TRAINED_DECISIONS = ("rollback", "failed")
RESET_TRADE_COUNT_DECISIONS = tuple(
    d for d in TRAINING_DECISIONS if d not in NO_MODEL_TRAINED_DECISIONS
)


def record_columns(alias: str | None = None) -> str:
    """The SELECT list of :data:`RECORD_FIELDS`, qualified by a table ``alias`` if given."""
    return ", ".join(f"{alias}.{f}" if alias else f for f in RECORD_FIELDS)


def row_to_record(row: Any) -> ModelRecord:
    """A :class:`ModelRecord` from a row selected with :func:`record_columns`."""
    return ModelRecord(
        id=int(row.id),
        version=row.version,
        path=row.path,
        is_champion=bool(row.is_champion),
        n_params=None if row.n_params is None else int(row.n_params),
        created_at=row.created_at,
    )


def to_json(name: str, value: Mapping[str, Any] | None) -> str | None:
    """Serialize for a JSONB column. NaN/Infinity are not JSON, so they are rejected up front
    instead of failing inside PostgreSQL."""
    if value is None:
        return None
    try:
        return json.dumps(dict(value), allow_nan=False)
    except ValueError as exc:
        raise ValueError(f"{name} must only hold finite numbers: {dict(value)!r}") from exc


def _constraint_name(exc: IntegrityError) -> str | None:
    diag = getattr(exc.orig, "diag", None)
    return getattr(diag, "constraint_name", None)


def read_champion(engine: Engine) -> ModelRecord | None:
    """The current champion registry row, or ``None`` if nothing has been promoted yet.

    The single-champion partial index guarantees at most one row matches, so inference always
    serves an unambiguous checkpoint. Torch-free like the rest of this module: the caller
    (:mod:`solia_ai.inference`) turns ``path`` into a loaded model.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT {record_columns()} FROM models WHERE is_champion")
        ).one_or_none()
    return None if row is None else row_to_record(row)


def last_train_cutoff(engine: Engine) -> datetime | None:
    """The ``started_at`` of the latest run that trained a model (its data cut-off), or ``None``
    before any did. Trades closed strictly after it are the ones not learned from yet; a run in
    flight, ``failed`` or ``rollback`` does not move it (same rule as the backend's count).
    """
    with engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT max(started_at) FROM training_events " "WHERE decision IN :decisions"
            ).bindparams(bindparam("decisions", expanding=True)),
            {"decisions": list(RESET_TRADE_COUNT_DECISIONS)},
        ).scalar_one()


def register_model(
    engine: Engine,
    *,
    version: str,
    path: str,
    metrics: Mapping[str, Any] | None = None,
    hyperparams: Mapping[str, Any] | None = None,
    n_params: int | None = None,
    is_champion: bool = False,
) -> ModelRecord:
    """Insert a ``models`` row and return it.

    When ``is_champion`` is set, any existing champion is demoted first (same transaction), so the
    single-champion index always holds. Raises :class:`ValueError` if ``version`` already exists
    (the column is ``UNIQUE``), if a concurrent run promoted another champion first, or if
    ``metrics``/``hyperparams`` hold non-finite numbers.
    """
    with engine.begin() as conn:
        return insert_model(
            conn,
            version=version,
            path=path,
            metrics=metrics,
            hyperparams=hyperparams,
            n_params=n_params,
            is_champion=is_champion,
        )


def insert_model(
    conn: Connection,
    *,
    version: str,
    path: str,
    metrics: Mapping[str, Any] | None = None,
    hyperparams: Mapping[str, Any] | None = None,
    n_params: int | None = None,
    is_champion: bool = False,
) -> ModelRecord:
    """:func:`register_model` inside the caller's transaction ``conn``, so a promotion can commit
    together with other writes (e.g. the retrain's ``training_events`` row) or not at all."""
    if not version or not version.strip():
        raise ValueError("version must be a non-empty string")
    params = {
        "version": version,
        "path": str(path),
        "is_champion": is_champion,
        "metrics": to_json("metrics", metrics),
        "hyperparams": to_json("hyperparams", hyperparams),
        "n_params": None if n_params is None else int(n_params),
    }
    try:
        # A savepoint: a constraint violation must not abort the caller's whole transaction.
        with conn.begin_nested():
            if is_champion:
                conn.execute(text("UPDATE models SET is_champion = FALSE WHERE is_champion"))
            row = conn.execute(
                text(
                    "INSERT INTO models (version, path, is_champion, metrics, hyperparams, "
                    "n_params) VALUES (:version, :path, :is_champion, CAST(:metrics AS JSONB), "
                    "CAST(:hyperparams AS JSONB), :n_params) RETURNING id, created_at"
                ),
                params,
            ).one()
    except IntegrityError as exc:
        if _constraint_name(exc) == CHAMPION_INDEX:
            raise ValueError(
                f"cannot promote {version!r}: another champion was promoted concurrently"
            ) from exc
        if _constraint_name(exc) == VERSION_CONSTRAINT:
            raise ValueError(f"model version {version!r} already exists in the registry") from exc
        raise
    return ModelRecord(
        id=int(row.id),
        version=version,
        path=str(path),
        is_champion=is_champion,
        n_params=params["n_params"],
        created_at=row.created_at,
    )
