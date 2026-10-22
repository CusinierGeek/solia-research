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

from sqlalchemy import text
from sqlalchemy.engine import Engine
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


CHAMPION_INDEX = "models_single_champion_idx"
VERSION_CONSTRAINT = "models_version_key"


def _to_json(name: str, value: Mapping[str, Any] | None) -> str | None:
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
    if not version or not version.strip():
        raise ValueError("version must be a non-empty string")
    params = {
        "version": version,
        "path": str(path),
        "is_champion": is_champion,
        "metrics": _to_json("metrics", metrics),
        "hyperparams": _to_json("hyperparams", hyperparams),
        "n_params": None if n_params is None else int(n_params),
    }
    try:
        with engine.begin() as conn:
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
