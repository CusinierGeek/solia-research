"""The internal FastAPI the backend calls (docs/API-CONTRACTS.md "ai-service").

Three endpoints, all JSON:

* ``POST /predict`` — **stateless** inference: the caller sends the token's recent feature window
  (docs/ARCHITECTURE.md §5 — the REST path reads features from the request, never Postgres) and
  gets back, per requested horizon, ``pump_prob`` / ``rug_prob`` / ``expected_return`` /
  ``confidence`` plus the ``model_version``. This is the moderate-cadence path; the per-event
  sniping path is the in-process ONNX export, not this.
* ``GET /model/status`` — the loaded champion's identity, parameter count and training metrics.
* ``GET /health`` — liveness.

The champion is loaded **lazily, cached and refreshed**: importing the app and serving
``/health`` never touch the database, so the process starts even before a model is promoted; the
first ``/predict`` or ``/model/status`` triggers the load and returns ``503`` (not a cached
failure) while none is available. Once loaded, the registry is re-checked at most every
``refresh_seconds`` and a newly promoted champion replaces the cached one (the checkpoint is only
re-read when the version changed); if that check fails, the cached champion keeps serving. Loads
are serialized by a lock, so concurrent first requests read the checkpoint once.

:func:`create_app` takes a loader so tests inject a tiny in-memory champion instead of a database.
Every error — including an unexpected one — uses the shared ``{"error": {"code", "message"}}``
shape.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException

from .features import FEATURE_COLUMNS, build_feature_matrix
from .inference import NO_CHAMPION, Predictor

logger = logging.getLogger(__name__)

Row = dict[str, float | None] | list[float]

DEFAULT_REFRESH_SECONDS = 30.0


@dataclass
class LoadedChampion:
    """What ``/predict`` and ``/model/status`` need: the predictor plus its registry metadata."""

    predictor: Predictor
    created_at: datetime | None = None
    metrics: dict[str, Any] | None = None

    @property
    def version(self) -> str:
        return self.predictor.model_version

    @property
    def param_count(self) -> int:
        return self.predictor.model.n_params


# Called with the currently cached champion (``None`` before the first load); returns it unchanged
# when the registry still points at the same version, else the newly promoted one. Raises
# ``LookupError`` while no champion is registered.
ChampionLoader = Callable[[LoadedChampion | None], LoadedChampion]


# --------------------------------------------------------------------------- request / response


class PredictRequest(BaseModel):
    """A token's recent feature window (oldest row first) plus the horizons to score.

    ``features`` is either a single feature object (the latest vector — a one-row window) or a
    non-empty list of rows: objects keyed by feature name (NULLs imputed to 0, exactly as training
    does) or plain arrays of numbers in :data:`FEATURE_COLUMNS` order. ``horizons``, when given,
    must name at least one horizon.
    """

    mint: str | None = None
    features: list[Row] | dict[str, float | None]
    horizons: list[str] | None = Field(default=None, min_length=1)

    @field_validator("features")
    @classmethod
    def _as_window(cls, value: list[Row] | dict[str, float | None]) -> list[Row]:
        rows = [value] if isinstance(value, dict) else value
        if not rows:
            raise ValueError("the feature window must hold at least one row")
        return rows


class ApiError(HTTPException):
    """An error carrying the contract's ``code`` so the handler can render it verbatim."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail={"code": code, "message": message})


def _window_matrix(rows: list[Row]) -> np.ndarray:
    """Turn the request's ``features`` rows into a ``[L, F]`` float32 window.

    Object rows go through :func:`build_feature_matrix` (same column/NULL rules as training);
    array rows must each hold exactly ``len(FEATURE_COLUMNS)`` numbers. Mixing the two is rejected.
    """
    if all(isinstance(row, dict) for row in rows):
        try:
            return build_feature_matrix(pd.DataFrame(list(rows)))
        except ValueError as exc:
            raise ApiError(400, "invalid_features", str(exc)) from exc
    if all(isinstance(row, list) for row in rows):
        width = len(FEATURE_COLUMNS)
        if any(len(row) != width for row in rows):
            raise ApiError(400, "invalid_features", f"each feature row must hold {width} values")
        matrix = np.asarray(rows, dtype=np.float32)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    raise ApiError(
        400, "invalid_features", "every feature row must be the same shape (all objects or arrays)"
    )


# --------------------------------------------------------------------------- app


def create_app(
    load_champion: ChampionLoader | None = None,
    *,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Build the ai-service app. ``load_champion`` resolves (and refreshes) the champion; the
    default reads the registry + checkpoint over one shared database engine. The registry is
    re-checked at most every ``refresh_seconds`` (``clock`` is injectable for tests)."""
    if refresh_seconds < 0:
        raise ValueError(f"refresh_seconds must be >= 0, got {refresh_seconds}")
    app = FastAPI(title="Solia ai-service", version="0.1.0")
    app.state.load_champion = load_champion or registry_loader()
    app.state.champion = None
    lock = threading.Lock()
    checked_at = 0.0

    def champion() -> LoadedChampion:
        nonlocal checked_at
        with lock:
            current = app.state.champion
            if current is not None and clock() - checked_at < refresh_seconds:
                return current
            try:
                app.state.champion = app.state.load_champion(current)
            except LookupError as exc:
                if current is None:
                    raise ApiError(503, "no_champion", str(exc)) from exc
                logger.warning("champion refresh found no champion; keeping %s", current.version)
            except Exception as exc:
                if current is None:
                    logger.exception("loading the champion failed")
                    raise ApiError(
                        503, "champion_unavailable", f"the champion could not be loaded: {exc}"
                    ) from exc
                logger.warning(
                    "champion refresh failed (%s); keeping %s", exc, current.version, exc_info=True
                )
            checked_at = clock()
            return app.state.champion

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model/status")
    def model_status() -> dict[str, Any]:
        loaded = champion()
        return {
            "champion_version": loaded.version,
            "created_at": loaded.created_at.isoformat() if loaded.created_at else None,
            "param_count": loaded.param_count,
            "metrics": loaded.metrics or {},
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        window = _window_matrix(request.features)  # a bad request is a 400, champion or not
        predictor = champion().predictor
        try:
            prediction = predictor.predict(window, request.horizons)
        except (ValueError, TypeError) as exc:
            raise ApiError(400, "invalid_request", str(exc)) from exc
        return {"mint": request.mint, **prediction.as_dict()}

    @app.exception_handler(HTTPException)
    def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = detail
        else:
            body = {"code": "http_error", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": body})

    @app.exception_handler(RequestValidationError)
    def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": _first_error(exc)}},
        )

    @app.exception_handler(Exception)
    def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "internal server error"}},
        )

    return app


def _first_error(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid request body"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = first.get("msg", "invalid value")
    return f"{location}: {message}" if location else message


def registry_loader(engine_factory: Callable[[], Any] | None = None) -> ChampionLoader:
    """The database-backed loader: reads the registry champion over **one** engine, created on
    first use (so importing the app needs no database) and reused by every later call.

    The checkpoint is only loaded when the registry's champion version differs from the cached
    one; the predictor is stamped with the registry version — the identity callers echo back.
    """
    engine: Any = None

    def load(current: LoadedChampion | None) -> LoadedChampion:
        nonlocal engine
        from .registry import read_champion

        if engine is None:
            from .db import make_engine

            engine = (engine_factory or make_engine)()
        record = read_champion(engine)
        if record is None:
            raise LookupError(NO_CHAMPION)
        if current is not None and current.version == record.version:
            return current
        predictor = Predictor.from_record(record)
        return LoadedChampion(
            predictor=predictor,
            created_at=record.created_at,
            metrics=dict(predictor.metadata.get("metrics") or {}),
        )

    return load


# Importable ASGI app for ``uvicorn solia_ai.api:app`` (the default, database-backed loader).
app = create_app()
