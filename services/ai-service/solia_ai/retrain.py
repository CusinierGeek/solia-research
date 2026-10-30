"""Continuous-training orchestrator: is a retrain due, run it end to end, or roll back.

One entry point for docs/ML-MODEL.md "Continuous training loop":

1. **Due?** Closed paper trades since the last run that trained a model
   (:func:`count_trades_since_last_train`, the backend's ``/model/status`` rule) against
   ``RETRAIN_EVERY_N_TRADES``. ``manual`` runs and ``--force`` skip the check.
2. **One run at a time.** A PostgreSQL session advisory lock (:data:`LOCK_KEY`) is taken for the
   whole run (and for a rollback); a second run is refused (:class:`RetrainBusyError`), not queued.
   A heartbeat keeps the lock's connection alive and checks it is still held; a run that lost it
   records no decision (:class:`RetrainLockLostError`).
3. **Audit trail.** A ``training_events`` row is written at start (trigger, champion) and completed
   at the end (``finished_at``, ``n_samples``, the gate's comparison, decision, notes). Whatever
   raises after the start ends the row as ``failed`` with the error in ``notes``.
4. **Train + evaluate** (:func:`train_and_evaluate`): ``features`` is read bounded
   (:func:`solia_ai.db.read_training_features`: the ``data.lookback_days`` window, at most
   ``data.max_rows`` rows of whole tokens; the bounds used are logged and recorded with the run and
   the candidate); the promotion gate's walk-forward held-out windows are cut once from it; one
   candidate is trained per gate seed (:func:`solia_ai.candidate.train_candidate`, pinned to the
   first window's start, so it never sees the held-out data); every candidate and the champion are
   scored on the same windows. The champion is a fixed model, so its metrics on a window are shared
   by every seed of that window. With **no champion** yet a single candidate is trained (the first
   seed): the seeds only matter for a paired comparison, and the first valid candidate is promoted
   anyway. Each candidate's start / end and every epoch (loss, duration, samples/s, ETA) are
   logged, and handed to ``on_progress`` (the API's ``/model/status``).
5. **Decide + record** (:func:`solia_ai.promotion.decide_promotion`): the first seed's candidate
   always gets a ``models`` row (with its metrics); a promotion flips ``is_champion`` in the same
   transaction that completes the ``training_events`` row. The API re-reads the registry champion
   every ``DEFAULT_REFRESH_SECONDS`` (:mod:`solia_ai.api`), so a promotion is served without a
   restart.

**Rollback** (:func:`rollback`) restores the champion the current one replaced (found through the
``promoted`` event that crowned it), after :func:`solia_ai.promotion.check_rollback` validated its
checkpoint, and records a ``rollback`` event. Refused (:class:`RollbackRefusedError`) when there is
no previous champion, its checkpoint is missing or it no longer fits the config.

**Retention** (:func:`prune`): after each run that recorded a decision, and on demand
(``python -m solia_ai.retrain prune [--dry-run] [--delete-orphans]``), old checkpoint files are
deleted under the same lock (:mod:`solia_ai.retention`; a dry run only reads, without it). A failed
prune is only logged: the run already committed.

Every step of a run is logged to stdout and to the ``logs`` table (service ``ai-service``); a skip
(not due) only to stdout. DB writes are raw
SQL; the data-pipeline tables are only read. Paper trading only: nothing here places an order.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from .config import ConfigError
from .db import add_data_bounds_arguments
from .registry import (
    ModelRecord,
    insert_model,
    last_train_cutoff,
    previous_champion,
    read_champion,
    to_json,
)
from .retention import (
    PruneError,
    PruneOutcome,
    PrunePlan,
    RetentionSettings,
    apply_prune,
    describe,
    fallback_directory,
    model_directory,
    plan_prune,
)

logger = logging.getLogger(__name__)

# progress(event): see train_and_evaluate (and solia_ai.train.fit for "epoch" events).
ProgressFn = Callable[[dict[str, Any]], None]

SERVICE = "ai-service"
THRESHOLD_ENV = "RETRAIN_EVERY_N_TRADES"
DEFAULT_THRESHOLD = 500  # same default as the backend (docker-compose.yml)
# Triggers a run can be started with; ``initial`` is the bootstrap ``train.py`` run.
RUN_TRIGGERS = ("n_trades", "scheduled", "manual")
# Session advisory-lock key (arbitrary, fixed) shared by every retrain and rollback.
LOCK_KEY = 0x50_4C_49_41_07_05
_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING}


class RetrainBusyError(RuntimeError):
    """Another retrain or rollback holds the lock: this one is refused, not queued."""


class RollbackRefusedError(RuntimeError):
    """The previous champion cannot be restored; ``reasons`` says why."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("rollback refused: " + "; ".join(self.reasons))


# --------------------------------------------------------------------------- trigger


def retrain_threshold(lookup: Callable[[str], str | None] = os.environ.get) -> int:
    """``RETRAIN_EVERY_N_TRADES`` (default :data:`DEFAULT_THRESHOLD`), a positive integer."""
    raw = lookup(THRESHOLD_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_THRESHOLD
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{THRESHOLD_ENV} must be a positive integer, got {raw!r}")
    return value


def count_trades_since_last_train(engine: Engine) -> int:
    """Closed trades closed strictly after the latest run that trained a model started, or every
    closed trade before any did — the backend's ``trades_since_last_train``."""
    cutoff = last_train_cutoff(engine)
    query = "SELECT count(*) FROM trades WHERE status = 'closed' AND closed_at IS NOT NULL"
    params: dict[str, Any] = {}
    if cutoff is not None:
        query += " AND closed_at > :since"
        params["since"] = cutoff
    with engine.connect() as conn:
        return int(conn.execute(text(query), params).scalar_one())


def is_due(trades: int, threshold: int, *, trigger: str, force: bool = False) -> bool:
    """A run is due at ``threshold`` closed trades; a ``manual`` or forced run always is."""
    if trigger not in RUN_TRIGGERS:
        raise ValueError(f"trigger must be one of {RUN_TRIGGERS}, got {trigger!r}")
    return force or trigger == "manual" or trades >= threshold


# --------------------------------------------------------------------------- lock


class RetrainLockLostError(RuntimeError):
    """The lock's connection died mid-run, so PostgreSQL released it: another run may be going."""


# Whether this session still holds :data:`LOCK_KEY` (a bigint key is split over classid/objid).
_HELD_SQL = text(
    "SELECT count(*) > 0 FROM pg_locks WHERE locktype = 'advisory' AND granted "
    "AND pid = pg_backend_pid() AND classid::bigint = :hi AND objid::bigint = :lo "
    "AND objsubid = 1"
)
_KEY_PARTS = {"hi": LOCK_KEY >> 32, "lo": LOCK_KEY & 0xFFFFFFFF}
HEARTBEAT_SECONDS = 60.0


class RetrainLock:
    """The held lock. A heartbeat keeps its otherwise idle connection alive through a long training
    (so no idle timeout drops it) and notices if it is lost anyway; :meth:`ensure_held` is checked
    before every registry write, so a run whose lock was released never records a decision."""

    def __init__(self, conn: Connection, heartbeat_seconds: float) -> None:
        self._conn = conn
        self._guard = threading.Lock()  # the heartbeat and ensure_held share one connection
        self._stop = threading.Event()
        self._lost: str | None = None
        self._thread = threading.Thread(
            target=self._beat, args=(heartbeat_seconds,), name="retrain-lock", daemon=True
        )

    def _check(self) -> None:
        if self._lost is not None:
            return
        try:
            with self._guard:
                held = self._conn.execute(_HELD_SQL, _KEY_PARTS).scalar_one()
                self._conn.commit()
        except Exception as exc:
            self._lost = f"its connection failed ({exc})"
            return
        if not held:
            self._lost = "PostgreSQL no longer holds it"

    def _beat(self, interval: float) -> None:
        while self._lost is None and not self._stop.wait(interval):
            self._check()

    def _start(self) -> None:
        self._thread.start()

    def _halt(self) -> None:
        self._stop.set()
        self._thread.join()

    def ensure_held(self) -> None:
        """Raise :class:`RetrainLockLostError` unless the lock is still held."""
        self._check()
        if self._lost is not None:
            raise RetrainLockLostError(
                f"the retrain lock was lost mid-run ({self._lost}): another run may have started, "
                "so this one records no decision"
            )


@contextmanager
def retrain_lock(
    engine: Engine, *, heartbeat_seconds: float = HEARTBEAT_SECONDS
) -> Iterator[RetrainLock]:
    """Hold the session advisory lock for the block; :class:`RetrainBusyError` if it is taken.

    The lock lives on one dedicated connection: released explicitly, or by PostgreSQL when that
    connection dies (it is then discarded from the pool rather than reused still holding it).
    """
    conn = engine.connect()
    try:
        acquired = conn.execute(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": LOCK_KEY}
        ).scalar_one()
        conn.commit()
        if not acquired:
            raise RetrainBusyError("another retrain or rollback is running; try again later")
        lock = RetrainLock(conn, heartbeat_seconds)
        lock._start()
        try:
            yield lock
        finally:
            lock._halt()
            try:
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": LOCK_KEY})
                conn.commit()
            except Exception as exc:
                # Dropping the connection releases the lock server-side; raising here would mask
                # the body's own error or fail a run that already committed.
                logger.warning(
                    "could not release the retrain lock, dropping its connection: %s", exc
                )
                conn.invalidate()
    finally:
        conn.close()


# --------------------------------------------------------------------------- audit


def db_log(
    engine: Engine, level: str, message: str, context: Mapping[str, Any] | None = None
) -> None:
    """Log to stdout and to the ``logs`` table. A failed ``logs`` insert is only reported: the
    audit trail lives in ``training_events``, so logging never fails a run."""
    logger.log(_LOG_LEVELS.get(level, logging.ERROR), message)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO logs (service, level, message, context) "
                    "VALUES (:service, :level, :message, CAST(:context AS JSONB))"
                ),
                {
                    "service": SERVICE,
                    "level": level,
                    "message": message,
                    "context": to_json("context", context),
                },
            )
    except Exception as exc:
        logger.warning("could not write the log line to the logs table: %s", exc)


def start_event(engine: Engine, *, trigger: str, champion_id: int | None) -> int:
    """Insert the in-flight ``training_events`` row (decision and ``finished_at`` NULL)."""
    with engine.begin() as conn:
        return int(
            conn.execute(
                text(
                    "INSERT INTO training_events (trigger, champion_model_id) "
                    "VALUES (:trigger, :champion) RETURNING id"
                ),
                {"trigger": trigger, "champion": champion_id},
            ).scalar_one()
        )


ABANDONED_NOTE = "abandoned: its process stopped before the run finished"


def close_abandoned_runs(engine: Engine) -> list[int]:
    """Record every in-flight ``training_events`` row as ``failed``; return their ids.

    Call it only while holding the retrain lock: a run holds that lock from its ``start_event`` to
    its decision, so under it any row still in flight belongs to a process that died (its daemon
    worker never reached :func:`_fail`)."""
    with engine.begin() as conn:
        ids = (
            conn.execute(
                text(
                    "UPDATE training_events SET finished_at = now(), decision = 'failed', "
                    "notes = :notes WHERE finished_at IS NULL RETURNING id"
                ),
                {"notes": ABANDONED_NOTE},
            )
            .scalars()
            .all()
        )
    return sorted(int(i) for i in ids)


def recover_abandoned_runs(engine: Engine) -> list[int]:
    """At startup: close the runs a previous process left in flight (see
    :func:`close_abandoned_runs`). Nothing is closed while another process holds the lock — its
    run is alive, and the next run closes whatever is left."""
    try:
        with retrain_lock(engine):
            ids = close_abandoned_runs(engine)
    except RetrainBusyError:
        return []
    if ids:
        db_log(
            engine,
            "warn",
            f"retrain run(s) {', '.join(f'#{i}' for i in ids)} abandoned by a stopped process: "
            "recorded as failed",
            {"training_event_ids": ids},
        )
    return ids


def finish_event(
    conn: Connection,
    event_id: int,
    *,
    decision: str,
    n_samples: int | None = None,
    candidate_id: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> None:
    """Complete an in-flight run inside the caller's transaction (decision + ``finished_at``)."""
    updated = conn.execute(
        text(
            "UPDATE training_events SET finished_at = now(), decision = :decision, "
            "n_samples = :n_samples, candidate_model_id = :candidate, "
            "metrics = CAST(:metrics AS JSONB), notes = :notes "
            "WHERE id = :id AND finished_at IS NULL"
        ),
        {
            "id": event_id,
            "decision": decision,
            "n_samples": n_samples,
            "candidate": candidate_id,
            "metrics": to_json("metrics", metrics),
            "notes": notes,
        },
    ).rowcount
    if updated != 1:
        raise RuntimeError(f"training event {event_id} is not an in-flight run")


# --------------------------------------------------------------------------- evaluation


@dataclass
class Evaluation:
    """A trained candidate and the paired gate runs of candidate and champion."""

    candidate: Any  # solia_ai.candidate.CandidateResult (the one registered)
    candidate_runs: dict[tuple[int, int], dict[str, float | None]]
    champion_runs: dict[tuple[int, int], dict[str, float | None]] | None
    windows: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


EvaluateFn = Callable[..., Evaluation]


def _column_view(model: Any) -> Any:
    """``model`` fed current-spec windows: a model trained on an ordered subset of the columns
    only sees those, as :class:`solia_ai.inference.Predictor` serves it (same
    :func:`solia_ai.inference.feature_index`)."""
    import torch
    from torch import nn

    from .inference import feature_index

    columns = feature_index(model.feature_columns)
    if columns is None:
        return model
    index = torch.tensor(columns)

    class ColumnView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model
            self.horizons = model.horizons
            self.seq_len = model.seq_len

        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> Any:
            return self.inner(x.index_select(-1, index), padding_mask)

    return ColumnView()


def _load_scorer(checkpoint: Mapping[str, Any]) -> tuple[Any, Any]:
    from .calibration import Calibration
    from .inference import build_model

    model = build_model(dict(checkpoint))
    return _column_view(model), Calibration.from_checkpoint(checkpoint, model.horizons)


def eligibility_note(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    """A warning when the champion was trained under another eligibility rule than ``config``'s
    (``None`` when the rules match): both models are scored on the config's eligible windows, and
    the bot trades the champion's population until a candidate trained under the new rule wins."""
    from .eligibility import EligibilityRule

    trained = EligibilityRule.of(checkpoint.get("hyperparams"))
    wanted = EligibilityRule.from_config(config)
    if trained == wanted:
        return None
    return (
        f"champion {checkpoint.get('version')} was trained under eligibility rule "
        f"[{trained.describe()}], the config's is [{wanted.describe()}]: both are scored on the "
        "config's eligible windows; the bot keeps trading the champion's population until a "
        "candidate trained under the config's rule is promoted"
    )


def load_champion_checkpoint(
    champion: ModelRecord | None, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The champion's checkpoint, checked comparable with ``config``. Run before any data is read,
    so a champion that can't be compared fails the run cheaply.

    Raises:
        ValueError: the checkpoint can't be loaded, or serves other horizons than the config.
    """
    from .inference import load_checkpoint, resolve_checkpoint_path

    if champion is None:
        return None
    try:
        checkpoint = load_checkpoint(resolve_checkpoint_path(champion.path))
    except Exception as exc:
        raise ValueError(
            f"champion {champion.version} checkpoint {champion.path} could not be loaded "
            f"({exc}): it can't be compared; restore it, roll back, or promote a model "
            "with solia_ai.train"
        ) from exc
    if list(checkpoint.get("horizons") or []) != list(config["horizons"]):
        raise ValueError(
            f"champion {champion.version} serves horizons {checkpoint.get('horizons')}, "
            f"the config {list(config['horizons'])}: no paired comparison is possible"
        )
    return checkpoint


def resize_windows(samples: Any, df: Any, config: Mapping[str, Any], seq_len: int) -> Any:
    """``samples`` (same windows, same order) holding up to ``seq_len`` rows each, so a champion
    trained on another ``seq_len`` is scored on windows shaped like its training ones."""
    from .dataset import config_samples

    if seq_len == samples.seq_len:
        return samples
    if seq_len < samples.seq_len:  # keep each window's most recent rows
        start = np.maximum(samples.start_row, samples.end_row - seq_len + 1)
        return replace(samples, seq_len=seq_len, start_row=start)
    # Longer: rebuilt with the same min_seq_len, the windows are the same with more history.
    model = dict(config["model"])
    model["min_seq_len"] = model.get("min_seq_len", model["seq_len"])
    model["seq_len"] = seq_len
    resized = config_samples(df, {**config, "model": model})
    if not np.array_equal(resized.end_row, samples.end_row):
        raise ValueError(f"windows rebuilt at seq_len {seq_len} don't match the gate's windows")
    return resized


def train_and_evaluate(
    df: Any,
    config: Mapping[str, Any],
    *,
    champion: ModelRecord | None,
    champion_checkpoint: Mapping[str, Any] | None = None,
    model_dir: str | os.PathLike[str] | None = None,
    device: Any = None,
    version: str | None = None,
    progress: ProgressFn | None = None,
) -> Evaluation:
    """Train one candidate per gate seed (a single one without a champion) and score them and the
    ``champion`` on the gate's walk-forward held-out windows (see the module doc). Only the first
    seed's checkpoint is kept.

    The champion is never scored in-sample: the held-out windows end after its own training data
    plus the embargo (the held-out share shrinks if fewer windows are that recent). It is fed
    windows of its own ``seq_len``, and only windows it would serve (at least its ``min_seq_len``
    rows) are scored, for both models, so the comparison stays paired. ``progress`` gets a
    ``candidate_start`` / ``candidate_end`` event around each candidate (``candidate`` 1-based,
    ``candidates``, ``seed``; the end adds ``best_epoch``, ``epochs_run`` and ``seconds``) and the
    fit's ``epoch`` events tagged with the same ``candidate`` / ``candidates`` / ``seed``.

    Raises:
        ValueError / IncompatibleCheckpointError: too little data (or none yet past the champion's
            training data), or a champion that can't be loaded or scored on the config's horizons
            (no paired comparison is possible).
    """
    from .ablation import run_metrics, walk_forward_windows
    from .candidate import CandidateSettings, champion_data_end_ts, train_candidate
    from .dataset import SequenceDataset, config_samples, embargo_seconds
    from .evaluate import ReferencePolicy, default_policy_horizon, evaluate
    from .inference import load_checkpoint, recorded_min_seq_len
    from .promotion import PromotionSettings
    from .train import default_version

    gate = PromotionSettings.from_config(config)
    settings = CandidateSettings.from_config(config)
    notes: list[str] = []

    champion_ckpt = champion_checkpoint
    if champion_ckpt is None and champion is not None:
        champion_ckpt = load_champion_checkpoint(champion, config)

    if champion_ckpt is not None and (note := eligibility_note(champion_ckpt, config)):
        notes.append(note)

    samples = config_samples(df, config)
    embargo = embargo_seconds(samples.horizons)
    scoreable = np.ones(len(samples), bool)
    eval_fraction = settings.heldout_fraction
    champion_scorer = champion_samples = fresh_after = None
    if champion_ckpt is not None:
        champion_scorer = _load_scorer(champion_ckpt)
        seq_len = champion_scorer[0].seq_len
        champion_samples = resize_windows(samples, df, config, seq_len)
        lengths = champion_samples.end_row - champion_samples.start_row + 1
        scoreable &= lengths >= recorded_min_seq_len(champion_ckpt, seq_len)
        seen = champion_data_end_ts(champion_ckpt)
        if seen is None:
            notes.append(
                f"champion {champion.version} does not record its training data's end: it may be "
                "scored on windows it was trained on, so the comparison may favour it"
            )
        else:
            fresh_after = seen + embargo
            fresh = samples.end_ts > fresh_after
            past = float(np.mean(fresh))
            if past == 0:
                raise ValueError(
                    f"no window ends after champion {champion.version}'s training data (up to "
                    f"{seen}) plus the {embargo}s embargo: wait for more data before retraining"
                )
            scoreable &= fresh
            if past < eval_fraction:
                eval_fraction = past
                notes.append(
                    f"held-out cut to the {past:.1%} of windows past champion "
                    f"{champion.version}'s training data"
                )
    windows = walk_forward_windows(
        samples.end_ts,
        n_windows=gate.windows,
        eval_fraction=eval_fraction,
        val_fraction=settings.val_fraction,
        embargo=embargo,
    )
    windows = [replace(w, eval_idx=w.eval_idx[scoreable[w.eval_idx]]) for w in windows]
    empty = [w.index for w in windows if len(w.eval_idx) == 0]
    if empty:
        raise ValueError(
            f"held-out window(s) {empty} hold no window the champion can be scored on: wait for "
            "more data or use fewer gate windows"
        )
    heldout_start = windows[0].eval_start_ts
    if fresh_after is not None:  # the candidate trains up to the champion's data, never past it
        heldout_start = max(heldout_start, fresh_after)
    policy = ReferencePolicy(horizon=default_policy_horizon(samples.horizons))

    def score(scorer: tuple[Any, Any], scored: Any) -> list[dict[str, float | None]]:
        model, calibration = scorer
        return [
            run_metrics(
                evaluate(
                    model,
                    SequenceDataset(scored, w.eval_idx),
                    policy=policy,
                    calibration=calibration,
                )
            )
            for w in windows
        ]

    base_seed = int(config["training"].get("seed", 0))
    n_seeds = gate.seeds if champion_ckpt is not None else 1
    if n_seeds < gate.seeds:
        notes.append(
            f"no champion: one candidate trained (seed {base_seed}) instead of {gate.seeds}: the "
            "gate's seeds only matter for a paired comparison"
        )
    seeds = [base_seed + i for i in range(n_seeds)]
    version = version or default_version()
    report = progress or (lambda event: None)
    candidate_runs: dict[tuple[int, int], dict[str, float | None]] = {}
    primary = None
    with tempfile.TemporaryDirectory(prefix="solia-retrain-") as scratch:
        for i, seed in enumerate(seeds):
            tag = {"candidate": i + 1, "candidates": len(seeds), "seed": seed}
            report({"event": "candidate_start", **tag})
            started = time.monotonic()
            cfg = copy.deepcopy(dict(config))
            cfg["training"] = {**cfg["training"], "seed": seed}
            result = train_candidate(
                df,
                cfg,
                champion=champion_ckpt,
                heldout_start_ts=heldout_start,
                model_dir=model_dir if i == 0 else scratch,
                version=version if i == 0 else f"{version}-seed{seed}",
                device=device,
                samples=samples,
                progress=lambda event, tag=tag: report({**event, **tag}),
            )
            report(
                {
                    "event": "candidate_end",
                    **tag,
                    "best_epoch": result.best_epoch + 1,
                    "epochs_run": len(result.history),
                    "seconds": time.monotonic() - started,
                }
            )
            primary = primary or result
            scorer = _load_scorer(load_checkpoint(result.checkpoint_path))
            for w, metrics in zip(windows, score(scorer, samples), strict=True):
                candidate_runs[(w.index, seed)] = metrics

    champion_runs = None
    if champion_scorer is not None:
        per_window = score(champion_scorer, champion_samples)
        champion_runs = {
            (w.index, seed): metrics
            for w, metrics in zip(windows, per_window, strict=True)
            for seed in seeds
        }
    if primary.warm_start.reasons:
        notes.append("warm start: " + "; ".join(primary.warm_start.reasons))
    return Evaluation(
        candidate=primary,
        candidate_runs=candidate_runs,
        champion_runs=champion_runs,
        windows=[w.describe() for w in windows],
        notes=notes,
    )


def describe_progress(event: Mapping[str, Any]) -> str:
    """One line for a training progress event (:func:`train_and_evaluate`)."""
    who = f"candidate {event.get('candidate', 1)}/{event.get('candidates', 1)}"
    if "seed" in event:
        who += f" (seed {event['seed']})"
    kind = event.get("event")
    if kind == "candidate_start":
        return f"{who} started"
    if kind == "candidate_end":
        return (
            f"{who} trained in {event['seconds']:.0f}s: best epoch {event['best_epoch']} of "
            f"{event['epochs_run']} run"
        )
    if kind == "epoch":
        from .train import describe_epoch

        return f"{who} {describe_epoch(event)}"
    return f"{who} {kind}"


def _is_finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def json_safe(value: Any) -> Any:
    """``value`` with every NaN/Infinity replaced by ``None`` (JSONB holds no non-finite number),
    so an undefined metric is recorded as missing instead of failing the run."""
    if isinstance(value, Mapping):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def heldout_means(runs: Mapping[Any, Mapping[str, float | None]]) -> dict[str, float | None]:
    """Mean of each metric over the runs where it is defined (``None`` where it never is)."""
    keys = sorted({k for metrics in runs.values() for k in metrics})
    out: dict[str, float | None] = {}
    for key in keys:
        values = [m[key] for m in runs.values() if _is_finite(m.get(key))]
        out[key] = float(np.mean(values)) if values else None
    return out


# --------------------------------------------------------------------------- run


@dataclass(frozen=True)
class RetrainOutcome:
    """What a run did: ``skipped`` (not due), ``promoted`` or ``rejected``."""

    status: str
    trades_since_last_train: int
    threshold: int
    event_id: int | None = None
    candidate_id: int | None = None
    candidate_version: str | None = None
    reasons: tuple[str, ...] = ()
    pruned: tuple[str, ...] = ()  # versions whose checkpoint the retention policy pruned


def run_retrain(
    engine: Engine,
    config: Mapping[str, Any] | None = None,
    *,
    trigger: str = "n_trades",
    force: bool = False,
    threshold: int | None = None,
    model_dir: str | os.PathLike[str] | None = None,
    device: Any = None,
    lookback_days: float | None = None,
    max_rows: int | None = None,
    evaluate_fn: EvaluateFn = train_and_evaluate,
    read_data_fn: Callable[..., Any] | None = None,
    load_champion_fn: Callable[..., Any] = load_champion_checkpoint,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    on_started: Callable[[int], None] | None = None,
    on_progress: Callable[[int, dict[str, Any]], None] | None = None,
) -> RetrainOutcome:
    """Run one retrain if due (see the module doc). ``lookback_days`` / ``max_rows`` override the
    config's ``data`` bounds (the CLI flags). ``evaluate_fn`` / ``read_data_fn`` (``(engine,
    DataSettings) -> (features, DataBounds)``) / ``load_champion_fn`` are injectable so tests stub
    the training. ``on_started`` is called with
    the ``training_events`` id as soon as the run is recorded (the API answers ``202`` with it
    while the run goes on); it is not called for a skipped run, and an error it raises is only
    logged. Every training progress event (:func:`train_and_evaluate`) is logged and handed to
    ``on_progress(event_id, event)`` (an error it raises is only logged too).

    Raises:
        RetrainBusyError: another run holds the lock (nothing is recorded).
        RetrainLockLostError: the lock was lost mid-run (the run is recorded as failed).
        ConfigError: invalid data bounds (before anything is locked or recorded).
        Exception: whatever failed the run, after its event was recorded as ``failed``.
    """
    from .config import load_config
    from .db import DataSettings
    from .promotion import PromotionSettings, decide_promotion

    cfg = load_config() if config is None else config
    threshold = retrain_threshold() if threshold is None else threshold
    if read_data_fn is None:
        from .db import read_training_features as read_data_fn
    gate = PromotionSettings.from_config(cfg)
    data_settings = DataSettings.from_config(cfg, lookback_days=lookback_days, max_rows=max_rows)

    with retrain_lock(engine, heartbeat_seconds=heartbeat_seconds) as lock:
        trades = count_trades_since_last_train(engine)
        if not is_due(trades, threshold, trigger=trigger, force=force):
            # stdout only: a scheduler polls often, and a skip changes nothing worth a logs row.
            logger.info(
                "retrain not due: %s/%s closed trades since the last training", trades, threshold
            )
            return RetrainOutcome("skipped", trades, threshold)

        abandoned = close_abandoned_runs(engine)  # under the lock: left by a dead process
        if abandoned:
            logger.warning("closed abandoned retrain run(s) %s as failed", abandoned)
        champion = read_champion(engine)
        event_id = start_event(engine, trigger=trigger, champion_id=champion and champion.id)
        if on_started is not None:
            try:
                on_started(event_id)
            except Exception:
                logger.exception("retrain #%s: the on_started callback failed", event_id)
        context = {"training_event_id": event_id, "trigger": trigger}

        def report(event: dict[str, Any]) -> None:
            db_log(
                engine,
                "info",
                f"retrain #{event_id}: {describe_progress(event)}",
                {**context, "progress": json_safe(event)},
            )
            if on_progress is not None:
                try:
                    on_progress(event_id, event)
                except Exception:
                    logger.exception("retrain #%s: the on_progress callback failed", event_id)

        db_log(
            engine,
            "info",
            f"retrain #{event_id} started ({trigger}{', forced' if force else ''}): {trades} "
            f"closed trades since the last training (threshold {threshold}), champion "
            f"{champion.version if champion else 'none'}",
            {**context, "trades_since_last_train": trades, "threshold": threshold},
        )
        try:
            champion_checkpoint = load_champion_fn(champion, cfg)  # fail before the data scan
            df, bounds = read_data_fn(engine, data_settings)
            db_log(
                engine,
                "info",
                f"retrain #{event_id}: {bounds.describe()}",
                {**context, "data_bounds": bounds.as_dict()},
            )
            evaluation = evaluate_fn(
                df,
                cfg,
                champion=champion,
                champion_checkpoint=champion_checkpoint,
                model_dir=model_dir,
                device=device,
                progress=report,
            )
            candidate = evaluation.candidate
            db_log(
                engine,
                "info",
                f"retrain #{event_id}: candidate {candidate.version} trained "
                f"({'warm start' if candidate.warm_start.used else 'scratch'}), samples "
                + ", ".join(f"{k}={v}" for k, v in candidate.counts.items()),
                {**context, "counts": candidate.counts, "windows": evaluation.windows},
            )
            for (window, seed), metrics in sorted(evaluation.candidate_runs.items()):
                theirs = (evaluation.champion_runs or {}).get((window, seed), {})
                db_log(
                    engine,
                    "info",
                    f"retrain #{event_id}: window {window} seed {seed} {gate.primary_key} "
                    f"candidate={metrics.get(gate.primary_key)} "
                    f"champion={theirs.get(gate.primary_key)}",
                    {**context, "window": window, "seed": seed, "candidate": metrics,
                     "champion": theirs or None},
                )  # fmt: skip
            for note in evaluation.notes:
                db_log(engine, "warn", f"retrain #{event_id}: {note}", context)

            verdict = decide_promotion(
                evaluation.candidate_runs,
                evaluation.champion_runs,
                gate,
                windows=evaluation.windows,
            )
            metrics = {
                **verdict.metrics(),
                "candidate_version": candidate.version,
                "champion_version": champion.version if champion else None,
                "warm_start": candidate.warm_start.as_dict(),
                "counts": candidate.counts,
                "trades_since_last_train": trades,
                "threshold": threshold,
                "data_bounds": bounds.as_dict(),
                "notes": evaluation.notes,
            }
            lock.ensure_held()
            with engine.begin() as conn:
                record = insert_model(
                    conn,
                    version=candidate.version,
                    path=str(candidate.checkpoint_path),
                    metrics=json_safe(
                        {
                            **candidate.metrics,
                            "heldout": heldout_means(evaluation.candidate_runs),
                            "data_bounds": bounds.as_dict(),
                        }
                    ),
                    hyperparams=candidate.hyperparams,
                    n_params=candidate.n_params,
                    is_champion=verdict.promote,
                )
                finish_event(
                    conn,
                    event_id,
                    decision=verdict.event_decision,
                    n_samples=candidate.n_samples,
                    candidate_id=record.id,
                    metrics=json_safe(metrics),
                    notes="; ".join(
                        verdict.reasons + tuple(evaluation.notes) + (bounds.describe(),)
                    ),
                )
        except BaseException as exc:  # an interrupt too: never leave the run in flight
            _fail(engine, event_id, exc)
            raise

        db_log(
            engine,
            "info",
            f"retrain #{event_id}: candidate {candidate.version} {verdict.event_decision} — "
            + "; ".join(verdict.reasons),
            {**context, "decision": verdict.event_decision, "reasons": list(verdict.reasons)},
        )
        pruned = _prune_after_run(engine, cfg, lock, model_dir, event_id)
        return RetrainOutcome(
            verdict.event_decision,
            trades,
            threshold,
            event_id=event_id,
            candidate_id=record.id,
            candidate_version=candidate.version,
            reasons=verdict.reasons,
            pruned=pruned,
        )


def _fail(engine: Engine, event_id: int, exc: BaseException) -> None:
    """End the run as ``failed`` with the error in ``notes`` (best effort: never masks ``exc``)."""
    error = f"{type(exc).__name__}: {exc}"
    try:
        with engine.begin() as conn:
            finish_event(conn, event_id, decision="failed", notes=error)
    except Exception as record_exc:
        logger.error("could not record training event %s as failed: %s", event_id, record_exc)
    db_log(
        engine,
        "error",
        f"retrain #{event_id} failed: {error}",
        {
            "training_event_id": event_id,
            "traceback": "".join(traceback.format_exception(exc))[-4000:],
        },
    )


# --------------------------------------------------------------------------- rollback


@dataclass(frozen=True)
class RollbackOutcome:
    event_id: int
    restored: ModelRecord
    replaced: ModelRecord


def rollback(engine: Engine, config: Mapping[str, Any] | None = None) -> RollbackOutcome:
    """Restore the previous champion and record a ``rollback`` event.

    Raises:
        RetrainBusyError: a retrain or rollback is running.
        RollbackRefusedError: no previous champion, its checkpoint missing or incompatible.
    """
    from .config import load_config
    from .inference import load_checkpoint, resolve_checkpoint_path
    from .promotion import check_rollback

    cfg = load_config() if config is None else config
    with retrain_lock(engine) as lock:
        current = read_champion(engine)
        previous = previous_champion(engine, current) if current else None
        checkpoint = None
        if previous is not None and previous.pruned_at is None:
            try:
                checkpoint = load_checkpoint(resolve_checkpoint_path(previous.path))
            except Exception as exc:
                logger.warning("checkpoint of %s unreadable: %s", previous.version, exc)
        check = check_rollback(current, previous, checkpoint=checkpoint, horizons=cfg["horizons"])
        if not check.possible:
            db_log(
                engine,
                "warn",
                "rollback refused: " + "; ".join(check.reasons),
                {"reasons": list(check.reasons)},
            )
            raise RollbackRefusedError(check.reasons)
        target = check.target
        notes = f"rolled back from {current.version} to {target.version}"
        lock.ensure_held()
        with engine.begin() as conn:
            demoted = conn.execute(
                text("UPDATE models SET is_champion = FALSE WHERE id = :id AND is_champion"),
                {"id": current.id},
            ).rowcount
            if demoted != 1:
                raise RuntimeError(f"{current.version} is no longer the champion; not rolled back")
            restored = conn.execute(
                text("UPDATE models SET is_champion = TRUE WHERE id = :id"), {"id": target.id}
            ).rowcount
            if restored != 1:  # raising rolls the demotion back: never zero champions
                raise RuntimeError(
                    f"{target.version} is no longer in the registry; not rolled back"
                )
            event_id = int(
                conn.execute(
                    text(
                        "INSERT INTO training_events (trigger, finished_at, candidate_model_id, "
                        "champion_model_id, decision, notes) VALUES ('manual', now(), :target, "
                        ":current, 'rollback', :notes) RETURNING id"
                    ),
                    {"target": target.id, "current": current.id, "notes": notes},
                ).scalar_one()
            )
        db_log(engine, "info", f"rollback #{event_id}: {notes}", {"training_event_id": event_id})
        return RollbackOutcome(
            event_id, replace(target, is_champion=True), replace(current, is_champion=False)
        )


# --------------------------------------------------------------------------- retention


def _logged_before(engine: Engine, level: str, message: str) -> bool:
    """Whether this exact line is already in the ``logs`` table (a failed read counts as no)."""
    try:
        with engine.connect() as conn:
            return (
                conn.execute(
                    text(
                        "SELECT 1 FROM logs WHERE service = :service AND level = :level "
                        "AND message = :message LIMIT 1"
                    ),
                    {"service": SERVICE, "level": level, "message": message},
                ).first()
                is not None
            )
    except Exception:
        return False


def _log_prune(engine: Engine, outcome: PruneOutcome, label: str) -> None:
    """Log what changed. Refused models and orphans are lasting states found again on every run:
    each one is warned about once (its message carries no run label), not after every retrain."""
    plan = outcome.plan
    context = {
        "model_dir": str(plan.directory),
        "pruned": list(outcome.marked),
        "missing": [m.version for m in plan.missing],
        "orphans": [str(p) for p in plan.orphans + plan.recent_orphans],
        "orphans_deleted": [str(p) for p in outcome.orphans_deleted],
        "refused": [{"version": m.version, "reason": why} for m, why in plan.refused],
    }
    if outcome.marked or outcome.orphans_deleted:
        db_log(
            engine,
            "info",
            f"{label}: pruned {len(outcome.marked)} checkpoint(s)"
            + (f" ({', '.join(outcome.marked)})" if outcome.marked else "")
            + (f", deleted {len(outcome.orphans_deleted)} orphan(s)" if outcome.orphans_deleted
               else ""),
            context,
        )  # fmt: skip
    deleted = set(outcome.orphans_deleted)
    warnings = [
        f"retention: orphan checkpoint {orphan} has no models row "
        "(kept; delete with prune --delete-orphans)"
        for orphan in plan.orphans
        if orphan not in deleted
    ] + [f"retention: {model.version} not pruned: {why}" for model, why in plan.refused]
    for message in warnings:
        if not _logged_before(engine, "warn", message):
            db_log(engine, "warn", message, context)
    for error in outcome.errors:
        db_log(engine, "error", f"{label}: {error}", context)


def _plan(
    engine: Engine, config: Mapping[str, Any], model_dir: str | os.PathLike[str] | None
) -> PrunePlan:
    return plan_prune(
        engine,
        RetentionSettings.from_config(config),
        model_directory(model_dir),
        fallback_dir=fallback_directory(model_dir),
    )


def _prune_locked(
    engine: Engine,
    config: Mapping[str, Any],
    lock: RetrainLock,
    model_dir: str | os.PathLike[str] | None,
    *,
    delete_orphans: bool,
    label: str,
) -> PruneOutcome:
    outcome = apply_prune(
        engine,
        _plan(engine, config, model_dir),
        delete_orphans=delete_orphans,
        ensure_held=lock.ensure_held,
    )
    _log_prune(engine, outcome, label)
    return outcome


def _prune_after_run(
    engine: Engine,
    config: Mapping[str, Any],
    lock: RetrainLock,
    model_dir: str | os.PathLike[str] | None,
    event_id: int,
) -> tuple[str, ...]:
    """Prune once a run is recorded; never raises (the run already committed)."""
    try:
        outcome = _prune_locked(
            engine,
            config,
            lock,
            model_dir,
            delete_orphans=False,
            label=f"retention after retrain #{event_id}",
        )
    except Exception as exc:
        db_log(
            engine,
            "warn",
            f"retention after retrain #{event_id} failed: {type(exc).__name__}: {exc}",
            {"training_event_id": event_id},
        )
        return ()
    return outcome.marked


def prune(
    engine: Engine,
    config: Mapping[str, Any] | None = None,
    *,
    model_dir: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    delete_orphans: bool = False,
) -> PruneOutcome:
    """Apply the retention policy now (see :mod:`solia_ai.retention`). A dry run changes nothing
    and only reads, so it runs without the retrain lock, even during a retrain.

    Raises:
        RetrainBusyError: a retrain or rollback is running (not for a dry run).
        PruneError: no registered checkpoint is found in the directory (nothing is changed).
    """
    from .config import load_config

    cfg = load_config() if config is None else config
    if dry_run:
        return apply_prune(engine, _plan(engine, cfg, model_dir), dry_run=True)
    with retrain_lock(engine) as lock:
        return _prune_locked(
            engine, cfg, lock, model_dir, delete_orphans=delete_orphans, label="retention"
        )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the model when due (promote or reject the candidate), roll back, or "
        "prune old checkpoints."
    )
    parser.add_argument(
        "command", nargs="?", choices=("run", "rollback", "prune"), default="run",
        help="run a retrain (default), restore the previous champion, or apply the checkpoint "
        "retention policy",
    )  # fmt: skip
    parser.add_argument("--trigger", choices=RUN_TRIGGERS, default="n_trades")
    parser.add_argument("--force", action="store_true", help="skip the closed-trades check")
    parser.add_argument(
        "--model-dir", help="checkpoint directory (default: $MODEL_DIR or ./models)"
    )
    add_data_bounds_arguments(parser)
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    parser.add_argument(
        "--dry-run", action="store_true", help="prune: list what would be deleted, delete nothing"
    )
    parser.add_argument(
        "--delete-orphans", action="store_true",
        help="prune: also delete checkpoint files that have no models row",
    )  # fmt: skip
    args = parser.parse_args(argv)

    from .db import make_engine

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = make_engine()
    try:
        if args.command == "rollback":
            done = rollback(engine)
            print(f"rolled back to {done.restored.version} (event {done.event_id})")
            return
        if args.command == "prune":
            pruned = prune(
                engine,
                model_dir=Path(args.model_dir) if args.model_dir else None,
                dry_run=args.dry_run,
                delete_orphans=args.delete_orphans,
            )
            print("\n".join(describe(pruned)))
            if pruned.errors:
                raise SystemExit(1)
            return
        if args.device is None:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        outcome = run_retrain(
            engine,
            trigger=args.trigger,
            force=args.force,
            model_dir=Path(args.model_dir) if args.model_dir else None,
            device=args.device,
            lookback_days=args.lookback_days,
            max_rows=args.max_rows,
        )
    except RetrainBusyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(75) from None  # EX_TEMPFAIL: retry later
    except (RollbackRefusedError, PruneError, ConfigError) as exc:
        raise SystemExit(str(exc)) from None
    if outcome.status == "skipped":
        print(
            f"not due: {outcome.trades_since_last_train}/{outcome.threshold} closed trades "
            "since the last training"
        )
    else:
        print(
            f"retrain #{outcome.event_id}: {outcome.candidate_version} {outcome.status} — "
            + "; ".join(outcome.reasons)
        )


if __name__ == "__main__":
    main()
