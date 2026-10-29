"""Retrains inside the API process: on demand (``POST /model/retrain``) and on a schedule.

:class:`RetrainService` owns the process's one retrain slot and runs every retrain on a worker
thread, never on a request thread or the event loop, so ``/predict`` and ``/health`` keep serving
during a training:

* **Manual** (:meth:`RetrainService.start`): a ``manual`` run is started in the background and the
  call returns its ``training_events`` id as soon as the run is recorded (the ``202``). A run
  already going — in this process (the slot) or in another one (the orchestrator's advisory lock,
  e.g. the CLI) — is refused with :class:`~solia_ai.retrain.RetrainBusyError` (the ``409``).
* **Scheduled** (:meth:`RetrainService.tick`, every ``RETRAIN_CHECK_INTERVAL_SECONDS`` while
  ``RETRAIN_SCHEDULER_ENABLED``): reads the closed-trade counter and starts an ``n_trades`` run once
  it reaches ``RETRAIN_EVERY_N_TRADES``. A tick never overlaps a running retrain (it skips while
  the slot is taken), and whatever fails — the database down, the run itself — is logged and
  retried on the next tick; the scheduler never dies of it.
* **Progress** (:meth:`RetrainService.progress`): the counter, the threshold and the latest run, for
  ``GET /model/status``, read at most every ``progress_seconds`` (a run starting or ending refreshes
  it). A failed read reports them as ``None`` (the backend then uses its own count) instead of
  failing the status.
* **Recovery** (:meth:`RetrainService.recover_in_background`, at startup): worker threads are
  daemons, so stopping the process mid-training leaves that run's ``training_events`` row in flight
  (no decision). Startup records such rows as ``failed`` — off the startup path, so the API serves
  even with the database down — and every later run does the same under its lock.

The orchestrator (:func:`solia_ai.retrain.run_retrain`) still re-checks the threshold under its
lock: the tick's read is only a cheap pre-check. Paper trading only: nothing here places an order.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .retrain import RetrainBusyError, RetrainOutcome, retrain_threshold

logger = logging.getLogger(__name__)

ENABLED_ENV = "RETRAIN_SCHEDULER_ENABLED"
INTERVAL_ENV = "RETRAIN_CHECK_INTERVAL_SECONDS"
DEFAULT_INTERVAL_SECONDS = 300.0
# How long ``start`` waits for the run to be recorded before answering ``503``. Must stay below the
# backend's ``AiService:RetrainTimeout`` (30 s), so the backend hears the answer instead of timing
# out on a run that did start.
DEFAULT_START_TIMEOUT_SECONDS = 20.0
# How long ``GET /model/status`` reuses a progress read (the champion's refresh interval).
DEFAULT_PROGRESS_SECONDS = 30.0

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

# run_fn(trigger=..., on_started=...) runs one retrain (``run_retrain`` over the shared engine).
RunFn = Callable[..., RetrainOutcome]
# count_fn() -> (closed trades since the last training, threshold); raises when the DB fails.
CountFn = Callable[[], tuple[int, int]]
# latest_fn() -> the newest training_events summary, or None before any run.
LatestFn = Callable[[], dict[str, Any] | None]
# recover_fn() -> the ids of the abandoned runs it recorded as failed.
RecoverFn = Callable[[], list[int]]


class RetrainStartError(RuntimeError):
    """The manual run could not be started (database down, or it was not recorded in time)."""


@dataclass(frozen=True)
class SchedulerSettings:
    """Whether the background trigger runs, and how often it checks the trade counter."""

    enabled: bool = False
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS

    @classmethod
    def from_env(cls, lookup: Callable[[str], str | None] = os.environ.get) -> SchedulerSettings:
        """``RETRAIN_SCHEDULER_ENABLED`` (default off; compose turns it on) and
        ``RETRAIN_CHECK_INTERVAL_SECONDS`` (default 300, > 0). An invalid value raises
        ``ValueError`` at startup rather than silently disabling the trigger."""
        raw_enabled = (lookup(ENABLED_ENV) or "").strip().lower()
        if raw_enabled not in _TRUE | _FALSE:
            raise ValueError(f"{ENABLED_ENV} must be true or false, got {lookup(ENABLED_ENV)!r}")
        raw_interval = (lookup(INTERVAL_ENV) or "").strip()
        interval = DEFAULT_INTERVAL_SECONDS
        if raw_interval:
            try:
                interval = float(raw_interval)
            except ValueError:
                interval = 0.0
            if not interval > 0 or interval == float("inf"):
                raise ValueError(
                    f"{INTERVAL_ENV} must be a positive number of seconds, got {raw_interval!r}"
                )
        return cls(enabled=raw_enabled in _TRUE, interval_seconds=interval)


class _Start:
    """Hand-off between ``start`` and its worker: the recorded id, or why the run never started."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.event_id: int | None = None
        self.error: BaseException | None = None
        self.outcome: RetrainOutcome | None = None

    def started(self, event_id: int) -> None:
        self.event_id = event_id
        self.done.set()


class RetrainService:
    """The process's retrain slot, manual starts, the scheduler, recovery and the progress read.

    ``count_fn`` / ``latest_fn`` may be ``None`` (no database): the scheduler then never runs and
    the progress is all ``None``."""

    def __init__(
        self,
        run_fn: RunFn,
        count_fn: CountFn | None,
        latest_fn: LatestFn | None,
        *,
        recover_fn: RecoverFn | None = None,
        settings: SchedulerSettings | None = None,
        start_timeout: float = DEFAULT_START_TIMEOUT_SECONDS,
        progress_seconds: float = DEFAULT_PROGRESS_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if progress_seconds < 0:
            raise ValueError(f"progress_seconds must be >= 0, got {progress_seconds}")
        self._run_fn = run_fn
        self._count_fn = count_fn
        self._latest_fn = latest_fn
        self._recover_fn = recover_fn
        self.settings = settings or SchedulerSettings()
        self._start_timeout = start_timeout
        self._progress_seconds = progress_seconds
        self._clock = clock
        self._slot = threading.Lock()  # held for the whole of a run, by whichever thread runs it
        # Each scheduler thread gets its own stop event: a stopped loop still finishing a tick is
        # never revived by a later start, which runs a fresh thread instead.
        self._stop = threading.Event()
        self._scheduler: threading.Thread | None = None
        self._progress_lock = threading.Lock()
        self._progress: dict[str, Any] | None = None
        self._progress_at = 0.0

    @classmethod
    def from_engine(cls, engine: Callable[[], Any]) -> RetrainService:
        """The database-backed service; ``engine()`` returns the process's shared engine."""
        from .config import load_config
        from .db import DataSettings
        from .registry import latest_training_event
        from .retrain import count_trades_since_last_train, recover_abandoned_runs, run_retrain

        threshold = retrain_threshold()  # fail fast on a bad RETRAIN_EVERY_N_TRADES
        # ... and on bad data bounds; each run re-reads them with the config (same as the CLI).
        DataSettings.from_config(load_config())

        def run_fn(*, trigger: str, on_started: Callable[[int], None] | None = None):
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            return run_retrain(
                engine(), trigger=trigger, threshold=threshold, device=device, on_started=on_started
            )

        return cls(
            run_fn,
            lambda: (count_trades_since_last_train(engine()), threshold),
            lambda: latest_training_event(engine()),
            recover_fn=lambda: recover_abandoned_runs(engine()),
            settings=SchedulerSettings.from_env(),
        )

    @classmethod
    def unavailable(cls) -> RetrainService:
        """No database behind it: manual runs are refused (``503``), nothing is scheduled and the
        progress is all ``None``. For an app built over its own champion loader."""

        def run_fn(**_: Any) -> RetrainOutcome:
            raise RuntimeError("no retrain service is configured")

        return cls(run_fn, None, None)

    @property
    def busy(self) -> bool:
        """Whether this process is running a retrain right now."""
        return self._slot.locked()

    # ----------------------------------------------------------------------- manual

    def start(self, trigger: str = "manual") -> int:
        """Start a run in the background; return its ``training_events`` id once recorded.

        Raises:
            RetrainBusyError: a retrain is already running (here or in another process).
            RetrainStartError: the run failed before it was recorded, or wasn't recorded within
                the start timeout (it may still start; the status shows it).
        """
        if not self._slot.acquire(blocking=False):
            raise RetrainBusyError("a retrain is already running; try again later")
        handoff = _Start()

        def on_started(event_id: int) -> None:
            self._invalidate_progress()  # the status shows the new run at once
            handoff.started(event_id)

        def work() -> None:
            try:
                handoff.outcome = self._run_fn(trigger=trigger, on_started=on_started)
            except BaseException as exc:
                handoff.error = exc
                if handoff.event_id is not None:  # already recorded as failed by the run
                    logger.error("retrain #%s failed: %s", handoff.event_id, exc)
            finally:
                self._slot.release()
                self._invalidate_progress()
                handoff.done.set()

        try:
            threading.Thread(target=work, name=f"retrain-{trigger}", daemon=True).start()
        except BaseException:
            self._slot.release()
            raise
        if not handoff.done.wait(self._start_timeout):
            raise RetrainStartError(f"the retrain was not recorded within {self._start_timeout:g}s")
        if handoff.event_id is not None:
            return handoff.event_id
        if isinstance(handoff.error, RetrainBusyError):
            raise handoff.error
        if handoff.error is not None:
            raise RetrainStartError(
                f"the retrain could not be started: {type(handoff.error).__name__}: "
                f"{handoff.error}"
            ) from handoff.error
        status = handoff.outcome.status if handoff.outcome else "unknown"
        raise RetrainStartError(f"the retrain ended without being recorded ({status})")

    # ----------------------------------------------------------------------- scheduled

    def tick(self) -> RetrainOutcome | None:
        """One scheduler check: start an ``n_trades`` run (and wait for it) if the counter reached
        the threshold. Returns the run's outcome, or ``None`` when nothing ran. Never raises."""
        if self._count_fn is None:
            return None
        if not self._slot.acquire(blocking=False):
            logger.debug("retrain check skipped: a retrain is running")
            return None
        ran = False
        try:
            trades, threshold = self._count_fn()
            if trades < threshold:
                logger.debug("retrain not due: %s/%s closed trades", trades, threshold)
                return None
            logger.info("retrain due: %s/%s closed trades", trades, threshold)
            ran = True
            return self._run_fn(
                trigger="n_trades", on_started=lambda _: self._invalidate_progress()
            )
        except RetrainBusyError as exc:
            logger.info("retrain check skipped: %s", exc)
        except Exception as exc:
            logger.warning(
                "retrain check failed (retrying next tick): %s: %s", type(exc).__name__, exc
            )
        finally:
            self._slot.release()
            if ran:
                self._invalidate_progress()
        return None

    def start_scheduler(self) -> bool:
        """Start the background trigger if enabled (idempotent); return whether it runs."""
        if not self.settings.enabled or self._count_fn is None:
            logger.info("retrain scheduler disabled (%s)", ENABLED_ENV)
            return False
        if self._scheduler is not None and not self._stop.is_set():
            return True
        stop = self._stop = threading.Event()
        self._scheduler = threading.Thread(
            target=self._loop, args=(stop,), name="retrain-scheduler", daemon=True
        )
        self._scheduler.start()
        logger.info("retrain scheduler checking every %gs", self.settings.interval_seconds)
        return True

    def stop_scheduler(self, timeout: float | None = 5.0) -> None:
        """Stop the trigger. A run it started keeps going on its daemon thread past ``timeout``;
        that loop then ends with the run (a later start never revives it)."""
        self._stop.set()
        if self._scheduler is not None:
            self._scheduler.join(timeout)
            self._scheduler = None

    def _loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.settings.interval_seconds):
            self.tick()

    # ----------------------------------------------------------------------- recovery

    def recover(self) -> list[int]:
        """Record the runs a stopped process left in flight as ``failed``; return their ids.
        Never raises: a failure is logged, and the next run closes them under its lock."""
        if self._recover_fn is None:
            return []
        try:
            ids = self._recover_fn()
        except Exception as exc:
            logger.warning("abandoned-run recovery failed: %s: %s", type(exc).__name__, exc)
            return []
        if ids:
            logger.warning("abandoned retrain run(s) %s recorded as failed", ids)
            self._invalidate_progress()
        return ids

    def recover_in_background(self) -> threading.Thread | None:
        """Run :meth:`recover` on a daemon thread: startup never waits on the database."""
        if self._recover_fn is None:
            return None
        thread = threading.Thread(target=self.recover, name="retrain-recover", daemon=True)
        thread.start()
        return thread

    # ----------------------------------------------------------------------- progress

    def progress(self) -> dict[str, Any]:
        """``trades_since_last_train`` + ``retrain_threshold`` (one consistent pair, both ``None``
        when the count fails) and ``latest_run`` (``None`` before any run or when its read
        fails). A read is reused for ``progress_seconds`` — a failed one too, so a slow or down
        database costs one attempt per interval rather than one per status call."""
        with self._progress_lock:
            fresh = self._clock() - self._progress_at < self._progress_seconds
            if self._progress is None or not fresh:
                self._progress = self._read_progress()
                self._progress_at = self._clock()
            return dict(self._progress)

    def _invalidate_progress(self) -> None:
        with self._progress_lock:
            self._progress = None

    def _read_progress(self) -> dict[str, Any]:
        trades = threshold = latest = None
        if self._count_fn is not None:
            try:
                trades, threshold = self._count_fn()
            except Exception as exc:
                logger.warning("retrain progress unavailable: %s: %s", type(exc).__name__, exc)
        if self._latest_fn is not None:
            try:
                latest = self._latest_fn()
            except Exception as exc:
                logger.warning("latest retrain run unavailable: %s: %s", type(exc).__name__, exc)
        return {
            "trades_since_last_train": trades,
            "retrain_threshold": threshold,
            "latest_run": latest,
        }
