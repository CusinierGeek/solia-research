"""The in-process retrain service: manual starts, the background trigger and the progress read
(stubbed runs and counters, no database)."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from solia_ai.retrain import RetrainBusyError, RetrainOutcome
from solia_ai.retrain_service import (
    DEFAULT_INTERVAL_SECONDS,
    RetrainService,
    RetrainStartError,
    SchedulerSettings,
)


class Counter:
    """A trade counter the test moves; ``fail`` makes the next reads raise like a DB blip."""

    def __init__(self, trades: int, threshold: int = 5) -> None:
        self.trades = trades
        self.threshold = threshold
        self.fail = 0

    def __call__(self) -> tuple[int, int]:
        if self.fail:
            self.fail -= 1
            raise ConnectionError("connection refused")
        return self.trades, self.threshold


class Runs:
    """A run_fn recording its triggers; ``gate`` (when set) blocks the run until released."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.triggers: list[str] = []
        self.gate = gate
        self.running = threading.Event()

    def __call__(self, *, trigger, on_started=None) -> RetrainOutcome:
        self.triggers.append(trigger)
        if on_started:
            on_started(len(self.triggers))
        self.running.set()
        if self.gate is not None:
            self.gate.wait(10)
        return RetrainOutcome("rejected", 0, 5, event_id=len(self.triggers))


def service(runs, counter, **kwargs) -> RetrainService:
    return RetrainService(runs, counter, lambda: None, **kwargs)


def wait_idle(svc: RetrainService) -> None:
    deadline = time.monotonic() + 5
    while svc.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not svc.busy


# --------------------------------------------------------------------------- tick


def test_the_trigger_fires_at_the_threshold_and_not_below():
    runs, counter = Runs(), Counter(trades=4, threshold=5)
    svc = service(runs, counter)

    assert svc.tick() is None
    assert runs.triggers == []

    counter.trades = 5
    outcome = svc.tick()

    assert runs.triggers == ["n_trades"]
    assert outcome is not None and outcome.event_id == 1
    assert not svc.busy  # the slot is released once the run ends


def test_a_tick_never_overlaps_a_running_retrain():
    gate = threading.Event()
    runs, counter = Runs(gate), Counter(trades=50)
    svc = service(runs, counter)
    try:
        assert svc.start() == 1  # a manual run holds the slot
        assert svc.tick() is None
        assert runs.triggers == ["manual"]
    finally:
        gate.set()
    wait_idle(svc)
    svc.tick()
    assert runs.triggers == ["manual", "n_trades"]


def test_a_tick_survives_a_database_blip_and_retries_next_tick(caplog):
    runs, counter = Runs(), Counter(trades=9)
    counter.fail = 1
    svc = service(runs, counter)

    with caplog.at_level(logging.WARNING):
        assert svc.tick() is None
    assert "retrying next tick" in caplog.text and "connection refused" in caplog.text
    assert not svc.busy

    svc.tick()
    assert runs.triggers == ["n_trades"]


def test_a_tick_survives_a_failed_run():
    def crash(*, trigger, on_started=None):
        raise RuntimeError("training exploded")

    svc = service(crash, Counter(trades=9))

    assert svc.tick() is None
    assert not svc.busy


def test_a_tick_skips_when_another_process_holds_the_lock(caplog):
    def busy(*, trigger, on_started=None):
        raise RetrainBusyError("another retrain or rollback is running")

    svc = service(busy, Counter(trades=9))

    with caplog.at_level(logging.INFO):
        assert svc.tick() is None
    assert "another retrain" in caplog.text


def test_the_scheduler_ticks_on_its_interval_and_stops():
    runs, counter = Runs(), Counter(trades=9)
    svc = service(runs, counter, settings=SchedulerSettings(enabled=True, interval_seconds=0.01))

    assert svc.start_scheduler()
    assert svc.start_scheduler()  # idempotent: still one scheduler
    assert runs.running.wait(5)
    svc.stop_scheduler()

    assert runs.triggers and set(runs.triggers) == {"n_trades"}
    count = len(runs.triggers)
    time.sleep(0.05)
    assert len(runs.triggers) == count  # stopped: no more ticks


def test_a_restart_after_a_timed_out_stop_runs_a_single_loop():
    gate = threading.Event()
    runs, counter = Runs(gate), Counter(trades=9)
    svc = service(runs, counter, settings=SchedulerSettings(enabled=True, interval_seconds=0.01))
    assert svc.start_scheduler()
    assert runs.running.wait(5)  # the old loop is inside a tick
    old = svc._scheduler

    svc.stop_scheduler(timeout=0.01)  # the join times out mid-training
    assert old.is_alive()
    assert svc.start_scheduler()
    new = svc._scheduler
    gate.set()  # the old tick ends

    old.join(5)
    assert not old.is_alive()  # never revived by the restart
    assert new is not old and new.is_alive()
    svc.stop_scheduler()


def test_a_disabled_scheduler_never_starts():
    runs = Runs()
    svc = service(runs, Counter(trades=9), settings=SchedulerSettings(enabled=False))

    assert not svc.start_scheduler()
    svc.stop_scheduler()  # harmless when never started
    assert runs.triggers == []


# --------------------------------------------------------------------------- start


def test_start_returns_the_recorded_id_while_the_run_goes_on():
    gate = threading.Event()
    runs = Runs(gate)
    svc = service(runs, Counter(trades=0))
    try:
        assert svc.start() == 1
        assert svc.busy
        with pytest.raises(RetrainBusyError):
            svc.start()
    finally:
        gate.set()
    wait_idle(svc)


def test_start_reports_a_run_that_failed_before_being_recorded():
    def down(*, trigger, on_started=None):
        raise ConnectionError("database down")

    svc = service(down, Counter(trades=0))

    with pytest.raises(RetrainStartError, match="database down"):
        svc.start()
    wait_idle(svc)


def test_start_reports_a_run_that_ended_without_being_recorded():
    svc = service(lambda **_: RetrainOutcome("skipped", 0, 5), Counter(trades=0))

    with pytest.raises(RetrainStartError, match="skipped"):
        svc.start()


def test_a_run_failing_after_it_was_recorded_still_returns_its_id():
    def fails_later(*, trigger, on_started=None):
        on_started(3)
        raise RuntimeError("training exploded")

    svc = service(fails_later, Counter(trades=0))

    assert svc.start() == 3
    wait_idle(svc)


# --------------------------------------------------------------------------- progress


def test_progress_reports_one_consistent_pair_and_the_latest_run():
    latest = {"id": 4, "trigger": "manual", "started_at": "x", "finished_at": None}
    svc = RetrainService(Runs(), Counter(trades=3, threshold=8), lambda: latest)

    assert svc.progress() == {
        "trades_since_last_train": 3,
        "retrain_threshold": 8,
        "latest_run": latest,
    }


def test_progress_is_null_when_the_reads_fail():
    counter = Counter(trades=3)
    counter.fail = 1

    def latest_down():
        raise ConnectionError("down")

    svc = RetrainService(Runs(), counter, latest_down)

    assert svc.progress() == {
        "trades_since_last_train": None,
        "retrain_threshold": None,
        "latest_run": None,
    }


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Calls:
    def __init__(self, fn) -> None:
        self.fn = fn
        self.n = 0

    def __call__(self):
        self.n += 1
        return self.fn()


def test_progress_is_reused_until_its_interval_elapses():
    clock, counter = Clock(), Calls(Counter(trades=3, threshold=8))
    latest = Calls(lambda: None)
    svc = RetrainService(Runs(), counter, latest, progress_seconds=30, clock=clock)

    for _ in range(5):  # a dashboard polling /model/status
        svc.progress()
    assert (counter.n, latest.n) == (1, 1)

    clock.now = 30.0
    svc.progress()
    assert (counter.n, latest.n) == (2, 2)


def test_a_failed_progress_read_is_reused_too():
    counter = Counter(trades=3)
    counter.fail = 5
    calls = Calls(counter)
    svc = RetrainService(Runs(), calls, lambda: None, clock=Clock())

    assert svc.progress()["trades_since_last_train"] is None
    assert svc.progress()["trades_since_last_train"] is None
    assert calls.n == 1  # a down database costs one attempt per interval


def test_a_started_run_refreshes_the_progress():
    counter = Calls(Counter(trades=3))
    svc = RetrainService(Runs(), counter, lambda: None, clock=Clock())
    svc.progress()

    svc.start()
    wait_idle(svc)
    svc.progress()

    assert counter.n == 2


def test_a_negative_progress_interval_is_refused():
    with pytest.raises(ValueError, match="progress_seconds"):
        RetrainService(Runs(), Counter(trades=0), lambda: None, progress_seconds=-1)


# --------------------------------------------------------------------------- recovery


def test_recover_reports_the_closed_runs():
    svc = RetrainService(Runs(), Counter(trades=0), lambda: None, recover_fn=lambda: [4, 9])

    assert svc.recover() == [4, 9]


def test_recover_never_raises(caplog):
    def down():
        raise ConnectionError("database down")

    svc = RetrainService(Runs(), Counter(trades=0), lambda: None, recover_fn=down)

    with caplog.at_level(logging.WARNING):
        assert svc.recover() == []
    assert "database down" in caplog.text


def test_recover_runs_in_the_background():
    called = threading.Event()

    def recover():
        called.set()
        return []

    svc = RetrainService(Runs(), Counter(trades=0), lambda: None, recover_fn=recover)

    thread = svc.recover_in_background()
    thread.join(5)
    assert called.is_set()
    assert service(Runs(), Counter(trades=0)).recover_in_background() is None  # nothing to do


# --------------------------------------------------------------------------- unavailable


def test_an_unavailable_service_refuses_runs_and_reports_no_progress():
    svc = RetrainService.unavailable()
    svc.settings = SchedulerSettings(enabled=True)

    with pytest.raises(RetrainStartError, match="no retrain service"):
        svc.start()
    assert svc.tick() is None
    assert not svc.start_scheduler()
    assert svc.recover() == []
    assert svc.progress() == {
        "trades_since_last_train": None,
        "retrain_threshold": None,
        "latest_run": None,
    }


# --------------------------------------------------------------------------- settings


def test_scheduler_settings_default_off_and_read_the_environment():
    assert SchedulerSettings.from_env({}.get) == SchedulerSettings(
        enabled=False, interval_seconds=DEFAULT_INTERVAL_SECONDS
    )
    env = {"RETRAIN_SCHEDULER_ENABLED": " True ", "RETRAIN_CHECK_INTERVAL_SECONDS": "60"}
    assert SchedulerSettings.from_env(env.get) == SchedulerSettings(True, 60.0)
    assert not SchedulerSettings.from_env({"RETRAIN_SCHEDULER_ENABLED": "off"}.get).enabled


@pytest.mark.parametrize("raw", ["maybe", "2"])
def test_an_invalid_enable_flag_is_refused(raw):
    with pytest.raises(ValueError, match="RETRAIN_SCHEDULER_ENABLED"):
        SchedulerSettings.from_env({"RETRAIN_SCHEDULER_ENABLED": raw}.get)


@pytest.mark.parametrize("raw", ["0", "-5", "soon", "inf", "nan"])
def test_an_invalid_interval_is_refused(raw):
    with pytest.raises(ValueError, match="RETRAIN_CHECK_INTERVAL_SECONDS"):
        SchedulerSettings.from_env({"RETRAIN_CHECK_INTERVAL_SECONDS": raw}.get)
