"""The retrain orchestrator: trigger rule, lock, run recording, promote/reject/fail, rollback.

The database tests run against the shared Testcontainers PostgreSQL (skipped locally without
Docker); training is stubbed there. One test runs the real train + evaluate step on tiny data.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sqlalchemy import create_engine, text
from test_candidate import candidate_config
from test_train import BOUNDS, make_features_df, small_config

from solia_ai import promotion, retrain
from solia_ai.candidate import WarmStart
from solia_ai.config import DATA_DEFAULTS, ConfigError
from solia_ai.dataset import config_samples
from solia_ai.db import DataSettings
from solia_ai.features import FEATURE_COLUMNS
from solia_ai.inference import feature_index
from solia_ai.promotion import PromotionSettings, decide_promotion
from solia_ai.registry import latest_training_event, read_champion, register_model
from solia_ai.retrain import (
    Evaluation,
    RetrainBusyError,
    RetrainLockLostError,
    RollbackRefusedError,
    count_trades_since_last_train,
    eligibility_note,
    heldout_means,
    is_due,
    json_safe,
    load_champion_checkpoint,
    resize_windows,
    retrain_lock,
    retrain_threshold,
    rollback,
    run_retrain,
    train_and_evaluate,
)
from solia_ai.train import train

T0 = datetime(2026, 1, 10, tzinfo=UTC)


def gate_config(**promotion) -> dict:
    cfg = candidate_config()
    cfg["promotion"] = {"windows": 2, "seeds": 2, "bootstrap_samples": 50, **promotion}
    return cfg


PRIMARY = PromotionSettings.from_config(gate_config()).primary_key


# --------------------------------------------------------------------------- trigger rule


def test_threshold_defaults_and_reads_the_environment():
    assert retrain_threshold(lambda _: None) == retrain.DEFAULT_THRESHOLD
    assert retrain_threshold(lambda _: " ") == retrain.DEFAULT_THRESHOLD
    assert retrain_threshold(lambda _: "250") == 250


@pytest.mark.parametrize("raw", ["0", "-3", "many", "2.5"])
def test_an_invalid_threshold_is_refused(raw):
    with pytest.raises(ValueError, match="RETRAIN_EVERY_N_TRADES"):
        retrain_threshold(lambda _: raw)


@pytest.mark.parametrize("trigger", ["n_trades", "scheduled"])
def test_a_run_is_due_at_the_threshold(trigger):
    assert not is_due(499, 500, trigger=trigger)
    assert is_due(500, 500, trigger=trigger)
    assert is_due(501, 500, trigger=trigger)


def test_manual_and_forced_runs_skip_the_check():
    assert is_due(0, 500, trigger="manual")
    assert is_due(0, 500, trigger="n_trades", force=True)


def test_an_unknown_trigger_is_refused():
    with pytest.raises(ValueError, match="trigger"):
        is_due(0, 1, trigger="initial")


def test_heldout_means_skip_undefined_values():
    runs = {(0, 0): {"a": 1.0, "b": None}, (0, 1): {"a": 3.0, "b": None}}
    assert heldout_means(runs) == {"a": 2.0, "b": None}


def test_heldout_means_skip_non_finite_values():
    runs = {(0, 0): {"a": float("nan")}, (0, 1): {"a": 3.0}, (1, 0): {"b": float("inf")}}
    assert heldout_means(runs) == {"a": 3.0, "b": None}


def test_json_safe_replaces_non_finite_numbers():
    value = {"a": float("nan"), "b": [1.0, float("-inf")], "c": {"d": 2}, "e": "x"}
    assert json_safe(value) == {"a": None, "b": [1.0, None], "c": {"d": 2}, "e": "x"}


class _UnlockFails:
    """A connection whose lock is granted but whose unlock raises."""

    def __init__(self):
        self.calls, self.invalidated, self.closed = 0, False, False

    def execute(self, *args):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("connection lost")
        return SimpleNamespace(scalar_one=lambda: True)

    def commit(self):
        pass

    def invalidate(self):
        self.invalidated = True

    def close(self):
        self.closed = True


def test_a_failed_unlock_never_masks_the_run():
    conn = _UnlockFails()
    engine = SimpleNamespace(connect=lambda: conn)
    with retrain_lock(engine):
        pass  # a committed run returns normally
    assert conn.invalidated and conn.closed
    conn = _UnlockFails()
    with pytest.raises(ValueError, match="the real error"):
        with retrain_lock(SimpleNamespace(connect=lambda: conn)):
            raise ValueError("the real error")


# --------------------------------------------------------------------------- database


@pytest.fixture(autouse=True)
def isolated_model_dir(tmp_path, monkeypatch):
    """Runs prune their checkpoint directory afterwards: never let it default to ./models."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "model-dir"))


@pytest.fixture
def engine(pg_url):
    eng = create_engine(pg_url)
    with eng.begin() as conn:
        conn.exec_driver_sql(
            "TRUNCATE training_events, models, logs, trades, tokens_history RESTART IDENTITY "
            "CASCADE"
        )
    yield eng
    eng.dispose()


def add_trades(engine, n: int, *, closed_at: datetime | None = T0, status: str = "closed"):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO tokens_history (mint) VALUES ('M') ON CONFLICT DO NOTHING"))
        for _ in range(n):
            conn.execute(
                text("INSERT INTO trades (mint, status, closed_at) VALUES ('M', :s, :c)"),
                {"s": status, "c": closed_at},
            )


def add_event(engine, decision: str | None, started_at: datetime, **ids) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text(
                "INSERT INTO training_events (started_at, finished_at, trigger, decision, "
                "candidate_model_id, champion_model_id) VALUES (:s, :f, 'manual', :d, :cand, "
                ":champ) RETURNING id"
            ),
            {
                "s": started_at,
                "f": started_at if decision else None,
                "d": decision,
                "cand": ids.get("candidate"),
                "champ": ids.get("champion"),
            },
        ).scalar_one()


def events(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT * FROM training_events ORDER BY id")).mappings().all()


def models(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT * FROM models ORDER BY id")).mappings().all()


def runs(value: float, *, windows: int = 2, seeds: int = 2) -> dict:
    return {
        (w, s): {PRIMARY: value + 0.001 * s, "policy/total_pnl": 0.0}
        for w in range(windows)
        for s in range(seeds)
    }


def stub(candidate_value: float, champion_value: float | None = None, *, version="cand"):
    """An evaluate_fn returning synthetic gate runs (``champion_value=None``: no champion runs)."""
    calls = []

    def evaluate_fn(
        df, config, *, champion, model_dir, device, champion_checkpoint=None, progress=None
    ):
        calls.append(champion)
        candidate = SimpleNamespace(
            version=version,
            checkpoint_path=Path(f"/models/{version}.pt"),
            metrics={"val_total": 1.25},
            hyperparams={"training": {"seed": 42}},
            n_params=123,
            n_samples=900,
            counts={"n_samples": 900, "n_train": 700, "n_val": 200, "n_heldout": 300},
            warm_start=WarmStart(True, False, None, ("no champion checkpoint",)),
        )
        theirs = None if champion is None or champion_value is None else runs(champion_value)
        ours = runs(candidate_value, seeds=1 if theirs is None else 2)  # one seed when unpaired
        return Evaluation(candidate, ours, theirs, [{"index": 0}, {"index": 1}])

    evaluate_fn.calls = calls
    return evaluate_fn


def run(engine, evaluate_fn, **kwargs):
    kwargs.setdefault("trigger", "manual")
    kwargs.setdefault("read_data_fn", lambda engine, data: (pd.DataFrame({"x": [1, 2]}), BOUNDS))
    kwargs.setdefault("load_champion_fn", lambda champion, config: None)
    return run_retrain(
        engine,
        gate_config(),
        threshold=kwargs.pop("threshold", 5),
        evaluate_fn=evaluate_fn,
        **kwargs,
    )


def terminate_lock_holder(engine):
    """Kill the backend holding the retrain lock, as a dropped connection would."""
    with engine.begin() as conn:
        killed = (
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid, 5000) FROM pg_locks "
                    "WHERE locktype = 'advisory' AND objid::bigint = :lo"
                ),
                {"lo": retrain.LOCK_KEY & 0xFFFFFFFF},
            )
            .scalars()
            .all()
        )
    assert killed == [True]


def test_count_follows_the_backend_rule(engine):
    add_trades(engine, 3, closed_at=T0)
    add_trades(engine, 2, closed_at=None, status="open")
    assert count_trades_since_last_train(engine) == 3
    add_event(engine, "rejected", T0 + timedelta(minutes=1))
    add_trades(engine, 4, closed_at=T0 + timedelta(minutes=2))
    assert count_trades_since_last_train(engine) == 4
    # A failed run, a rollback and a run in flight trained nothing: the count is not reset.
    for decision in ("failed", "rollback", None):
        add_event(engine, decision, T0 + timedelta(minutes=3))
    assert count_trades_since_last_train(engine) == 4


def test_a_run_below_the_threshold_is_skipped_unrecorded(engine):
    add_trades(engine, 4)
    fn = stub(0.5)
    outcome = run(engine, fn, trigger="n_trades")
    assert (outcome.status, outcome.trades_since_last_train, outcome.threshold) == ("skipped", 4, 5)
    assert fn.calls == [] and events(engine) == []
    with engine.connect() as conn:  # a scheduler tick that does nothing leaves no logs row
        assert conn.execute(text("SELECT count(*) FROM logs")).scalar() == 0


def test_a_run_at_the_threshold_trains(engine):
    add_trades(engine, 5)
    outcome = run(engine, stub(0.5), trigger="n_trades")
    assert outcome.status == "promoted"
    assert events(engine)[0]["trigger"] == "n_trades"


def test_a_manual_or_forced_run_bypasses_the_threshold(engine):
    assert run(engine, stub(0.5), trigger="manual").status == "promoted"
    outcome = run(engine, stub(0.5, version="cand2"), trigger="scheduled", force=True)
    assert outcome.status != "skipped"


def test_on_started_gets_the_recorded_event_id_before_the_run_ends(engine):
    seen = []

    def on_started(event_id):
        row = events(engine)[-1]
        seen.append((event_id, row["id"], row["decision"]))  # recorded, still in flight

    outcome = run(engine, stub(0.5), on_started=on_started)

    assert seen == [(outcome.event_id, outcome.event_id, None)]


def test_on_started_is_not_called_for_a_skipped_run_and_its_errors_are_only_logged(engine):
    seen = []
    assert run(engine, stub(0.5), trigger="n_trades", on_started=seen.append).status == "skipped"
    assert seen == []

    def broken(event_id):
        raise RuntimeError("callback bug")

    assert run(engine, stub(0.5), on_started=broken).status == "promoted"


def test_latest_training_event_summarizes_the_newest_run(engine):
    assert latest_training_event(engine) is None
    add_event(engine, "rejected", T0)
    newest = add_event(engine, None, T0 + timedelta(minutes=5))

    assert latest_training_event(engine) == {
        "id": newest,
        "trigger": "manual",
        "started_at": (T0 + timedelta(minutes=5)).isoformat(),
        "finished_at": None,
        "decision": None,
    }


def test_the_api_service_reads_progress_from_the_database(engine, monkeypatch):
    from solia_ai.retrain_service import RetrainService

    monkeypatch.setenv("RETRAIN_EVERY_N_TRADES", "7")
    add_trades(engine, 3)
    event = add_event(engine, "promoted", T0 - timedelta(minutes=1))

    progress = RetrainService.from_engine(lambda: engine).progress()

    assert (progress["trades_since_last_train"], progress["retrain_threshold"]) == (3, 7)
    assert progress["latest_run"]["id"] == event
    assert progress["latest_run"]["decision"] == "promoted"


def test_a_concurrent_run_is_refused_not_queued(engine):
    fn = stub(0.5)
    with retrain_lock(engine):
        with pytest.raises(RetrainBusyError):
            run(engine, fn)
        with pytest.raises(RetrainBusyError):
            rollback(engine, gate_config())
    assert fn.calls == [] and events(engine) == []
    assert run(engine, fn).status == "promoted"  # the lock was released


def test_recovery_records_runs_left_in_flight_as_failed(engine):
    done = add_event(engine, "rejected", T0 - timedelta(hours=2))
    stale = add_event(engine, None, T0 - timedelta(hours=1))  # its process died mid-training

    assert retrain.recover_abandoned_runs(engine) == [stale]

    rows = {row["id"]: row for row in events(engine)}
    assert rows[stale]["decision"] == "failed" and rows[stale]["finished_at"] is not None
    assert rows[stale]["notes"] == retrain.ABANDONED_NOTE
    assert rows[done]["decision"] == "rejected"  # finished runs are left alone
    assert retrain.recover_abandoned_runs(engine) == []  # idempotent
    assert latest_training_event(engine)["decision"] == "failed"  # no longer "running"


def test_recovery_leaves_a_live_run_alone(engine):
    live = add_event(engine, None, T0)
    with retrain_lock(engine):  # another process is training
        assert retrain.recover_abandoned_runs(engine) == []
    assert events(engine)[0]["id"] == live and events(engine)[0]["decision"] is None


def test_a_run_closes_an_abandoned_run_before_recording_its_own(engine):
    stale = add_event(engine, None, T0 - timedelta(hours=1))

    outcome = run(engine, stub(0.5))

    rows = {row["id"]: row for row in events(engine)}
    assert rows[stale]["decision"] == "failed"
    assert rows[outcome.event_id]["decision"] == "promoted"


def test_a_lost_lock_is_detected_and_released(engine):
    with retrain_lock(engine) as lock:
        lock.ensure_held()
        terminate_lock_holder(engine)
        with pytest.raises(RetrainLockLostError, match="lost mid-run"):
            lock.ensure_held()
    with retrain_lock(engine):  # the dead connection released it
        pass


def test_the_heartbeat_notices_a_lost_lock(engine):
    with retrain_lock(engine, heartbeat_seconds=0.05) as lock:
        terminate_lock_holder(engine)
        deadline = time.monotonic() + 5
        while lock._lost is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert lock._lost is not None


def test_a_run_that_lost_its_lock_records_no_decision(engine):
    old = register_model(engine, version="old", path="/old.pt", is_champion=True)
    inner = stub(0.9, 0.1)

    def evaluate_fn(df, config, **kwargs):
        terminate_lock_holder(engine)  # e.g. dropped during a long training
        return inner(df, config, **kwargs)

    with pytest.raises(RetrainLockLostError):
        run(engine, evaluate_fn)
    (event,) = events(engine)
    assert event["decision"] == "failed" and "lost mid-run" in event["notes"]
    assert len(models(engine)) == 1 and read_champion(engine).id == old.id


def test_an_unloadable_champion_fails_before_the_data_is_read(engine, tmp_path):
    register_model(engine, version="old", path=str(tmp_path / "gone.pt"), is_champion=True)
    reads = []
    with pytest.raises(ValueError, match="could not be loaded"):
        run(
            engine,
            stub(0.5),
            load_champion_fn=load_champion_checkpoint,
            read_data_fn=lambda engine, data: reads.append(data),
        )
    assert reads == []
    assert events(engine)[0]["decision"] == "failed"


def test_training_progress_is_logged_and_handed_to_on_progress(engine):
    add_trades(engine, 7)
    inner = stub(0.5)
    epoch = {"event": "epoch", "epoch": 2, "epochs": 5, "train_loss": 0.5, "val_loss": 0.75,
             "seconds": 3.0, "samples_per_s": 1200.0, "eta_seconds": 9.0,
             "since_improved": 1, "patience": 3}  # fmt: skip
    tag = {"candidate": 1, "candidates": 1, "seed": 42}

    def evaluate_fn(df, config, *, progress, **kwargs):
        progress({"event": "candidate_start", **tag})
        progress({**epoch, **tag, "val_loss": float("nan")})  # not JSON: logged as null
        progress({"event": "candidate_end", **tag, "best_epoch": 1, "epochs_run": 2,
                  "seconds": 6.0})  # fmt: skip
        return inner(df, config, **kwargs)

    seen = []

    def on_progress(event_id, event):
        seen.append((event_id, event["event"]))
        raise RuntimeError("a broken listener never fails the run")

    outcome = run(engine, evaluate_fn, on_progress=on_progress)

    assert outcome.status == "promoted"
    assert seen == [(outcome.event_id, k) for k in ("candidate_start", "epoch", "candidate_end")]
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT message, context FROM logs WHERE context ? 'progress' ORDER BY id")
        ).all()
    messages = [m for m, _ in rows]
    assert messages[0] == f"retrain #{outcome.event_id}: candidate 1/1 (seed 42) started"
    assert messages[1].startswith(f"retrain #{outcome.event_id}: candidate 1/1 (seed 42) epoch 2/5")
    assert "ETA 9s" in messages[1]
    assert "trained in 6s: best epoch 1 of 2 run" in messages[2]
    assert rows[1][1]["progress"]["val_loss"] is None
    assert rows[1][1]["training_event_id"] == outcome.event_id


def test_first_candidate_is_promoted_and_fully_recorded(engine):
    add_trades(engine, 7)
    outcome = run(engine, stub(0.5))
    assert outcome.status == "promoted"
    (event,) = events(engine)
    (model,) = models(engine)
    assert event["decision"] == "promoted" and event["finished_at"] is not None
    assert event["champion_model_id"] is None
    assert event["candidate_model_id"] == model["id"] == outcome.candidate_id
    assert event["n_samples"] == 900
    assert event["metrics"]["initial"] is True
    assert event["metrics"]["trades_since_last_train"] == 7
    assert event["metrics"]["warm_start"]["reasons"] == ["no champion checkpoint"]
    assert "first valid candidate" in event["notes"]
    assert model["is_champion"] and model["version"] == "cand"
    assert model["metrics"]["val_total"] == 1.25
    assert model["metrics"]["heldout"][PRIMARY] == pytest.approx(0.5)  # one seed: no champion
    assert model["hyperparams"] == {"training": {"seed": 42}} and model["n_params"] == 123
    assert count_trades_since_last_train(engine) == 0


def test_a_better_candidate_replaces_the_champion(engine):
    old = register_model(engine, version="old", path="/old.pt", is_champion=True)
    fn = stub(0.6, 0.4)
    outcome = run(engine, fn)
    assert fn.calls[0].id == old.id
    assert outcome.status == "promoted"
    (event,) = events(engine)
    assert event["champion_model_id"] == old.id
    assert event["metrics"]["primary_lift"]["mean"] == pytest.approx(0.2)
    assert read_champion(engine).version == "cand"
    assert [m["is_champion"] for m in models(engine)] == [False, True]


def test_a_worse_candidate_is_rejected_but_registered(engine):
    old = register_model(engine, version="old", path="/old.pt", is_champion=True)
    outcome = run(engine, stub(0.4, 0.6))
    assert outcome.status == "rejected" and outcome.reasons
    (event,) = events(engine)
    assert event["decision"] == "rejected"
    assert "does not clear the margin" in event["notes"]
    assert read_champion(engine).id == old.id
    candidate = models(engine)[1]
    assert not candidate["is_champion"] and candidate["metrics"]["heldout"]
    assert event["candidate_model_id"] == candidate["id"]


def test_a_non_finite_candidate_is_rejected_and_registered(engine):
    old = register_model(engine, version="old", path="/old.pt", is_champion=True)
    outcome = run(engine, stub(float("nan"), 0.5))
    assert outcome.status == "rejected"
    (event,) = events(engine)
    assert event["decision"] == "rejected" and event["candidate_model_id"] == outcome.candidate_id
    candidate = models(engine)[1]
    assert not candidate["is_champion"] and candidate["metrics"]["heldout"][PRIMARY] is None
    assert read_champion(engine).id == old.id


def test_a_crash_ends_the_run_as_failed(engine):
    old = register_model(engine, version="old", path="/old.pt", is_champion=True)

    def boom(df, config, **kwargs):
        raise ValueError("too few samples to train a candidate")

    with pytest.raises(ValueError, match="too few samples"):
        run(engine, boom)
    (event,) = events(engine)
    assert event["decision"] == "failed" and event["finished_at"] is not None
    assert event["champion_model_id"] == old.id and event["candidate_model_id"] is None
    assert event["notes"] == "ValueError: too few samples to train a candidate"
    assert len(models(engine)) == 1 and read_champion(engine).id == old.id


def test_a_registration_error_rolls_back_the_promotion_and_fails_the_run(engine):
    register_model(engine, version="cand", path="/dup.pt", is_champion=True)  # same version
    with pytest.raises(ValueError, match="already exists"):
        run(engine, stub(0.9, 0.1))
    (event,) = events(engine)
    assert event["decision"] == "failed" and "already exists" in event["notes"]
    assert read_champion(engine).path == "/dup.pt"


def test_every_step_is_logged_to_the_logs_table(engine):
    register_model(engine, version="old", path="/old.pt", is_champion=True)
    run(engine, stub(0.4, 0.6))
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT service, level, message FROM logs ORDER BY id")).all()
    assert {r.service for r in rows} == {"ai-service"}
    text_ = "\n".join(r.message for r in rows)
    assert "started (manual)" in text_ and "champion old" in text_
    assert "n_samples=900" in text_
    assert f"window 1 seed 1 {PRIMARY}" in text_
    assert "rejected" in text_ and "does not clear the margin" in text_


def test_a_failure_is_logged_as_an_error(engine):
    def boom(df, config, **kwargs):
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        run(engine, boom)
    with engine.connect() as conn:
        level, message = conn.execute(
            text("SELECT level, message FROM logs WHERE level = 'error'")
        ).one()
    assert "failed: RuntimeError: disk full" in message


# --------------------------------------------------------------------------- rollback


def fake_checkpoint(path: Path, version: str, *, horizons=None) -> Path:
    torch.save(
        {
            "model_state": {},
            "version": version,
            "feature_columns": FEATURE_COLUMNS,
            "horizons": horizons or gate_config()["horizons"],
        },
        path,
    )
    return path


def promote_chain(engine, tmp_path, versions):
    """Each version promoted in turn by a retrain event that records the champion it replaced."""
    previous = None
    for i, version in enumerate(versions):
        path = fake_checkpoint(tmp_path / f"{version}.pt", version)
        record = register_model(engine, version=version, path=str(path), is_champion=True)
        add_event(
            engine,
            "promoted",
            T0 + timedelta(minutes=i),
            candidate=record.id,
            champion=previous and previous.id,
        )
        previous = record
    return previous


def test_rollback_restores_the_previous_champion(engine, tmp_path):
    promote_chain(engine, tmp_path, ["a", "b"])
    add_trades(engine, 3, closed_at=T0 + timedelta(hours=1))
    done = rollback(engine, gate_config())
    assert (done.restored.version, done.replaced.version) == ("a", "b")
    assert done.restored.is_champion and not done.replaced.is_champion
    assert read_champion(engine).version == "a"
    event = events(engine)[-1]
    assert event["decision"] == "rollback" and event["finished_at"] is not None
    assert (event["candidate_model_id"], event["champion_model_id"]) == (
        done.restored.id,
        done.replaced.id,
    )
    assert count_trades_since_last_train(engine) == 3  # a rollback trained nothing


def test_rolling_back_twice_walks_further_back(engine, tmp_path):
    promote_chain(engine, tmp_path, ["a", "b", "c"])
    assert rollback(engine, gate_config()).restored.version == "b"
    assert rollback(engine, gate_config()).restored.version == "a"
    with pytest.raises(RollbackRefusedError, match="no previous champion"):
        rollback(engine, gate_config())
    assert read_champion(engine).version == "a"


def test_rollback_never_leaves_zero_champions(engine, tmp_path, monkeypatch):
    promote_chain(engine, tmp_path, ["a", "b"])
    real = promotion.check_rollback

    def vanished_target(*args, **kwargs):  # the target row deleted after the check
        check = real(*args, **kwargs)
        return replace(check, target=replace(check.target, id=9999))

    monkeypatch.setattr(promotion, "check_rollback", vanished_target)
    with pytest.raises(RuntimeError, match="no longer in the registry"):
        rollback(engine, gate_config())
    assert read_champion(engine).version == "b"
    assert events(engine)[-1]["decision"] == "promoted"


def test_rollback_is_refused_without_a_champion(engine):
    with pytest.raises(RollbackRefusedError, match="no current champion"):
        rollback(engine, gate_config())


def test_rollback_is_refused_when_the_checkpoint_is_missing(engine, tmp_path):
    promote_chain(engine, tmp_path, ["a", "b"])
    (tmp_path / "a.pt").unlink()
    with pytest.raises(RollbackRefusedError, match="not found"):
        rollback(engine, gate_config())
    assert read_champion(engine).version == "b"
    assert events(engine)[-1]["decision"] == "promoted"  # nothing recorded


def test_rollback_is_refused_when_the_previous_champion_is_incompatible(engine, tmp_path):
    promote_chain(engine, tmp_path, ["a", "b"])
    fake_checkpoint(tmp_path / "a.pt", "a", horizons=["5m"])
    with pytest.raises(RollbackRefusedError, match="horizons"):
        rollback(engine, gate_config())
    assert read_champion(engine).version == "b"
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM logs WHERE level = 'warn'")).scalar() == 1


# --------------------------------------------------------------------------- real step


@pytest.fixture(scope="module")
def df():
    return make_features_df()


def test_train_and_evaluate_pairs_every_seeded_candidate_with_the_champion(df, tmp_path):
    early = df[df["ts"] < df["ts"].min() + pd.Timedelta(seconds=200)]
    trained = train(early, small_config(), model_dir=tmp_path / "champ", register=False)
    champion = SimpleNamespace(version=trained.version, path=str(trained.checkpoint_path))
    cfg = gate_config()

    events = []
    evaluation = train_and_evaluate(
        df, cfg, champion=champion, model_dir=tmp_path / "out", device="cpu", version="cand",
        progress=events.append,
    )  # fmt: skip
    starts = [e for e in events if e["event"] == "candidate_start"]
    assert starts == [
        {"event": "candidate_start", "candidate": 1, "candidates": 2, "seed": 42},
        {"event": "candidate_start", "candidate": 2, "candidates": 2, "seed": 43},
    ]
    keys = {(w, s) for w in range(2) for s in (42, 43)}
    assert set(evaluation.candidate_runs) == set(evaluation.champion_runs) == keys
    assert evaluation.candidate_runs[(0, 42)][PRIMARY] is not None
    # The champion is one fixed model: a window's metrics are shared by its seeds.
    assert evaluation.champion_runs[(0, 42)] == evaluation.champion_runs[(0, 43)]
    assert evaluation.candidate_runs[(0, 42)] != evaluation.candidate_runs[(0, 43)]
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["cand.pt"]  # extra seeds dropped
    assert evaluation.candidate.version == "cand" and len(evaluation.windows) == 2
    decision = decide_promotion(
        evaluation.candidate_runs, evaluation.champion_runs, PromotionSettings.from_config(cfg)
    )
    assert decision.reasons  # a valid, paired comparison
    assert not any("eligibility" in note for note in evaluation.notes)  # same (disabled) rule


def test_without_a_champion_a_single_candidate_is_trained_and_reported(df, tmp_path):
    cfg = gate_config()  # 2 gate seeds: only a paired comparison needs them
    events = []

    evaluation = train_and_evaluate(
        df, cfg, champion=None, model_dir=tmp_path / "out", device="cpu", version="cand",
        progress=events.append,
    )  # fmt: skip

    assert set(evaluation.candidate_runs) == {(0, 42), (1, 42)}
    assert evaluation.champion_runs is None
    assert any("one candidate trained (seed 42) instead of 2" in n for n in evaluation.notes)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "candidate_start" and kinds[-1] == "candidate_end"
    epochs = [e for e in events if e["event"] == "epoch"]
    assert len(epochs) == len(evaluation.candidate.history) > 0
    assert all(e["candidate"] == 1 and e["candidates"] == 1 and e["seed"] == 42 for e in epochs)
    end = events[-1]
    assert end["epochs_run"] == len(epochs) and end["best_epoch"] >= 1 and end["seconds"] > 0
    decision = decide_promotion(evaluation.candidate_runs, None, PromotionSettings.from_config(cfg))
    assert decision.promote and decision.initial  # the gate takes the single seed


def test_train_and_evaluate_notes_a_champion_trained_under_another_rule(df, tmp_path):
    early = df[df["ts"] < df["ts"].min() + pd.Timedelta(seconds=200)]
    trained = train(early, small_config(), model_dir=tmp_path / "champ", register=False)
    champion = SimpleNamespace(version=trained.version, path=str(trained.checkpoint_path))
    cfg = gate_config()
    cfg["eligibility"] = {"min_volume_sol": 1e-9}  # any traded token: keeps nearly every window

    evaluation = train_and_evaluate(
        df, cfg, champion=champion, model_dir=tmp_path / "out", device="cpu"
    )

    (note,) = [n for n in evaluation.notes if "eligibility" in n]
    assert trained.version in note and "volume>=1e-09 SOL" in note


def test_train_and_evaluate_scores_a_champion_on_its_own_window_shape(df, tmp_path):
    early = df[df["ts"] < df["ts"].min() + pd.Timedelta(seconds=200)]
    cfg = small_config()
    cfg["model"] = {**cfg["model"], "seq_len": 4, "min_seq_len": 3}
    trained = train(early, cfg, model_dir=tmp_path / "champ", register=False)
    champion = SimpleNamespace(version=trained.version, path=str(trained.checkpoint_path))
    evaluation = train_and_evaluate(
        df, gate_config(), champion=champion, model_dir=tmp_path / "out", device="cpu"
    )
    assert evaluation.champion_runs and evaluation.champion_runs[(0, 42)][PRIMARY] is not None


def test_train_and_evaluate_refuses_a_champion_trained_on_all_the_data(df, tmp_path):
    trained = train(df, small_config(), model_dir=tmp_path / "champ", register=False)
    champion = SimpleNamespace(version=trained.version, path=str(trained.checkpoint_path))
    with pytest.raises(ValueError, match="wait for more data"):
        train_and_evaluate(df, gate_config(), champion=champion, model_dir=tmp_path / "out")


def test_resize_windows_keeps_the_same_windows(df):
    cfg = gate_config()
    samples = config_samples(df, cfg)
    for seq_len in (samples.seq_len - 2, samples.seq_len + 3):
        resized = resize_windows(samples, df, cfg, seq_len)
        assert resized.seq_len == seq_len
        np.testing.assert_array_equal(resized.end_row, samples.end_row)
        lengths = resized.end_row - resized.start_row + 1
        assert lengths.max() <= seq_len
    assert resize_windows(samples, df, cfg, samples.seq_len) is samples


def test_feature_index_feeds_an_earlier_spec_its_own_columns():
    assert feature_index(FEATURE_COLUMNS) is None
    assert feature_index(FEATURE_COLUMNS[:3]) == [0, 1, 2]
    assert feature_index(["not-a-column"]) is None


def test_train_and_evaluate_refuses_an_unloadable_champion(df, tmp_path):
    champion = SimpleNamespace(version="gone", path=str(tmp_path / "gone.pt"))
    with pytest.raises(ValueError, match="could not be loaded"):
        train_and_evaluate(df, gate_config(), champion=champion, model_dir=tmp_path)


def test_main_reports_a_skipped_run_and_a_busy_lock(monkeypatch, capsys):
    monkeypatch.setattr("solia_ai.db.make_engine", lambda: "engine")
    outcome = retrain.RetrainOutcome("skipped", 3, 500)
    seen = {}

    def fake_run(engine, **kwargs):
        seen.update(kwargs)
        return outcome

    monkeypatch.setattr(retrain, "run_retrain", fake_run)
    retrain.main(["--trigger", "scheduled", "--force", "--device", "cpu"])
    assert "not due: 3/500" in capsys.readouterr().out
    assert (seen["trigger"], seen["force"]) == ("scheduled", True)

    def busy(engine, **kwargs):
        raise RetrainBusyError("another retrain or rollback is running")

    monkeypatch.setattr(retrain, "run_retrain", busy)
    with pytest.raises(SystemExit) as exit_:
        retrain.main(["--device", "cpu"])
    assert exit_.value.code == 75


def test_main_rollback_refusal_exits_non_zero(monkeypatch):
    monkeypatch.setattr("solia_ai.db.make_engine", lambda: "engine")

    def refused(engine):
        raise RollbackRefusedError(["no previous champion to roll back to"])

    monkeypatch.setattr(retrain, "rollback", refused)
    with pytest.raises(SystemExit, match="no previous champion"):
        retrain.main(["rollback"])


# --------------------------------------------------------------------------- bounded features read


def test_a_run_reads_the_configured_bounds_and_records_them(engine):
    reads = []

    def read(engine, data):
        reads.append(data)
        return pd.DataFrame({"x": [1, 2]}), BOUNDS

    cfg = gate_config()
    cfg["data"] = {"lookback_days": 3, "max_rows": 777}
    run_retrain(engine, cfg, trigger="manual", evaluate_fn=stub(0.5), read_data_fn=read,
                load_champion_fn=lambda champion, config: None)  # fmt: skip
    assert reads == [DataSettings(3.0, 777)]
    (event,) = events(engine)
    (model,) = models(engine)
    assert event["metrics"]["data_bounds"] == BOUNDS.as_dict()
    assert BOUNDS.describe() in event["notes"]
    assert model["metrics"]["data_bounds"] == BOUNDS.as_dict()
    with engine.connect() as conn:
        logged = conn.execute(
            text("SELECT context FROM logs WHERE message LIKE '%features window%'")
        ).scalar_one()
    assert logged["data_bounds"] == BOUNDS.as_dict()


def test_overrides_win_over_the_configured_bounds(engine):
    reads = []

    def read(engine, data):
        reads.append(data)
        return pd.DataFrame(), BOUNDS

    run(engine, stub(0.5), lookback_days=1.5, max_rows=42, read_data_fn=read)
    assert reads == [DataSettings(1.5, 42)]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_rows": 0}, "'data.max_rows' must be >= 1"),
        ({"lookback_days": 0}, "'data.lookback_days' must be > 0"),
        ({"lookback_days": 0.001}, "shorter than the promotion gate needs"),  # 86 s < 3 x 1 m
    ],
)
def test_invalid_bounds_fail_before_anything_is_recorded(engine, override, message):
    add_trades(engine, 7)
    with pytest.raises(ConfigError, match=message):
        run(engine, stub(0.5), **override)
    assert events(engine) == [] and models(engine) == []


def test_the_default_reader_bounds_the_database_read(engine):
    """The path the background trigger and POST /model/retrain take: no reader injected."""
    newest = datetime(2026, 1, 10, tzinfo=UTC)
    with engine.begin() as conn:
        for mint, first, n in [("OLD", newest - timedelta(days=5), 3), ("NEW", newest, 1),
                               ("MID", newest - timedelta(hours=8), 4)]:  # fmt: skip
            conn.execute(text("INSERT INTO tokens_history (mint) VALUES (:m)"), {"m": mint})
            for i in range(n):
                conn.execute(
                    text(
                        "INSERT INTO features (mint, ts, is_migrated, price, volume, holders, "
                        "buy_events, sell_events) VALUES (:m, :ts, false, 0.1, 1, 1, 1, 1)"
                    ),
                    {"m": mint, "ts": first + timedelta(minutes=i)},
                )
    seen = []

    def evaluate_fn(df, config, **kwargs):
        seen.append(df)
        return stub(0.5)(df, config, **kwargs)

    run_retrain(engine, gate_config(), trigger="manual", evaluate_fn=evaluate_fn,
                load_champion_fn=lambda champion, config: None)  # fmt: skip
    (df,) = seen  # the champion and every candidate are scored on this one frame
    assert sorted(df["mint"].unique()) == ["MID", "NEW"]  # OLD is past the 2-day window
    (event,) = events(engine)
    bounds = event["metrics"]["data_bounds"]
    assert bounds["rows_read"] == 5 and bounds["tokens_kept"] == 2
    assert bounds["window_end"] == newest.isoformat()
    assert bounds["lookback_days"] == DATA_DEFAULTS["lookback_days"]


def test_main_passes_the_bound_flags_and_refuses_invalid_ones(monkeypatch):
    monkeypatch.setattr("solia_ai.db.make_engine", lambda: "engine")
    seen = {}

    def fake_run(engine, **kwargs):
        seen.update(kwargs)
        return retrain.RetrainOutcome("skipped", 3, 500)

    monkeypatch.setattr(retrain, "run_retrain", fake_run)
    retrain.main(["--device", "cpu", "--lookback-days", "1.5", "--max-rows", "1000"])
    assert (seen["lookback_days"], seen["max_rows"]) == (1.5, 1000)
    retrain.main(["--device", "cpu"])
    assert (seen["lookback_days"], seen["max_rows"]) == (None, None)  # the config decides

    def invalid(engine, **kwargs):
        raise ConfigError("'data.max_rows' must be >= 1, got 0")

    monkeypatch.setattr(retrain, "run_retrain", invalid)
    with pytest.raises(SystemExit, match="'data.max_rows' must be >= 1"):
        retrain.main(["--device", "cpu", "--max-rows", "0"])


def test_the_background_service_fails_fast_on_invalid_bounds(monkeypatch):
    import solia_ai.config
    from solia_ai.retrain_service import RetrainService

    bad = gate_config()
    bad["data"] = {"max_rows": -5}
    monkeypatch.setattr(solia_ai.config, "load_config", lambda: bad)
    with pytest.raises(ConfigError, match="'data.max_rows' must be >= 1"):
        RetrainService.from_engine(lambda: None)


def test_a_champion_trained_under_another_eligibility_rule_is_reported():
    cfg = {"eligibility": {"min_unique_traders": 3}}
    checkpoint = {"version": "old", "hyperparams": {"model": {}}}  # recorded before the rule

    note = eligibility_note(checkpoint, cfg)

    assert note is not None
    assert "champion old" in note
    assert "disabled (every token)" in note
    assert "traders>=3" in note


def test_a_champion_trained_under_the_config_s_rule_is_not_reported():
    rule = {"min_unique_traders": 3, "min_volume_sol": 0.0, "min_token_age_sec": 0.0}
    checkpoint = {"version": "v", "hyperparams": {"eligibility": rule}}

    assert eligibility_note(checkpoint, {"eligibility": {"min_unique_traders": 3}}) is None
    assert eligibility_note({"hyperparams": {}}, {}) is None  # both disabled
