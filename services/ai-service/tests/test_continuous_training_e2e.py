"""Sprint 7 Definition of Done, end to end: the ai-service app over a real PostgreSQL.

Everything is the production path — the FastAPI app and its lifespan, the background scheduler,
``RETRAIN_EVERY_N_TRADES`` / ``SOLIA_AI_CONFIG`` / ``MODEL_DIR`` from the environment, features
read back from the ``features`` table, a real (tiny) champion and real candidate training and
scoring, the registry, retention, rollback — except one thing: the gate's metrics. A candidate
trained on the same synthetic data is not reliably better or worse than its champion, so the
evaluation step runs for real and then sets the primary metric (candidate = champion + ``lift``,
every other metric equal to the champion's) to make the decision deterministic.

Runs against the shared Testcontainers PostgreSQL (skipped locally without Docker, required in
CI). Kept cheap: tiny model, a few epochs, one seed over two held-out windows.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from test_retrain import gate_config
from test_train import make_features_df, small_config

from solia_ai.api import create_app, registry_loader
from solia_ai.features import DB_FEATURE_COLUMNS, FEATURE_COLUMNS
from solia_ai.promotion import PromotionSettings
from solia_ai.registry import latest_training_event, read_champion, register_model
from solia_ai.retrain import (
    _KEY_PARTS,
    count_trades_since_last_train,
    recover_abandoned_runs,
    retrain_threshold,
    rollback,
    run_retrain,
    train_and_evaluate,
)
from solia_ai.retrain_service import RetrainService, SchedulerSettings
from solia_ai.train import train

N_TRADES = 5  # RETRAIN_EVERY_N_TRADES
TICK_SECONDS = 0.2  # RETRAIN_CHECK_INTERVAL_SECONDS
RUN_TIMEOUT_SECONDS = 180.0
CONFIG = gate_config(windows=2, seeds=1)
PRIMARY = PromotionSettings.from_config(CONFIG).primary_key

# Whether anyone holds the retrain lock: retrain._HELD_SQL's predicate, for any session.
_LOCK_HELD_SQL = text(
    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "
    "AND classid::bigint = :hi AND objid::bigint = :lo AND objsubid = 1"
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("SOLIA_AI_CONFIG", str(config_path))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("RETRAIN_EVERY_N_TRADES", str(N_TRADES))
    monkeypatch.setenv("RETRAIN_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("RETRAIN_CHECK_INTERVAL_SECONDS", str(TICK_SECONDS))
    return tmp_path


@pytest.fixture
def engine(pg_url, env):
    """A database holding the data-pipeline's features and a champion trained on their start."""
    eng = create_engine(pg_url)
    with eng.begin() as conn:
        conn.exec_driver_sql(
            "TRUNCATE training_events, models, logs, trades, features, tokens_history "
            "RESTART IDENTITY CASCADE"
        )
    df = make_features_df()
    with eng.begin() as conn:
        for mint in df["mint"].unique():
            conn.execute(text("INSERT INTO tokens_history (mint) VALUES (:m)"), {"m": mint})
    rows = df[["mint", "ts", *DB_FEATURE_COLUMNS]].astype({"is_migrated": bool})
    rows.to_sql("features", eng, if_exists="append", index=False)

    early = df[df["ts"] < df["ts"].min() + pd.Timedelta(seconds=200)]
    trained = train(early, small_config(), version="champion-0", register=False, device="cpu")
    register_model(eng, version="champion-0", path=str(trained.checkpoint_path), is_champion=True)
    yield eng
    eng.dispose()


class GateError(AssertionError):
    """One of the gate's checks failed (on the retrain's worker thread)."""


class Gate:
    """The real train + evaluate step, with the primary metric set to champion + ``lift``.

    It runs on the retrain's worker thread, where a failed check only records the run as
    ``failed`` (and leaves the counter due, so the scheduler retrains again): the failures are
    kept in ``errors`` for the test thread to fail on at once, with their message."""

    def __init__(self) -> None:
        self.lift = 0.1
        self.real_runs = 0
        self.errors: list[str] = []

    def __call__(self, df, config, **kwargs):
        evaluation = train_and_evaluate(df, config, **kwargs)
        self.real_runs += 1
        if evaluation.champion_runs is None:
            self._fail("the candidate was not paired against the champion")
        if set(evaluation.candidate_runs) != set(evaluation.champion_runs):
            self._fail(
                f"candidate runs {sorted(evaluation.candidate_runs)} != "
                f"champion runs {sorted(evaluation.champion_runs)}"
            )
        champion = {
            run: {**metrics, PRIMARY: 0.5} for run, metrics in evaluation.champion_runs.items()
        }
        candidate = {run: {**m, PRIMARY: 0.5 + self.lift} for run, m in champion.items()}
        return replace(evaluation, candidate_runs=candidate, champion_runs=champion)

    def _fail(self, message: str) -> None:
        self.errors.append(message)
        raise GateError(message)


@pytest.fixture
def gate():
    return Gate()


class Harness:
    """The service ``RetrainService.from_engine`` builds — same functions, settings from the
    environment — with the gate as ``run_retrain``'s ``evaluate_fn``, plus what the test thread
    waits on: startup recovery done, scheduler checks made, runs returned (lock released)."""

    def __init__(self, engine, gate: Gate) -> None:
        self.engine = engine
        self.gate = gate
        self.threshold = retrain_threshold()
        self.recovered = threading.Event()
        self.scheduler_checks = 0
        self.returned_runs = 0
        self._changed = threading.Condition()
        self.service = RetrainService(
            self._run,
            self._count,
            lambda: latest_training_event(engine),
            recover_fn=self._recover,
            settings=SchedulerSettings.from_env(),
        )

    def _run(self, *, trigger, on_started=None, on_progress=None):
        try:
            return run_retrain(
                self.engine,
                trigger=trigger,
                threshold=self.threshold,
                device="cpu",
                on_started=on_started,
                on_progress=on_progress,
                evaluate_fn=self.gate,
            )
        finally:  # run_retrain returns once its decision, logs and pruning are done and unlocked
            self._bump("returned_runs")

    def _count(self):
        counted = (count_trades_since_last_train(self.engine), self.threshold)
        if threading.current_thread().name == "retrain-scheduler":  # not a /model/status read
            self._bump("scheduler_checks")
        return counted

    def _recover(self):
        try:
            return recover_abandoned_runs(self.engine)
        finally:
            self.recovered.set()

    def _bump(self, counter: str) -> None:
        with self._changed:
            setattr(self, counter, getattr(self, counter) + 1)
            self._changed.notify_all()

    def _wait(self, predicate, timeout: float, failure: str) -> None:
        with self._changed:
            done = self._changed.wait_for(lambda: predicate() or self.gate.errors, timeout)
        if self.gate.errors:
            pytest.fail(f"the gate failed on the retrain thread: {self.gate.errors}")
        if not done:
            pytest.fail(failure)

    def wait_for_scheduler_checks(self, n: int) -> None:
        """``n`` more scheduler reads of the counter, from now."""
        target = self.scheduler_checks + n
        timeout = 50 * TICK_SECONDS
        self._wait(
            lambda: self.scheduler_checks >= target,
            timeout,
            f"the scheduler made fewer than {n} checks within {timeout}s",
        )

    def wait_for_returned_run(self, count: int = 1):
        """The ``count``-th training event, once its run has returned."""
        self._wait(
            lambda: self.returned_runs >= count,
            RUN_TIMEOUT_SECONDS,
            f"run #{count} did not return within {RUN_TIMEOUT_SECONDS}s",
        )
        rows = events(self.engine)
        assert len(rows) >= count, rows
        return rows[count - 1]

    def wait_for_idle(self, timeout: float = RUN_TIMEOUT_SECONDS) -> None:
        """No run in flight in this process, and the retrain lock free."""
        deadline = time.monotonic() + timeout
        while self.service.busy and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.service.busy:
            pytest.fail(f"a retrain was still running after {timeout}s")
        wait_for_lock_release(self.engine)


@pytest.fixture
def harness(engine, gate):
    return Harness(engine, gate)


@pytest.fixture
def client(harness):
    """The ai-service app as ``uvicorn solia_ai.api:app`` builds it, over the test database."""
    app = create_app(
        registry_loader(lambda: harness.engine), retrain=harness.service, refresh_seconds=0
    )
    with TestClient(app) as test_client:  # the lifespan starts recovery and the scheduler
        if not harness.recovered.wait(10.0):
            pytest.fail("startup recovery did not finish within 10s")
        wait_for_lock_release(harness.engine)
        yield test_client
    # The lifespan stopped the scheduler, but a run it started may still be going: let it end
    # before the engine is disposed and the next test truncates the shared database.
    harness.wait_for_idle()


def wait_for_lock_release(engine, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            if conn.execute(_LOCK_HELD_SQL, _KEY_PARTS).scalar_one() == 0:
                return
        time.sleep(0.05)
    pytest.fail(f"the retrain lock was still held after {timeout}s")


def close_trades(engine, n: int) -> None:
    with engine.begin() as conn:
        for _ in range(n):
            conn.execute(
                text("INSERT INTO trades (mint, status, closed_at) VALUES ('M0', 'closed', now())")
            )


def events(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT * FROM training_events ORDER BY id")).mappings().all()


def models(engine):
    with engine.connect() as conn:
        return {
            row["version"]: row
            for row in conn.execute(text("SELECT * FROM models ORDER BY id")).mappings()
        }


def status(client):
    response = client.get("/model/status")
    assert response.status_code == 200, response.text
    return response.json()


def automatic_run(harness, count: int = 1):
    """Close ``N_TRADES`` trades (one short first: nothing may start) and await the run."""
    close_trades(harness.engine, N_TRADES - 1)
    # Two reads from now: at least one whole check saw the N-1 trades, and started nothing.
    harness.wait_for_scheduler_checks(2)
    assert harness.returned_runs == count - 1
    assert len(events(harness.engine)) == count - 1
    assert count_trades_since_last_train(harness.engine) == N_TRADES - 1  # /model/status caches it
    close_trades(harness.engine, 1)
    return harness.wait_for_returned_run(count)


def test_n_closed_trades_start_a_retrain_that_promotes_a_better_candidate(
    client, engine, gate, harness
):
    before = status(client)
    assert before["champion_version"] == "champion-0"
    assert (before["trades_since_last_train"], before["retrain_threshold"]) == (0, N_TRADES)

    event = automatic_run(harness)

    assert (event["trigger"], event["decision"]) == ("n_trades", "promoted")
    assert gate.real_runs == 1
    registry = models(engine)
    champion = read_champion(engine)
    assert champion.id == event["candidate_model_id"] != registry["champion-0"]["id"]
    assert event["champion_model_id"] == registry["champion-0"]["id"]
    assert not registry["champion-0"]["is_champion"]
    assert event["n_samples"] > 0 and event["metrics"]["primary"] == PRIMARY
    after = status(client)
    assert after["champion_version"] == champion.version  # served without a restart
    assert after["trades_since_last_train"] == 0
    assert after["latest_run"]["id"] == event["id"]
    assert after["latest_run"]["decision"] == "promoted"
    response = client.post("/predict", json={"features": [[0.0] * len(FEATURE_COLUMNS)] * 3})
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == champion.version


def test_a_worse_candidate_is_rejected_and_the_champion_kept(client, engine, gate, harness):
    gate.lift = -0.1

    event = automatic_run(harness)

    assert (event["trigger"], event["decision"]) == ("n_trades", "rejected")
    assert "does not clear the margin" in event["notes"]
    assert read_champion(engine).version == "champion-0"
    candidate = next(m for m in models(engine).values() if m["id"] == event["candidate_model_id"])
    assert not candidate["is_champion"]  # kept in the registry, metrics and all
    assert candidate["metrics"]["heldout"][PRIMARY] == pytest.approx(0.4)
    after = status(client)
    assert after["champion_version"] == "champion-0"
    assert after["trades_since_last_train"] == 0  # a rejected run trained on those trades too
    assert after["latest_run"]["decision"] == "rejected"


def test_rollback_restores_the_prior_champion(client, engine, harness):
    promoted = automatic_run(harness)
    assert promoted["decision"] == "promoted"
    close_trades(engine, 2)

    done = rollback(engine)  # the config from SOLIA_AI_CONFIG, as the CLI does

    assert done.restored.version == "champion-0"
    assert read_champion(engine).version == "champion-0"
    event = events(engine)[-1]
    assert (event["decision"], event["trigger"]) == ("rollback", "manual")
    assert (event["candidate_model_id"], event["champion_model_id"]) == (
        done.restored.id,
        promoted["candidate_model_id"],
    )
    after = status(client)
    assert after["champion_version"] == "champion-0"
    assert after["trades_since_last_train"] == 2  # a rollback trained nothing


def test_a_manual_retrain_is_accepted_and_recorded_as_manual(client, engine, gate, harness):
    # The body the backend's SoliaAiClient sends for POST /api/v1/model/retrain.
    response = client.post("/model/retrain", json={"trigger": "manual"})

    assert response.status_code == 202, response.text
    event_id = response.json()["training_event_id"]
    event = harness.wait_for_returned_run()
    assert event["id"] == event_id
    assert (event["trigger"], event["decision"]) == ("manual", "promoted")  # below the threshold
    assert gate.real_runs == 1
    assert status(client)["latest_run"]["id"] == event_id
