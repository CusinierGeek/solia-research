"""The ai-service FastAPI surface, exercised through the in-memory TestClient (no database)."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_inference import saved_checkpoint

import solia_ai.api
import solia_ai.registry
from solia_ai.api import (
    LoadedChampion,
    check_model_dir_writable,
    create_app,
    lazy_engine,
    registry_loader,
)
from solia_ai.features import DB_FEATURE_COLUMNS, FEATURE_COLUMNS, VENUE_COLUMNS, VENUE_OTHER
from solia_ai.inference import Predictor
from solia_ai.model import IncompatibleCheckpointError, SoliaTransformer
from solia_ai.retrain import RetrainOutcome
from solia_ai.retrain_service import RetrainService

HORIZONS = ("30s", "1m", "5m")
N_FEATURES = len(FEATURE_COLUMNS)


def _predictor(version: str = "test-v1") -> Predictor:
    model = SoliaTransformer(
        n_features=N_FEATURES,
        horizons=HORIZONS,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        seq_len=8,
    )
    model.fit_normalizer(np.random.default_rng(0).normal(size=(64, N_FEATURES)).astype("f4"))
    return Predictor(model.eval(), model_version=version)


def _champion(version: str = "test-v1") -> LoadedChampion:
    return LoadedChampion(
        predictor=_predictor(version),
        created_at=datetime(2026, 4, 20, 12, tzinfo=UTC),
        metrics={"val_total": 0.3},
    )


def _static_loader(current=None) -> LoadedChampion:
    return current or _champion()


LATEST_RUN = {
    "id": 7,
    "trigger": "n_trades",
    "started_at": "2026-04-21T08:00:00+00:00",
    "finished_at": "2026-04-21T08:20:00+00:00",
    "decision": "rejected",
}


def _service(run_fn=None, *, trades: int = 142, threshold: int = 500, **kwargs) -> RetrainService:
    """A retrain service over stubs (no database): ``run_fn`` defaults to a run recorded as #17."""

    def recorded(*, trigger, on_started=None, on_progress=None):
        if on_started:
            on_started(17)
        return RetrainOutcome("rejected", trades, threshold, event_id=17)

    return RetrainService(
        run_fn or recorded, lambda: (trades, threshold), lambda: LATEST_RUN, **kwargs
    )


def _client(loader=None, **kwargs) -> TestClient:
    kwargs.setdefault("retrain", _service())
    return TestClient(create_app(loader or _static_loader, **kwargs))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _dict_rows(n: int = 3) -> list[dict[str, float]]:
    return [{col: float(i + 1) for col in FEATURE_COLUMNS} for i in range(n)]


def _list_rows(n: int = 3) -> list[list[float]]:
    return [[float(i + 1)] * N_FEATURES for i in range(n)]


# --------------------------------------------------------------------------- health / status


def test_health_is_ok_without_touching_the_model():
    loaded = False

    def loader(current):
        nonlocal loaded
        loaded = True
        return _champion()

    response = _client(loader).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert loaded is False  # health never forces the champion to load


def test_model_status_reports_the_champion():
    response = _client().get("/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["champion_version"] == "test-v1"
    assert body["created_at"] == "2026-04-20T12:00:00+00:00"
    assert body["param_count"] == _predictor().model.n_params
    assert "n_params" not in body  # one parameter count, from the loaded model
    assert body["metrics"] == {"val_total": 0.3}
    assert body["seq_len"] == 8
    assert body["min_seq_len"] == 1  # no hyperparams recorded: any non-empty window is accepted
    # ... nor an eligibility rule: a model from before the rule was trained on every token.
    assert body["eligibility"] == {
        "min_unique_traders": 0,
        "min_volume_sol": 0.0,
        "min_token_age_sec": 0.0,
    }
    assert body["trades_since_last_train"] == 142
    assert body["retrain_threshold"] == 500
    assert body["latest_run"] == LATEST_RUN


def test_model_status_reports_no_progress_when_the_count_fails():
    def down():
        raise ConnectionError("database down")

    service = RetrainService(lambda **_: None, down, down)
    body = _client(retrain=service).get("/model/status").json()

    assert body["champion_version"] == "test-v1"  # the champion is still served
    # Both counters null together, so the backend falls back to its own consistent pair.
    assert body["trades_since_last_train"] is None
    assert body["retrain_threshold"] is None
    assert body["latest_run"] is None


def test_model_status_without_any_run_has_a_null_latest_run():
    service = RetrainService(lambda **_: None, lambda: (0, 500), lambda: None)
    body = _client(retrain=service).get("/model/status").json()

    assert (body["trades_since_last_train"], body["retrain_threshold"]) == (0, 500)
    assert body["latest_run"] is None


# --------------------------------------------------------------------------- retrain


def test_retrain_returns_202_with_the_training_event_id():
    calls = []

    def run_fn(*, trigger, on_started=None, on_progress=None):
        calls.append(trigger)
        on_started(17)
        return RetrainOutcome("promoted", 0, 500, event_id=17)

    response = _client(retrain=_service(run_fn)).post("/model/retrain", json={"trigger": "manual"})

    assert response.status_code == 202
    assert response.json() == {"training_event_id": 17}
    assert calls == ["manual"]


def test_retrain_answers_202_before_the_run_finishes_and_409_while_it_runs():
    release = threading.Event()
    finished = threading.Event()

    def long_run(*, trigger, on_started=None, on_progress=None):
        on_started(21)
        release.wait(10)
        finished.set()
        return RetrainOutcome("rejected", 0, 500, event_id=21)

    service = _service(long_run)
    client = _client(retrain=service)
    try:
        first = client.post("/model/retrain", json={"trigger": "manual"})
        assert first.status_code == 202 and first.json() == {"training_event_id": 21}
        assert service.busy and not finished.is_set()

        second = client.post("/model/retrain", json={"trigger": "manual"})
        assert second.status_code == 409
        assert second.json()["error"]["code"] == "retrain_in_progress"

        # Training runs off the request threads: predict and health keep answering meanwhile.
        started = time.monotonic()
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json={"features": _list_rows()}).status_code == 200
        assert time.monotonic() - started < 5
    finally:
        release.set()
    assert finished.wait(5)
    deadline = time.monotonic() + 5
    while service.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.post("/model/retrain", json={"trigger": "manual"}).status_code == 202


def test_retrain_is_409_when_another_process_holds_the_lock():
    from solia_ai.retrain import RetrainBusyError

    def busy(*, trigger, on_started=None, on_progress=None):
        raise RetrainBusyError("another retrain or rollback is running; try again later")

    response = _client(retrain=_service(busy)).post("/model/retrain", json={"trigger": "manual"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "retrain_in_progress"


def test_retrain_is_503_when_the_run_cannot_be_started():
    def down(*, trigger, on_started=None, on_progress=None):
        raise ConnectionError("database down")

    response = _client(retrain=_service(down)).post("/model/retrain", json={"trigger": "manual"})

    assert response.status_code == 503
    body = response.json()["error"]
    assert body["code"] == "retrain_unavailable" and "database down" in body["message"]


def test_retrain_is_503_when_the_run_is_not_recorded_in_time():
    release = threading.Event()

    def stuck(*, trigger, on_started=None, on_progress=None):
        release.wait(10)

    service = _service(stuck, start_timeout=0.05)
    try:
        response = _client(retrain=service).post("/model/retrain", json={"trigger": "manual"})
    finally:
        release.set()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "retrain_unavailable"


@pytest.mark.parametrize(
    "body",
    [{}, {"trigger": "n_trades"}, {"trigger": "MANUAL"}, {"trigger": 1}, [], "manual"],
)
def test_retrain_rejects_an_invalid_body(body):
    calls = []
    service = _service(lambda **kw: calls.append(kw))

    response = _client(retrain=service).post("/model/retrain", json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert calls == []


def test_retrain_without_a_body_is_rejected():
    response = _client().post("/model/retrain")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_the_app_lifespan_starts_and_stops_the_scheduler():
    from solia_ai.retrain_service import SchedulerSettings

    service = _service(settings=SchedulerSettings(enabled=True, interval_seconds=3600))
    with _client(retrain=service):
        assert service._scheduler is not None and service._scheduler.is_alive()
    assert service._scheduler is None


def test_the_app_lifespan_recovers_abandoned_runs_in_the_background():
    recovered = threading.Event()

    def recover():
        recovered.set()
        return [3]

    with _client(retrain=_service(recover_fn=recover)) as client:
        assert recovered.wait(5)
        assert client.get("/health").status_code == 200


def test_a_custom_loader_without_a_retrain_service_touches_no_database(monkeypatch):
    # An invalid retrain env would fail the database-backed service at build time.
    monkeypatch.setenv("RETRAIN_SCHEDULER_ENABLED", "maybe")
    monkeypatch.setenv("RETRAIN_EVERY_N_TRADES", "nope")
    monkeypatch.setattr(
        RetrainService, "from_engine", lambda engine: pytest.fail("built the database service")
    )

    with TestClient(create_app(_static_loader)) as client:
        body = client.get("/model/status").json()
        retrain = client.post("/model/retrain", json={"trigger": "manual"})

    assert body["champion_version"] == "test-v1"
    assert body["trades_since_last_train"] is None and body["latest_run"] is None
    assert retrain.status_code == 503
    assert retrain.json()["error"]["code"] == "retrain_unavailable"


# --------------------------------------------------------------------------- predict


def test_predict_returns_the_full_contract_per_horizon():
    response = _client().post("/predict", json={"mint": "So1x", "features": _dict_rows()})

    assert response.status_code == 200
    body = response.json()
    assert body["mint"] == "So1x"
    assert body["model_version"] == "test-v1"
    assert [p["horizon"] for p in body["predictions"]] == list(HORIZONS)
    for p in body["predictions"]:
        assert set(p) == {"horizon", "pump_prob", "rug_prob", "expected_return", "confidence"}
        assert 0.0 <= p["pump_prob"] <= 1.0
        assert 0.0 <= p["confidence"] <= 100.0


def _scored_windows(monkeypatch) -> list[np.ndarray]:
    """Spy on the windows the champion is asked to score."""
    seen: list[np.ndarray] = []
    original = Predictor.predict

    def spy(self, window, *args, **kwargs):
        seen.append(np.asarray(window))
        return original(self, window, *args, **kwargs)

    monkeypatch.setattr(Predictor, "predict", spy)
    return seen


def _venue_rows(*venues) -> list[dict]:
    return [{**{c: 1.0 for c in DB_FEATURE_COLUMNS}, "venue": v} for v in venues]


def test_predict_encodes_the_raw_venue_string_like_training(monkeypatch):
    # The backend sends the stored columns plus venue; the one-hot is derived server-side.
    seen = _scored_windows(monkeypatch)

    response = _client().post(
        "/predict", json={"features": _venue_rows(" PumpSwap ", "new-amm", None)}
    )

    assert response.status_code == 200
    (window,) = seen
    hot = window[:, [FEATURE_COLUMNS.index(c) for c in VENUE_COLUMNS]]
    expected = np.zeros_like(hot)
    expected[0, VENUE_COLUMNS.index("venue_pumpswap")] = 1.0
    expected[1:, VENUE_COLUMNS.index(VENUE_OTHER)] = 1.0  # unknown venue and NULL, as training
    np.testing.assert_array_equal(hot, expected)


def test_predict_rejects_rows_that_name_different_features():
    rows = _venue_rows("pumpfun", "pumpswap")
    del rows[1]["venue"]  # would otherwise be scored as venue_other, silently

    response = _client().post("/predict", json={"features": rows})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_features"
    assert (
        "row 1" in response.json()["error"]["message"]
        and "venue" in response.json()["error"]["message"]
    )


def test_predict_rejects_a_raw_venue_beside_its_one_hot_columns():
    rows = [{**row, "venue_pumpfun": 1.0} for row in _venue_rows("pumpswap")]

    response = _client().post("/predict", json={"features": rows})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_features"
    assert "venue_pumpfun" in response.json()["error"]["message"]


def test_predict_rejects_a_non_string_venue():
    response = _client().post("/predict", json={"features": _venue_rows(3.0)})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_features"


def test_predict_rejects_a_non_numeric_string_in_a_numeric_column():
    rows = _dict_rows()
    rows[0]["volume"] = "lots"

    response = _client().post("/predict", json={"features": rows})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_features"


def test_predict_accepts_array_rows():
    response = _client().post("/predict", json={"features": _list_rows()})

    assert response.status_code == 200
    assert response.json()["mint"] is None


def test_predict_accepts_a_single_feature_object_as_a_one_row_window():
    # docs/API-CONTRACTS.md: "features": { /* latest vector */ }
    response = _client().post("/predict", json={"mint": "So1x", "features": _dict_rows(1)[0]})

    assert response.status_code == 200
    assert [p["horizon"] for p in response.json()["predictions"]] == list(HORIZONS)


def _champion_with_min_seq_len(min_seq_len: int) -> LoadedChampion:
    predictor = _predictor()
    predictor.metadata["hyperparams"] = {"model": {"seq_len": 8, "min_seq_len": min_seq_len}}
    return LoadedChampion(predictor=predictor)


def test_predict_rejects_a_window_shorter_than_the_champion_min_seq_len():
    champion = _champion_with_min_seq_len(4)
    client = _client(lambda current: current or champion)

    response = client.post("/predict", json={"features": _dict_rows(3)})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "window_too_short"
    assert "at least 4" in error["message"]


def test_predict_accepts_a_window_of_exactly_min_seq_len_rows():
    champion = _champion_with_min_seq_len(4)
    client = _client(lambda current: current or champion)

    assert client.post("/predict", json={"features": _dict_rows(4)}).status_code == 200


def test_predict_accepts_a_window_longer_than_seq_len():
    response = _client().post("/predict", json={"features": _dict_rows(20)})  # seq_len is 8

    assert response.status_code == 200


def test_predict_rejects_an_empty_horizon_list():
    response = _client().post("/predict", json={"features": _dict_rows(), "horizons": []})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert "horizons" in response.json()["error"]["message"]


def test_predict_rejects_a_bare_number_list_as_features():
    response = _client().post("/predict", json={"features": [1.0, 2.0]})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_an_unexpected_error_keeps_the_error_shape():
    champion = _champion()

    def boom(*_args, **_kwargs):
        raise RuntimeError("kaboom")

    champion.predictor.predict = boom
    client = TestClient(
        create_app(lambda current: champion, retrain=_service()), raise_server_exceptions=False
    )

    response = client.post("/predict", json={"features": _list_rows()})

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "internal server error"}
    }


def test_predict_restricts_to_requested_horizons():
    response = _client().post(
        "/predict", json={"features": _dict_rows(), "horizons": ["5m", "30s"]}
    )

    assert response.status_code == 200
    assert [p["horizon"] for p in response.json()["predictions"]] == ["5m", "30s"]


def test_predict_rejects_an_unknown_horizon():
    response = _client().post("/predict", json={"features": _dict_rows(), "horizons": ["7d"]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "7d" in response.json()["error"]["message"]


def test_predict_rejects_an_array_row_of_the_wrong_width():
    bad = [[1.0, 2.0, 3.0]]  # not N_FEATURES wide

    response = _client().post("/predict", json={"features": bad})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_features"


def test_predict_rejects_rows_missing_a_feature_column():
    incomplete = [{"price": 1.0}]  # missing every other FEATURE_COLUMN

    response = _client().post("/predict", json={"features": incomplete})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_features"


def test_predict_rejects_an_empty_window_with_the_error_shape():
    response = _client().post("/predict", json={"features": []})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


# --------------------------------------------------------------------------- no champion


def test_endpoints_return_503_until_a_champion_is_promoted():
    def loader(current):
        raise LookupError("no champion is registered")

    client = _client(loader)

    assert client.get("/health").status_code == 200  # liveness is independent
    for request in (
        lambda: client.get("/model/status"),
        lambda: client.post("/predict", json={"features": _list_rows()}),
    ):
        response = request()
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "no_champion"


def test_a_bad_request_is_a_400_even_without_a_champion():
    def loader(current):
        raise LookupError("no champion is registered")

    response = _client(loader).post("/predict", json={"features": [[1.0, 2.0]]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_features"


@pytest.mark.parametrize(
    "failure",
    [OSError("could not connect to server"), FileNotFoundError("champ.pt"), ValueError("corrupt")],
)
def test_a_failed_champion_load_is_a_503_with_the_error_shape(failure):
    def loader(current):
        raise failure

    client = _client(loader)

    for response in (
        client.get("/model/status"),
        client.post("/predict", json={"features": _list_rows()}),
    ):
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "champion_unavailable"


def test_an_incompatible_champion_is_a_503_naming_the_retrain():
    def loader(current):
        raise IncompatibleCheckpointError("trained on 19 feature columns: retrain")

    client = _client(loader)

    for response in (
        client.get("/model/status"),
        client.post("/predict", json={"features": _list_rows()}),
    ):
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "champion_incompatible"
        assert "retrain" in error["message"]


def test_an_incompatible_new_champion_keeps_the_cached_one_serving():
    now = [0.0]
    calls = 0

    def loader(current):
        nonlocal calls
        calls += 1
        if current is None:
            return _champion()
        raise IncompatibleCheckpointError("trained on 19 feature columns")

    client = _client(loader, clock=lambda: now[0])
    assert client.get("/model/status").status_code == 200
    now[0] += 3600  # past the refresh interval

    assert client.get("/model/status").status_code == 200
    assert calls == 2


def test_the_champion_is_cached_between_refreshes():
    calls = 0

    def loader(current):
        nonlocal calls
        calls += 1
        return current or _champion()

    client = _client(loader, clock=FakeClock())
    client.get("/model/status")
    client.post("/predict", json={"features": _list_rows()})
    client.get("/model/status")

    assert calls == 1


def test_a_newly_promoted_champion_is_served_after_the_refresh_interval():
    registry = {"version": "v1"}
    seen = []

    def loader(current):
        seen.append(current.version if current else None)
        if current is not None and current.version == registry["version"]:
            return current
        return _champion(registry["version"])

    clock = FakeClock()
    client = _client(loader, refresh_seconds=30, clock=clock)
    assert client.get("/model/status").json()["champion_version"] == "v1"

    registry["version"] = "v2"  # promoted in the registry
    clock.now += 10
    assert client.get("/model/status").json()["champion_version"] == "v1"  # not re-checked yet

    clock.now += 30
    assert client.get("/model/status").json()["champion_version"] == "v2"
    body = client.post("/predict", json={"features": _list_rows()}).json()
    assert body["model_version"] == "v2"
    assert seen == [None, "v1"]  # the refresh hands the loader the cached champion


@pytest.mark.parametrize("failure", [LookupError("demoted"), OSError("database down")])
def test_a_failed_refresh_keeps_serving_the_cached_champion(failure):
    calls = 0

    def loader(current):
        nonlocal calls
        calls += 1
        if current is None:
            return _champion("v1")
        raise failure

    clock = FakeClock()
    client = _client(loader, refresh_seconds=5, clock=clock)
    client.get("/model/status")
    clock.now += 10

    response = client.get("/model/status")

    assert response.status_code == 200
    assert response.json()["champion_version"] == "v1"
    clock.now += 1  # the failed check still resets the interval: no retry storm
    client.get("/model/status")
    assert calls == 2


def test_concurrent_first_requests_load_the_champion_once():
    calls = 0

    def loader(current):
        nonlocal calls
        if current is not None:
            return current
        calls += 1
        time.sleep(0.05)  # a slow checkpoint read, so the requests overlap
        return _champion()

    client = _client(loader, clock=FakeClock())
    statuses = []

    def hit():
        statuses.append(client.get("/model/status").status_code)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert statuses == [200] * 8
    assert calls == 1


def test_create_app_rejects_a_negative_refresh_interval():
    with pytest.raises(ValueError, match="refresh_seconds"):
        create_app(_static_loader, refresh_seconds=-1)


# --------------------------------------------------------------------------- registry loader


def _record(path, version="champ-1"):
    return SimpleNamespace(
        version=version, path=str(path), created_at=datetime(2026, 5, 1, tzinfo=UTC)
    )


def _counting_factory(engines):
    def factory():
        engines.append(object())
        return engines[-1]

    return factory


def test_registry_loader_reuses_one_engine_and_skips_unchanged_versions(tmp_path, monkeypatch):
    _, path = saved_checkpoint(tmp_path, version="champ-1")
    engines = []
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: _record(path))
    loader = registry_loader(lazy_engine(_counting_factory(engines)))

    first = loader(None)
    again = loader(first)

    assert first.version == "champ-1"
    assert first.predictor.model_version == "champ-1"
    assert first.created_at == datetime(2026, 5, 1, tzinfo=UTC)
    assert first.metrics == {}
    assert again is first  # same version: the checkpoint is not re-read
    assert len(engines) == 1  # one engine for every call


def test_registry_loader_loads_a_new_champion_version(tmp_path, monkeypatch):
    _, path = saved_checkpoint(tmp_path, version="champ-1")
    records = {"current": _record(path)}
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: records["current"])
    loader = registry_loader(lazy_engine(object))
    first = loader(None)

    records["current"] = _record(path, version="champ-2")
    second = loader(first)

    assert second is not first
    assert second.version == "champ-2"


def test_registry_loader_without_a_champion_raises_lookup_error(monkeypatch):
    engines = []
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: None)
    loader = registry_loader(lazy_engine(_counting_factory(engines)))

    for _ in range(3):  # polled before the first promotion
        with pytest.raises(LookupError, match="no champion"):
            loader(None)

    assert len(engines) == 1  # no engine leaked per call


# --------------------------------------------------------------------------- MODEL_DIR


def test_model_dir_check_passes_when_unset_missing_or_writable(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert check_model_dir_writable()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "absent"))
    assert check_model_dir_writable()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert check_model_dir_writable()
    assert list(tmp_path.iterdir()) == []  # the probe leaves nothing behind


def test_unwritable_model_dir_warns_without_failing_startup(tmp_path, monkeypatch, caplog):
    def denied(**kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(solia_ai.api.tempfile, "TemporaryFile", denied)

    assert not check_model_dir_writable()
    assert "is not writable" in caplog.text and "chown" in caplog.text
    assert _client().get("/health").json() == {"status": "ok"}  # serving still starts


def test_model_status_reports_the_champion_s_eligibility_rule():
    predictor = _predictor()
    rule = {"min_unique_traders": 3, "min_volume_sol": 0.5, "min_token_age_sec": 0.0}
    predictor.metadata["hyperparams"] = {"eligibility": rule}
    champion = LoadedChampion(predictor=predictor)

    response = _client(lambda current: current or champion).get("/model/status")

    assert response.json()["eligibility"] == rule
