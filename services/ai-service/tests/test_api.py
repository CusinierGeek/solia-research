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
from solia_ai.api import LoadedChampion, check_model_dir_writable, create_app, registry_loader
from solia_ai.features import FEATURE_COLUMNS
from solia_ai.inference import Predictor
from solia_ai.model import IncompatibleCheckpointError, SoliaTransformer

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


def _client(loader=None, **kwargs) -> TestClient:
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
    client = TestClient(create_app(lambda current: champion), raise_server_exceptions=False)

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
    loader = registry_loader(_counting_factory(engines))

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
    loader = registry_loader(object)
    first = loader(None)

    records["current"] = _record(path, version="champ-2")
    second = loader(first)

    assert second is not first
    assert second.version == "champ-2"


def test_registry_loader_without_a_champion_raises_lookup_error(monkeypatch):
    engines = []
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: None)
    loader = registry_loader(_counting_factory(engines))

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
