"""Inference: the output contract, checkpoint loading, and ONNX export + parity."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from test_train import small_config

import solia_ai.db
import solia_ai.inference
import solia_ai.registry
from solia_ai.dataset import pad_window
from solia_ai.features import FEATURE_COLUMNS
from solia_ai.inference import (
    OUTPUT_NAMES,
    HorizonPrediction,
    Prediction,
    Predictor,
    build_model,
    derive_confidence,
    export_onnx,
    load_checkpoint,
    main,
    onnx_parity,
)
from solia_ai.model import SoliaTransformer
from solia_ai.train import build_checkpoint, save_checkpoint

N_FEATURES = len(FEATURE_COLUMNS)
HORIZONS = ("30s", "1m", "5m")


def tiny(seq_len: int = 6, horizons=HORIZONS) -> SoliaTransformer:
    model = SoliaTransformer(
        n_features=N_FEATURES,
        horizons=horizons,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,  # deterministic: no dropout noise between runs
        seq_len=seq_len,
    )
    model.fit_normalizer(np.random.default_rng(0).normal(size=(64, N_FEATURES)).astype("f4"))
    return model.eval()


def saved_checkpoint(tmp_path, *, version: str | None = "solia-test"):
    """Write a real checkpoint and return ``(model, path)``."""
    cfg = small_config()
    model = SoliaTransformer.from_config(cfg)
    model.fit_normalizer(np.random.default_rng(1).normal(size=(64, N_FEATURES)).astype("f4"))
    checkpoint = build_checkpoint(model, cfg, version=version or "nameless", metrics={})
    if version is None:
        del checkpoint["version"]
    path = tmp_path / "model.pt"
    save_checkpoint(checkpoint, path)
    return model.eval(), path


def window(rows: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, N_FEATURES)).astype("f4")


# --------------------------------------------------------------------------- confidence


def test_confidence_is_zero_at_a_coin_flip_and_max_when_decisive():
    assert derive_confidence(0.5, 0.5) == pytest.approx(0.0)
    assert derive_confidence(1.0, 0.0) == pytest.approx(100.0)
    assert derive_confidence(0.0, 1.0) == pytest.approx(100.0)


def test_confidence_is_monotonic_with_the_probability_margin():
    probs = np.array([0.5, 0.6, 0.7, 0.9, 1.0])
    confidence = derive_confidence(probs, np.full_like(probs, 0.5))

    assert np.all(np.diff(confidence) > 0)
    assert np.all((confidence >= 0) & (confidence <= 100))


def test_confidence_takes_the_more_decisive_of_pump_and_rug():
    assert derive_confidence(0.5, 0.9) == pytest.approx(derive_confidence(0.9, 0.5))
    assert derive_confidence(0.8, 0.2) == pytest.approx(60.0)  # both margins 0.3 -> 200*0.3


def test_confidence_preserves_array_shape():
    out = derive_confidence(np.full((2, 3), 0.75), np.full((2, 3), 0.5))

    assert out.shape == (2, 3)
    assert np.allclose(out, 50.0)  # 200 * |0.75 - 0.5|


def test_temperature_softens_or_sharpens_confidence():
    base = derive_confidence(0.9, 0.5)

    assert derive_confidence(0.9, 0.5, temperature=2.0) < base  # softer -> less confident
    assert derive_confidence(0.9, 0.5, temperature=0.5) > base  # sharper -> more confident
    assert derive_confidence(0.9, 0.5, temperature=1.0) == pytest.approx(base)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        derive_confidence(0.9, 0.1, temperature=0.0)


# --------------------------------------------------------------------------- checkpoints


def test_build_model_reproduces_the_checkpointed_outputs(tmp_path):
    model, path = saved_checkpoint(tmp_path)
    x = torch.randn(2, 6, N_FEATURES)

    restored = build_model(load_checkpoint(path))

    assert not restored.training  # eval mode, ready to serve
    assert torch.equal(restored(x)["pump_logits"], model(x)["pump_logits"])


def test_load_checkpoint_rejects_a_non_checkpoint(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"something": "else"}, path)

    with pytest.raises(ValueError, match="not a Solia checkpoint"):
        load_checkpoint(path)


class _Payload:
    """Stands in for an arbitrary object planted in a checkpoint file."""


def test_load_checkpoint_refuses_arbitrary_pickled_objects(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"model_state": {}, "payload": _Payload()}, path)

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)


def test_build_model_refuses_a_checkpoint_from_another_horizon_set(tmp_path):
    _, path = saved_checkpoint(tmp_path)
    checkpoint = load_checkpoint(path)
    checkpoint["hyperparams"]["horizons"] = ["1m", "30s"]  # reordered vs the trained weights

    with pytest.raises(ValueError, match="horizons"):
        build_model(checkpoint)


# --------------------------------------------------------------------------- Predictor


def test_predict_single_window_returns_the_full_contract():
    predictor = Predictor(tiny(), model_version="v-test")

    result = predictor.predict(window(6))

    assert isinstance(result, Prediction)
    assert result.model_version == "v-test"
    assert [p.horizon for p in result.predictions] == list(HORIZONS)
    for item in result.predictions:
        assert 0.0 <= item.pump_prob <= 1.0
        assert 0.0 <= item.rug_prob <= 1.0
        assert 0.0 <= item.confidence <= 100.0
        assert np.isfinite(item.expected_return)


def test_predict_matches_the_model_directly():
    model = tiny()
    predictor = Predictor(model, model_version="v")
    rows = window(4, seed=3)

    result = predictor.predict(rows)

    x, mask = pad_window(rows, model.seq_len)
    out = model.predict(torch.from_numpy(x)[None], torch.from_numpy(mask)[None])
    assert result.predictions[0].pump_prob == pytest.approx(float(out["pump_prob"][0, 0]))
    assert result.predictions[2].expected_return == pytest.approx(
        float(out["expected_return"][0, 2])
    )


def test_predict_a_batch_returns_one_prediction_per_window():
    predictor = Predictor(tiny(), model_version="v")

    results = predictor.predict([window(6, 1), window(3, 2), window(6, 3)])

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(len(r.predictions) == len(HORIZONS) for r in results)


def test_predict_restricts_and_orders_the_requested_horizons():
    predictor = Predictor(tiny(), model_version="v")
    rows = window(6, seed=5)

    full = predictor.predict(rows)
    subset = predictor.predict(rows, horizons=["5m", "30s"])

    assert [p.horizon for p in subset.predictions] == ["5m", "30s"]
    by_horizon = {p.horizon: p.pump_prob for p in full.predictions}
    assert subset.predictions[0].pump_prob == pytest.approx(by_horizon["5m"])
    assert subset.predictions[1].pump_prob == pytest.approx(by_horizon["30s"])


def test_predict_rejects_an_unknown_horizon():
    predictor = Predictor(tiny(), model_version="v")

    with pytest.raises(ValueError, match="unknown horizon"):
        predictor.predict(window(6), horizons=["5m", "4h"])


def test_predict_rejects_a_bare_string_of_horizons():
    predictor = Predictor(tiny(), model_version="v")

    with pytest.raises(TypeError, match="sequence of names"):
        predictor.predict(window(6), horizons="5m")


def test_predict_rejects_duplicate_horizons():
    predictor = Predictor(tiny(), model_version="v")

    with pytest.raises(ValueError, match="duplicate horizon"):
        predictor.predict(window(6), horizons=["1m", "1m"])


def test_predict_rejects_an_empty_window():
    predictor = Predictor(tiny(), model_version="v")

    with pytest.raises(ValueError, match="non-empty"):
        predictor.predict(np.zeros((0, N_FEATURES), dtype="f4"))


def test_predict_accepts_a_short_window_via_padding():
    predictor = Predictor(tiny(seq_len=6), model_version="v")

    result = predictor.predict(window(1))  # a single snapshot, left-padded to seq_len

    assert len(result.predictions) == len(HORIZONS)


def test_predict_rejects_a_window_with_the_wrong_feature_count():
    predictor = Predictor(tiny(), model_version="v")

    with pytest.raises(ValueError, match=rf"\[L, {N_FEATURES}\]"):
        predictor.predict(np.zeros((6, N_FEATURES + 1), dtype="f4"))


def test_predict_on_an_empty_batch_returns_an_empty_list():
    assert Predictor(tiny(), model_version="v").predict([]) == []


def test_predictor_temperature_flows_into_confidence():
    model = tiny()
    rows = window(6, seed=7)

    sharp = Predictor(model, model_version="v", temperature=0.5).predict(rows)
    soft = Predictor(model, model_version="v", temperature=2.0).predict(rows)

    assert sharp.predictions[0].confidence >= soft.predictions[0].confidence


def test_predictor_serves_calibrated_probabilities_consistent_with_confidence():
    model = tiny()
    rows = window(6, seed=7)
    raw = Predictor(model, model_version="v").predict(rows)

    calibrated = Predictor(model, model_version="v", temperature=0.5).predict(rows)

    for before, after in zip(raw.predictions, calibrated.predictions, strict=True):
        # the served probabilities themselves are temperature-scaled...
        assert float(derive_confidence(after.pump_prob, 0.5)) == pytest.approx(
            float(derive_confidence(before.pump_prob, 0.5, temperature=0.5)), abs=1e-4
        )
        # ...and the confidence is exactly the documented formula over them
        assert after.confidence == pytest.approx(
            float(derive_confidence(after.pump_prob, after.rug_prob)), abs=1e-4
        )


def test_predictor_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        Predictor(tiny(), model_version="v", temperature=0.0)


def test_prediction_as_dict_matches_the_api_contract():
    item = HorizonPrediction(
        "1m", pump_prob=0.72, rug_prob=0.08, expected_return=0.35, confidence=81.5
    )
    payload = Prediction("v3", [item]).as_dict()

    assert payload == {
        "model_version": "v3",
        "predictions": [
            {
                "horizon": "1m",
                "pump_prob": 0.72,
                "rug_prob": 0.08,
                "expected_return": 0.35,
                "confidence": 81.5,
            }
        ],
    }


# --------------------------------------------------------------------------- from_checkpoint


def test_from_checkpoint_uses_the_checkpoints_version(tmp_path):
    _, path = saved_checkpoint(tmp_path, version="solia-42")

    predictor = Predictor.from_checkpoint(path)

    assert predictor.model_version == "solia-42"
    assert predictor.horizons == tuple(small_config()["horizons"])


def test_from_checkpoint_falls_back_to_the_filename_when_unversioned(tmp_path):
    _, path = saved_checkpoint(tmp_path, version=None)

    assert Predictor.from_checkpoint(path).model_version == path.stem


def test_from_champion_loads_the_registered_checkpoint(tmp_path, monkeypatch):
    _, path = saved_checkpoint(tmp_path, version="champ-1")
    record = SimpleNamespace(version="champ-1", path=str(path))
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: record)

    predictor = Predictor.from_champion(engine=object())

    assert predictor.model_version == "champ-1"


def test_from_champion_without_a_champion_raises(monkeypatch):
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: None)

    with pytest.raises(LookupError, match="no champion"):
        Predictor.from_champion(engine=object())


# --------------------------------------------------------------------------- ONNX export


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    model = tiny()
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx(model, path, model_version="v-onnx", verify=True)
    return model, path


def test_export_writes_the_contract_graph_with_version_metadata(exported):
    _, path = exported

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert [o.name for o in graph.graph.output] == list(OUTPUT_NAMES)
    metadata = _metadata(path)
    assert metadata["model_version"] == "v-onnx"
    assert float(metadata["temperature"]) == 1.0


def test_onnx_matches_pytorch_within_tolerance(exported):
    model, path = exported

    assert onnx_parity(model, path) < 1e-4


@pytest.mark.parametrize("batch", [1, 3])
def test_onnx_graph_runs_with_a_dynamic_batch(exported, batch):
    model, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(batch).normal(size=(batch, model.seq_len, N_FEATURES)).astype("f4")
    mask = np.zeros((batch, model.seq_len), dtype=bool)

    pump, rug, ret, conf = session.run(list(OUTPUT_NAMES), {"features": x, "padding_mask": mask})

    assert pump.shape == rug.shape == ret.shape == conf.shape == (batch, len(HORIZONS))
    assert np.all((pump >= 0) & (pump <= 1))
    assert np.all((conf >= 0) & (conf <= 100))


def test_export_removes_the_file_when_parity_fails(tmp_path):
    path = tmp_path / "bad.onnx"

    # tol=0 can never be met (float runtimes differ by ~1e-8), so verification must fail...
    with pytest.raises(ValueError, match="parity failed"):
        export_onnx(tiny(), path, verify=True, tol=0.0)

    assert not path.exists()  # ...and the unverified graph must not be left behind


def test_export_keeps_the_file_when_verification_cannot_run(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("onnxruntime is broken")  # not a parity verdict

    monkeypatch.setattr(solia_ai.inference, "onnx_parity", broken)
    path = tmp_path / "kept.onnx"

    with pytest.raises(RuntimeError, match="broken"):
        export_onnx(tiny(), path, verify=True)

    assert path.exists()


def _metadata(path):
    return {m.key: m.value for m in onnx.load(str(path)).metadata_props}


def _run_onnx(path, x, mask):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(list(OUTPUT_NAMES), {"features": x, "padding_mask": mask})
    return dict(zip(OUTPUT_NAMES, outputs, strict=True))


def _field(prediction, name):
    return [getattr(p, name) for p in prediction.predictions]


def test_onnx_scores_a_fully_padded_row_as_neutral_not_nan(exported):
    model, path = exported
    x = np.random.default_rng(9).normal(size=(2, model.seq_len, N_FEATURES)).astype("f4")
    mask = np.zeros((2, model.seq_len), dtype=bool)
    mask[0, :] = True  # row 0: a token with no snapshot yet

    out = _run_onnx(path, x, mask)

    assert all(np.all(np.isfinite(v)) for v in out.values())
    assert np.allclose(out["pump_prob"][0], 0.5)
    assert np.allclose(out["rug_prob"][0], 0.5)
    assert np.allclose(out["expected_return"][0], 0.0)
    assert np.allclose(out["confidence"][0], 0.0)
    # the valid row in the same batch is unaffected by its neighbour
    expected = Predictor(model, model_version="v").predict(x[1])
    assert out["pump_prob"][1] == pytest.approx(_field(expected, "pump_prob"), abs=1e-4)


def test_onnx_confidence_matches_the_predictor(exported):
    model, path = exported
    rows = window(model.seq_len, seed=11)

    out = _run_onnx(path, rows[None], np.zeros((1, model.seq_len), dtype=bool))

    expected = Predictor(model, model_version="v").predict(rows)
    assert out["confidence"][0] == pytest.approx(_field(expected, "confidence"), abs=1e-2)


def test_onnx_bakes_in_the_calibration_temperature(tmp_path):
    model = tiny()
    path = export_onnx(model, tmp_path / "calibrated.onnx", temperature=0.5)  # parity-checked
    rows = window(model.seq_len, seed=12)

    out = _run_onnx(path, rows[None], np.zeros((1, model.seq_len), dtype=bool))

    expected = Predictor(model, model_version="v", temperature=0.5).predict(rows)
    assert out["pump_prob"][0] == pytest.approx(_field(expected, "pump_prob"), abs=1e-4)
    assert out["confidence"][0] == pytest.approx(_field(expected, "confidence"), abs=1e-2)
    assert float(_metadata(path)["temperature"]) == 0.5


# --------------------------------------------------------------------------- CLI


def test_cli_exports_a_checkpoint_with_its_version(tmp_path):
    _, ckpt = saved_checkpoint(tmp_path, version="solia-7")
    out = tmp_path / "out.onnx"

    main(["--checkpoint", str(ckpt), "--out", str(out)])

    assert _metadata(out)["model_version"] == "solia-7"


def test_cli_champion_is_stamped_with_the_registry_version(tmp_path, monkeypatch):
    # The registry row's version wins over the one embedded in the file, exactly as /predict.
    _, ckpt = saved_checkpoint(tmp_path, version="embedded-name")
    record = SimpleNamespace(version="registry-name", path=str(ckpt))
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda: object())
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: record)
    out = tmp_path / "champ.onnx"

    main(["--champion", "--out", str(out), "--temperature", "0.8"])

    metadata = _metadata(out)
    assert metadata["model_version"] == "registry-name"
    assert float(metadata["temperature"]) == 0.8


def test_cli_without_a_champion_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda: object())
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: None)
    out = tmp_path / "none.onnx"

    with pytest.raises(SystemExit, match="no champion"):
        main(["--champion", "--out", str(out)])

    assert not out.exists()


def test_cli_no_verify_skips_the_parity_check(tmp_path, monkeypatch):
    _, ckpt = saved_checkpoint(tmp_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("parity check ran despite --no-verify")

    monkeypatch.setattr(solia_ai.inference, "onnx_parity", must_not_run)
    out = tmp_path / "unverified.onnx"

    main(["--checkpoint", str(ckpt), "--out", str(out), "--no-verify"])

    assert out.exists()


def test_cli_rejects_a_non_positive_temperature(tmp_path):
    _, ckpt = saved_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="temperature must be > 0"):
        main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "x.onnx"), "--temperature", "0"])


def test_cli_requires_a_source(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "x.onnx")])
