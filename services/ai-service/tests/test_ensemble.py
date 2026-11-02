"""#204: seed ensemble champion (``model.ensemble_size``) — N members averaged, calibrated and
saved as one model; N = 1 is today's single model."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_train import make_features_df, small_config

import solia_ai.ablation as ablation
import solia_ai.search as search
import solia_ai.train as train_mod
from solia_ai.candidate import (
    CandidateSettings,
    resolve_warm_start,
    warm_start_problems,
    warm_start_state,
)
from solia_ai.config import ConfigError, validate_config
from solia_ai.dataset import SequenceDataset, config_samples, temporal_split
from solia_ai.evaluate import ReferencePolicy
from solia_ai.inference import Predictor, build_model, export_onnx, load_checkpoint, onnx_parity
from solia_ai.model import (
    IncompatibleCheckpointError,
    SoliaEnsemble,
    SoliaTransformer,
    ensemble_size,
    member_state,
)
from solia_ai.train import build_checkpoint, fit, fit_one, train


def sized(n: int | None) -> dict:
    cfg = small_config()
    if n is not None:
        cfg["model"] = {**cfg["model"], "ensemble_size": n}
    return cfg


def split(cfg):
    samples = config_samples(make_features_df(), cfg)
    train_idx, val_idx = temporal_split(samples.end_ts, val_fraction=0.25, embargo_seconds=60)
    return SequenceDataset(samples, train_idx), SequenceDataset(samples, val_idx)


def members(n: int = 3, seed: int = 0) -> list[SoliaTransformer]:
    out = []
    for k in range(n):
        torch.manual_seed(seed + k)
        out.append(SoliaTransformer.from_config(small_config()).eval())
    return out


def inputs(batch: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    model = members(1)[0]
    return torch.randn(batch, model.seq_len, model.n_features, generator=g), None


def assert_same_state(a, b):
    left, right = a.state_dict(), b.state_dict()
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right[key], rtol=0, atol=0, msg=key)


# --- the ensemble model ---


def test_the_members_are_averaged_in_logit_space():
    parts = members(3)
    ensemble = SoliaEnsemble(parts).eval()
    x, mask = inputs()

    out = ensemble(x, mask)
    each = [m(x, mask) for m in parts]
    for key in ("pump_logits", "rug_logits", "expected_return"):
        torch.testing.assert_close(out[key], torch.stack([o[key] for o in each]).mean(0))
    torch.testing.assert_close(ensemble.predict(x)["pump_prob"], torch.sigmoid(out["pump_logits"]))
    torch.testing.assert_close(ensemble.encode_logits(x)["rug_logits"], out["rug_logits"])


def test_the_ensemble_exposes_the_single_model_interface():
    parts = members(2)
    ensemble = SoliaEnsemble(parts)
    one = parts[0]
    assert ensemble.ensemble_size == 2 and one.ensemble_size == 1
    for name in ("horizons", "feature_columns", "seq_len", "n_features", "n_horizons"):
        assert getattr(ensemble, name) == getattr(one, name)
    assert ensemble.n_params == 2 * one.n_params
    assert torch.equal(ensemble.feature_mask, one.feature_mask)


def test_an_ensemble_needs_two_alike_members():
    with pytest.raises(ValueError, match="at least 2"):
        SoliaEnsemble(members(1))
    other = copy.deepcopy(small_config())
    other["horizons"] = ["30s"]
    other["labeling"]["pump_threshold"] = {"30s": 0.1}
    other["labeling"]["rug_threshold"] = {"30s": 0.3}
    with pytest.raises(ValueError, match="horizons"):
        SoliaEnsemble([members(1)[0], SoliaTransformer.from_config(other)])
    masked = SoliaTransformer.from_config({**small_config(), "features": {"groups": ["price"]}})
    with pytest.raises(ValueError, match="feature mask"):
        SoliaEnsemble([members(1)[0], masked])


def test_ensemble_size_defaults_to_one():
    assert ensemble_size(small_config()) == 1
    assert ensemble_size(sized(3)) == 3
    assert SoliaEnsemble.from_config(sized(3)).ensemble_size == 3


# --- training ---


def test_size_one_trains_exactly_as_today():
    train_ds, val_ds = split(small_config())
    today = fit_one(train_ds, val_ds, small_config(), device="cpu")
    one = fit(train_ds, val_ds, sized(1), device="cpu")

    assert isinstance(one.model, SoliaTransformer) and one.members == ()
    assert one.history == today.history and one.metrics == today.metrics
    assert_same_state(one.model, today.model)
    assert "ensemble_size" not in build_checkpoint(one.model, sized(1), version="v")


def test_members_are_trained_under_consecutive_seeds_and_averaged():
    cfg = sized(3)
    train_ds, val_ds = split(cfg)
    events = []

    result = fit(train_ds, val_ds, cfg, device="cpu", progress=events.append)

    assert isinstance(result.model, SoliaEnsemble) and len(result.members) == 3
    seed = cfg["training"]["seed"]
    for k, member in enumerate(result.model.members):
        alone = copy.deepcopy(small_config())
        alone["training"]["seed"] = seed + k
        assert_same_state(member, fit_one(train_ds, val_ds, alone, device="cpu").model)
    assert {(e["member"], e["members"]) for e in events} == {(1, 3), (2, 3), (3, 3)}
    assert "member 2/3 epoch" in train_mod.describe_epoch(events[-1] | {"member": 2})
    assert result.best_epoch == result.members[0].best_epoch


def test_the_ensemble_metrics_are_its_own_validation_losses():
    cfg = sized(2)
    train_ds, val_ds = split(cfg)
    result = fit(train_ds, val_ds, cfg, device="cpu")

    loader = train_mod.make_loaders(train_ds, val_ds, batch_size=8, balanced=False)[1]
    loss_fn = train_mod.SoliaLoss.from_config(
        cfg, pos_weight=train_mod.resolve_pos_weight(train_ds, cfg)
    )
    expected = train_mod._validate(result.model, loss_fn, loader, torch.device("cpu"))
    assert result.metrics == pytest.approx({f"val_{k}": v for k, v in expected.items()})


# --- calibration, artifact, serving ---


@pytest.fixture(scope="module")
def trained_ensemble(tmp_path_factory):
    directory = tmp_path_factory.mktemp("models")
    calls = []
    real = train_mod.fit_calibration_window

    def spy(model, val_ds, cfg):
        calls.append(model)
        return real(model, val_ds, cfg)

    train_mod.fit_calibration_window = spy
    try:
        result = train(make_features_df(), config=sized(2), model_dir=directory, register=False,
                       version="ens-2", device="cpu")  # fmt: skip
    finally:
        train_mod.fit_calibration_window = real
    return result, calls


def test_the_ensemble_is_calibrated_once_as_one_model(trained_ensemble):
    _, calls = trained_ensemble
    assert len(calls) == 1 and isinstance(calls[0], SoliaEnsemble)


def test_the_ensemble_is_saved_as_one_artifact_and_reloads(trained_ensemble):
    result, _ = trained_ensemble
    checkpoint = load_checkpoint(result.checkpoint_path)

    assert checkpoint["ensemble_size"] == 2
    assert checkpoint["hyperparams"]["model"]["ensemble_size"] == 2
    model = build_model(checkpoint)
    assert isinstance(model, SoliaEnsemble) and model.ensemble_size == 2

    predictor = Predictor.from_checkpoint(result.checkpoint_path)
    window = np.random.default_rng(0).normal(size=(model.seq_len, model.n_features))
    prediction = predictor.predict([window.astype(np.float32)])[0]
    assert prediction.model_version == "ens-2" and len(prediction.predictions) == 2


def test_a_single_model_champion_keeps_serving(tmp_path):
    result = train(make_features_df(), config=small_config(), model_dir=tmp_path, register=False,
                   version="single", device="cpu")  # fmt: skip
    checkpoint = load_checkpoint(result.checkpoint_path)
    assert "ensemble_size" not in checkpoint
    assert type(build_model(checkpoint)) is SoliaTransformer


def test_an_ensemble_state_loaded_as_one_model_fails_clearly():
    state = SoliaEnsemble(members(3)).state_dict()
    with pytest.raises(IncompatibleCheckpointError, match="3-member ensemble"):
        SoliaTransformer.from_config(small_config()).load_state_dict(state)


def test_an_ensemble_of_another_size_is_refused():
    state = SoliaEnsemble(members(3)).state_dict()
    with pytest.raises((IncompatibleCheckpointError, RuntimeError)):
        SoliaEnsemble(members(2)).load_state_dict(state)


def test_member_state_picks_one_member():
    parts = members(2)
    state = SoliaEnsemble(parts).state_dict()
    fresh = SoliaTransformer.from_config(small_config())
    fresh.load_state_dict(member_state(state, 1))
    assert_same_state(fresh, parts[1])
    with pytest.raises(IncompatibleCheckpointError, match="no ensemble member 5"):
        member_state(state, 5)


def test_the_onnx_export_holds_every_member(tmp_path):
    ensemble = SoliaEnsemble(members(2)).eval()
    path = tmp_path / "ensemble.onnx"
    export_onnx(ensemble, path, model_version="ens", verify=True)
    assert onnx_parity(ensemble, path) < 1e-4


# --- warm start ---


def champion_checkpoint(n: int, cfg=None):
    cfg = cfg or sized(n)
    model = SoliaEnsemble(members(n)) if n > 1 else members(1)[0]
    checkpoint = build_checkpoint(model, cfg, version=f"champ-{n}", data_end_ts=0)
    return checkpoint, model


def test_a_same_size_ensemble_champion_seeds_each_member():
    checkpoint, champion = champion_checkpoint(2)
    assert warm_start_problems(checkpoint, sized(2)) == []

    cfg = sized(2)
    cfg["training"]["epochs"] = 1
    train_ds, val_ds = split(cfg)
    result = fit(train_ds, val_ds, cfg, device="cpu", init_state=warm_start_state(checkpoint))
    assert isinstance(result.model, SoliaEnsemble)


@pytest.mark.parametrize(("champion_n", "config_n"), [(1, 3), (3, 1), (2, 3)])
def test_another_size_falls_back_to_scratch_or_errors(champion_n, config_n):
    checkpoint, _ = champion_checkpoint(champion_n)
    cfg = sized(config_n)
    problems = warm_start_problems(checkpoint, cfg)
    assert any("ensemble_size" in p for p in problems)

    scratch = CandidateSettings.from_config({**cfg, "candidate": {"on_incompatible": "scratch"}})
    decision, state = resolve_warm_start(checkpoint, cfg, scratch)
    assert not decision.used and state is None
    error = CandidateSettings.from_config({**cfg, "candidate": {"on_incompatible": "error"}})
    with pytest.raises(IncompatibleCheckpointError, match="ensemble_size"):
        resolve_warm_start(checkpoint, cfg, error)


# --- config, search, scoring ---


@pytest.mark.parametrize("value", [0, -1, 1.5, "3", True])
def test_config_rejects_a_bad_ensemble_size(value):
    with pytest.raises(ConfigError, match="model.ensemble_size"):
        validate_config(sized(value), "test")


def test_the_ensemble_size_is_searchable():
    assert search.apply_params(small_config(), {"ensemble_size": 3})["model"]["ensemble_size"] == 3
    with pytest.raises(ValueError, match=">= 1"):
        search.check_space({"ensemble_size": {"choices": [0]}}, small_config())


def test_the_evaluation_scores_the_ensemble_as_a_whole():
    cfg = sized(2)
    samples = config_samples(make_features_df(), cfg)
    idx = np.arange(len(samples))
    train_idx, val_idx = temporal_split(samples.end_ts, val_fraction=0.25, embargo_seconds=60)
    best_epoch, metrics = ablation.train_and_score(
        SequenceDataset(samples, train_idx), SequenceDataset(samples, val_idx),
        SequenceDataset(samples, idx[-20:]), cfg, seed=0,
        policy=ReferencePolicy(horizon="1m"), device="cpu",
    )  # fmt: skip
    assert best_epoch >= 0 and "pump/1m/pr_auc" in metrics
