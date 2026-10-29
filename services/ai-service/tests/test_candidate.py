"""Candidate training: refreshed time split, warm start from the champion, never registered."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import torch
from test_train import make_features_df, small_config

import solia_ai.candidate as candidate_module
import solia_ai.registry
from solia_ai.candidate import (
    CandidateSettings,
    candidate_split,
    champion_data_end_ts,
    main,
    resolve_warm_start,
    train_candidate,
    warm_start_leak,
    warm_start_problems,
)
from solia_ai.config import CANDIDATE_DEFAULTS, DEFAULT_EVAL_FRACTION, ConfigError
from solia_ai.dataset import config_samples, embargo_seconds
from solia_ai.model import IncompatibleCheckpointError
from solia_ai.train import fit, train


def candidate_config(**candidate) -> dict:
    cfg = small_config()
    cfg["calibration"] = {"method": "platt", "min_samples": 5, "min_class_samples": 2}
    cfg["candidate"] = {"min_train_samples": 5, "warm_start_epochs": 2, **candidate}
    return cfg


@pytest.fixture(scope="module")
def df():
    return make_features_df()


@pytest.fixture(scope="module")
def champion(df, tmp_path_factory):
    """A champion checkpoint trained (unregistered) on the first half of the synthetic data, as
    the previous retrain would have: its data ends well before the candidate's held-out window."""
    early = df[df["ts"] < df["ts"].min() + pd.Timedelta(seconds=200)]
    result = train(
        early, small_config(), model_dir=tmp_path_factory.mktemp("champion"), register=False
    )
    return torch.load(result.checkpoint_path, weights_only=True)


@pytest.fixture(scope="module")
def late_champion(df, tmp_path_factory):
    """A champion trained on all of the data: its weights have seen the held-out window."""
    result = train(df, small_config(), model_dir=tmp_path_factory.mktemp("late"), register=False)
    return torch.load(result.checkpoint_path, weights_only=True)


# --- settings ---


def test_settings_default_every_omitted_knob():
    cfg = small_config()

    assert CandidateSettings.from_config(cfg) == CandidateSettings(**CANDIDATE_DEFAULTS)
    cfg["candidate"] = {"warm_start": False}
    assert CandidateSettings.from_config(cfg).warm_start is False
    cfg["candidate"] = {"on_incompatible": "ignore"}
    with pytest.raises(ConfigError, match="candidate.on_incompatible"):
        CandidateSettings.from_config(cfg)
    for factor in (0, 1.5):
        cfg["candidate"] = {"warm_start_lr_factor": factor}
        with pytest.raises(ConfigError, match="warm_start_lr_factor"):
            CandidateSettings.from_config(cfg)


def test_heldout_fraction_defaults_to_the_walk_forward_eval_share():
    from solia_ai import ablation
    from solia_ai.config import load_config

    assert ablation.DEFAULT_EVAL_FRACTION is DEFAULT_EVAL_FRACTION
    assert CANDIDATE_DEFAULTS["heldout_fraction"] == DEFAULT_EVAL_FRACTION
    # The shipped config doesn't pin its own copy, so the two shares can't drift apart.
    assert "heldout_fraction" not in load_config()["candidate"]
    assert CandidateSettings.from_config(load_config()).heldout_fraction == DEFAULT_EVAL_FRACTION


# --- time split ---


def test_split_never_trains_on_or_labels_into_the_heldout_window(df):
    samples = config_samples(df, small_config())
    embargo = embargo_seconds(samples.horizons)

    split = candidate_split(samples, heldout_fraction=0.3, min_train_samples=5)

    ends = samples.end_ts
    train_ends, val_ends = ends[split.train.indices], ends[split.val.indices]
    heldout_ends = ends[split.heldout.indices]
    assert train_ends.max() < split.heldout_start_ts < heldout_ends.min()
    assert max(train_ends.max(), val_ends.max()) + embargo <= split.heldout_start_ts
    assert train_ends.max() + embargo <= val_ends.min()  # temporal validation, embargoed too
    assert heldout_ends.max() == ends.max()  # the held-out window is the most recent one
    counts = split.counts()
    assert (
        counts["n_samples"]
        == counts["n_train"] + counts["n_val"]
        == len(train_ends) + len(val_ends)
    )
    assert counts["n_heldout"] == len(heldout_ends)


def test_split_matches_the_walk_forward_cut(df):
    from solia_ai.ablation import walk_forward_windows

    samples = config_samples(df, small_config())
    embargo = embargo_seconds(samples.horizons)

    split = candidate_split(samples, heldout_fraction=0.3, min_train_samples=5)
    first = walk_forward_windows(samples.end_ts, n_windows=1, eval_fraction=0.3, embargo=embargo)[0]

    assert split.heldout_start_ts == first.eval_start_ts
    assert set(split.heldout.indices) == set(first.eval_idx)


def test_split_honours_an_explicit_heldout_start(df):
    samples = config_samples(df, small_config())
    start = int(np.quantile(samples.end_ts, 0.5))

    split = candidate_split(samples, min_train_samples=5, heldout_start_ts=start)

    assert split.heldout_start_ts == start
    assert samples.end_ts[split.heldout.indices].min() > start
    assert samples.end_ts[split.train.indices].max() <= start - embargo_seconds(samples.horizons)


def test_too_few_samples_is_a_clear_error(df):
    samples = config_samples(df, small_config())

    with pytest.raises(ValueError, match="too few samples to train a candidate"):
        candidate_split(samples, min_train_samples=len(samples))


def test_split_refuses_empty_data_and_an_empty_heldout_window(df):
    samples = config_samples(df, small_config())
    empty = config_samples(df.iloc[:0], small_config())

    with pytest.raises(ValueError, match="no labeled window"):
        candidate_split(empty)
    with pytest.raises(ValueError, match="no labeled window ends after"):
        candidate_split(samples, heldout_start_ts=int(samples.end_ts.max()))
    with pytest.raises(ValueError, match="heldout_fraction"):
        candidate_split(samples, heldout_fraction=1.0)


# --- warm start compatibility ---


def test_a_matching_champion_is_compatible(champion):
    assert warm_start_problems(champion, small_config()) == []


def _mutated(champion, **changes):
    out = dict(champion)
    out.update(changes)
    return out


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda c: c["model"].update(d_model=32, ff_dim=64), "model.d_model is 16"),
        (lambda c: c["model"].update(n_layers=2), "model.n_layers is 1"),
        (lambda c: c.update(horizons=["30s", "5m"]), "horizons"),
        (lambda c: c.update(features={"groups": ["price"]}), "enabled feature groups differ"),
    ],
)
def test_a_config_change_makes_the_champion_incompatible(champion, change, problem):
    cfg = small_config()
    change(cfg)

    problems = warm_start_problems(champion, cfg)

    assert any(problem in p for p in problems), problems


def test_a_feature_spec_change_makes_the_champion_incompatible(champion):
    narrower = _mutated(champion, feature_columns=list(champion["feature_columns"])[:-1])
    unrecorded = _mutated(champion, feature_columns=None)

    assert any("n_features" in p for p in warm_start_problems(narrower, small_config()))
    assert any("does not record" in p for p in warm_start_problems(unrecorded, small_config()))


def test_incompatible_champion_falls_back_to_scratch_logged_and_recorded(champion, caplog):
    cfg = small_config()
    cfg["model"]["n_layers"] = 2
    settings = CandidateSettings.from_config(candidate_config())

    with caplog.at_level(logging.WARNING, logger="solia_ai.candidate"):
        decision, state = resolve_warm_start(champion, cfg, settings)

    assert state is None
    assert decision.requested and not decision.used
    assert decision.champion_version == champion["version"]
    assert any("n_layers" in r for r in decision.reasons)
    assert "falls back to scratch" in caplog.text


def test_incompatible_champion_raises_when_configured_to(champion):
    cfg = small_config()
    cfg["model"]["n_layers"] = 2
    settings = CandidateSettings.from_config(candidate_config(on_incompatible="error"))

    with pytest.raises(IncompatibleCheckpointError, match="n_layers"):
        resolve_warm_start(champion, cfg, settings)


def test_no_champion_or_warm_start_off_trains_from_scratch(champion):
    on = CandidateSettings.from_config(candidate_config())
    off = CandidateSettings.from_config(candidate_config(warm_start=False))

    no_champion, state = resolve_warm_start(None, small_config(), on)
    assert state is None and not no_champion.used and no_champion.reasons
    disabled, state = resolve_warm_start(champion, small_config(), off)
    assert state is None and not disabled.requested and not disabled.reasons


# --- training ---


def test_fit_from_a_champion_state_starts_from_its_weights(df, champion):
    cfg = small_config()
    cfg["training"] = {**cfg["training"], "learning_rate": 1e-12, "epochs": 1, "warmup_steps": 1}
    split = candidate_split(config_samples(df, cfg), min_train_samples=5)
    state = champion["model_state"]

    warm = fit(split.train, split.val, cfg, init_state=state).model.state_dict()
    scratch = fit(split.train, split.val, cfg).model.state_dict()

    normalizer = {"normalizer.mean", "normalizer.std"}
    tensors = [k for k, v in state.items() if torch.is_tensor(v) and k not in normalizer]
    assert all(torch.allclose(warm[k].float(), state[k].float(), atol=1e-6) for k in tensors)
    assert not torch.allclose(scratch["input_proj.weight"], state["input_proj.weight"])
    # The normalizer is refit on the refreshed rows, exactly as a scratch candidate's.
    for key in normalizer:
        assert torch.equal(warm[key], scratch[key])
    assert not torch.equal(warm["normalizer.mean"], state["normalizer.mean"])


def test_warm_start_candidate_is_seeded_from_the_champion(df, champion, tmp_path, monkeypatch):
    seen = {}

    def spy(train_ds, val_ds, config, **kwargs):
        seen["init_state"] = kwargs.get("init_state")
        seen["epochs"] = config["training"]["epochs"]
        seen["learning_rate"] = config["training"]["learning_rate"]
        return fit(train_ds, val_ds, config, **kwargs)

    monkeypatch.setattr(candidate_module, "fit", spy)
    cfg = candidate_config(warm_start_lr_factor=0.25)

    result = train_candidate(df, cfg, champion=champion, model_dir=tmp_path)

    assert result.warm_start.used and result.warm_start.champion_version == champion["version"]
    assert seen["epochs"] == 2  # the warm-start budget, not training.epochs
    # The already-trained weights are fine-tuned at a fraction of the peak learning rate.
    assert seen["learning_rate"] == pytest.approx(cfg["training"]["learning_rate"] * 0.25)
    # The configured hyperparams are recorded; the warm start's effective ones sit beside them.
    assert result.hyperparams["training"] == cfg["training"]
    assert result.hyperparams["warm_start"] == {
        "champion_version": champion["version"],
        "epochs": 2,
        "learning_rate": pytest.approx(seen["learning_rate"]),
    }
    saved = torch.load(result.checkpoint_path, weights_only=True)
    assert saved["hyperparams"]["training"]["epochs"] == cfg["training"]["epochs"]
    assert saved["hyperparams"]["warm_start"]["epochs"] == 2
    init = seen["init_state"]
    assert all(
        torch.equal(init[k], v) for k, v in champion["model_state"].items() if torch.is_tensor(v)
    )


def test_candidate_is_saved_with_its_split_but_never_registered(
    df, champion, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("a candidate must not be registered")

    monkeypatch.setattr(solia_ai.registry, "register_model", forbidden)

    result = train_candidate(df, candidate_config(), champion=champion, model_dir=tmp_path)

    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert result.checkpoint_path.parent == tmp_path.resolve()
    assert not hasattr(result, "model_id")
    assert checkpoint["candidate"]["warm_start"]["used"] is True
    assert checkpoint["candidate"]["n_samples"] == result.n_samples > 0
    assert checkpoint["candidate"]["heldout_start_ts"] == result.heldout_start_ts
    assert result.train_end_ts < result.heldout_start_ts
    # Calibrated on its own validation split, before the held-out window.
    assert checkpoint["calibration"] is not None
    assert result.train_end_ts < checkpoint["calibration_end_ts"] < result.heldout_start_ts
    # data_end_ts covers the whole validation split (early stopping included), embargo before
    # the held-out window.
    embargo = embargo_seconds(candidate_config()["horizons"])
    assert checkpoint["data_end_ts"] == result.data_end_ts
    assert checkpoint["calibration_end_ts"] <= result.data_end_ts
    assert result.data_end_ts + embargo <= result.heldout_start_ts
    assert {"n_params", "metrics", "hyperparams"} <= set(vars(result)) and result.n_params > 0


def test_scratch_candidate_is_the_plain_training_loop_on_the_split(df, tmp_path):
    cfg = candidate_config(warm_start=False)
    split = candidate_split(config_samples(df, cfg), min_train_samples=5)

    result = train_candidate(df, cfg, model_dir=tmp_path)

    plain = fit(split.train, split.val, cfg)
    assert not result.warm_start.used
    assert result.metrics == plain.metrics and result.best_epoch == plain.best_epoch
    assert result.counts == split.counts()


def test_candidate_is_reproducible_for_a_seed(df, champion, tmp_path):
    first = train_candidate(df, candidate_config(), champion=champion, model_dir=tmp_path / "a")
    second = train_candidate(df, candidate_config(), champion=champion, model_dir=tmp_path / "b")

    assert first.metrics == second.metrics


def test_incompatible_champion_trains_a_scratch_candidate(df, champion, tmp_path):
    cfg = candidate_config()
    cfg["model"]["n_layers"] = 2

    result = train_candidate(df, cfg, champion=champion, model_dir=tmp_path)

    assert not result.warm_start.used and result.warm_start.reasons
    assert result.hyperparams["training"]["epochs"] == cfg["training"]["epochs"]
    assert "warm_start" not in result.hyperparams


def test_too_few_samples_fails_before_training(df, tmp_path, monkeypatch):
    monkeypatch.setattr(candidate_module, "fit", lambda *a, **k: pytest.fail("trained anyway"))

    with pytest.raises(ValueError, match="too few samples"):
        train_candidate(df, candidate_config(min_train_samples=10**6), model_dir=tmp_path)
    assert not any(tmp_path.iterdir())


# --- CLI ---


def test_main_reads_the_champion_and_honours_scratch(df, champion, tmp_path, monkeypatch, capsys):
    import solia_ai.db
    import solia_ai.inference

    record = type("Record", (), {"path": "champion.pt"})()
    calls = {}
    monkeypatch.setattr(candidate_module, "load_config", lambda: candidate_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda: "engine")
    monkeypatch.setattr(solia_ai.db, "read_features", lambda engine, limit=None: df)
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: record)
    monkeypatch.setattr(solia_ai.inference, "load_checkpoint", lambda path: champion)

    def spy(frame, cfg, **kwargs):
        calls["warm_start"] = cfg["candidate"].get("warm_start", True)
        calls["champion"] = kwargs["champion"]
        return train_candidate(frame, cfg, **kwargs)

    monkeypatch.setattr(candidate_module, "train_candidate", spy)

    main(["--model-dir", str(tmp_path / "warm"), "--device", "cpu"])
    assert calls == {"warm_start": True, "champion": champion}
    assert "warm start from" in capsys.readouterr().out

    main(["--scratch", "--model-dir", str(tmp_path / "scratch"), "--device", "cpu"])
    assert calls == {"warm_start": False, "champion": None}
    assert "(scratch)" in capsys.readouterr().out


def test_main_falls_back_to_scratch_when_the_champion_cannot_be_loaded(
    df, tmp_path, monkeypatch, caplog
):
    import solia_ai.db
    import solia_ai.inference

    record = type("Record", (), {"path": "gone.pt"})()
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda: "engine")
    monkeypatch.setattr(solia_ai.db, "read_features", lambda engine, limit=None: df)
    monkeypatch.setattr(solia_ai.registry, "read_champion", lambda engine: record)

    def missing(path):
        raise FileNotFoundError(f"no checkpoint at {path}")

    monkeypatch.setattr(solia_ai.inference, "load_checkpoint", missing)

    monkeypatch.setattr(candidate_module, "load_config", lambda: candidate_config())
    with caplog.at_level(logging.WARNING, logger="solia_ai.candidate"):
        main(["--model-dir", str(tmp_path / "scratch"), "--device", "cpu"])
    assert "gone.pt could not be loaded" in caplog.text
    assert any((tmp_path / "scratch").iterdir())  # a scratch candidate was still trained

    monkeypatch.setattr(
        candidate_module, "load_config", lambda: candidate_config(on_incompatible="error")
    )
    with pytest.raises(IncompatibleCheckpointError, match="could not be loaded"):
        main(["--model-dir", str(tmp_path / "error"), "--device", "cpu"])


# --- held-out leakage through the champion ---


def test_champion_data_end_covers_validation_and_legacy_checkpoints(champion):
    assert champion["data_end_ts"] > champion["train_end_ts"]  # the validation split counts
    assert champion_data_end_ts(champion) == champion["data_end_ts"]

    legacy = _mutated(champion, data_end_ts=None, train_end_ts=100, calibration_end_ts=250)
    assert champion_data_end_ts(legacy) == 250
    assert champion_data_end_ts(_mutated(legacy, calibration_end_ts=None)) == 100
    unrecorded = _mutated(legacy, train_end_ts=None, calibration_end_ts=None)
    assert champion_data_end_ts(unrecorded) is None


def test_a_champion_whose_data_reaches_the_heldout_window_leaks(champion):
    end = champion["data_end_ts"]

    assert warm_start_leak(champion, end + 60, 60) is None  # ends exactly at start - embargo
    assert "leak held-out windows" in warm_start_leak(champion, end + 59, 60)
    unrecorded = _mutated(champion, data_end_ts=None, train_end_ts=None, calibration_end_ts=None)
    assert "does not record" in warm_start_leak(unrecorded, end + 10**6, 60)


def test_a_leaking_champion_falls_back_to_scratch_or_raises(champion, caplog):
    start = champion["data_end_ts"]  # the held-out window starts inside the champion's data
    scratch = CandidateSettings.from_config(candidate_config())
    error = CandidateSettings.from_config(candidate_config(on_incompatible="error"))

    with caplog.at_level(logging.WARNING, logger="solia_ai.candidate"):
        decision, state = resolve_warm_start(
            champion, small_config(), scratch, heldout_start_ts=start, embargo=60
        )
    assert state is None and decision.requested and not decision.used
    assert any("leak" in r for r in decision.reasons) and "falls back" in caplog.text
    with pytest.raises(IncompatibleCheckpointError, match="leak"):
        resolve_warm_start(champion, small_config(), error, heldout_start_ts=start, embargo=60)


def test_a_champion_trained_on_the_heldout_window_never_seeds_the_candidate(
    df, late_champion, tmp_path, monkeypatch
):
    seen = {}

    def spy(train_ds, val_ds, config, **kwargs):
        seen["init_state"] = kwargs.get("init_state")
        return fit(train_ds, val_ds, config, **kwargs)

    monkeypatch.setattr(candidate_module, "fit", spy)

    result = train_candidate(df, candidate_config(), champion=late_champion, model_dir=tmp_path)

    assert seen["init_state"] is None
    assert not result.warm_start.used
    assert any("leak" in r for r in result.warm_start.reasons)


def test_an_unloadable_champion_falls_back_to_scratch_or_raises(caplog):
    why = "champion checkpoint x.pt could not be loaded: corrupt"
    scratch = CandidateSettings.from_config(candidate_config())
    error = CandidateSettings.from_config(candidate_config(on_incompatible="error"))

    with caplog.at_level(logging.WARNING, logger="solia_ai.candidate"):
        decision, state = resolve_warm_start(None, small_config(), scratch, unavailable=why)
    assert state is None and decision.requested and not decision.used
    assert decision.reasons == (why,) and why in caplog.text
    with pytest.raises(IncompatibleCheckpointError, match="corrupt"):
        resolve_warm_start(None, small_config(), error, unavailable=why)


def test_evaluation_never_scores_the_candidates_validation_windows(df, tmp_path):
    from solia_ai.evaluate import holdout_dataset
    from solia_ai.inference import Predictor

    cfg = candidate_config(warm_start=False)
    cfg["calibration"] = {"method": "none"}  # no calibration_end_ts to lean on
    result = train_candidate(df, cfg, model_dir=tmp_path)
    predictor = Predictor.from_checkpoint(result.checkpoint_path)

    held_out = holdout_dataset(df, cfg, predictor)

    ends = held_out.samples.end_ts[held_out.indices]
    assert ends.min() > result.data_end_ts + embargo_seconds(cfg["horizons"])
