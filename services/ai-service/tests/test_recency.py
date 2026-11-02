"""#203: optional recency weighting of the training loss (``training.recency_half_life_hours``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from test_train import make_features_df, small_config

import solia_ai.search as search
import solia_ai.train as train
from solia_ai.config import ConfigError, load_config, validate_config
from solia_ai.dataset import SequenceDataset, config_samples, recency_weights, temporal_split
from solia_ai.loss import SoliaLoss, focal_bce_loss

HOUR = 3600


def targets(n: int = 4, h: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "pump": (torch.rand(n, h, generator=g) < 0.5).float(),
        "rug": (torch.rand(n, h, generator=g) < 0.5).float(),
        "expected_return": torch.randn(n, h, generator=g),
        "mask": (torch.rand(n, h, generator=g) < 0.8).float(),
    }


def outputs(n: int = 4, h: int = 2, seed: int = 1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "pump_logits": torch.randn(n, h, generator=g),
        "rug_logits": torch.randn(n, h, generator=g),
        "expected_return": torch.randn(n, h, generator=g),
    }


def loss_fn(**kwargs) -> SoliaLoss:
    return SoliaLoss(("30s", "1m"), alpha=0.5, focal_gamma=2.0, **kwargs)


# --- the weights ---


def test_one_half_life_old_weighs_half_the_newest():
    w = recency_weights(np.array([0, 12 * HOUR, 24 * HOUR]), 12)
    np.testing.assert_allclose(w[1] / w[2], 0.5, rtol=1e-6)
    np.testing.assert_allclose(w[0] / w[2], 0.25, rtol=1e-6)


def test_weights_are_monotonic_in_age_and_average_one():
    end_ts = np.sort(np.random.default_rng(0).integers(0, 4 * 24 * HOUR, 500))
    w = recency_weights(end_ts, 24)
    assert w.dtype == np.float32
    assert np.all(np.diff(w) >= 0)  # older (earlier end) never weighs more
    np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-5)


def test_age_is_measured_to_the_given_reference():
    w = recency_weights(np.array([0, 6 * HOUR]), 6, reference_ts=12 * HOUR)
    np.testing.assert_allclose(w[0] / w[1], 0.5, rtol=1e-6)
    with pytest.raises(ValueError, match="after the recency reference"):
        recency_weights(np.array([0, 13 * HOUR]), 6, reference_ts=12 * HOUR)


def test_no_sample_gives_no_weight():
    assert recency_weights(np.array([], np.int64), 24).shape == (0,)


@pytest.mark.parametrize("half_life", [0, -1, float("nan"), float("inf"), True, "24"])
def test_a_bad_half_life_is_refused(half_life):
    with pytest.raises(ValueError, match="half_life_hours"):
        recency_weights(np.array([0, 1]), half_life)


# --- the dataset ---


def test_off_leaves_the_dataset_as_it_is():
    samples = config_samples(make_features_df(), small_config())
    ds = SequenceDataset(samples)
    assert ds.with_recency(None) is ds
    assert "weight" not in ds.batch([0, 1])[2] and "weight" not in ds[0][2]


def test_weights_follow_the_indices_in_items_and_batches():
    samples = config_samples(make_features_df(), small_config())
    indices = np.arange(len(samples))[::3]
    ds = SequenceDataset(samples, indices).with_recency(1)

    expected = recency_weights(samples.end_ts[indices], 1)
    np.testing.assert_array_equal(ds.sample_weight, expected)
    batch = ds.batch([2, 0, 5])[2]["weight"]
    np.testing.assert_array_equal(batch.numpy(), expected[[2, 0, 5]])
    assert float(ds[5][2]["weight"]) == pytest.approx(float(expected[5]))


@pytest.mark.parametrize(
    ("weights", "message"), [([1.0], "shape"), ([1.0, -1.0, 1.0], ">= 0"), ([1, np.nan, 1], ">= 0")]
)
def test_bad_sample_weights_are_refused(weights, message):
    samples = config_samples(make_features_df(), small_config())
    with pytest.raises(ValueError, match=message):
        SequenceDataset(samples, np.arange(3), np.array(weights, np.float32))


# --- the loss ---


def test_unit_weights_give_the_unweighted_loss():
    out, tgt = outputs(), targets()
    plain = loss_fn()(out, tgt)
    weighted = loss_fn()(out, {**tgt, "weight": torch.ones(4)})
    for key, value in plain.items():
        torch.testing.assert_close(weighted[key], value, rtol=0, atol=0)


def test_sample_weights_compose_with_pos_weight_focal_and_horizon_weights():
    out, tgt = outputs(), targets()
    pos = {"pump": torch.tensor([2.0, 3.0]), "rug": torch.tensor([1.5, 0.5])}
    fn = loss_fn(pos_weight=pos, horizon_weights=[1.0, 2.0])
    sample = torch.tensor([0.25, 1.0, 2.0, 0.75])

    got = fn(out, {**tgt, "weight": sample})

    weight = tgt["mask"] * torch.tensor([1.0, 2.0]) * sample[:, None]  # multiplied, not replaced
    per = focal_bce_loss(out["pump_logits"], tgt["pump"], alpha=0.5, gamma=2.0,
                         pos_weight=pos["pump"])  # fmt: skip
    torch.testing.assert_close(got["pump"], (per * weight).sum() / weight.sum())


def test_a_heavier_sample_pulls_the_loss_towards_its_own():
    out, tgt = outputs(n=2), targets(n=2)
    tgt["mask"] = torch.ones(2, 2)
    fn = loss_fn()

    def alone(i):
        return fn({k: v[i : i + 1] for k, v in out.items()},
                  {k: v[i : i + 1] for k, v in tgt.items()})["total"]  # fmt: skip

    per_sample = [alone(0), alone(1)]
    heavy_first = fn(out, {**tgt, "weight": torch.tensor([9.0, 1.0])})["total"]
    torch.testing.assert_close(heavy_first, 0.9 * per_sample[0] + 0.1 * per_sample[1])


# --- training ---


def split(cfg):
    samples = config_samples(make_features_df(), cfg)
    train_idx, val_idx = temporal_split(samples.end_ts, val_fraction=0.25, embargo_seconds=60)
    return SequenceDataset(samples, train_idx), SequenceDataset(samples, val_idx)


def with_half_life(hours):
    cfg = small_config()
    cfg["training"] = {**cfg["training"], "recency_half_life_hours": hours}
    return cfg


def test_only_the_training_loader_is_weighted(monkeypatch):
    seen = {}
    real = train.make_loaders

    def spy(train_ds, val_ds, **kwargs):
        seen["train"], seen["val"] = train_ds, val_ds
        return real(train_ds, val_ds, **kwargs)

    monkeypatch.setattr(train, "make_loaders", spy)
    train_ds, val_ds = split(small_config())
    train.fit(train_ds, val_ds, with_half_life(0.01), device="cpu")

    assert seen["train"].sample_weight is not None and seen["val"].sample_weight is None
    np.testing.assert_array_equal(
        seen["train"].sample_weight,
        recency_weights(train_ds.samples.end_ts[train_ds.indices], 0.01),
    )
    assert train_ds.sample_weight is None  # the caller's dataset is untouched


def test_validation_loss_stays_unweighted():
    cfg = with_half_life(0.01)
    train_ds, val_ds = split(cfg)
    result = train.fit(train_ds, val_ds, cfg, device="cpu")

    loader = train.make_loaders(train_ds, val_ds, batch_size=8, balanced=False)[1]
    fn = SoliaLoss.from_config(cfg, pos_weight=train.resolve_pos_weight(train_ds, cfg))
    unweighted = train._validate(result.model, fn, loader, torch.device("cpu"))
    assert result.metrics["val_total"] == pytest.approx(unweighted["total"], rel=1e-6)


def test_off_trains_on_the_unweighted_dataset_itself(monkeypatch):
    seen = {}
    real = train.make_loaders

    def spy(train_ds, val_ds, **kwargs):
        seen["train"] = train_ds
        return real(train_ds, val_ds, **kwargs)

    monkeypatch.setattr(train, "make_loaders", spy)
    train_ds, val_ds = split(small_config())
    train.fit(train_ds, val_ds, small_config(), device="cpu")
    assert seen["train"] is train_ds and train_ds.sample_weight is None


def test_a_short_half_life_changes_what_is_learned():
    train_ds, val_ds = split(small_config())
    a = train.fit(train_ds, val_ds, small_config(), device="cpu")
    b = train.fit(train_ds, val_ds, with_half_life(0.01), device="cpu")
    assert a.history[0]["train_loss"] != b.history[0]["train_loss"]


# --- config & search ---


@pytest.mark.parametrize("value", [0, -3, "24", True, float("inf")])
def test_config_rejects_a_bad_half_life_naming_the_key(value):
    with pytest.raises(ConfigError, match="training.recency_half_life_hours"):
        validate_config(with_half_life(value), "test")


def test_config_accepts_a_half_life_or_none_at_all():
    validate_config(with_half_life(24), "test")
    validate_config(small_config(), "test")
    assert "recency_half_life_hours" not in load_config()["training"]  # shipped: off


def test_the_half_life_is_searchable():
    cfg = search.apply_params(small_config(), {"recency_half_life_hours": 12})
    assert cfg["training"]["recency_half_life_hours"] == 12
    with pytest.raises(ValueError, match="must be > 0"):
        search.check_space({"recency_half_life_hours": {"choices": [0]}}, small_config())


def test_the_shipped_recency_space_is_valid():
    path = Path(search.__file__).parent / "sweeps" / "recency.yaml"
    space = search.check_space(search.load_space(path), load_config(), strategy="grid")
    assert space == {"recency_half_life_hours": {"choices": [12, 24, 48]}}
