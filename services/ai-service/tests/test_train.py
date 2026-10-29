"""The training loop end-to-end on a small in-memory dataset (no database)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import solia_ai.db
import solia_ai.registry
import solia_ai.train as train_module
from solia_ai.calibration import Calibration
from solia_ai.config import DATA_DEFAULTS
from solia_ai.dataset import SequenceDataset, make_datasets, positive_weights
from solia_ai.db import DataBounds, DataSettings
from solia_ai.features import BASELINE_GROUPS, FEATURE_COLUMNS
from solia_ai.loss import SoliaLoss
from solia_ai.model import SoliaTransformer
from solia_ai.train import (
    TrainResult,
    _cosine_warmup,
    _validate,
    calibration_split,
    default_version,
    fit_calibration_window,
    main,
    resolve_pos_weight,
    train,
    training_feature_rows,
)

N_FEATURES = len(FEATURE_COLUMNS)


# What a stubbed bounded read reports (solia_ai.db.read_training_features).
BOUNDS = DataBounds(
    lookback_days=2.0,
    max_rows=100,
    window_start=datetime(2026, 1, 8, tzinfo=UTC),
    window_end=datetime(2026, 1, 10, tzinfo=UTC),
    rows_in_window=3,
    tokens_in_window=2,
    tokens_partial=0,
    tokens_capped=0,
    tokens_kept=2,
    rows_read=2,
)


def small_config() -> dict:
    """A fast, valid config: tiny model, short horizons, a handful of epochs."""
    return {
        "model": {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "ff_dim": 32,
            "dropout": 0.0,
            "seq_len": 6,
            "min_seq_len": 2,
        },
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "epochs": 4,
            "batch_size": 8,
            "warmup_steps": 2,
            "grad_clip": 1.0,
            "early_stopping_patience": 3,
            "seed": 42,
        },
        "loss": {
            "alpha": 0.5,
            "focal_gamma": 2.0,
            "strategy_weight": 1.0,
            "regression_weight": 1.0,
        },
        "horizons": ["30s", "1m"],
        "labeling": {
            "max_gap_fraction": 0.5,
            "min_gap_seconds": 10,
            "pump_threshold": {"30s": 0.1, "1m": 0.2},
            "rug_threshold": {"30s": 0.3, "1m": 0.4},
        },
    }


def make_features_df(n_tokens: int = 6, n_rows: int = 80, step_s: int = 5, seed: int = 0):
    """A synthetic ``features`` frame spanning well beyond the 60s embargo, with injected pumps."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-04-20T12:00:00Z")
    frames = []
    for token in range(n_tokens):
        ts = base + pd.to_timedelta(np.arange(n_rows) * step_s, unit="s")
        returns = rng.normal(0.0, 0.05, n_rows)
        returns[n_rows // 3] += 0.8 if token % 2 == 0 else -0.5  # a pump or a dump mid-run
        price = (0.001 * np.cumprod(1.0 + returns)).clip(min=1e-6)
        columns = {col: rng.normal(0.0, 1.0, n_rows) for col in FEATURE_COLUMNS}
        columns["price"] = price
        columns["liquidity"] = (50.0 + rng.normal(0.0, 5.0, n_rows)).clip(min=1.0)
        columns["is_migrated"] = (rng.random(n_rows) < 0.1).astype(float)
        frame = pd.DataFrame(columns)
        frame["mint"] = f"M{token}"
        frame["ts"] = ts
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --- helpers ---


def test_default_version_is_sortable_and_timestamped():
    stamp = default_version(pd.Timestamp("2026-04-20T12:00:00Z").to_pydatetime())

    assert stamp == "solia-20260420T120000000000Z"


def test_default_versions_differ_within_the_same_second():
    first = default_version(pd.Timestamp("2026-04-20T12:00:00.000001Z").to_pydatetime())
    second = default_version(pd.Timestamp("2026-04-20T12:00:00.000002Z").to_pydatetime())

    assert first < second


def test_cosine_warmup_ramps_up_then_decays():
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = _cosine_warmup(optimizer, warmup_steps=3, total_steps=13)

    lrs = []
    for _ in range(13):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] < lrs[2] <= lrs[3]  # linear warmup for the first 3 steps
    assert lrs[3] > lrs[-1]  # then cosine decay
    assert lrs[-1] < 0.05  # decayed almost to zero by the final step


def test_cosine_warmup_never_outlasts_half_the_schedule():
    # warmup_steps sized for a long run, applied to a short (e.g. warm-start) budget.
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = _cosine_warmup(optimizer, warmup_steps=500, total_steps=10)

    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[4] == pytest.approx(1.0)  # peak reached after half the steps
    assert lrs[-1] < lrs[4]  # and the rate still decays


def test_training_feature_rows_covers_only_training_windows():
    df = make_features_df()
    cfg = small_config()
    from solia_ai.dataset import make_datasets

    train_ds, val_ds = make_datasets(df, cfg, val_fraction=0.3)
    rows = training_feature_rows(train_ds)

    assert rows.shape[1] == N_FEATURES
    assert 0 < len(rows) <= len(train_ds.samples.features)


# --- end-to-end ---


def test_train_runs_end_to_end_and_writes_a_checkpoint(tmp_path):
    result = train(
        make_features_df(),
        small_config(),
        model_dir=tmp_path,
        register=False,
        version="test-run",
    )

    assert result.checkpoint_path == tmp_path / "test-run.pt"
    assert result.checkpoint_path.exists()
    assert result.n_params > 0
    assert 0 <= result.best_epoch < len(result.history)
    assert np.isfinite(result.metrics["val_total"])
    assert result.history and all("val_total" in row for row in result.history)
    assert result.model_id is None  # no engine -> no registry row


def test_checkpoint_is_self_describing_and_reloads(tmp_path):
    result = train(
        make_features_df(), small_config(), model_dir=tmp_path, register=False, version="reload"
    )

    checkpoint = torch.load(result.checkpoint_path, weights_only=False)
    assert checkpoint["version"] == "reload"
    assert isinstance(checkpoint["train_end_ts"], int)  # evaluation's held-out boundary
    assert checkpoint["horizons"] == ["30s", "1m"]
    assert checkpoint["feature_columns"] == FEATURE_COLUMNS
    # The enabled groups travel with the checkpoint: candidates stay masked by default.
    assert checkpoint["hyperparams"]["features"] == {"groups": list(BASELINE_GROUPS)}

    shape = checkpoint["hyperparams"]["model"]
    model = SoliaTransformer(
        n_features=len(checkpoint["feature_columns"]),
        horizons=checkpoint["horizons"],
        d_model=shape["d_model"],
        n_heads=shape["n_heads"],
        n_layers=shape["n_layers"],
        ff_dim=shape["ff_dim"],
        dropout=shape["dropout"],
        seq_len=shape["seq_len"],
    )
    model.load_state_dict(checkpoint["model_state"])

    prediction = model.predict(torch.randn(2, shape["seq_len"], N_FEATURES))
    assert prediction["pump_prob"].shape == (2, 2)
    assert all(torch.isfinite(v).all() for v in prediction.values())


def test_early_stopping_caps_epochs_when_validation_stalls(tmp_path):
    cfg = small_config()
    cfg["training"]["epochs"] = 30
    cfg["training"]["early_stopping_patience"] = 1
    cfg["training"]["learning_rate"] = 0.0  # frozen weights -> val loss never improves

    result = train(make_features_df(), cfg, model_dir=tmp_path, register=False)

    assert len(result.history) < 30  # stopped early instead of running all epochs
    assert result.best_epoch == 0


def test_train_rejects_data_too_short_for_the_split(tmp_path):
    short = make_features_df(n_tokens=2, n_rows=10)  # span < the 60s embargo

    with pytest.raises(ValueError, match="empty split"):
        train(short, small_config(), model_dir=tmp_path, register=False)


# --- divergence ---


def _forbid_registration(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("register_model must not be called")

    monkeypatch.setattr(solia_ai.registry, "register_model", fail)


def test_a_never_finite_validation_loss_fails_without_saving_or_registering(tmp_path, monkeypatch):
    nan = float("nan")
    monkeypatch.setattr(
        train_module,
        "_validate",
        lambda *a, **k: {"pump": nan, "rug": nan, "return": nan, "total": nan},
    )
    _forbid_registration(monkeypatch)

    with pytest.raises(RuntimeError, match="never finite"):
        train(make_features_df(), small_config(), model_dir=tmp_path, engine=object())

    assert list(tmp_path.iterdir()) == []


# --- checkpoint files ---


def test_an_existing_checkpoint_is_never_overwritten(tmp_path, monkeypatch):
    first = train(make_features_df(), small_config(), model_dir=tmp_path, version="v1")
    original = first.checkpoint_path.read_bytes()
    _forbid_registration(monkeypatch)

    with pytest.raises(FileExistsError, match="already exists"):
        train(
            make_features_df(seed=1),
            small_config(),
            model_dir=tmp_path,
            engine=object(),
            version="v1",
        )

    assert first.checkpoint_path.read_bytes() == original


def test_a_failed_registration_removes_the_new_checkpoint(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("model version 'v1' already exists in the registry")

    monkeypatch.setattr(solia_ai.registry, "register_model", reject)

    with pytest.raises(ValueError, match="already exists"):
        train(make_features_df(), small_config(), model_dir=tmp_path, engine=object(), version="v1")

    assert list(tmp_path.iterdir()) == []


def test_the_checkpoint_path_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = train(make_features_df(), small_config(), model_dir="models", register=False)

    assert result.checkpoint_path.is_absolute()
    assert result.checkpoint_path == (tmp_path / "models" / f"{result.version}.pt").resolve()
    assert result.checkpoint_path.exists()


# --- class imbalance is corrected once ---


@pytest.mark.parametrize(("balanced", "expects_pos_weight"), [(True, False), (False, True)])
def test_pos_weight_only_applies_without_balanced_sampling(
    tmp_path, monkeypatch, balanced, expects_pos_weight
):
    seen = []
    original = SoliaLoss.from_config.__func__

    def spy(cls, config, *, pos_weight=None):
        seen.append(pos_weight)
        return original(cls, config, pos_weight=pos_weight)

    monkeypatch.setattr(SoliaLoss, "from_config", classmethod(spy))
    cfg = small_config()
    cfg["training"]["balanced_sampling"] = balanced

    train(make_features_df(), cfg, model_dir=tmp_path, register=False)

    (pos_weight,) = seen
    assert (pos_weight is not None) is expects_pos_weight


# --- validation loss ---


def _batch(mask, seed):
    gen = torch.Generator().manual_seed(seed)
    b, h = mask.shape
    x = torch.randn(b, 6, N_FEATURES, generator=gen)
    targets = {
        "pump": torch.randint(0, 2, (b, h), generator=gen).float(),
        "rug": torch.randint(0, 2, (b, h), generator=gen).float(),
        "expected_return": torch.randn(b, h, generator=gen),
        "mask": mask,
    }
    return x, torch.zeros(b, 6, dtype=torch.bool), targets


def test_validate_is_the_masked_mean_over_the_whole_set():
    torch.manual_seed(0)
    model = SoliaTransformer.from_config(small_config())
    loss_fn = SoliaLoss.from_config(small_config())
    batches = [
        _batch(torch.ones(4, 2), seed=1),  # fully labeled
        _batch(torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), seed=2),
        _batch(torch.zeros(4, 2), seed=3),  # no labels: must not drag the mean down
    ]

    result = _validate(model, loss_fn, batches, torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        x = torch.cat([b[0] for b in batches])
        padding = torch.cat([b[1] for b in batches])
        targets = {k: torch.cat([b[2][k] for b in batches]) for k in batches[0][2]}
        expected = loss_fn(model(x, padding), targets)
    for key, value in expected.items():
        assert result[key] == pytest.approx(float(value), rel=1e-5)


def test_validate_rejects_a_split_without_labels():
    model = SoliaTransformer.from_config(small_config())
    loss_fn = SoliaLoss.from_config(small_config())

    with pytest.raises(ValueError, match="no labeled horizon"):
        _validate(model, loss_fn, [_batch(torch.zeros(4, 2), seed=1)], torch.device("cpu"))


# --- CLI ---


def test_main_prints_the_trained_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda: "engine")
    reads = []
    monkeypatch.setattr(
        solia_ai.db,
        "read_training_features",
        lambda engine, data: reads.append(data) or (pd.DataFrame(), BOUNDS),
    )
    calls = []

    def fake_train(df, **kwargs):
        calls.append(kwargs)
        return TrainResult(
            version="v9",
            checkpoint_path=tmp_path / "v9.pt",
            n_params=10,
            best_epoch=2,
            metrics={"val_total": 0.5},
            hyperparams={},
        )

    monkeypatch.setattr(train_module, "train", fake_train)

    main(["--version", "v9", "--no-register", "--device", "cpu", "--max-rows", "1234"])

    assert reads == [DataSettings(DATA_DEFAULTS["lookback_days"], 1234)]  # the flag wins
    assert calls[0]["data_bounds"] == BOUNDS.as_dict()
    assert "trained v9: 10 params, best epoch 2, val_total=0.5000" in capsys.readouterr().out
    assert calls[0]["register"] is False
    assert calls[0]["version"] == "v9"


def test_checkpoint_records_the_last_training_window_end(tmp_path):
    df, cfg = make_features_df(), small_config()
    result = train(df, cfg, model_dir=tmp_path, register=False, version="cutoff")

    train_ds, val_ds = make_datasets(df, cfg)
    checkpoint = torch.load(result.checkpoint_path, weights_only=False)

    assert checkpoint["train_end_ts"] == int(train_ds.samples.end_ts[train_ds.indices].max())
    assert checkpoint["train_end_ts"] < int(val_ds.samples.end_ts[val_ds.indices].min())
    # data_end_ts: the latest window seen at all, early-stopping validation included.
    assert checkpoint["data_end_ts"] == int(val_ds.samples.end_ts[val_ds.indices].max())


# --- calibration & class weighting ---


def test_checkpoint_records_a_calibration_fit_before_the_evaluated_windows(tmp_path):
    cfg = small_config()
    cfg["calibration"] = {
        "method": "platt",
        "fraction": 0.5,
        "min_samples": 5,
        "min_class_samples": 2,
    }
    df = make_features_df()

    result = train(df, cfg, model_dir=tmp_path, register=False)

    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    _, val_ds = make_datasets(df, cfg, val_fraction=0.2)
    val_ends = val_ds.samples.end_ts[val_ds.indices]
    calibration = Calibration.from_checkpoint(checkpoint, cfg["horizons"])
    assert calibration.method == "platt"
    assert checkpoint["train_end_ts"] < checkpoint["calibration_end_ts"] < int(val_ends.max())
    assert checkpoint["hyperparams"]["calibration"] == cfg["calibration"]


def test_calibration_method_none_saves_no_calibration(tmp_path):
    cfg = small_config()
    cfg["calibration"] = {"method": "none"}

    result = train(make_features_df(), cfg, model_dir=tmp_path, register=False)

    checkpoint = torch.load(result.checkpoint_path, weights_only=True)
    assert checkpoint["calibration"] is None and checkpoint["calibration_end_ts"] is None


def test_calibration_split_is_the_earliest_share_of_the_validation_windows():
    _, val_ds = make_datasets(make_features_df(), small_config(), val_fraction=0.3)

    cal_ds = calibration_split(val_ds, 0.5)

    ends = val_ds.samples.end_ts
    rest = np.setdiff1d(val_ds.indices, cal_ds.indices)
    assert 0 < len(cal_ds) < len(val_ds)
    assert set(cal_ds.indices) <= set(val_ds.indices)
    assert ends[cal_ds.indices].max() < ends[rest].min()
    with pytest.raises(ValueError, match="calibration fraction"):
        calibration_split(val_ds, 1.0)


def _windows_ending_at(*ends: int) -> SequenceDataset:
    return SequenceDataset(SimpleNamespace(end_ts=np.array(ends)), np.arange(len(ends)))


def test_calibration_split_leaves_windows_tied_on_the_cutoff_out_of_a_whole_split():
    # The median end is the latest one: "<= cutoff" would take every window and leave nothing
    # after the calibration window to evaluate.
    val_ds = _windows_ending_at(100, 200, 200, 200)

    cal_ds = calibration_split(val_ds, 0.5)

    assert cal_ds.indices.tolist() == [0]


@pytest.mark.parametrize("ends", [(100,), (100, 100, 100)])
def test_calibration_split_refuses_a_split_with_a_single_end_time(ends):
    with pytest.raises(ValueError, match="too short to hold a calibration window"):
        calibration_split(_windows_ending_at(*ends), 0.5)


def test_a_too_short_validation_split_keeps_the_identity_with_a_warning():
    cfg = small_config()
    cfg["calibration"] = {"method": "platt"}
    model = SoliaTransformer.from_config(cfg)

    with pytest.warns(UserWarning, match="calibration skipped"):
        result = fit_calibration_window(model, _windows_ending_at(100, 100), cfg)

    assert result == (None, None)


def test_fitting_the_calibration_leaves_the_model_untouched():
    cfg = small_config()
    cfg["calibration"] = {"method": "platt", "min_samples": 5, "min_class_samples": 2}
    _, val_ds = make_datasets(make_features_df(), cfg, val_fraction=0.3)
    model = SoliaTransformer.from_config(cfg).train()
    before = {k: v.clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}

    calibration, end_ts = fit_calibration_window(model, val_ds, cfg)

    assert calibration is not None and end_ts is not None
    assert model.training
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fitting_the_calibration_keeps_a_gpu_model_on_its_device():
    cfg = small_config()
    _, val_ds = make_datasets(make_features_df(), cfg, val_fraction=0.3)
    model = SoliaTransformer.from_config(cfg).cuda()

    fit_calibration_window(model, val_ds, cfg)

    assert all(p.device.type == "cuda" for p in model.parameters())


def _natural(pos_weight=None) -> dict:
    cfg = small_config()
    cfg["training"]["balanced_sampling"] = False
    if pos_weight is not None:
        cfg["loss"]["pos_weight"] = pos_weight
    return cfg


def test_resolve_pos_weight_follows_the_config():
    train_ds, _ = make_datasets(make_features_df(), small_config(), val_fraction=0.2)
    n_h = len(small_config()["horizons"])

    assert resolve_pos_weight(train_ds, small_config()) is None  # balanced: corrected once
    auto = resolve_pos_weight(train_ds, _natural())  # natural sampling defaults to auto
    measured = positive_weights(train_ds)
    assert all(torch.equal(auto[h], measured[h]) for h in ("pump", "rug"))
    assert resolve_pos_weight(train_ds, _natural("none")) is None
    fixed = resolve_pos_weight(train_ds, _natural(3.0))
    assert torch.equal(fixed["pump"], torch.full((n_h,), 3.0))
    assert torch.equal(fixed["rug"], torch.full((n_h,), 3.0))
    per_head = resolve_pos_weight(train_ds, _natural({"pump": 5.0}))
    assert torch.equal(per_head["pump"], torch.full((n_h,), 5.0))
    assert torch.equal(per_head["rug"], torch.ones(n_h))
    # applied as configured, never silently clipped (validation bounds it instead)
    assert torch.equal(
        resolve_pos_weight(train_ds, _natural(1000))["pump"], torch.full((n_h,), 1000.0)
    )
