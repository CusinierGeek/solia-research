"""Evaluation: the metric functions on synthetic data, and the report end-to-end on a tiny model."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr
from test_train import make_features_df, small_config

import solia_ai.evaluate
from solia_ai.calibration import Calibration
from solia_ai.dataset import embargo_seconds, make_datasets
from solia_ai.evaluate import (
    DEFAULT_POLICY_HORIZON,
    EvaluationReport,
    ReferencePolicy,
    apply_calibration,
    calibrate,
    classification_metrics,
    default_policy_horizon,
    evaluate,
    expected_calibration_error,
    holdout_dataset,
    regression_metrics,
    score_logits,
)
from solia_ai.model import SoliaTransformer

# --------------------------------------------------------------------------- calibration


def test_ece_is_zero_when_probabilities_match_the_frequency():
    # Half the samples at p=0.0 and all negative, half at p=1.0 and all positive: perfect.
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    y_true = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)


def test_ece_grows_with_overconfidence():
    # Claim p=1.0 but only half are positive: a 0.5 calibration gap.
    y_prob = np.ones(10)
    y_true = np.array([1.0, 0.0] * 5)

    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.5)


def test_ece_is_nan_on_empty_input():
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))


# --------------------------------------------------------------------------- classification


def test_classification_metrics_reward_perfect_separation():
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = classification_metrics(y_true, y_prob)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["n"] == 4
    assert metrics["positives"] == 2
    assert metrics["base_rate"] == pytest.approx(0.5)


def test_auc_is_none_when_one_class_is_absent():
    metrics = classification_metrics(np.zeros(5), np.linspace(0, 1, 5))

    assert metrics["roc_auc"] is None  # undefined, not zero
    assert metrics["pr_auc"] is None
    assert metrics["base_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- regression


def test_regression_metrics_on_a_perfect_prediction():
    y_true = np.array([-0.2, 0.0, 0.3, 0.5, 1.0])

    metrics = regression_metrics(y_true, y_true.copy())

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["ic"] == pytest.approx(1.0)
    assert metrics["corr"] == pytest.approx(1.0)


def test_regression_ic_is_negative_for_an_inverted_signal():
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    metrics = regression_metrics(y_true, -y_true)

    assert metrics["ic"] == pytest.approx(-1.0)
    assert metrics["mae"] > 0


def test_regression_correlation_is_none_for_a_constant_prediction():
    metrics = regression_metrics(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.5, 0.5]))

    assert metrics["ic"] is None
    assert metrics["corr"] is None
    assert metrics["mae"] is not None  # MAE is still well-defined


# --------------------------------------------------------------------------- reference policy


def _scored(pump, rug, ret, mask):
    """Single-horizon prediction/truth arrays as ``score_dataset`` would produce them."""
    col = lambda a: np.asarray(a, dtype=np.float32).reshape(-1, 1)  # noqa: E731
    predictions = {"pump_prob": col(pump), "rug_prob": col(rug)}
    truth = {"expected_return": col(ret), "mask": col(mask)}
    return predictions, truth


def test_reference_policy_enters_only_on_confident_non_rug_windows():
    # rows:        enter?   reason
    #  0  p .8 r .1   yes
    #  1  p .9 r .5   no    (rug above threshold)
    #  2  p .3 r .1   no    (pump below threshold)
    #  3  p .7 r .2   yes
    #  4  p .95 r .1  no    (unlabeled: mask 0)
    predictions, truth = _scored(
        pump=[0.8, 0.9, 0.3, 0.7, 0.95],
        rug=[0.1, 0.5, 0.1, 0.2, 0.1],
        ret=[0.4, 0.0, 0.0, -0.1, 1.0],
        mask=[1, 1, 1, 1, 0],
    )
    policy = ReferencePolicy(horizon="30s", pump_threshold=0.6, rug_threshold=0.3)

    result = policy.simulate(predictions, truth, ("30s",))

    assert result["n_candidates"] == 4  # the labeled rows
    assert result["n_trades"] == 2  # rows 0 and 3
    assert result["win_rate"] == pytest.approx(0.5)  # 0.4 wins, -0.1 loses
    assert result["total_pnl"] == pytest.approx(0.3)
    assert result["avg_pnl"] == pytest.approx(0.15)


def test_reference_policy_reports_none_when_no_window_qualifies():
    predictions, truth = _scored(pump=[0.1, 0.2], rug=[0.1, 0.1], ret=[0.5, 0.5], mask=[1, 1])
    policy = ReferencePolicy(horizon="30s")

    result = policy.simulate(predictions, truth, ("30s",))

    assert result["n_trades"] == 0
    assert result["win_rate"] is None
    assert result["total_pnl"] == pytest.approx(0.0)
    assert result["baseline_avg_return"] == pytest.approx(0.5)


def test_reference_policy_defaults_are_the_swept_entry_rule():
    # The bot's defaults (BotOptions / BotHostOptions) are the same rule: 15m, pump >= 0.4, no rug
    # gate (docs/ML-MODEL.md "Choosing the entry rule", 2026-11-02 sweep).
    policy = ReferencePolicy(horizon=DEFAULT_POLICY_HORIZON)

    assert (policy.horizon, policy.pump_threshold, policy.rug_threshold) == ("15m", 0.4, 1.0)


def test_the_default_reference_policy_enters_whatever_the_rug_head_says():
    policy = ReferencePolicy(horizon=DEFAULT_POLICY_HORIZON)
    assert policy.rug_threshold == 1.0  # rug_prob <= 1 always holds: no rug gate


def test_default_policy_horizon_falls_back_to_the_first_served_horizon():
    assert default_policy_horizon(["30s", "1m", "15m"]) == "15m"
    assert default_policy_horizon(("30s", "1m")) == "30s"
    with pytest.raises(ValueError, match="no horizons"):
        default_policy_horizon([])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pump_threshold": 0.0}, "pump_threshold"),  # a zero gate enters on every window
        ({"pump_threshold": 1.1}, "pump_threshold"),
        ({"pump_threshold": float("nan")}, "pump_threshold"),
        ({"rug_threshold": -0.1}, "rug_threshold"),
        ({"rug_threshold": 1.5}, "rug_threshold"),
        ({"rug_threshold": float("inf")}, "rug_threshold"),
    ],
)
def test_reference_policy_rejects_invalid_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReferencePolicy(horizon="15m", **kwargs)


def test_reference_policy_rejects_an_unknown_horizon():
    predictions, truth = _scored(pump=[0.8], rug=[0.1], ret=[0.4], mask=[1])

    with pytest.raises(ValueError, match="not one of the model's"):
        ReferencePolicy(horizon="5m").simulate(predictions, truth, ("30s", "1m"))


# --------------------------------------------------------------------------- end-to-end


def _trained_tiny():
    cfg = small_config()
    df = make_features_df(n_tokens=8, n_rows=90, seed=3)
    train_ds, val_ds = make_datasets(df, cfg, val_fraction=0.3)
    model = SoliaTransformer.from_config(cfg)
    rows = train_ds.samples.features
    model.fit_normalizer(rows)
    return model.eval(), val_ds


def test_scored_logits_calibrate_into_aligned_predictions_and_truth():
    model, val_ds = _trained_tiny()

    logits, truth = score_logits(model, val_ds, batch_size=4)
    predictions = apply_calibration(logits)

    n_h = len(val_ds.samples.horizons)
    for array in (*predictions.values(), *truth.values()):
        assert array.shape == (len(val_ds), n_h)
    assert np.all((predictions["pump_prob"] >= 0) & (predictions["pump_prob"] <= 1))
    assert np.all((predictions["confidence"] >= 0) & (predictions["confidence"] <= 100))


def test_evaluation_scores_the_same_bounded_return_as_serving():
    logits = {
        "pump_logits": np.zeros((2, 1)),
        "rug_logits": np.zeros((2, 1)),
        "expected_return": np.array([[-1.1328], [0.25]]),  # a legacy head's out-of-contract value
    }

    predictions = apply_calibration(logits)

    assert predictions["expected_return"].tolist() == [[-1.0], [0.25]]


def test_evaluate_builds_a_report_over_every_horizon():
    model, val_ds = _trained_tiny()

    report = evaluate(model, val_ds, model_version="tiny-v1")

    assert isinstance(report, EvaluationReport)
    assert report.model_version == "tiny-v1"
    assert report.n_samples == len(val_ds)
    horizons = list(val_ds.samples.horizons)
    assert report.horizons == horizons
    for horizon in horizons:
        assert set(report.pump[horizon]) >= {"roc_auc", "pr_auc", "ece", "base_rate"}
        assert set(report.expected_return[horizon]) >= {"ic", "corr", "mae"}
    assert report.policy["horizon"] == horizons[0]


def test_report_as_dict_is_json_safe_and_nulls_non_finite(tmp_path):
    model, val_ds = _trained_tiny()
    report = evaluate(model, val_ds, model_version="tiny-v1")
    # Force a non-finite metric to prove as_dict() renders it as JSON null, not a crash.
    report.pump[report.horizons[0]]["roc_auc"] = float("nan")

    payload = report.as_dict()
    text = json.dumps(payload)  # would raise if a NaN leaked through

    assert payload["pump"][report.horizons[0]]["roc_auc"] is None
    assert "tiny-v1" in text


def test_report_render_and_write_round_trip(tmp_path):
    model, val_ds = _trained_tiny()
    report = evaluate(model, val_ds, model_version="tiny-v1")

    rendered = report.render()
    path = report.write(tmp_path / "report.json")

    assert "Evaluation" in rendered
    assert "Reference policy" in rendered
    on_disk = json.loads(path.read_text("utf-8"))
    assert on_disk["model_version"] == "tiny-v1"
    assert on_disk["horizons"] == list(val_ds.samples.horizons)


# --------------------------------------------------------------------------- review fixes


def test_ic_ranks_ties_by_their_average_rank():
    # Many realized returns tie at exactly 0 (no move); ordinal ranks would make the IC depend on
    # the input order, average ranks match scipy's Spearman whatever the order.
    y_true = np.array([0.0, 0.0, 0.0, 0.0, 0.5, -0.2, 0.0, 1.0])
    y_pred = np.array([0.1, 0.3, 0.2, 0.4, 0.6, 0.0, 0.5, 0.9])
    expected = spearmanr(y_pred, y_true).statistic

    ics = []
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(y_true.size)
        ics.append(regression_metrics(y_true[order], y_pred[order])["ic"])

    assert ics == pytest.approx([expected] * 5)


def test_ic_is_none_when_ties_leave_a_constant_series():
    metrics = regression_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))

    assert metrics["ic"] is None


def test_evaluate_refuses_a_dataset_labeled_for_other_horizons():
    model, _ = _trained_tiny()
    cfg = small_config()
    cfg["horizons"] = ["1m", "30s"]  # same horizons, different order: columns would misalign
    _, val_ds = make_datasets(
        make_features_df(n_tokens=8, n_rows=90, seed=3), cfg, val_fraction=0.3
    )

    with pytest.raises(ValueError, match="model predicts"):
        evaluate(model, val_ds)


def test_evaluate_checks_the_policy_horizon_before_scoring(monkeypatch):
    model, val_ds = _trained_tiny()

    def never(*_args, **_kwargs):
        raise AssertionError("scored the dataset before validating the policy")

    monkeypatch.setattr(solia_ai.evaluate, "score_logits", never)

    with pytest.raises(ValueError, match="not one of the model's"):
        evaluate(model, val_ds, policy=ReferencePolicy(horizon="7d"))


def test_report_has_no_unused_history_field():
    model, val_ds = _trained_tiny()

    assert "history" not in evaluate(model, val_ds).as_dict()


def _predictor_stub(model, **metadata):
    return SimpleNamespace(horizons=model.horizons, metadata=metadata)


def test_holdout_keeps_only_windows_after_the_training_data_and_embargo():
    model, _ = _trained_tiny()
    cfg = small_config()
    df = make_features_df(n_tokens=8, n_rows=90, seed=3)
    train_ds, _ = make_datasets(df, cfg, val_fraction=0.3)
    train_end_ts = int(train_ds.samples.end_ts[train_ds.indices].max())

    # A different val share than training used must not pull training windows back in.
    held_out = holdout_dataset(
        df, cfg, _predictor_stub(model, train_end_ts=train_end_ts), val_fraction=0.9
    )

    ends = held_out.samples.end_ts[held_out.indices]
    assert len(held_out) > 0
    assert ends.min() > train_end_ts + embargo_seconds(cfg["horizons"])
    assert not set(held_out.indices) & set(train_ds.indices)


def test_holdout_labels_follow_the_checkpoint_horizons_not_the_config():
    model, _ = _trained_tiny()
    cfg = small_config()
    df = make_features_df(n_tokens=8, n_rows=90, seed=3)
    drifted = {**cfg, "horizons": ["1m"]}  # config.yaml edited since the model was trained

    held_out = holdout_dataset(df, drifted, _predictor_stub(model, train_end_ts=0))

    assert held_out.samples.horizons == tuple(model.horizons)
    evaluate(model, held_out)  # aligned: no horizon mismatch


def test_holdout_falls_back_to_the_split_with_a_warning_for_legacy_checkpoints(capsys):
    model, _ = _trained_tiny()
    cfg = small_config()
    df = make_features_df(n_tokens=8, n_rows=90, seed=3)

    held_out = holdout_dataset(df, cfg, _predictor_stub(model), val_fraction=0.3)

    _, expected = make_datasets(df, cfg, val_fraction=0.3)
    assert list(held_out.indices) == list(expected.indices)
    assert "train_end_ts" in capsys.readouterr().err


def test_holdout_rejects_data_that_ends_before_the_training_cutoff():
    model, _ = _trained_tiny()
    cfg = small_config()
    df = make_features_df(n_tokens=8, n_rows=90, seed=3)
    far_future = int(df["ts"].max().timestamp()) + 10_000

    with pytest.raises(ValueError, match="no window ends after"):
        holdout_dataset(df, cfg, _predictor_stub(model, train_end_ts=far_future))


# --- calibration ---


def test_evaluate_reports_the_ece_before_and_after_calibration():
    model, val_ds = _trained_tiny()
    n = len(model.horizons)
    calibration = Calibration(
        tuple(model.horizons), (3.0,) * n, (-1.0,) * n, (0.3,) * n, (0.5,) * n, "platt"
    )

    report = evaluate(model, val_ds, calibration=calibration)
    raw = evaluate(model, val_ds)

    assert report.calibration == calibration.as_dict()
    for head in ("pump", "rug"):
        for h in model.horizons:
            after, before = getattr(report, head)[h], report.uncalibrated[head][h]
            assert before == getattr(raw, head)[h]  # "before" = the uncalibrated scoring
            assert after["roc_auc"] == pytest.approx(before["roc_auc"])  # ranking untouched
            assert after["ece"] != pytest.approx(before["ece"])
    assert "ECE before -> after" in report.render()
    json.dumps(report.as_dict())


@pytest.mark.parametrize("bias", [10.0, -10.0])
def test_a_calibration_at_its_bounds_does_not_tie_scores_or_move_the_ranking(bias):
    # T = 0.1 and |b| = 10 push logit/T + b past float32's sigmoid saturation (~17); scoring in
    # float64 keeps distinct scores distinct, so ROC-AUC / PR-AUC stay those of the raw scores.
    model, val_ds = _trained_tiny()
    n = len(model.horizons)
    calibration = Calibration(
        tuple(model.horizons), (0.1,) * n, (bias,) * n, (0.1,) * n, (bias,) * n, "platt"
    )

    report = evaluate(model, val_ds, calibration=calibration)

    for head in ("pump", "rug"):
        for h in model.horizons:
            after, before = getattr(report, head)[h], report.uncalibrated[head][h]
            for key in ("roc_auc", "pr_auc"):
                if before[key] is not None:
                    assert after[key] == pytest.approx(before[key], abs=1e-12)


def test_an_identity_calibration_scores_the_classification_metrics_once(monkeypatch):
    model, val_ds = _trained_tiny()
    calls = []
    real = solia_ai.evaluate.classification_metrics

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(solia_ai.evaluate, "classification_metrics", counting)
    n = len(model.horizons)

    report = evaluate(model, val_ds)
    assert len(calls) == 2 * n  # pump + rug per horizon, not again for "uncalibrated"
    assert report.uncalibrated == {"pump": report.pump, "rug": report.rug}

    calls.clear()
    evaluate(model, val_ds, temperature=2.0)
    assert len(calls) == 4 * n


def test_identity_calibration_leaves_the_metrics_unchanged():
    model, val_ds = _trained_tiny()

    report = evaluate(model, val_ds)

    assert report.calibration["method"] == "none"
    for head in ("pump", "rug"):
        assert report.uncalibrated[head] == getattr(report, head)


def test_calibrate_fits_on_the_given_window():
    model, val_ds = _trained_tiny()

    cal = calibrate(model, val_ds, method="temperature", min_samples=1)

    assert cal.horizons == tuple(model.horizons) and cal.method == "temperature"
    assert all(t > 0 for t in cal.pump_temperature + cal.rug_temperature)
    assert cal.pump_bias == (0.0,) * len(model.horizons)


def test_holdout_starts_after_the_calibration_window():
    model, _ = _trained_tiny()
    cfg = small_config()
    df = make_features_df(n_tokens=8, n_rows=90, seed=3)
    train_ds, val_ds = make_datasets(df, cfg, val_fraction=0.5)
    train_end_ts = int(train_ds.samples.end_ts[train_ds.indices].max())
    val_ends = np.sort(val_ds.samples.end_ts[val_ds.indices])
    calibration_end_ts = int(val_ends[len(val_ends) // 4])

    held_out = holdout_dataset(
        df,
        cfg,
        _predictor_stub(model, train_end_ts=train_end_ts, calibration_end_ts=calibration_end_ts),
    )

    ends = held_out.samples.end_ts[held_out.indices]
    assert ends.min() > calibration_end_ts + embargo_seconds(cfg["horizons"])


def test_evaluate_reports_the_entry_selections_and_run_metrics_flatten_them():
    from solia_ai.ablation import run_metrics
    from solia_ai.entry_selection import RUG_THRESHOLD_KEY, entry_key, entry_keys

    model, val_ds = _trained_tiny()
    policy = ReferencePolicy(horizon=val_ds.samples.horizons[0], rug_threshold=1.0)

    report = evaluate(model, val_ds, policy=policy, entry_top_fraction=0.5)

    horizons = list(val_ds.samples.horizons)
    for h, horizon in enumerate(horizons):
        gated, every = report.entry[horizon]["gated"], report.entry[horizon]["pump_only"]
        labeled = int((val_ds.samples.mask[val_ds.indices][:, h] == 1).sum())
        assert gated["rug_threshold"] == 1.0 and every["rug_threshold"] is None
        assert gated["pool_windows"] == every["pool_windows"] == labeled  # rug <= 1 keeps all
        assert every["windows"] == -(-labeled // 2) and 1 <= every["tokens"] <= 8
    metrics = run_metrics(report)
    assert set(entry_keys(horizons)) <= set(metrics)
    assert metrics[RUG_THRESHOLD_KEY] == 1.0
    assert metrics[entry_key("entry", horizons[0], "windows")] == float(
        report.entry[horizons[0]]["gated"]["windows"]
    )
    assert "Entry selection" in report.render()
