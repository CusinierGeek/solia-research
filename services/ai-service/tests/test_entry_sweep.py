"""Entry-rule sweep: per-rule metrics on a hand-checked fixture, top-X %, empty selections,
the recommendation, the report and invalid grids."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from solia_ai import entry_sweep
from solia_ai.entry_sweep import (
    bootstrap_mean_ci,
    build_report,
    main,
    recommend,
    render,
    score_windows,
    selection_metrics,
    sweep,
    sweep_horizon,
    write_report,
)

# Six labeled windows of one horizon:
#   i  pump  rug  conf  pump?  return
#   0  0.50  0.05  90    1     +1.0
#   1  0.40  0.40  20    1     +0.5
#   2  0.30  0.10  80    0     -0.2
#   3  0.20  0.05  90    0     +0.1
#   4  0.05  0.60  20    0     -1.0
#   5  0.01  0.02  96    0      0.0
PUMP = np.array([0.50, 0.40, 0.30, 0.20, 0.05, 0.01])
RUG = np.array([0.05, 0.40, 0.10, 0.05, 0.60, 0.02])
CONF = np.array([90.0, 20.0, 80.0, 90.0, 20.0, 96.0])
LABEL = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
RET = np.array([1.0, 0.5, -0.2, 0.1, -1.0, 0.0])


def _sweep(**kwargs):
    defaults = {
        "pump_thresholds": (0.3,),
        "rug_thresholds": (None, 0.2),
        "top_fractions": (0.5,),
        "min_trades": 2,
        "bootstrap_samples": 200,
    }
    return sweep_horizon(PUMP, RUG, CONF, LABEL, RET, **{**defaults, **kwargs})


def _rule(result, kind, rug):
    return next(r for r in result["rules"] if r["kind"] == kind and r["rug_threshold"] == rug)


def test_threshold_rule_metrics():
    result = _sweep()

    no_gate = _rule(result, "threshold", None)  # pump >= 0.3: rows 0, 1, 2
    assert no_gate["n_trades"] == 3
    assert no_gate["precision"] == pytest.approx(2 / 3)
    assert no_gate["precision_lift"] == pytest.approx((2 / 3) / (2 / 6))
    assert no_gate["win_rate"] == pytest.approx(2 / 3)
    assert no_gate["mean_return"] == pytest.approx(1.3 / 3)
    assert no_gate["median_return"] == pytest.approx(0.5)
    assert no_gate["total_pnl"] == pytest.approx(1.3)

    gated = _rule(result, "threshold", 0.2)  # + rug <= 0.2: rows 0, 2
    assert gated["n_trades"] == 2
    assert gated["precision"] == pytest.approx(0.5)
    assert gated["total_pnl"] == pytest.approx(0.8)


def test_baseline_is_buy_everything():
    base = _sweep()["baseline"]

    assert base["n_trades"] == 6
    assert base["precision"] == pytest.approx(2 / 6)
    assert base["mean_return"] == pytest.approx(0.4 / 6)
    assert _sweep()["base_rate"] == pytest.approx(2 / 6)


def test_top_fraction_selects_the_highest_scores_and_reports_the_implied_threshold():
    result = _sweep()

    top = _rule(result, "top", None)  # top 50 % of 6 = rows 0, 1, 2
    assert top["n_trades"] == 3
    assert top["pump_threshold"] == pytest.approx(0.30)
    assert top["total_pnl"] == pytest.approx(1.3)
    # The rug gate filters the top slice, it doesn't re-rank: rows 0 and 2 remain.
    assert _rule(result, "top", 0.2)["n_trades"] == 2


def test_top_fraction_rounds_up_to_at_least_one_window():
    result = _sweep(top_fractions=(0.01,))

    top = _rule(result, "top", None)
    assert top["n_trades"] == 1
    assert top["pump_threshold"] == pytest.approx(0.50)


def test_empty_selection_has_no_ratios_and_zero_pnl():
    result = _sweep(pump_thresholds=(0.9,))

    empty = _rule(result, "threshold", None)
    assert empty["n_trades"] == 0
    assert empty["precision"] is None
    assert empty["win_rate"] is None
    assert empty["mean_return"] is None
    assert empty["total_pnl"] == 0.0
    assert empty["mean_return_ci"] == [None, None]
    assert empty["enough_trades"] is False


def test_empty_horizon_is_handled():
    empty = np.array([])
    result = sweep_horizon(empty, empty, empty, empty, empty, top_fractions=(0.1,))

    assert result["n_candidates"] == 0
    assert result["base_rate"] is None
    assert all(r["n_trades"] == 0 for r in result["rules"])
    assert recommend(result) is None


def test_confidence_gate_removes_trades_and_counts_them():
    result = _sweep(min_confidence=60)

    no_gate = _rule(result, "threshold", None)  # row 1 has confidence 20
    assert no_gate["n_trades"] == 2
    assert no_gate["blocked_by_confidence"] == 1
    assert result["confidence_pass_rate"] == pytest.approx(4 / 6)


def test_enough_trades_follows_min_trades():
    result = _sweep(min_trades=3)

    assert _rule(result, "threshold", None)["enough_trades"] is True
    assert _rule(result, "threshold", 0.2)["enough_trades"] is False


def test_bootstrap_ci_brackets_the_mean_and_is_undefined_below_two_values():
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 1.0, size=500)

    low, high = bootstrap_mean_ci(values, samples=500, ci=0.95, rng=rng)
    assert low < values.mean() < high
    assert bootstrap_mean_ci(np.array([1.0]), samples=10, ci=0.95, rng=rng) == (None, None)


def test_total_pnl_ci_scales_the_mean_ci():
    m = selection_metrics(
        RET, LABEL, base_rate=0.5, samples=200, ci=0.9, rng=np.random.default_rng(1)
    )

    assert m["total_pnl_ci"][0] == pytest.approx(m["mean_return_ci"][0] * 6)
    assert m["total_pnl_ci"][1] == pytest.approx(m["mean_return_ci"][1] * 6)


def test_recommend_prefers_precision_among_rules_that_clear_the_baseline():
    def row(precision, n, low):
        return {
            "precision": precision,
            "n_trades": n,
            "enough_trades": n >= 10,
            "mean_return_ci": [low, low + 1],
        }

    sweep_result = {
        "baseline": {"mean_return": 0.0},
        "rules": [
            row(0.9, 5, 0.5),  # too few trades
            row(0.6, 50, -0.1),  # most precise with enough trades, but CI under the baseline
            row(0.4, 100, 0.1),
            row(0.3, 300, 0.2),
        ],
    }

    best = recommend(sweep_result)
    assert best["precision"] == 0.4
    assert best["beats_baseline"] is True


def test_recommend_falls_back_to_precision_when_nothing_clears_the_baseline():
    sweep_result = {
        "baseline": {"mean_return": 0.5},
        "rules": [
            {"precision": 0.2, "n_trades": 20, "enough_trades": True, "mean_return_ci": [0, 1]},
            {"precision": 0.3, "n_trades": 20, "enough_trades": True, "mean_return_ci": [0, 1]},
        ],
    }

    best = recommend(sweep_result)
    assert best["precision"] == 0.3
    assert best["beats_baseline"] is False


def _matrix_inputs():
    # Two horizons; the second masks out the last two windows.
    mask = np.ones((6, 2))
    mask[4:, 1] = 0
    predictions = {
        "pump_prob": np.stack([PUMP, PUMP], axis=1),
        "rug_prob": np.stack([RUG, RUG], axis=1),
        "confidence": np.stack([CONF, CONF], axis=1),
    }
    truth = {
        "pump": np.stack([LABEL, LABEL], axis=1),
        "expected_return": np.stack([RET, RET], axis=1),
        "mask": mask,
    }
    return predictions, truth


def test_sweep_scores_each_horizon_on_its_labeled_windows():
    predictions, truth = _matrix_inputs()

    result = sweep(
        predictions, truth, ["1m", "5m"], pump_thresholds=(0.3,), min_trades=2, top_fractions=()
    )

    assert result["1m"]["n_candidates"] == 6
    assert result["5m"]["n_candidates"] == 4
    assert result["1m"]["recommended"] is not None


def test_report_round_trips_to_json_and_renders(tmp_path):
    predictions, truth = _matrix_inputs()
    settings = {"min_trades": 2, "min_confidence": 0.0, "ci": 0.95}
    by_horizon = sweep(predictions, truth, ["1m", "5m"], pump_thresholds=(0.3, 0.9), min_trades=2)
    report = build_report("v1", 6, by_horizon, settings)

    json_path, table_path = write_report(report, tmp_path / "sweep.json")

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["model_version"] == "v1"
    assert set(loaded["horizons"]) == {"1m", "5m"}
    table = table_path.read_text(encoding="utf-8")
    assert "buy everything" in table and "recommended:" in table
    # Rules under min_trades are hidden unless asked for.
    assert "p>=0.9" not in table
    assert "p>=0.9" in render(report, all_rules=True)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pump_thresholds": (1.5,)}, "pump threshold"),
        ({"pump_thresholds": (float("nan"),)}, "pump threshold"),
        ({"rug_thresholds": (-0.1,)}, "rug threshold"),
        ({"top_fractions": (0.0,)}, "top fraction"),
        ({"top_fractions": (1.2,)}, "top fraction"),
        ({"min_confidence": 101.0}, "min_confidence"),
        ({"min_trades": 0}, "min_trades"),
        ({"bootstrap_samples": 0}, "bootstrap_samples"),
        ({"ci": 1.0}, "ci"),
    ],
)
def test_invalid_grid_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _sweep(**kwargs)


def test_mismatched_arrays_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        sweep_horizon(PUMP, RUG[:3], CONF, LABEL, RET)


def test_cli_rejects_a_bad_grid_before_touching_the_database():
    with pytest.raises(SystemExit, match="pump threshold"):
        main(["--champion", "--pump-thresholds", "0.2,2"])


def test_scoring_moves_the_model_to_the_requested_device(monkeypatch):
    # The "meta" device stands in for the GPU, so this also runs on a machine without one.
    class FakePredictor:
        model = torch.nn.Linear(1, 1)
        calibration = 1.0

    seen = []

    def fake_score(model, dataset):
        seen.append(next(model.parameters()).device.type)
        return {"pump_logits": dataset}, {"mask": dataset}

    monkeypatch.setattr(entry_sweep, "score_logits", fake_score)
    monkeypatch.setattr(entry_sweep, "apply_calibration", lambda logits, calibration: logits)

    predictions, truth = score_windows(FakePredictor(), "windows", "meta")

    assert seen == ["meta"]
    assert predictions == {"pump_logits": "windows"}
    assert truth == {"mask": "windows"}
