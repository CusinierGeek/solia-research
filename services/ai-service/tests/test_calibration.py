"""Post-hoc calibration: fitting, the identity fallbacks, and (de)serialization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from solia_ai.calibration import Calibration, fit_calibration, fit_head, settings
from solia_ai.evaluate import expected_calibration_error

HORIZONS = ("30s", "1m")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def synthetic(n: int = 4000, *, seed: int = 0, base: float = -1.5):
    """True logits and labels drawn from them: ``sigmoid(true)`` is perfectly calibrated."""
    rng = np.random.default_rng(seed)
    true = rng.normal(base, 1.5, size=n)
    labels = (rng.random(n) < sigmoid(true)).astype(np.float64)
    return true, labels


def calibrated_probs(logits, t, b):
    return sigmoid(logits / t + b)


@pytest.mark.parametrize(("distortion", "expected_t"), [(4.0, 4.0), (0.25, 0.25)])
def test_temperature_fit_undoes_over_and_under_confidence(distortion, expected_t):
    true, labels = synthetic()
    logits = true * distortion  # x4 = overconfident, x0.25 = underconfident

    t, b = fit_head(logits, labels, method="temperature")

    assert b == 0.0
    assert t == pytest.approx(expected_t, rel=0.15)
    before = expected_calibration_error(labels, sigmoid(logits))
    after = expected_calibration_error(labels, calibrated_probs(logits, t, b))
    assert after < before / 2


def test_platt_also_corrects_a_base_rate_shift_temperature_cannot():
    true, labels = synthetic()
    logits = true + 2.0  # e.g. balanced batches: probabilities biased toward the event rate

    t_only, _ = fit_head(logits, labels, method="temperature")
    t, b = fit_head(logits, labels, method="platt")

    assert t == pytest.approx(1.0, abs=0.15)
    assert b == pytest.approx(-2.0, abs=0.2)
    platt_ece = expected_calibration_error(labels, calibrated_probs(logits, t, b))
    temp_ece = expected_calibration_error(labels, calibrated_probs(logits, t_only, 0.0))
    assert platt_ece < temp_ece
    assert platt_ece < expected_calibration_error(labels, sigmoid(logits)) / 2


def test_a_separable_window_keeps_the_fit_inside_the_non_saturating_bounds():
    # Perfectly separable labels push T -> 0 and b -> +-inf; the bounds stop the map at a mild one.
    logits = np.linspace(-3.0, 3.0, 400)
    labels = (logits > 0).astype(np.float64)

    t, b = fit_head(logits, labels, method="platt")

    assert 0.1 <= t <= 10.0 and -10.0 <= b <= 10.0
    assert t == pytest.approx(0.1)  # pinned at the bound, not driven to 0
    scores = torch.sigmoid(torch.tensor(logits, dtype=torch.float64) / t + b)
    assert len(torch.unique(scores)) == len(logits)  # no saturation ties in float64


@pytest.mark.parametrize("minority_label", [1.0, 0.0])
def test_a_rare_minority_class_keeps_the_identity(minority_label):
    # 200 labeled samples clear min_samples, but 3 events (or 3 non-events) can't support a fit.
    true, _ = synthetic(n=200)
    labels = np.full(200, 1.0 - minority_label)
    labels[:3] = minority_label

    assert fit_head(true, labels, method="platt", min_class_samples=10) == (1.0, 0.0)
    assert fit_head(true, labels, method="platt", min_class_samples=3) != (1.0, 0.0)


def test_calibration_never_changes_the_ranking():
    true, labels = synthetic(seed=3)
    logits = true * 3.0 + 1.0

    t, b = fit_head(logits, labels, method="platt")

    assert t > 0
    assert roc_auc_score(labels, calibrated_probs(logits, t, b)) == pytest.approx(
        roc_auc_score(labels, sigmoid(logits))
    )


@pytest.mark.parametrize(
    ("labels", "kwargs"),
    [
        (np.zeros(200), {}),  # a single class: nothing to calibrate against
        (np.ones(200), {}),
        (np.r_[np.zeros(20), np.ones(10)], {}),  # fewer than min_samples
        (np.r_[np.zeros(100), np.ones(100)], {"method": "none"}),
    ],
)
def test_fit_head_keeps_the_identity_when_it_cannot_or_must_not_fit(labels, kwargs):
    logits = np.linspace(-3, 3, labels.size)

    assert fit_head(logits, labels, **kwargs) == (1.0, 0.0)


def test_fit_head_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_head(np.zeros(100), np.r_[np.zeros(50), np.ones(50)], method="isotonic")


def test_fit_calibration_fits_each_head_and_horizon_on_labeled_samples_only():
    true, labels = synthetic(n=3000)
    n = true.size
    logits = {
        "pump_logits": np.stack([true * 4.0, true], axis=1),
        "rug_logits": np.stack([true, true * 0.25], axis=1),
    }
    mask = np.ones((n, 2))
    mask[: n // 2, 1] = 0  # the masked half of 1m gets garbage labels that must be ignored
    pump_1m = labels.copy()
    pump_1m[: n // 2] = 1.0
    truth = {
        "pump": np.stack([labels, pump_1m], axis=1),
        "rug": np.stack([labels, labels], axis=1),
        "mask": mask,
    }

    cal = fit_calibration(logits, truth, HORIZONS, method="temperature")

    assert cal.horizons == HORIZONS and cal.method == "temperature"
    assert cal.pump_temperature[0] == pytest.approx(4.0, rel=0.15)
    assert cal.pump_temperature[1] == pytest.approx(1.0, rel=0.2)  # garbage half ignored
    assert cal.rug_temperature[1] == pytest.approx(0.25, rel=0.2)
    assert cal.pump_bias == (0.0, 0.0)


# --------------------------------------------------------------------------- the object


def sample_calibration() -> Calibration:
    return Calibration(HORIZONS, (2.0, 0.5), (-1.0, 0.25), (1.5, 1.0), (0.0, -0.5), "platt")


def test_round_trips_through_its_dict_form():
    cal = sample_calibration()

    assert Calibration.from_dict(cal.as_dict(), HORIZONS) == cal


def test_a_checkpoint_without_calibration_loads_as_the_identity():
    cal = Calibration.from_checkpoint({"model_state": {}}, HORIZONS)

    assert cal.is_identity and cal.method == "none"
    assert Calibration.from_checkpoint({"calibration": None}, HORIZONS).is_identity


def test_from_dict_refuses_other_horizons_and_malformed_data():
    data = sample_calibration().as_dict()

    with pytest.raises(ValueError, match="model predicts"):
        Calibration.from_dict(data, ("30s", "5m"))
    with pytest.raises(ValueError, match="malformed calibration"):
        Calibration.from_dict({"horizons": list(HORIZONS), "pump": {}}, HORIZONS)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pump_temperature": (0.0, 1.0)}, "temperature must be > 0"),
        ({"rug_temperature": (1.0, -2.0)}, "temperature must be > 0"),
        ({"pump_bias": (float("nan"), 0.0)}, "must be finite"),
        ({"rug_bias": (0.0,)}, "has 1 entries, expected 2"),
        ({"method": "isotonic"}, "unknown calibration method"),
    ],
)
def test_invalid_calibrations_are_rejected(kwargs, message):
    fields = {
        "pump_temperature": (1.0, 1.0),
        "pump_bias": (0.0, 0.0),
        "rug_temperature": (1.0, 1.0),
        "rug_bias": (0.0, 0.0),
        "method": "platt",
    } | kwargs
    with pytest.raises(ValueError, match=message):
        Calibration(HORIZONS, **fields)


def test_uniform_and_identity():
    assert Calibration.identity(HORIZONS).uniform_temperature == 1.0
    uniform = Calibration.uniform(HORIZONS, 0.5)
    assert uniform.uniform_temperature == 0.5 and not uniform.is_identity
    assert sample_calibration().uniform_temperature is None
    with pytest.raises(ValueError, match="temperature must be > 0"):
        Calibration.uniform(HORIZONS, 0.0)


def test_params_are_per_horizon_tensors():
    temperature, bias = sample_calibration().params("pump")

    assert torch.equal(temperature, torch.tensor([2.0, 0.5]))
    assert torch.equal(bias, torch.tensor([-1.0, 0.25]))
    with pytest.raises(ValueError, match="unknown head"):
        sample_calibration().params("return")


@pytest.mark.parametrize(
    ("pump_t", "pump_b", "method"),
    [
        ((1.0, 1.0), (0.0, 0.0), "none"),
        ((2.0, 1.0), (0.0, 0.0), "temperature"),
        ((2.0, 1.0), (0.5, 0.0), "platt"),
    ],
)
def test_a_missing_method_is_inferred_from_the_parameters(pump_t, pump_b, method):
    data = sample_calibration().as_dict()
    data["pump"] = {"temperature": list(pump_t), "bias": list(pump_b)}
    data["rug"] = {"temperature": [1.0, 1.0], "bias": [0.0, 0.0]}
    del data["method"]

    assert Calibration.from_dict(data, HORIZONS).method == method


def test_check_horizons_refuses_a_calibration_for_other_horizons():
    cal = sample_calibration()

    assert cal.check_horizons(HORIZONS) is cal
    with pytest.raises(ValueError, match=r"is for horizons \['30s', '1m'\]"):
        cal.check_horizons(("1m", "30s"))


def test_settings_fill_in_the_defaults():
    assert settings({}) == {
        "method": "platt",
        "fraction": 0.5,
        "min_samples": 50,
        "min_class_samples": 10,
    }
    assert settings({"calibration": {"method": "none"}})["method"] == "none"
