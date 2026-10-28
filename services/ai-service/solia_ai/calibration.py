"""Post-hoc probability calibration, per head (pump / rug) and horizon.

The bot acts on probability thresholds (docs/ML-MODEL.md "Policy layer"), so the served
probabilities must mean what they say. A trained network rarely does that on its own — and with
class-balanced batches (``training.balanced_sampling``) its probabilities are biased toward the
50% event rate it saw, not the real base rate. Calibration fixes that after training, without
touching the weights:

    p = sigmoid(logit / temperature + bias)

fitted per head and horizon by minimizing the log-loss on a **calibration window** — the earliest
part of the validation split, strictly before the windows ``evaluate`` scores (see
:mod:`solia_ai.train`), so the reported ECE is never measured on the data it was fit on.

* ``temperature`` — temperature scaling: only ``temperature`` is fit (``bias = 0``).
* ``platt`` — Platt scaling: ``temperature`` and ``bias`` (also corrects a base-rate shift, which a
  temperature alone cannot).
* ``none`` — identity (``temperature = 1``, ``bias = 0``).

Both maps are strictly increasing (``temperature > 0``): calibration only rescales the
probabilities, never re-ranks them. In floating point a sigmoid saturates to exactly 0 or 1 at large
``|logit / T + b|``, which would tie distinct scores, so the fit is bounded to a mild map
(``T`` in [0.1, 10], ``|b| <= 10``) and :mod:`solia_ai.evaluate` scores in float64: ROC-AUC /
PR-AUC are then unchanged for any logit the model realistically emits. A head/horizon with too few
labeled samples, or too few of its minority class (a rare event with a handful of positives),
keeps the identity rather than fitting noise. A checkpoint saved before calibration existed loads
as the identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from .config import CALIBRATION_DEFAULT_METHOD as DEFAULT_METHOD
from .config import CALIBRATION_METHODS as METHODS

HEADS = ("pump", "rug")
DEFAULT_FRACTION = 0.5
DEFAULT_MIN_SAMPLES = 50
DEFAULT_MIN_CLASS_SAMPLES = 10

# Bounds keep a degenerate window (perfectly separable labels) from driving the temperature to 0
# or the bias to infinity — which would saturate the sigmoid and tie distinct scores; a sane fit
# sits well inside them (a 10x sharpening, a base-rate shift of e^10 in odds).
_LOG_T_BOUNDS = (math.log(0.1), math.log(10.0))
_BIAS_BOUNDS = (-10.0, 10.0)


def _check_temperature(temperature: float) -> float:
    if not temperature > 0:  # also rejects NaN
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return float(temperature)


@dataclass(frozen=True)
class Calibration:
    """Per-horizon ``temperature`` / ``bias`` for each head, in the model's horizon order."""

    horizons: tuple[str, ...]
    pump_temperature: tuple[float, ...]
    pump_bias: tuple[float, ...]
    rug_temperature: tuple[float, ...]
    rug_bias: tuple[float, ...]
    method: str = "none"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method {self.method!r} (known: {METHODS})")
        n = len(self.horizons)
        for name in ("pump_temperature", "pump_bias", "rug_temperature", "rug_bias"):
            values = getattr(self, name)
            if len(values) != n:
                raise ValueError(f"calibration {name} has {len(values)} entries, expected {n}")
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"calibration {name} must be finite, got {list(values)}")
        for name in ("pump_temperature", "rug_temperature"):
            for value in getattr(self, name):
                _check_temperature(value)

    @classmethod
    def identity(cls, horizons: Sequence[str]) -> Calibration:
        return cls.uniform(horizons, 1.0)

    @classmethod
    def uniform(cls, horizons: Sequence[str], temperature: float) -> Calibration:
        """One temperature for every head and horizon, no bias (the legacy ``--temperature``)."""
        temperature = _check_temperature(temperature)
        n = len(horizons)
        method = "none" if temperature == 1.0 else "temperature"
        return cls(
            tuple(horizons), (temperature,) * n, (0.0,) * n, (temperature,) * n, (0.0,) * n, method
        )

    @property
    def is_identity(self) -> bool:
        return all(t == 1.0 for t in self.pump_temperature + self.rug_temperature) and all(
            b == 0.0 for b in self.pump_bias + self.rug_bias
        )

    def check_horizons(self, horizons: Sequence[str]) -> Calibration:
        """``self``, if it was fit for exactly the model's ``horizons`` (in order); raises
        otherwise rather than mis-applying a horizon's parameters to another."""
        if tuple(self.horizons) != tuple(horizons):
            raise ValueError(
                f"calibration is for horizons {list(self.horizons)} but the model predicts "
                f"{list(horizons)}"
            )
        return self

    @property
    def uniform_temperature(self) -> float | None:
        """The single temperature when every head/horizon shares one and no bias is set."""
        temps = set(self.pump_temperature + self.rug_temperature)
        if len(temps) == 1 and all(b == 0.0 for b in self.pump_bias + self.rug_bias):
            return temps.pop()
        return None

    def params(self, head: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(temperature[H], bias[H])`` float32 tensors of ``head``, broadcast over the batch."""
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}")
        temperature = getattr(self, f"{head}_temperature")
        bias = getattr(self, f"{head}_bias")
        return torch.tensor(temperature, dtype=torch.float32), torch.tensor(
            bias, dtype=torch.float32
        )

    def as_dict(self) -> dict[str, Any]:
        """The JSON-able form stored in checkpoints (plain lists/floats: ``weights_only`` safe)."""
        return {
            "method": self.method,
            "horizons": list(self.horizons),
            **{
                head: {
                    "temperature": list(getattr(self, f"{head}_temperature")),
                    "bias": list(getattr(self, f"{head}_bias")),
                }
                for head in HEADS
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], horizons: Sequence[str]) -> Calibration:
        """Rebuild from :meth:`as_dict`, re-checked against the model's ``horizons``. A missing
        ``method`` is inferred from the parameters (identity -> ``none``, no bias ->
        ``temperature``, else ``platt``), so a fitted map is never labeled as no calibration."""
        try:
            fields = {
                f"{head}_{key}": tuple(float(v) for v in data[head][key])
                for head in HEADS
                for key in ("temperature", "bias")
            }
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed calibration: {exc!r}") from None
        recorded = tuple(data.get("horizons", ()))
        method = data.get("method")
        if method is None:
            biased = any(b != 0.0 for head in HEADS for b in fields[f"{head}_bias"])
            scaled = any(t != 1.0 for head in HEADS for t in fields[f"{head}_temperature"])
            method = "platt" if biased else "temperature" if scaled else "none"
        return cls(recorded, method=str(method), **fields).check_horizons(horizons)

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any], horizons: Sequence[str]) -> Calibration:
        """The checkpoint's calibration, or the identity for one saved before calibration."""
        data = checkpoint.get("calibration")
        return cls.identity(horizons) if data is None else cls.from_dict(data, horizons)


# --------------------------------------------------------------------------- fitting


def _log_loss(logits: np.ndarray, labels: np.ndarray, log_t: float, bias: float) -> float:
    z = logits / math.exp(log_t) + bias
    # log(1 + e^z) - y*z, computed stably.
    return float(np.mean(np.logaddexp(0.0, z) - labels * z))


def fit_head(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = DEFAULT_METHOD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_class_samples: int = DEFAULT_MIN_CLASS_SAMPLES,
) -> tuple[float, float]:
    """``(temperature, bias)`` minimizing the log-loss of ``sigmoid(logits / T + b)``.

    Returns the identity ``(1.0, 0.0)`` for ``method='none'``, fewer than ``min_samples`` samples,
    or fewer than ``min_class_samples`` of the minority class (a single class included): a fit
    driven by one or two positives would only learn to push every probability toward 0.
    """
    if method not in METHODS:
        raise ValueError(f"unknown calibration method {method!r} (known: {METHODS})")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    positives = int(labels.sum())
    minority = min(positives, labels.size - positives)
    if method == "none" or labels.size < min_samples or minority < max(1, min_class_samples):
        return 1.0, 0.0

    if method == "temperature":
        result = minimize(
            lambda v: _log_loss(logits, labels, v[0], 0.0),
            x0=[0.0],
            method="L-BFGS-B",
            bounds=[_LOG_T_BOUNDS],
        )
        return math.exp(float(result.x[0])), 0.0
    result = minimize(
        lambda v: _log_loss(logits, labels, v[0], v[1]),
        x0=[0.0, 0.0],
        method="L-BFGS-B",
        bounds=[_LOG_T_BOUNDS, _BIAS_BOUNDS],
    )
    return math.exp(float(result.x[0])), float(result.x[1])


def fit_calibration(
    logits: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    horizons: Sequence[str],
    *,
    method: str = DEFAULT_METHOD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_class_samples: int = DEFAULT_MIN_CLASS_SAMPLES,
) -> Calibration:
    """Fit every head/horizon on its labeled samples (``truth['mask'] == 1``).

    ``logits`` holds the raw ``pump_logits`` / ``rug_logits`` ``[N, H]`` and ``truth`` the
    ``pump`` / ``rug`` / ``mask`` labels, aligned row for row.
    """
    horizons = tuple(horizons)
    fields: dict[str, list[float]] = {}
    for head in HEADS:
        temps, biases = [], []
        for h in range(len(horizons)):
            labeled = truth["mask"][:, h] == 1
            t, b = fit_head(
                logits[f"{head}_logits"][:, h][labeled],
                truth[head][:, h][labeled],
                method=method,
                min_samples=min_samples,
                min_class_samples=min_class_samples,
            )
            temps.append(t)
            biases.append(b)
        fields[f"{head}_temperature"] = temps
        fields[f"{head}_bias"] = biases
    return Calibration(horizons, method=method, **{k: tuple(v) for k, v in fields.items()})


def settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``calibration`` section of a loaded config, with defaults filled in."""
    section = dict(config.get("calibration") or {})
    return {
        "method": section.get("method", DEFAULT_METHOD),
        "fraction": float(section.get("fraction", DEFAULT_FRACTION)),
        "min_samples": int(section.get("min_samples", DEFAULT_MIN_SAMPLES)),
        "min_class_samples": int(section.get("min_class_samples", DEFAULT_MIN_CLASS_SAMPLES)),
    }
