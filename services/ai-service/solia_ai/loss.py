"""The training loss: a weighted, masked, multi-task / multi-horizon objective.

Mirrors :class:`solia_ai.model.SoliaTransformer`'s outputs (per horizon): ``pump`` / ``rug``
classification and ``expected_return`` regression (see docs/ML-MODEL.md):

* **pump / rug** — ``alpha * BCE + (1 - alpha) * Focal(gamma)``. The focal term down-weights the
  easy, abundant negatives so the rare pumps/rugs still drive the gradient; an optional
  per-horizon ``pos_weight`` (from :func:`solia_ai.dataset.positive_weights`) up-weights the
  positive class on top of that.
* **expected_return** — Huber (smooth L1), robust to the heavy tails of token returns.

Every term is **masked**: a horizon contributes only where the dataset could label it
(``targets["mask"] == 1``), so unlabeled horizons never inject a spurious gradient. Terms are a
mask-weighted mean, then combined with per-task weights (``strategy_weight`` for pump+rug,
``regression_weight`` for the return) and optional per-horizon weights — all from ``config.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

TASKS = ("pump", "rug", "return")


def focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """``alpha * BCE + (1 - alpha) * Focal(gamma)`` per element (no reduction).

    ``logits``/``target`` share a shape; ``pos_weight`` (optional) broadcasts over the last axis
    and scales the positive class in the BCE term. With ``alpha=1`` this is plain BCE; with
    ``gamma=0`` the focal term degenerates to BCE too.
    """
    # Focal uses the unweighted cross-entropy: p_t = exp(-ce) is the probability of the true class,
    # computed from ce for numerical stability (no separate sigmoid that could over/underflow).
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (
        ce
        if pos_weight is None
        else F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight
        )
    )
    focal = (1.0 - torch.exp(-ce)) ** gamma * ce
    return alpha * bce + (1.0 - alpha) * focal


def _masked_mean(per_element: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of ``[B, H]`` losses by ``[B, H]`` weights; 0 (graph-safe) when all-zero."""
    denom = weight.sum()
    return (per_element * weight).sum() / torch.where(denom > 0, denom, torch.ones_like(denom))


def _horizon_weight_tensor(
    horizons: Sequence[str], spec: Mapping[str, float] | Sequence[float] | None
) -> torch.Tensor:
    if spec is None:
        return torch.ones(len(horizons))
    if isinstance(spec, Mapping):
        unknown = [str(h) for h in spec if h not in horizons]
        if unknown:
            raise ValueError(
                f"horizon_weights has unknown horizon(s): {', '.join(unknown)} "
                f"(expected a subset of {', '.join(horizons)})"
            )
        values = [float(spec.get(h, 1.0)) for h in horizons]
    else:
        values = [float(v) for v in spec]
        if len(values) != len(horizons):
            raise ValueError(f"horizon_weights has {len(values)} entries, expected {len(horizons)}")
    if any(v < 0 for v in values):
        raise ValueError(f"horizon_weights must be non-negative, got {values}")
    return torch.tensor(values, dtype=torch.float32)


class SoliaLoss(nn.Module):
    """Weighted sum of the masked pump/rug (BCE+Focal) and return (Huber) terms.

    :meth:`forward` takes the model's output dict (``pump_logits`` / ``rug_logits`` /
    ``expected_return``) and the dataset's target dict (``pump`` / ``rug`` / ``expected_return`` /
    ``mask``) and returns a dict of scalar tensors: the three task components plus ``total`` (the
    value to back-propagate). Weights, ``pos_weight`` and ``horizon_weights`` register as buffers,
    so ``loss.to(device)`` moves them alongside the model.
    """

    def __init__(
        self,
        horizons: Sequence[str],
        *,
        alpha: float,
        focal_gamma: float,
        task_weights: Mapping[str, float] | None = None,
        horizon_weights: Mapping[str, float] | Sequence[float] | None = None,
        pos_weight: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        horizons = tuple(horizons)
        if not horizons:
            raise ValueError("at least one horizon is required")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {focal_gamma}")

        self.horizons = horizons
        self.alpha = float(alpha)
        self.gamma = float(focal_gamma)

        weights = {task: 1.0 for task in TASKS} | dict(task_weights or {})
        unknown = set(weights) - set(TASKS)
        if unknown:
            raise ValueError(f"unknown task weight(s): {', '.join(sorted(unknown))}")
        if any(weights[task] < 0 for task in TASKS):
            raise ValueError(f"task weights must be non-negative, got {weights}")
        self.task_weights = {task: float(weights[task]) for task in TASKS}

        self.register_buffer("horizon_weights", _horizon_weight_tensor(horizons, horizon_weights))
        for name in ("pump", "rug"):
            weight = None if pos_weight is None else pos_weight.get(name)
            if weight is not None:
                weight = torch.as_tensor(weight, dtype=torch.float32)
                if weight.shape != (len(horizons),):
                    raise ValueError(
                        f"{name} pos_weight shape {tuple(weight.shape)} != ({len(horizons)},)"
                    )
            self.register_buffer(f"{name}_pos_weight", weight)

    def label_weight(self, targets: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[B, H]`` weight of each entry in the masked means: the label mask times the
        per-horizon weights. Its sum is the denominator every component shares."""
        return targets["mask"] * self.horizon_weights

    def forward(
        self, outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        weight = self.label_weight(targets)
        components = {
            "pump": _masked_mean(
                focal_bce_loss(
                    outputs["pump_logits"],
                    targets["pump"],
                    alpha=self.alpha,
                    gamma=self.gamma,
                    pos_weight=self.pump_pos_weight,
                ),
                weight,
            ),
            "rug": _masked_mean(
                focal_bce_loss(
                    outputs["rug_logits"],
                    targets["rug"],
                    alpha=self.alpha,
                    gamma=self.gamma,
                    pos_weight=self.rug_pos_weight,
                ),
                weight,
            ),
            "return": _masked_mean(
                F.smooth_l1_loss(
                    outputs["expected_return"], targets["expected_return"], reduction="none"
                ),
                weight,
            ),
        }
        components["total"] = sum(
            self.task_weights[task] * components[task] for task in TASKS
        )  # type: ignore[assignment]
        return components

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        pos_weight: Mapping[str, torch.Tensor] | None = None,
    ) -> SoliaLoss:
        """Build from a loaded ``config.yaml`` (``strategy_weight`` weights pump+rug, the
        classification signal; ``regression_weight`` weights the return)."""
        loss = config["loss"]
        task_weights = {
            "pump": loss["strategy_weight"],
            "rug": loss["strategy_weight"],
            "return": loss["regression_weight"],
        }
        return cls(
            tuple(config["horizons"]),
            alpha=loss["alpha"],
            focal_gamma=loss["focal_gamma"],
            task_weights=task_weights,
            horizon_weights=loss.get("horizon_weights"),
            pos_weight=pos_weight,
        )
