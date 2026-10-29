"""Promotion gate: does a freshly trained candidate replace the champion?

Promoting is the riskiest step of continuous training: a *luck-driven false champion* degrades
every later paper trade. The gate follows docs/ML-MODEL.md "Promotion gate", with the same rules as
the ablation harness and the hyper-parameter search:

* **Same runs, paired.** Candidate and champion are scored on the same rolling held-out windows
  (:func:`solia_ai.ablation.walk_forward_windows`) under the same seeds, so each ``(window, seed)``
  run gives one paired lift *candidate − champion* (positive = better, ECE negated). Metric names
  are the ablation's (:func:`solia_ai.ablation.run_metrics`): ``<head>/<horizon>/<metric>`` and
  ``policy/<metric>``, ``None`` where undefined.
* **Primary metric is predictive** (default PR-AUC of the pump head on the main horizon). The mean
  lift must clear ``margin`` *and* the lift must clear it on at least ``min_win_share`` of the runs
  — a lift carried by one lucky window is not a lift.
* **Guard rails.** The candidate is rejected when calibration (ECE) or any other predictive metric
  (ROC-AUC / PR-AUC on any head and horizon) regresses beyond its tolerance, on average over runs.
* **P&L is a sanity check, never a reason to promote.** The reference-policy P&L lift is reported
  with a bootstrap confidence interval (over runs); with ``pnl_veto`` a lift whose whole interval
  lies below zero rejects the candidate.
* **No champion** → the first *valid* candidate is promoted (``initial``). A candidate with a
  missing or non-finite primary metric, a non-finite value anywhere, or runs that are not exactly
  ``windows`` distinct windows × ``seeds`` distinct seeds is rejected, never promoted; so is one
  compared with a champion whose stored metrics are non-finite (the comparison cannot be trusted).

Everything here is pure (no torch, no database, no training): the caller (the continuous-training
loop) scores both models and writes :meth:`PromotionDecision.metrics` to ``training_events``.
:func:`check_rollback` validates the opposite move — restoring the previous champion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import PROMOTION_DEFAULTS, PROMOTION_HEADS, validate_config
from .features import FEATURE_COLUMNS, is_ordered_subset
from .lift import PREDICTIVE_METRICS, lift, summarize
from .registry import ModelRecord

PNL_METRIC = "policy/total_pnl"
# Slack on the win-share comparison, so a rounded two thirds (0.6667) accepts 6 wins out of 9.
WIN_SHARE_TOLERANCE = 1e-3
# Checkpoint metadata a rollback target must carry to be verified (solia_ai.train writes them all).
ROLLBACK_METADATA = ("version", "feature_columns", "horizons")

RunKey = tuple[int, int]  # (window index, seed)
RunMetrics = Mapping[RunKey, Mapping[str, float | None]]


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class PromotionSettings:
    """The ``config.yaml`` ``promotion`` knobs (see the file for each one's meaning)."""

    horizons: tuple[str, ...]
    metric: str
    head: str
    horizon: str
    margin: float
    windows: int
    seeds: int
    min_win_share: float
    ece_tolerance: float
    secondary_tolerance: float
    pnl_veto: bool
    bootstrap_samples: int
    confidence: float
    seed: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PromotionSettings:
        """Validated settings (:class:`solia_ai.config.ConfigError` on bad values); an omitted
        knob keeps its default, an omitted horizon is the middle configured one."""
        validate_config(config)
        horizons = tuple(config["horizons"])
        knobs = {**PROMOTION_DEFAULTS, **config.get("promotion", {})}
        if knobs["horizon"] is None:
            knobs["horizon"] = horizons[len(horizons) // 2]
        return cls(horizons=horizons, **knobs)

    @property
    def primary_key(self) -> str:
        return f"{self.head}/{self.horizon}/{self.metric}"

    @property
    def expected_runs(self) -> int:
        return self.windows * self.seeds

    def guard_keys(self) -> list[str]:
        """The predictive metrics checked for regression: every head, horizon and family except
        the primary one (never P&L, never the uncalibrated ``raw/`` twins)."""
        return [
            name
            for head in PROMOTION_HEADS
            for horizon in self.horizons
            for metric in PREDICTIVE_METRICS
            if (name := f"{head}/{horizon}/{metric}") != self.primary_key
        ]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["horizons"] = list(self.horizons)
        return out


# --------------------------------------------------------------------------- statistics


def bootstrap_ci(
    values: Sequence[float], *, samples: int, confidence: float, seed: int
) -> dict[str, Any]:
    """Mean of ``values`` with a percentile-bootstrap confidence interval of that mean.

    ``verdict`` is ``better`` / ``worse`` when the whole interval lies above / below zero,
    ``inconclusive`` otherwise; fewer than two values have no interval (``low``/``high`` ``None``).
    """
    data = np.asarray(values, dtype=np.float64)
    n = int(data.size)
    out: dict[str, Any] = {
        "n": n,
        "mean": float(data.mean()) if n else None,
        "confidence": confidence,
    }
    if n < 2:
        return {**out, "low": None, "high": None, "verdict": "inconclusive"}
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, n, size=(samples, n))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = (float(q) for q in np.quantile(means, [tail, 1.0 - tail]))
    verdict = "better" if low > 0 else "worse" if high < 0 else "inconclusive"
    return {**out, "low": low, "high": high, "verdict": verdict}


def _defined(value: Any) -> bool:
    return value is not None and math.isfinite(value)


# --------------------------------------------------------------------------- decision


@dataclass(frozen=True)
class PromotionDecision:
    """The gate's verdict: ``promote`` or not, why, and the comparison to audit."""

    promote: bool
    initial: bool  # promoted because there was no champion yet
    reasons: tuple[str, ...]
    summary: dict[str, Any]

    @property
    def decision(self) -> str:
        return "promote" if self.promote else "reject"

    @property
    def event_decision(self) -> str:
        """The ``training_events.decision`` value (:data:`solia_ai.registry.TRAINING_DECISIONS`)."""
        return "promoted" if self.promote else "rejected"

    def metrics(self) -> dict[str, Any]:
        """The JSON object stored in ``training_events.metrics`` (finite numbers only)."""
        return {
            "decision": self.decision,
            "initial": self.initial,
            "reasons": list(self.reasons),
            **self.summary,
        }


def decide_promotion(
    candidate: RunMetrics,
    champion: RunMetrics | None,
    settings: PromotionSettings,
    *,
    windows: Sequence[Mapping[str, Any]] | None = None,
) -> PromotionDecision:
    """Compare ``candidate`` with ``champion`` (``None`` = no champion yet), both keyed by
    ``(window, seed)`` run. ``windows`` (e.g. :meth:`solia_ai.ablation.Window.describe`) is only
    recorded, so the audit shows which held-out data each decision used.

    Raises:
        ValueError: no candidate run, or the champion was scored on other runs than the candidate
            (an unpaired comparison is a caller bug, not a verdict).
    """
    if not candidate:
        raise ValueError("no candidate run to decide on")
    if champion is not None and set(champion) != set(candidate):
        raise ValueError(
            "candidate and champion must be scored on the same (window, seed) runs: "
            f"{sorted(candidate)} vs {sorted(champion)}"
        )
    runs = sorted(candidate)
    key = settings.primary_key
    summary: dict[str, Any] = {
        "settings": settings.as_dict(),
        "primary": key,
        "n_runs": len(runs),
        "windows": [dict(w) for w in windows or ()],
    }

    def reject(*reasons: str) -> PromotionDecision:
        return PromotionDecision(False, False, tuple(reasons), summary)

    def non_finite(who: str, scored: RunMetrics) -> list[str]:
        return [
            f"{who} {name} is not finite on run {run}: {value!r}"
            for run in runs
            for name, value in scored[run].items()
            if value is not None and not math.isfinite(value)
        ]

    invalid = non_finite("candidate", candidate)
    invalid += [
        f"candidate {key} is missing on run {run}"
        for run in runs
        if candidate[run].get(key) is None
    ]
    if invalid:
        return reject(*invalid)
    n_windows = len({window for window, _ in runs})
    n_seeds = len({seed for _, seed in runs})
    if (n_windows, n_seeds, len(runs)) != (
        settings.windows,
        settings.seeds,
        settings.expected_runs,
    ):
        return reject(
            f"{len(runs)} run(s) over {n_windows} window(s) x {n_seeds} seed(s): the gate needs "
            f"windows x seeds = {settings.windows} x {settings.seeds} distinct runs"
        )

    if champion is None:
        values = [candidate[run][key] for run in runs]
        summary["candidate"] = {"mean": float(np.mean(values))}
        return PromotionDecision(True, True, ("no champion yet: first valid candidate",), summary)

    corrupt = non_finite("champion", champion)
    if corrupt:
        return reject(*corrupt)
    guard_keys = settings.guard_keys()
    missing = sorted(
        {
            name
            for run in runs
            for name in guard_keys
            if _defined(champion[run].get(name)) and candidate[run].get(name) is None
        }
    )
    if missing:
        return reject(f"candidate lacks metric(s) the champion has: {', '.join(missing)}")
    undefined = [run for run in runs if not _defined(champion[run].get(key))]
    if undefined:
        return reject(f"champion {key} is undefined on run(s) {undefined}: no paired comparison")

    reasons: list[str] = []

    # Primary: mean lift over the margin, and on enough of the runs.
    lifts = [lift(candidate[run][key], champion[run][key], key) for run in runs]
    stats = summarize(lifts)
    mean = stats["mean"]
    wins = sum(value > settings.margin for value in lifts)
    win_share = wins / len(lifts)
    summary["primary_lift"] = {
        **stats,
        "wins": wins,
        "win_share": win_share,
        "runs": [
            {
                "window": run[0],
                "seed": run[1],
                "candidate": candidate[run][key],
                "champion": champion[run][key],
                "lift": value,
            }
            for run, value in zip(runs, lifts, strict=True)
        ],
    }
    if not mean > settings.margin:
        reasons.append(f"{key} mean lift {mean:+.4f} does not clear the margin {settings.margin}")
    if win_share + WIN_SHARE_TOLERANCE < settings.min_win_share:
        reasons.append(
            f"{key} clears the margin on {wins}/{len(lifts)} runs "
            f"(needs a share of {settings.min_win_share:.4g})"
        )

    # Guard rails: no calibration or secondary predictive regression beyond tolerance.
    guards: dict[str, float] = {}
    for name in guard_keys:
        paired = [
            lift(candidate[run][name], champion[run][name], name)
            for run in runs
            if _defined(candidate[run].get(name)) and _defined(champion[run].get(name))
        ]
        if not paired:
            continue
        guards[name] = float(np.mean(paired))
        tolerance = (
            settings.ece_tolerance if name.endswith("/ece") else settings.secondary_tolerance
        )
        if -guards[name] > tolerance:
            reasons.append(f"{name} regresses by {-guards[name]:.4f} (tolerance {tolerance})")
    summary["guard_lifts"] = guards

    # P&L: sanity check only — can veto, never promote.
    pnl = [
        candidate[run][PNL_METRIC] - champion[run][PNL_METRIC]
        for run in runs
        if _defined(candidate[run].get(PNL_METRIC)) and _defined(champion[run].get(PNL_METRIC))
    ]
    summary["pnl_lift"] = bootstrap_ci(
        pnl,
        samples=settings.bootstrap_samples,
        confidence=settings.confidence,
        seed=settings.seed,
    )
    if settings.pnl_veto and summary["pnl_lift"]["verdict"] == "worse":
        ci = summary["pnl_lift"]
        reasons.append(
            f"reference-policy P&L lift is significantly negative "
            f"({ci['confidence']:.0%} CI [{ci['low']:+.4f}, {ci['high']:+.4f}])"
        )

    if reasons:
        return reject(*reasons)
    return PromotionDecision(
        True,
        False,
        (
            f"{key} lift {mean:+.4f} over margin {settings.margin} on {wins}/{len(lifts)} "
            "runs, no guard-rail regression",
        ),
        summary,
    )


# --------------------------------------------------------------------------- rollback


@dataclass(frozen=True)
class RollbackCheck:
    """Whether the previous champion can be restored (``target``), and if not, why."""

    possible: bool
    reasons: tuple[str, ...]
    target: ModelRecord | None


def check_rollback(
    current: ModelRecord | None,
    previous: ModelRecord | None,
    *,
    checkpoint: Mapping[str, Any] | None,
    horizons: Sequence[str],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> RollbackCheck:
    """Validate rolling back from the ``current`` champion to the ``previous`` one.

    ``checkpoint`` is the previous champion's loaded checkpoint (its metadata suffices; ``None``
    when the file is missing). Possible only when both champions exist and differ, the checkpoint
    is that version's, its feature spec still loads on ``feature_columns`` (an ordered subset, as
    :func:`solia_ai.inference.build_model` requires) and it serves the configured ``horizons``.
    A checkpoint that does not record its version, feature spec or horizons cannot be verified,
    so it is refused rather than trusted; one the retention policy pruned (``pruned_at``) is gone.
    """
    reasons: list[str] = []
    if current is None:
        reasons.append("no current champion to roll back from")
    if previous is None:
        reasons.append("no previous champion to roll back to")
        return RollbackCheck(False, tuple(reasons), None)
    if current is not None and previous.id == current.id:
        reasons.append(f"model {previous.version} is already the champion")
    if previous.pruned_at is not None:
        reasons.append(
            f"checkpoint of {previous.version} was pruned by the retention policy at "
            f"{previous.pruned_at.isoformat()}"
        )
    elif checkpoint is None:
        reasons.append(f"checkpoint of {previous.version} not found at {previous.path}")
    else:
        unrecorded = [name for name in ROLLBACK_METADATA if checkpoint.get(name) is None]
        if unrecorded:
            reasons.append(
                f"checkpoint at {previous.path} does not record its {', '.join(unrecorded)}: "
                "cannot verify it"
            )
        version = checkpoint.get("version")
        if version is not None and version != previous.version:
            reasons.append(f"checkpoint at {previous.path} is {version}, not {previous.version}")
        columns = checkpoint.get("feature_columns")
        if columns is not None and not is_ordered_subset(columns, feature_columns):
            reasons.append(
                f"{previous.version}'s feature spec no longer loads on the current one: "
                "retrain instead"
            )
        trained = checkpoint.get("horizons")
        if trained is not None and list(trained) != list(horizons):
            reasons.append(
                f"{previous.version} serves horizons {list(trained)}, not {list(horizons)}"
            )
    return RollbackCheck(not reasons, tuple(reasons), previous if not reasons else None)
