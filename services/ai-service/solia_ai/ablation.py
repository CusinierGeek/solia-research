"""Ablation harness: does a feature group actually make the model better?

Every enrichment must earn its place by **measured lift** (docs/ML-MODEL.md "Ablation"). This
module trains the model with feature groups (:data:`solia_ai.features.FEATURE_GROUPS`) enabled or
disabled and scores every variant on the **same** held-out, most-recent windows, so the only thing
that differs between two runs is the feature set:

* **Variants** — the baseline (:data:`BASELINE_GROUPS`, the original 19 columns), plus one variant
  per group under test: a group outside the baseline is *added* to it, a baseline group is *dropped*
  from it. A disabled group's columns are zeroed in the input (no signal); the architecture and
  everything else stay identical.
* **Windows** — walk-forward: the most recent ``eval_fraction`` of the labeled windows is cut into
  ``n_windows`` consecutive time slices. Each slice is scored by a model trained only on what ends
  before it, minus the label embargo; early stopping uses a temporal validation split *of that
  training pool*, never the scored slice. No shuffling across time, so nothing leaks.
* **Seeds** — every (window, variant) is trained under each seed, so a lift has to survive both
  eval-sample luck and initialization luck.

Metrics are not re-implemented: each run goes through :func:`solia_ai.evaluate.evaluate`. The
primary metrics are predictive — ROC-AUC, PR-AUC, ECE per head and horizon — and the
reference-policy P&L is secondary (the promotion-gate rule). The lift of a group is the paired
difference *with group − without group* over the same (window, seed) runs, oriented so positive
is better (ECE is negated). It is flagged significant only when it clears the margin by more than
one standard deviation of those paired differences.

Each run is calibrated (``config.yaml`` ``calibration``) on its early-stopping validation split —
never the scored slice — and scored calibrated, as served. The raw (uncalibrated) metrics are kept
too, and the report's ``calibration`` section pairs calibrated vs raw over every run: a strictly
increasing map can't move ROC-AUC / PR-AUC, so only ECE should change.

Ablation runs never save a checkpoint, register a model, or touch ``is_champion``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import settings as calibration_settings
from .dataset import SequenceDataset, config_samples, embargo_seconds, temporal_split
from .evaluate import EvaluationReport, ReferencePolicy, evaluate, fmt, json_safe
from .features import BASELINE_GROUPS, FEATURE_GROUPS, check_group_coverage
from .lift import PREDICTIVE_METRICS, lift, summarize
from .train import fit, fit_calibration_window

HEADS = ("pump", "rug")
POLICY_METRICS = ("total_pnl", "avg_pnl")
RAW_PREFIX = "raw/"

DEFAULT_WINDOWS = 3
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_EVAL_FRACTION = 0.3
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_MARGIN = 0.01
DEFAULT_PNL_MARGIN = 0.0


# --------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Variant:
    """A feature set to train: ``groups`` enabled, ``group`` the one under test (``None`` for the
    baseline), ``mode`` ``'baseline'`` / ``'add'`` / ``'drop'``."""

    name: str
    groups: tuple[str, ...]
    group: str | None = None
    mode: str = "baseline"


def plan_variants(
    candidates: Iterable[str] | None = None, *, baseline: Sequence[str] = BASELINE_GROUPS
) -> list[Variant]:
    """The baseline followed by one variant per candidate group (default: every declared group).

    A candidate outside ``baseline`` is added to it (``+group``); a baseline group is dropped from
    it (``-group``). Raises on an unknown group, an empty baseline, or when dropping would leave no
    feature at all.
    """
    baseline = tuple(dict.fromkeys(baseline))
    if not baseline:
        raise ValueError("the baseline needs at least one feature group")
    candidates = list(FEATURE_GROUPS if candidates is None else dict.fromkeys(candidates))
    unknown = [g for g in [*baseline, *candidates] if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature group(s): {', '.join(unknown)} "
            f"(known: {', '.join(FEATURE_GROUPS)})"
        )
    variants = [Variant("baseline", baseline)]
    for group in candidates:
        if group in baseline:
            rest = tuple(g for g in baseline if g != group)
            if not rest:
                raise ValueError(f"dropping {group!r} would leave the model with no feature")
            variants.append(Variant(f"-{group}", rest, group, "drop"))
        else:
            variants.append(Variant(f"+{group}", (*baseline, group), group, "add"))
    return variants


def check_variants(variants: Sequence[Variant]) -> None:
    """Raise unless ``variants`` starts with the baseline and every name is unique (runs, summaries
    and lifts are keyed by name)."""
    if not variants or variants[0].group is not None:
        raise ValueError("the first variant must be the baseline")
    names = [v.name for v in variants]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate variant name(s): {', '.join(duplicates)}")


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class Window:
    """One walk-forward fold: sample indices to train, early-stop and score on."""

    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    eval_idx: np.ndarray
    eval_start_ts: int  # scored windows end strictly after this
    eval_end_ts: int  # ... and at or before this

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "n_train": int(len(self.train_idx)),
            "n_val": int(len(self.val_idx)),
            "n_eval": int(len(self.eval_idx)),
            "eval_start": _iso(self.eval_start_ts),
            "eval_end": _iso(self.eval_end_ts),
        }


def walk_forward_windows(
    end_ts: np.ndarray,
    *,
    n_windows: int = DEFAULT_WINDOWS,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    embargo: int = 0,
    eval_end_ts: np.ndarray | None = None,
) -> list[Window]:
    """Cut the most recent ``eval_fraction`` of samples (by ``end_ts``) into ``n_windows`` slices.

    Slice ``k`` scores the samples ending in ``(b_k, b_k+1]``; its training pool is every sample
    ending at or before ``b_k - embargo``, split by time into train / early-stopping validation.
    With ``eval_end_ts`` the slices (bounds and ``eval_idx``) come from that separate scored set
    instead, and only the training pool from ``end_ts`` — a model trained on other labels is then
    still scored on the same windows. Raises when a slice, or its train / validation split, is
    empty.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if not 0 < eval_fraction < 1:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    end_ts = np.asarray(end_ts, dtype=np.int64)
    scored = end_ts if eval_end_ts is None else np.asarray(eval_end_ts, dtype=np.int64)
    if len(end_ts) == 0 or len(scored) == 0:
        raise ValueError("no labeled window to ablate on: the feature data is empty or too short")

    quantiles = 1.0 - eval_fraction + eval_fraction * np.arange(n_windows + 1) / n_windows
    bounds = np.quantile(scored.astype(np.float64), quantiles)
    bounds[-1] = scored.max()  # the last slice runs up to the most recent sample, inclusive
    windows = []
    for k in range(n_windows):
        lo, hi = bounds[k], bounds[k + 1]
        eval_idx = np.where((scored > lo) & (scored <= hi))[0]
        pool = np.where(end_ts <= lo - embargo)[0]
        if len(eval_idx) == 0 or len(pool) == 0:
            raise ValueError(
                f"window {k} is empty (eval={len(eval_idx)}, train pool={len(pool)} after the "
                f"{embargo}s embargo); use fewer windows or a longer time range"
            )
        train_rel, val_rel = temporal_split(
            end_ts[pool], val_fraction=val_fraction, embargo_seconds=embargo
        )
        if len(train_rel) == 0 or len(val_rel) == 0:
            raise ValueError(
                f"window {k}: empty train/validation split of its {len(pool)}-sample training "
                "pool; use fewer windows or a longer time range"
            )
        windows.append(
            Window(k, pool[train_rel], pool[val_rel], eval_idx, int(math.floor(lo)), int(hi))
        )
    return windows


# --------------------------------------------------------------------------- metrics & lift


def metric_keys(horizons: Sequence[str]) -> list[str]:
    """Flat metric names: ``<head>/<horizon>/<metric>``, then ``policy/<metric>``."""
    keys = [f"{head}/{h}/{m}" for head in HEADS for h in horizons for m in PREDICTIVE_METRICS]
    return keys + [f"policy/{m}" for m in POLICY_METRICS]


def raw_key(key: str) -> str:
    """The uncalibrated twin of a predictive metric key."""
    return f"{RAW_PREFIX}{key}"


def run_metrics(report: EvaluationReport) -> dict[str, float | None]:
    """The ablation metrics of one evaluation (``None`` where a metric is undefined), plus the
    uncalibrated predictive metrics under :func:`raw_key`."""
    out: dict[str, float | None] = {}
    for head in HEADS:
        per_horizon = getattr(report, head)
        raw = report.uncalibrated.get(head, {})
        for h in report.horizons:
            for m in PREDICTIVE_METRICS:
                out[f"{head}/{h}/{m}"] = _finite(per_horizon[h][m])
                if h in raw:
                    out[raw_key(f"{head}/{h}/{m}")] = _finite(raw[h][m])
    for m in POLICY_METRICS:
        out[f"policy/{m}"] = _finite(report.policy[m])
    return out


def decide(lifts: Iterable[float | None], margin: float) -> dict[str, Any]:
    """Summarize paired lifts (positive = better) and flag them against ``margin``.

    ``better`` / ``worse`` when the mean clears ``±margin`` by more than one standard deviation of
    the lifts, ``not significant`` otherwise — always so for a single comparable run, which has no
    spread to clear — and ``n/a`` when no run could be compared.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    stats = summarize(lifts)
    if stats["n"] == 0:
        return {**stats, "verdict": "n/a", "significant": None}
    if stats["n"] == 1:
        return {**stats, "verdict": "not significant", "significant": False}
    mean, std = stats["mean"], stats["std"]
    if mean - std > margin:
        verdict = "better"
    elif mean + std < -margin:
        verdict = "worse"
    else:
        verdict = "not significant"
    return {**stats, "verdict": verdict, "significant": verdict != "not significant"}


def paired_lifts(
    with_group: Sequence[float | None], without_group: Sequence[float | None], metric: str
) -> list[float | None]:
    """Per-run lift of having the group, oriented so positive is better; ``None`` when either run
    left the metric undefined (e.g. a single-class horizon)."""
    return [
        None if a is None or b is None else lift(a, b, metric)
        for a, b in zip(with_group, without_group, strict=True)
    ]


# --------------------------------------------------------------------------- report


@dataclass
class RunResult:
    variant: str
    window: int
    seed: int
    best_epoch: int
    metrics: dict[str, float | None]


@dataclass
class AblationReport:
    """Every run plus, per variant, metric mean ± std and, per tested group, its lift."""

    generated_at: str
    settings: dict[str, Any]
    horizons: list[str]
    windows: list[dict[str, Any]]
    variants: list[dict[str, Any]]
    runs: list[RunResult] = field(default_factory=list)
    summary: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    lifts: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    # Per predictive metric: calibrated - raw over every run (positive = better, ECE negated).
    calibration: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    def render(self) -> str:
        """A readable summary: one table per tested group, primary metrics first."""
        s = self.settings
        base = self.variants[0]["name"]
        lines = [
            f"Ablation — {len(self.windows)} window(s) x {len(s['seeds'])} seed(s), "
            f"margin {s['margin']} (P&L {s['pnl_margin']}); lift = with group - without, "
            "positive = better (ECE negated)",
            f"generated {self.generated_at}",
        ]
        for w in self.windows:
            lines.append(
                f"  window {w['index']}: eval {w['eval_start']} -> {w['eval_end']} "
                f"({w['n_eval']} scored, {w['n_train']} train, {w['n_val']} early-stop val)"
            )
        for variant in self.variants:
            if variant["group"] is None:
                continue
            name = variant["name"]
            lines += [
                "",
                f"{name}  ({variant['mode']} group '{variant['group']}' vs {base})",
                f"{'metric':>22} | {base:>17} | {name:>17} | {'lift':>17} | verdict",
            ]
            for key, gain in self.lifts[name].items():
                lines.append(
                    f"{key:>22} | {_pm(self.summary[base][key]):>17} | "
                    f"{_pm(self.summary[name][key]):>17} | {_pm(gain):>17} | {gain['verdict']}"
                )
        if self.calibration:
            method = s.get("calibration", {}).get("method", "?")
            lines += [
                "",
                f"Calibration ({method}): calibrated - raw over all runs, positive = better",
                f"{'metric':>22} | {'effect':>17} | verdict",
            ]
            for key, effect in self.calibration.items():
                lines.append(f"{key:>22} | {_pm(effect):>17} | {effect['verdict']}")
        return "\n".join(lines)


def _pm(stats: Mapping[str, Any]) -> str:
    if stats["mean"] is None:
        return "n/a"
    return f"{fmt(stats['mean'])} ± {fmt(stats['std'])}"


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), UTC).isoformat()


# --------------------------------------------------------------------------- orchestration


def run_ablation(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    variants: Sequence[Variant] | None = None,
    n_windows: int = DEFAULT_WINDOWS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    margin: float = DEFAULT_MARGIN,
    pnl_margin: float = DEFAULT_PNL_MARGIN,
    policy: ReferencePolicy | None = None,
    device: Any = None,
    progress: Callable[[str], None] | None = None,
) -> AblationReport:
    """Train and score every variant on every (window, seed); see the module doc.

    Raises ``ValueError`` on an unusable setup (bad settings, no seed, unknown group, duplicate
    variant name, empty window, a policy horizon the config doesn't label) before any training
    starts.
    """
    seeds = check_settings(
        n_windows=n_windows,
        seeds=seeds,
        eval_fraction=eval_fraction,
        val_fraction=val_fraction,
        margin=margin,
        pnl_margin=pnl_margin,
    )
    variants = list(plan_variants() if variants is None else variants)
    check_variants(variants)

    # Every variant's groups, not just the config's: a candidate group needs re-replayed rows too.
    check_group_coverage(df, [g for v in variants for g in v.groups])
    samples = config_samples(df, config)
    horizons = list(samples.horizons)
    policy = policy or ReferencePolicy(horizon=horizons[0])
    policy.horizon_index(horizons)
    windows = walk_forward_windows(
        samples.end_ts,
        n_windows=n_windows,
        eval_fraction=eval_fraction,
        val_fraction=val_fraction,
        embargo=embargo_seconds(horizons),
    )

    runs: list[RunResult] = []
    total = len(windows) * len(seeds) * len(variants)
    for variant in variants:
        for window in windows:
            train_ds = SequenceDataset(samples, window.train_idx)
            val_ds = SequenceDataset(samples, window.val_idx)
            eval_ds = SequenceDataset(samples, window.eval_idx)
            for seed in seeds:
                if progress:
                    progress(
                        f"[{len(runs) + 1}/{total}] {variant.name} window {window.index} "
                        f"seed {seed}"
                    )
                cfg = copy.deepcopy(dict(config))
                cfg["features"] = {"groups": list(variant.groups)}  # the model masks the rest
                best_epoch, metrics = train_and_score(
                    train_ds, val_ds, eval_ds, cfg, seed=seed, policy=policy, device=device
                )
                runs.append(RunResult(variant.name, window.index, seed, best_epoch, metrics))

    keys = metric_keys(horizons)
    by_variant = {v.name: _ordered(runs, v.name, windows, seeds) for v in variants}
    summary = {
        name: {k: summarize(r.metrics[k] for r in rs) for k in keys}
        for name, rs in by_variant.items()
    }
    lifts: dict[str, dict[str, dict[str, Any]]] = {}
    base = by_variant[variants[0].name]
    for variant in variants[1:]:
        other = by_variant[variant.name]
        with_g, without_g = (other, base) if variant.mode == "add" else (base, other)
        lifts[variant.name] = {
            k: decide(
                paired_lifts([r.metrics[k] for r in with_g], [r.metrics[k] for r in without_g], k),
                pnl_margin if k.startswith("policy/") else margin,
            )
            for k in keys
        }

    predictive = [k for k in keys if not k.startswith("policy/")]
    calibration_effect = {
        k: decide(
            paired_lifts(
                [r.metrics.get(k) for r in runs], [r.metrics.get(raw_key(k)) for r in runs], k
            ),
            margin,
        )
        for k in predictive
    }

    return AblationReport(
        generated_at=datetime.now(UTC).isoformat(),
        settings={
            "n_windows": n_windows,
            "seeds": seeds,
            "eval_fraction": eval_fraction,
            "val_fraction": val_fraction,
            "margin": margin,
            "pnl_margin": pnl_margin,
            "policy": {
                "horizon": policy.horizon,
                "pump_threshold": policy.pump_threshold,
                "rug_threshold": policy.rug_threshold,
            },
            "embargo_seconds": embargo_seconds(horizons),
            "calibration": calibration_settings(config),
        },
        horizons=horizons,
        windows=[w.describe() for w in windows],
        variants=[asdict(v) for v in variants],
        runs=runs,
        summary=summary,
        lifts=lifts,
        calibration=calibration_effect,
    )


def train_and_score(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    eval_ds: SequenceDataset,
    config: Mapping[str, Any],
    *,
    seed: int,
    policy: ReferencePolicy,
    device: Any = None,
) -> tuple[int, dict[str, float | None]]:
    """One run: fit under ``seed`` (early-stopping on ``val_ds``), calibrate on ``val_ds``, score
    ``eval_ds`` calibrated. Returns the best epoch and :func:`run_metrics`. Never saves anything."""
    cfg = copy.deepcopy(dict(config))
    cfg["training"] = {**cfg["training"], "seed": seed}
    fitted = fit(train_ds, val_ds, cfg, device=device)
    model = fitted.model.cpu()
    calibration, _ = fit_calibration_window(model, val_ds, cfg)
    report = evaluate(model, eval_ds, policy=policy, calibration=calibration)
    return fitted.best_epoch, run_metrics(report)


def check_settings(
    *,
    n_windows: int,
    seeds: Iterable[int],
    eval_fraction: float,
    val_fraction: float,
    margin: float,
    pnl_margin: float,
) -> list[int]:
    """Validate the run settings without touching any data; returns the de-duplicated seeds."""
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    if not seeds:
        raise ValueError("at least one seed is required")
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    for name, value in (("eval_fraction", eval_fraction), ("val_fraction", val_fraction)):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1), got {value}")
    for name, value in (("margin", margin), ("pnl_margin", pnl_margin)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    return seeds


def _ordered(
    runs: Sequence[RunResult], variant: str, windows: Sequence[Window], seeds: Sequence[int]
) -> list[RunResult]:
    """A variant's runs in (window, seed) order, so two variants pair up run for run."""
    index = {(r.window, r.seed): r for r in runs if r.variant == variant}
    return [index[(w.index, s)] for w in windows for s in seeds]


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure per-feature-group lift over walk-forward windows and seeds. "
        "Never saves or registers a model."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        help=f"groups to test (default: all of {', '.join(FEATURE_GROUPS)}); a baseline group "
        "is dropped, any other is added",
    )
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=DEFAULT_EVAL_FRACTION,
        help="most recent share of windows scored, split into --windows slices",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="early-stopping share of each window's training pool",
    )
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="predictive metrics")
    parser.add_argument("--pnl-margin", type=float, default=DEFAULT_PNL_MARGIN)
    parser.add_argument("--policy-horizon", help="reference-policy horizon (default: first)")
    parser.add_argument("--pump-threshold", type=float, default=ReferencePolicy.pump_threshold)
    parser.add_argument("--rug-threshold", type=float, default=ReferencePolicy.rug_threshold)
    parser.add_argument("--limit", type=int, help="cap the number of feature rows read")
    parser.add_argument("--out", help="write the JSON report to this path")
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    import torch

    from .config import load_config
    from .db import make_engine, read_features

    config = load_config()
    try:
        variants = plan_variants(args.groups)
        policy = ReferencePolicy(
            horizon=args.policy_horizon or config["horizons"][0],
            pump_threshold=args.pump_threshold,
            rug_threshold=args.rug_threshold,
        )
        policy.horizon_index(config["horizons"])
        check_settings(  # fail before reading anything
            n_windows=args.windows,
            seeds=args.seeds,
            eval_fraction=args.eval_fraction,
            val_fraction=args.val_fraction,
            margin=args.margin,
            pnl_margin=args.pnl_margin,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    df = read_features(make_engine(), limit=args.limit)
    try:
        report = run_ablation(
            df,
            config,
            variants=variants,
            n_windows=args.windows,
            seeds=args.seeds,
            eval_fraction=args.eval_fraction,
            val_fraction=args.val_fraction,
            margin=args.margin,
            pnl_margin=args.pnl_margin,
            policy=policy,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            progress=lambda msg: print(msg, file=sys.stderr),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(report.render())
    if args.out:
        print(f"\nreport written to {report.write(args.out)}")


if __name__ == "__main__":
    main()
