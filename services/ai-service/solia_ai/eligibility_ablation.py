"""Eligibility ablation: which token eligibility thresholds should the model train and trade on?

Filtering only helps if a model trained on the eligible tokens predicts *the tokens the bot will
trade* better than one trained on everything. So for each candidate rule ``S``
(:class:`solia_ai.eligibility.EligibilityRule`) every (window, seed) run pairs two models scored on
the **same** windows — the held-out windows ``S`` accepts:

* ``all`` — trained (and early-stopped, calibrated) on every window: today's behaviour. Trained
  once per (window, seed) and scored on each rule's population;
* ``S`` — trained on the windows ``S`` accepts only.

The walk-forward slices (:func:`solia_ai.ablation.walk_forward_windows`) are cut once, by time,
over every labeled window, so every rule is scored on the same time ranges; a rule only removes
windows from them. Lift = ``S - all`` over the paired runs, positive = better (ECE negated),
judged like the feature ablation (:func:`solia_ai.ablation.decide`). Predictive metrics are
primary; per rule the report also gives the samples it keeps (training cost), the tokens and
``features`` rows they come from (what a filtered read would load), and the reference-policy trades
and P&L on its population (secondary).

Samples are built once with the rule disabled; each rule is a mask over their recorded anchor
inputs (``Samples.eligibility_inputs``), the very rule :func:`solia_ai.dataset.build_samples`
applies. Never saves or registers a model.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation import (
    DEFAULT_MARGIN,
    DEFAULT_PNL_MARGIN,
    DEFAULT_SEEDS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WINDOWS,
    Window,
    _pm,
    check_settings,
    decide,
    metric_keys,
    paired_lifts,
    run_metrics,
    walk_forward_windows,
)
from .config import DEFAULT_EVAL_FRACTION
from .dataset import Samples, SequenceDataset, config_samples, embargo_seconds
from .eligibility import EligibilityRule
from .evaluate import ReferencePolicy, default_policy_horizon, evaluate, json_safe
from .lift import summarize
from .train import fit, fit_calibration_window

BASELINE = "all"
TRADES_KEY = "policy/n_trades"

# The default rules under test: unique traders (the issue's population stats put 53 % of tokens at
# <= 1 trader, 72 % at <= 3), window SOL volume, and both.
DEFAULT_RULES: tuple[EligibilityRule, ...] = (
    EligibilityRule(min_unique_traders=2),
    EligibilityRule(min_unique_traders=4),
    EligibilityRule(min_unique_traders=10),
    EligibilityRule(min_volume_sol=1.0),
    EligibilityRule(min_unique_traders=4, min_volume_sol=1.0),
)


def rule_name(rule: EligibilityRule) -> str:
    parts = []
    if rule.min_unique_traders:
        parts.append(f"traders>={rule.min_unique_traders}")
    if rule.min_volume_sol:
        parts.append(f"vol>={rule.min_volume_sol:g}")
    if rule.min_token_age_sec:
        parts.append(f"age>={rule.min_token_age_sec:g}")
    return ",".join(parts) or BASELINE


def parse_rule(text: str) -> EligibilityRule:
    """``traders=4,volume=1.0,age=30`` (any subset) -> the rule; ``ValueError`` when invalid."""
    keys = {
        "traders": "min_unique_traders",
        "volume": "min_volume_sol",
        "age": "min_token_age_sec",
    }
    values: dict[str, Any] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, sep, raw = part.partition("=")
        if not sep or name.strip() not in keys:
            raise ValueError(f"bad rule part {part!r}: expected traders=N, volume=X or age=S")
        key = keys[name.strip()]
        try:
            values[key] = int(raw) if key == "min_unique_traders" else float(raw)
        except ValueError:
            raise ValueError(f"bad rule part {part!r}: {raw!r} is not a number") from None
    rule = EligibilityRule.from_mapping(values)
    if not rule.enabled:
        raise ValueError(f"rule {text!r} filters nothing: it is the '{BASELINE}' baseline")
    return rule


@dataclass
class RuleRun:
    rule: str
    window: int
    seed: int
    best_epoch: int | None  # None for the baseline (its epoch is in the baseline run)
    metrics: dict[str, float | None]  # the rule's model on the rule's population
    baseline: dict[str, float | None]  # the `all` model on the same population


@dataclass
class EligibilityAblationReport:
    generated_at: str
    settings: dict[str, Any]
    horizons: list[str]
    windows: list[dict[str, Any]]
    rules: list[dict[str, Any]]  # name, thresholds, population counts
    runs: list[RuleRun] = field(default_factory=list)
    summary: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    baseline_summary: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    lifts: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def render(self, keys: Sequence[str] | None = None) -> str:
        """Population per rule, then one table per rule (``keys``: the metrics to show)."""
        s = self.settings
        lines = [
            f"Eligibility ablation — {len(self.windows)} window(s) x {len(s['seeds'])} seed(s), "
            f"margin {s['margin']}; lift = rule's model - '{BASELINE}' model on the rule's "
            "population, positive = better (ECE negated)",
            f"generated {self.generated_at}",
            "",
            f"{'rule':>24} | {'samples':>9} | {'share':>6} | {'tokens':>7} | {'rows':>9} | scored",
        ]
        for r in self.rules:
            lines.append(
                f"{r['name']:>24} | {r['n_samples']:>9} | {r['sample_share']:>6.1%} | "
                f"{r['n_tokens']:>7} | {r['n_rows']:>9} | {r['n_eval']}"
            )
        for r in self.rules:
            name = r["name"]
            if name not in self.lifts:
                continue
            lines += [
                "",
                name,
                f"{'metric':>22} | {BASELINE:>17} | {'rule':>17} | {'lift':>17} | verdict",
            ]
            for key, gain in self.lifts[name].items():
                if keys is not None and key not in keys:
                    continue
                lines.append(
                    f"{key:>22} | {_pm(self.baseline_summary[name][key]):>17} | "
                    f"{_pm(self.summary[name][key]):>17} | {_pm(gain):>17} | {gain['verdict']}"
                )
        return "\n".join(lines)


def population(samples: Samples, accepted: np.ndarray, rows_per_token: Mapping[str, int]):
    """What a rule keeps: samples, the tokens they come from and those tokens' feature rows."""
    tokens = pd.unique(samples.mint[accepted])
    return {
        "n_samples": int(accepted.sum()),
        "sample_share": float(accepted.mean()) if len(accepted) else 0.0,
        "n_tokens": int(len(tokens)),
        "n_rows": int(sum(rows_per_token.get(m, 0) for m in tokens)),
    }


def subset(window: Window, accepted: np.ndarray) -> Window:
    """``window`` restricted to the accepted samples (same time bounds)."""
    return Window(
        window.index,
        window.train_idx[accepted[window.train_idx]],
        window.val_idx[accepted[window.val_idx]],
        window.eval_idx[accepted[window.eval_idx]],
        window.eval_start_ts,
        window.eval_end_ts,
    )


def run_eligibility_ablation(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    rules: Sequence[EligibilityRule] = DEFAULT_RULES,
    n_windows: int = DEFAULT_WINDOWS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    margin: float = DEFAULT_MARGIN,
    pnl_margin: float = DEFAULT_PNL_MARGIN,
    policy: ReferencePolicy | None = None,
    device: Any = None,
    progress: Callable[[str], None] | None = None,
) -> EligibilityAblationReport:
    """Train and score the baseline and every rule on every (window, seed); see the module doc.

    Raises ``ValueError`` before any training on bad settings, no rule, a duplicate or disabled
    rule, or a rule leaving a window's train / validation / scored set empty.
    """
    seeds = check_settings(
        n_windows=n_windows,
        seeds=seeds,
        eval_fraction=eval_fraction,
        val_fraction=val_fraction,
        margin=margin,
        pnl_margin=pnl_margin,
    )
    rules = list(rules)
    if not rules:
        raise ValueError("at least one eligibility rule is required")
    names = [rule_name(r) for r in rules]
    if BASELINE in names:
        raise ValueError(f"a disabled rule is the '{BASELINE}' baseline, not a rule to test")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate rule(s): {names}")
    for rule in rules:
        rule.check_columns(df)

    cfg = copy.deepcopy(dict(config))
    cfg.pop("eligibility", None)  # every window: each rule is a mask over them
    samples = config_samples(df, cfg)
    horizons = list(samples.horizons)
    policy = policy or ReferencePolicy(horizon=default_policy_horizon(horizons))
    policy.horizon_index(horizons)
    windows = walk_forward_windows(
        samples.end_ts,
        n_windows=n_windows,
        eval_fraction=eval_fraction,
        val_fraction=val_fraction,
        embargo=embargo_seconds(horizons),
    )
    rows_per_token = df.groupby("mint").size().to_dict()
    accepted = {
        name: rule.accepts(samples.eligibility_inputs)
        for name, rule in zip(names, rules, strict=True)
    }
    folds = {name: [subset(w, accepted[name]) for w in windows] for name in names}
    for name, fs in folds.items():
        for w in fs:
            empty = [k for k, idx in (("train", w.train_idx), ("val", w.val_idx),
                                      ("scored", w.eval_idx)) if len(idx) == 0]  # fmt: skip
            if empty:
                raise ValueError(
                    f"rule {name}: window {w.index} keeps no {'/'.join(empty)} sample; "
                    "use a looser rule, fewer windows or more data"
                )

    def fit_and_calibrate(window: Window, seed: int):
        run_cfg = copy.deepcopy(cfg)
        run_cfg["training"] = {**run_cfg["training"], "seed": seed}
        val_ds = SequenceDataset(samples, window.val_idx)
        fitted = fit(SequenceDataset(samples, window.train_idx), val_ds, run_cfg, device=device)
        model = fitted.model.cpu()
        calibration, _ = fit_calibration_window(model, val_ds, run_cfg)
        return fitted.best_epoch, model, calibration

    def score(model, calibration, eval_idx: np.ndarray) -> dict[str, float | None]:
        report = evaluate(
            model, SequenceDataset(samples, eval_idx), policy=policy, calibration=calibration
        )
        return {**run_metrics(report), TRADES_KEY: float(report.policy["n_trades"])}

    runs: list[RuleRun] = []
    total = len(windows) * len(seeds) * (1 + len(rules))
    done = 0
    for window in windows:
        for seed in seeds:
            done += 1
            if progress:
                progress(f"[{done}/{total}] {BASELINE} window {window.index} seed {seed}")
            _, base_model, base_cal = fit_and_calibrate(window, seed)
            for name in names:
                done += 1
                if progress:
                    progress(f"[{done}/{total}] {name} window {window.index} seed {seed}")
                fold = folds[name][window.index]
                best_epoch, model, cal = fit_and_calibrate(fold, seed)
                runs.append(
                    RuleRun(
                        name,
                        window.index,
                        seed,
                        best_epoch,
                        score(model, cal, fold.eval_idx),
                        score(base_model, base_cal, fold.eval_idx),
                    )
                )

    keys = [*metric_keys(horizons), TRADES_KEY]
    summary, baseline_summary, lifts = {}, {}, {}
    for name in names:
        rs = [r for r in runs if r.rule == name]
        summary[name] = {k: summarize(r.metrics.get(k) for r in rs) for k in keys}
        baseline_summary[name] = {k: summarize(r.baseline.get(k) for r in rs) for k in keys}
        lifts[name] = {
            k: decide(
                paired_lifts([r.metrics.get(k) for r in rs], [r.baseline.get(k) for r in rs], k),
                pnl_margin if k.startswith("policy/") else margin,
            )
            for k in keys
            if k != TRADES_KEY
        }

    every = np.ones(len(samples), bool)
    rule_rows = [
        {
            "name": BASELINE,
            "thresholds": EligibilityRule().as_dict(),
            **population(samples, every, rows_per_token),
            "n_eval": sum(len(w.eval_idx) for w in windows),
        }
    ]
    for name, rule in zip(names, rules, strict=True):
        rule_rows.append(
            {
                "name": name,
                "thresholds": rule.as_dict(),
                **population(samples, accepted[name], rows_per_token),
                "n_eval": sum(len(w.eval_idx) for w in folds[name]),
            }
        )

    return EligibilityAblationReport(
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
            "feature_rows": int(len(df)),
        },
        horizons=horizons,
        windows=[w.describe() for w in windows],
        rules=rule_rows,
        runs=runs,
        summary=summary,
        baseline_summary=baseline_summary,
        lifts=lifts,
    )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure the lift of token eligibility rules over walk-forward windows and "
        "seeds (rule's model vs every-token model, on the rule's population). Never saves or "
        "registers a model."
    )
    parser.add_argument(
        "--rule",
        action="append",
        metavar="SPEC",
        help="a rule to test, e.g. traders=4,volume=1.0,age=30 (repeatable; default: "
        + "; ".join(rule_name(r) for r in DEFAULT_RULES)
        + ")",
    )
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="predictive metrics")
    parser.add_argument("--pnl-margin", type=float, default=DEFAULT_PNL_MARGIN)
    parser.add_argument(
        "--policy-horizon", help="reference-policy horizon (default: 15m if served, else first)"
    )
    parser.add_argument("--out", help="write the JSON report to this path")
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--epochs", type=int, help="override training.epochs (a faster sweep)")
    from .db import add_data_bounds_arguments

    add_data_bounds_arguments(parser)
    args = parser.parse_args(argv)

    import torch

    from .config import load_config
    from .db import DataSettings, make_engine, read_training_features

    config = load_config()
    if args.epochs is not None:
        config["training"] = {**config["training"], "epochs": args.epochs}
    try:
        rules = [parse_rule(spec) for spec in args.rule] if args.rule else list(DEFAULT_RULES)
        policy = ReferencePolicy(
            horizon=args.policy_horizon or default_policy_horizon(config["horizons"])
        )
        policy.horizon_index(config["horizons"])
        settings = DataSettings.from_config(
            config, lookback_days=args.lookback_days, max_rows=args.max_rows
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    df, bounds = read_training_features(make_engine(), settings)
    print(bounds.describe(), file=sys.stderr)
    try:
        report = run_eligibility_ablation(
            df,
            config,
            rules=rules,
            n_windows=args.windows,
            seeds=args.seeds,
            eval_fraction=args.eval_fraction,
            val_fraction=args.val_fraction,
            margin=args.margin,
            pnl_margin=args.pnl_margin,
            policy=policy,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            progress=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(report.render())
    if args.out:
        print(f"\nreport written to {report.write(args.out)}")


if __name__ == "__main__":
    main()
