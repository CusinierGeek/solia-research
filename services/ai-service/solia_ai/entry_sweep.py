"""Choose the entry rule from data: sweep entry thresholds over a model's held-out windows.

The bot enters when ``pump_prob >= MinPumpProb`` and ``rug_prob <= MaxRugProb`` (docs/ML-MODEL.md
"Policy layer"). Calibrated probabilities of a rare event are *low* — pumps are ~1 % of windows —
so a threshold picked before calibration (0.6) may never fire. This read-only sweep scores the
champion once over its held-out windows and reports, per horizon and per entry rule:

* **absolute thresholds** — every ``pump_prob >= P`` x ``rug_prob <= R`` of a grid (``R = None``:
  no rug gate);
* **top-X %** — the ``X`` share of labeled windows with the highest ``pump_prob`` (optionally under
  a rug gate too); the implied pump threshold (the lowest selected score) is reported, since that
  is what the bot is configured with.

Each rule's selection is measured on **predictive quality first** — ``precision`` (share of real
pumps) and its lift over the base rate — then on the realized return at the horizon: win rate, mean
/ median return, total P&L and a bootstrap CI of the mean (total = n x mean, so its CI scales the
same). Returns are heavy-tailed: one capped +1000 % label can make a 50-trade rule look great, hence
``min_trades`` and the CI. The buy-everything baseline is reported alongside. A confidence gate
(``min_confidence``, the bot's ``MinConfidence``) is applied on top and the trades it removes are
counted, since a rarely-satisfied gate blocks entries the same way a high threshold does.

:func:`recommend` picks, per horizon, the most precise rule with at least ``min_trades`` trades
whose mean-return CI lies above the baseline mean (falling back to the most precise rule with
enough trades when none does). It is a starting point for the bot's defaults, not an automatic
deployment: the probability scale shifts with every promoted champion, so re-run the sweep after a
promotion (``python -m solia_ai.entry_sweep --champion --out ...``).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import fmt, json_safe

DEFAULT_PUMP_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
DEFAULT_RUG_THRESHOLDS: tuple[float | None, ...] = (None, 0.5, 0.3, 0.2, 0.1, 0.05)
DEFAULT_TOP_FRACTIONS = (0.0001, 0.0005, 0.001, 0.005, 0.01)
DEFAULT_MIN_TRADES = 100
DEFAULT_MIN_CONFIDENCE = 0.0
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CI = 0.95
# Bootstrap draws are generated in chunks of about this many indices, so the buy-everything
# baseline (~10^6 windows) never materializes a samples x n matrix.
_BOOTSTRAP_CHUNK = 20_000_000


# --------------------------------------------------------------------------- validation


def _check_probability(name: str, value: float | None) -> None:
    if value is None:
        return
    if not (isinstance(value, int | float) and math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")


def _check_grid(
    pump_thresholds: Sequence[float],
    rug_thresholds: Sequence[float | None],
    top_fractions: Sequence[float],
    min_confidence: float,
    min_trades: int,
    bootstrap_samples: int,
    ci: float,
) -> None:
    for p in pump_thresholds:
        _check_probability("pump threshold", p)
    for r in rug_thresholds:
        _check_probability("rug threshold", r)
    for x in top_fractions:
        if not (math.isfinite(x) and 0.0 < x <= 1.0):
            raise ValueError(f"top fraction must be in (0, 1], got {x!r}")
    if not (math.isfinite(min_confidence) and 0.0 <= min_confidence <= 100.0):
        raise ValueError(f"min_confidence must be in [0, 100], got {min_confidence!r}")
    if min_trades < 1:
        raise ValueError(f"min_trades must be at least 1, got {min_trades!r}")
    if bootstrap_samples < 1:
        raise ValueError(f"bootstrap_samples must be at least 1, got {bootstrap_samples!r}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")


# --------------------------------------------------------------------------- metrics


def bootstrap_mean_ci(
    values: np.ndarray, *, samples: int, ci: float, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    """Percentile-bootstrap CI of the mean of ``values``; ``(None, None)`` under two values."""
    n = int(values.size)
    if n < 2:
        return None, None
    per_chunk = max(1, _BOOTSTRAP_CHUNK // n)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, per_chunk):
        k = min(per_chunk, samples - start)
        means[start : start + k] = values[rng.integers(0, n, size=(k, n))].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def selection_metrics(
    returns: np.ndarray,
    pumps: np.ndarray,
    *,
    base_rate: float | None,
    samples: int,
    ci: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Precision / win rate / return statistics of one selection of windows (may be empty)."""
    returns = np.asarray(returns, dtype=np.float64)
    n = int(returns.size)
    if n == 0:
        return {
            "n_trades": 0,
            "precision": None,
            "precision_lift": None,
            "win_rate": None,
            "mean_return": None,
            "median_return": None,
            "total_pnl": 0.0,
            "mean_return_ci": [None, None],
            "total_pnl_ci": [None, None],
        }
    precision = float(np.asarray(pumps, dtype=np.float64).mean())
    low, high = bootstrap_mean_ci(returns, samples=samples, ci=ci, rng=rng)
    return {
        "n_trades": n,
        "precision": precision,
        "precision_lift": (precision / base_rate) if base_rate else None,
        "win_rate": float((returns > 0).mean()),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "total_pnl": float(returns.sum()),
        "mean_return_ci": [low, high],
        "total_pnl_ci": [None if low is None else low * n, None if high is None else high * n],
    }


# --------------------------------------------------------------------------- sweep


def sweep_horizon(
    pump_prob: np.ndarray,
    rug_prob: np.ndarray,
    confidence: np.ndarray,
    pump_label: np.ndarray,
    realized: np.ndarray,
    *,
    pump_thresholds: Sequence[float] = DEFAULT_PUMP_THRESHOLDS,
    rug_thresholds: Sequence[float | None] = DEFAULT_RUG_THRESHOLDS,
    top_fractions: Sequence[float] = DEFAULT_TOP_FRACTIONS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_trades: int = DEFAULT_MIN_TRADES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> dict[str, Any]:
    """Every rule of the grid over one horizon's **labeled** windows (the arrays are already
    filtered to them). Returns the baseline (buy everything) and one row per rule."""
    _check_grid(
        pump_thresholds,
        rug_thresholds,
        top_fractions,
        min_confidence,
        min_trades,
        bootstrap_samples,
        ci,
    )
    pump_prob, rug_prob, confidence, pump_label, realized = (
        np.asarray(a, dtype=np.float64)
        for a in (pump_prob, rug_prob, confidence, pump_label, realized)
    )
    n = int(pump_prob.size)
    if not all(a.shape == (n,) for a in (rug_prob, confidence, pump_label, realized)):
        raise ValueError("the sweep's arrays must be 1-D and of the same length")
    rng = np.random.default_rng(seed)
    base_rate = float(pump_label.mean()) if n else None
    confident = confidence >= min_confidence

    def measure(selected: np.ndarray) -> dict[str, Any]:
        gated = selected & confident
        metrics = selection_metrics(
            realized[gated],
            pump_label[gated],
            base_rate=base_rate,
            samples=bootstrap_samples,
            ci=ci,
            rng=rng,
        )
        metrics["blocked_by_confidence"] = int(selected.sum() - gated.sum())
        metrics["enough_trades"] = metrics["n_trades"] >= min_trades
        return metrics

    rows: list[dict[str, Any]] = []
    for r in rug_thresholds:
        rug_ok = np.ones(n, dtype=bool) if r is None else rug_prob <= r
        for p in pump_thresholds:
            rule = {"kind": "threshold", "pump_threshold": p, "rug_threshold": r}
            rows.append({**rule, **measure((pump_prob >= p) & rug_ok)})
        # Top-X %: rank by pump_prob over every labeled window (the rug gate filters after), so the
        # implied pump threshold is a fixed score cut-off the bot can be configured with.
        order = np.argsort(-pump_prob, kind="stable")
        for x in top_fractions:
            k = min(n, math.ceil(x * n)) if n else 0
            top = np.zeros(n, dtype=bool)
            top[order[:k]] = True
            rows.append(
                {
                    "kind": "top",
                    "top_fraction": x,
                    "pump_threshold": float(pump_prob[order[k - 1]]) if k else None,
                    "rug_threshold": r,
                    **measure(top & rug_ok),
                }
            )

    baseline = selection_metrics(
        realized, pump_label, base_rate=base_rate, samples=bootstrap_samples, ci=ci, rng=rng
    )
    return {
        "n_candidates": n,
        "base_rate": base_rate,
        "pump_prob_quantiles": _quantiles(pump_prob),
        "rug_prob_quantiles": _quantiles(rug_prob),
        "confidence_pass_rate": float(confident.mean()) if n else None,
        "baseline": baseline,
        "rules": rows,
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    keys = ("p50", "p90", "p99", "p99.9", "max")
    if values.size == 0:
        return dict.fromkeys(keys)
    qs = np.quantile(values, [0.5, 0.9, 0.99, 0.999, 1.0])
    return {k: float(q) for k, q in zip(keys, qs, strict=True)}


def recommend(horizon_sweep: Mapping[str, Any]) -> dict[str, Any] | None:
    """The suggested rule of one horizon (see the module doc), or ``None`` when no rule has
    ``min_trades`` trades. Ties on precision go to the rule with more trades."""
    rows = [r for r in horizon_sweep["rules"] if r["enough_trades"]]
    if not rows:
        return None
    baseline_mean = horizon_sweep["baseline"]["mean_return"]

    def beats_baseline(row: Mapping[str, Any]) -> bool:
        low = row["mean_return_ci"][0]
        return low is not None and baseline_mean is not None and low > baseline_mean

    pool = [r for r in rows if beats_baseline(r)] or rows
    best = max(pool, key=lambda r: (r["precision"], r["n_trades"]))
    return {**best, "beats_baseline": beats_baseline(best)}


def sweep(
    predictions: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    horizons: Sequence[str],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """:func:`sweep_horizon` on every horizon's labeled windows (``[N, H]`` arrays, as
    :func:`solia_ai.evaluate.score_logits` / ``apply_calibration`` return them)."""
    out: dict[str, dict[str, Any]] = {}
    for h, horizon in enumerate(horizons):
        labeled = truth["mask"][:, h] == 1
        result = sweep_horizon(
            predictions["pump_prob"][:, h][labeled],
            predictions["rug_prob"][:, h][labeled],
            predictions["confidence"][:, h][labeled],
            truth["pump"][:, h][labeled],
            truth["expected_return"][:, h][labeled],
            **kwargs,
        )
        result["recommended"] = recommend(result)
        out[horizon] = result
    return out


# --------------------------------------------------------------------------- report


def build_report(
    model_version: str, n_samples: int, by_horizon: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    return json_safe(
        {
            "model_version": model_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "n_samples": n_samples,
            "settings": dict(settings),
            "horizons": dict(by_horizon),
        }
    )


def _rule_label(row: Mapping[str, Any]) -> str:
    rug = "-" if row["rug_threshold"] is None else f"{row['rug_threshold']:g}"
    if row["kind"] == "top":
        pump = "n/a" if row["pump_threshold"] is None else f"{row['pump_threshold']:.3f}"
        return f"top {row['top_fraction'] * 100:g}% (p>={pump}) r<={rug}"
    return f"p>={row['pump_threshold']:g} r<={rug}"


def render(report: Mapping[str, Any], *, all_rules: bool = False) -> str:
    """A text table per horizon: the baseline, the rules with enough trades (``all_rules``: every
    rule) and the recommendation."""
    lines = [
        f"Entry-rule sweep — model {report['model_version']} "
        f"({report['n_samples']} held-out windows), generated {report['generated_at']}",
        f"min_trades {report['settings']['min_trades']}, "
        f"min_confidence {report['settings']['min_confidence']}, "
        f"{report['settings']['ci'] * 100:g}% bootstrap CI",
    ]
    header = (
        f"{'rule':<34} {'trades':>7} {'prec':>7} {'lift':>6} {'win':>6} {'mean':>8} "
        f"{'median':>8} {'total':>9} {'mean CI':>19} {'conf-x':>6}"
    )
    for horizon, h in report["horizons"].items():
        q = h["pump_prob_quantiles"]
        lines += [
            "",
            f"[{horizon}] {h['n_candidates']} labeled, base rate {fmt(h['base_rate'])}, "
            f"pump_prob p50 {fmt(q['p50'])} p99 {fmt(q['p99'])} max {fmt(q['max'])}",
            header,
            _row("buy everything", h["baseline"]),
        ]
        for row in h["rules"]:
            if all_rules or row["enough_trades"]:
                lines.append(_row(_rule_label(row), row))
        rec = h["recommended"]
        lines.append(
            "recommended: none (no rule with enough trades)"
            if rec is None
            else f"recommended: {_rule_label(rec)}"
            + ("" if rec["beats_baseline"] else " (mean-return CI does not clear the baseline)")
        )
    return "\n".join(lines)


def _row(label: str, m: Mapping[str, Any]) -> str:
    low, high = m["mean_return_ci"]
    ci = "n/a" if low is None else f"[{low:+.4f},{high:+.4f}]"
    lift = "n/a" if m.get("precision_lift") is None else f"{m['precision_lift']:.1f}x"
    return (
        f"{label:<34} {m['n_trades']:>7} {fmt(m['precision']):>7} {lift:>6} "
        f"{fmt(m['win_rate']):>6} {fmt(m['mean_return']):>8} {fmt(m['median_return']):>8} "
        f"{m['total_pnl']:>9.2f} {ci:>19} {m.get('blocked_by_confidence', 0):>6}"
    )


def write_report(report: Mapping[str, Any], path: str | Path) -> tuple[Path, Path]:
    """The JSON report at ``path`` and its rendered table next to it (``.txt``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = path.with_suffix(".txt")
    table.write_text(render(report) + "\n", encoding="utf-8")
    return path, table


# --------------------------------------------------------------------------- CLI


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def _rugs(text: str) -> list[float | None]:
    return [None if v.strip().lower() == "none" else float(v) for v in text.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep entry thresholds over a checkpoint's held-out windows (read-only)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="path to a .pt checkpoint")
    source.add_argument("--champion", action="store_true", help="the registry's champion")
    parser.add_argument("--out", help="write the JSON report here (+ the table as .txt)")
    parser.add_argument(
        "--lookback-days", type=float, help="override data.lookback_days (the bounded read)"
    )
    parser.add_argument("--max-rows", type=int, help="override data.max_rows (the bounded read)")
    parser.add_argument("--pump-thresholds", type=_floats, default=list(DEFAULT_PUMP_THRESHOLDS))
    parser.add_argument(
        "--rug-thresholds",
        type=_rugs,
        default=list(DEFAULT_RUG_THRESHOLDS),
        help="comma-separated; 'none' = no rug gate",
    )
    parser.add_argument("--top-fractions", type=_floats, default=list(DEFAULT_TOP_FRACTIONS))
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--ci", type=float, default=DEFAULT_CI)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-rules", action="store_true", help="print rules under min_trades")
    args = parser.parse_args(argv)

    settings = {
        "pump_thresholds": args.pump_thresholds,
        "rug_thresholds": args.rug_thresholds,
        "top_fractions": args.top_fractions,
        "min_confidence": args.min_confidence,
        "min_trades": args.min_trades,
        "bootstrap_samples": args.bootstrap_samples,
        "ci": args.ci,
        "seed": args.seed,
    }
    try:  # fail on a bad grid before reading or scoring anything
        _check_grid(
            args.pump_thresholds,
            args.rug_thresholds,
            args.top_fractions,
            args.min_confidence,
            args.min_trades,
            args.bootstrap_samples,
            args.ci,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    from .config import load_config
    from .db import DataSettings, make_engine, read_training_features
    from .evaluate import apply_calibration, holdout_dataset, score_logits
    from .inference import Predictor

    engine = make_engine()
    if args.champion:
        try:
            predictor = Predictor.from_champion(engine)
        except LookupError as exc:
            raise SystemExit(str(exc)) from None
    else:
        predictor = Predictor.from_checkpoint(args.checkpoint)

    # The retrain's bounded read (whole tokens of the newest lookback_days, at most max_rows rows):
    # the full table would not fit in memory. Its windows after the checkpoint's data are scored.
    config = load_config()
    data_settings = DataSettings.from_config(
        config, lookback_days=args.lookback_days, max_rows=args.max_rows
    )
    df, _ = read_training_features(engine, data_settings)
    dataset = holdout_dataset(df, config, predictor)
    logits, truth = score_logits(predictor.model, dataset)
    predictions = apply_calibration(logits, predictor.calibration)
    by_horizon = sweep(predictions, truth, list(predictor.horizons), **settings)
    report = build_report(predictor.model_version, len(dataset), by_horizon, settings)
    print(render(report, all_rules=args.all_rules))
    if args.out:
        json_path, table_path = write_report(report, args.out)
        print(f"\nreport written to {json_path} and {table_path}")


if __name__ == "__main__":
    main()
