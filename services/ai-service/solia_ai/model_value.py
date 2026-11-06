"""Does the model earn its place? Its entries against entries that don't use it (#252).

The exit simulation (:mod:`solia_ai.exit_sim`) says what a model-based rule earns, not whether the
model is the reason. This read-only command walks entries that don't use the model through the very
same exits, costs, tokens and period, on the windows the model never trained on:

- ``random``: as many entries as the model's, drawn at random, many times.
- ``matched``: for each model entry, a random window of another token with the same venue, token age
  and volatility, many times. The model mostly finds volatile tokens, so this asks whether it picks
  better than chance among comparable windows.
- ``breakout``, ``buy_pressure``, ``volatility``: plain rules, each at the score that gives as many
  trades as the model. That score is read over the whole period, which gives the plain rules
  hindsight the model doesn't have: a model that still wins does so against a favoured opponent.

Nothing is written but the report, and paper only: nothing is ever executed. docs/ML-MODEL.md
("Is the model useful?") fixes how to read it. It compares one period: that reading asks for two.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .evaluate import fmt, json_safe
from .exit_sim import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CI,
    DEFAULT_HORIZON,
    FILLED,
    REFERENCE_RULE,
    Execution,
    ExitRule,
    Outcome,
    Paths,
    Windows,
    _utc,
    cell_metrics,
    path_rows,
    score_windows,
    simulate_position,
    simulate_rule,
    uncensored,
)
from .liquidity import LiquidityFloor

# The frozen paper-test setting (docs/ML-MODEL.md, "Exit-rule simulation").
EXIT_RULE = REFERENCE_RULE
DEFAULT_PUMP_THRESHOLD = 0.5
DEFAULT_DRAWS = 200
EDGE_MIN_TRADES = 300  # the edge criterion's condition 2

BREAKOUT_LOOKBACK_MARKS = 30  # about 5 minutes of ~10 s marks
BREAKOUT_MIN_MARKS = 6  # about 1 minute: under that, there is no high to break
PLAIN_RULES = ("breakout", "buy_pressure", "volatility")
RANDOM_ARMS = ("random", "matched")
AGE_EDGES_S = (300.0, 900.0, 3600.0)
VOLATILITY_BINS = 4
MATCH_LEVELS = ("venue+age+volatility", "venue+age", "venue", "any")


# --------------------------------------------------------------------------- signals


def breakout_signal(paths: Paths) -> np.ndarray:
    """Per mark, how far the price stands above the token's highest of the previous marks. It reads
    only earlier marks. NaN without enough history."""
    out = np.full(len(paths.price), np.nan)
    for start, stop in paths.spans.values():
        previous = pd.Series(paths.price[start:stop]).shift(1)
        high = previous.rolling(BREAKOUT_LOOKBACK_MARKS, min_periods=BREAKOUT_MIN_MARKS).max()
        out[start:stop] = paths.price[start:stop] / high.to_numpy(np.float64) - 1.0
    return out


def plain_signals(df: pd.DataFrame, paths: Paths) -> dict[str, np.ndarray]:
    """Every plain rule's score per mark of ``paths``, read from ``df``'s row of that mark."""
    missing = {"buy_volume_ratio"} - set(df.columns)
    if missing:
        raise ValueError(f"the plain rules need column(s): {', '.join(sorted(missing))}")
    order = path_rows(df)[0]
    pressure = pd.to_numeric(df["buy_volume_ratio"], errors="coerce").to_numpy(np.float64)[order]
    return {
        "breakout": breakout_signal(paths),
        "buy_pressure": pressure,
        "volatility": paths.volatility.copy(),
    }


# --------------------------------------------------------------------------- entries


def without_model(windows: Windows) -> Windows:
    """The same windows with the upside gate always open, so that no decision reads the model."""
    return replace(windows, expected_return=np.ones(len(windows), np.float64))


def entries_with_rows(
    paths: Paths, windows: Windows, rule: ExitRule, execution: Execution
) -> list[tuple[int, Any, Outcome]]:
    """Like :func:`~solia_ai.exit_sim.simulate_rule`, keeping each filled trade's decision row."""
    out: list[tuple[int, Any, Outcome]] = []
    busy_until: dict[Any, float] = {}
    for i in np.lexsort((windows.row,)):
        mint, row = windows.mint[i], int(windows.row[i])
        if paths.ts[row] < busy_until.get(mint, -math.inf):
            continue
        outcome = simulate_position(
            paths, row, paths.spans[mint][1], float(windows.expected_return[i]), rule, execution
        )
        busy_until[mint] = outcome.free_at
        if outcome.status == FILLED:
            out.append((row, mint, outcome))
    return out


def _filled(entries: Sequence[tuple[Any, Outcome]]) -> list[tuple[Any, Outcome]]:
    return [(m, o) for m, o in entries if o.status == FILLED]


def top_windows_for_trades(
    paths: Paths,
    windows: Windows,
    signal: np.ndarray,
    target_trades: int,
    rule: ExitRule,
    execution: Execution,
    rng: np.random.Generator,
) -> Windows:
    """The windows with the highest ``signal`` that give about ``target_trades``. A rule that keeps
    firing on one token takes few trades per window, so the window count alone wouldn't compare."""
    values = signal[windows.row]
    usable = np.flatnonzero(np.isfinite(values))
    if not len(usable) or target_trades < 1:
        return windows.take(np.zeros(len(windows), bool))
    # Random tie-break: a ratio capped at 1 ties on many windows, and frame order means nothing.
    ranked = usable[np.lexsort((rng.random(len(usable)), -values[usable]))]

    def count(k: int) -> int:
        return len(_filled(simulate_rule(paths, windows.take(ranked[:k]), rule, execution)))

    # Double, then bisect: a run costs what its trades cost, so stay far from all the windows.
    high = min(target_trades, len(ranked))
    while high < len(ranked) and count(high) < target_trades:
        high = min(high * 2, len(ranked))
    low = high // 2 + 1 if high > target_trades else 1
    best, best_gap = high, math.inf
    while low <= high:
        k = (low + high) // 2
        trades = count(k)
        gap = abs(trades - target_trades)
        if gap < best_gap or (gap == best_gap and k < best):
            best, best_gap = k, gap
        if trades == target_trades:
            break
        if trades < target_trades:
            low = k + 1
        else:
            high = k - 1
    return windows.take(ranked[:best])


# --------------------------------------------------------------------------- matching


def strata(df: pd.DataFrame, paths: Paths, windows: Windows) -> list[np.ndarray]:
    """Per level of :data:`MATCH_LEVELS`, each window's stratum code, finest first. Volatility bins
    are the windows' own quantiles, and an unknown value has its own bin."""
    missing = {"token_age_sec", "venue"} - set(df.columns)
    if missing:
        raise ValueError(f"the matching needs column(s): {', '.join(sorted(missing))}")
    order = path_rows(df)[0]
    age = pd.to_numeric(df["token_age_sec"], errors="coerce").to_numpy(np.float64)[order]
    venue = df["venue"].to_numpy(dtype=object)[order][windows.row]
    venue_code = pd.factorize(pd.Series(venue, dtype=object), use_na_sentinel=False)[0]

    age_w = age[windows.row]
    age_bin = np.searchsorted(AGE_EDGES_S, age_w, side="right")
    age_bin[~np.isfinite(age_w)] = len(AGE_EDGES_S) + 1

    vol = paths.volatility[windows.row]
    known = vol[np.isfinite(vol)]
    vol_bin = np.full(len(vol), VOLATILITY_BINS, np.int64)
    if len(known):
        edges = np.quantile(known, np.linspace(0, 1, VOLATILITY_BINS + 1)[1:-1])
        vol_bin[np.isfinite(vol)] = np.searchsorted(edges, vol[np.isfinite(vol)], side="right")

    by_age = venue_code * (len(AGE_EDGES_S) + 2) + age_bin
    return [
        by_age * (VOLATILITY_BINS + 1) + vol_bin,
        by_age,
        venue_code,
        np.zeros(len(vol), np.int64),
    ]


class Matcher:
    """Draws, for a window, a random window of another token in its finest stratum that has one."""

    def __init__(self, levels: Sequence[np.ndarray], mint: np.ndarray) -> None:
        self.levels = levels
        self.mint = mint
        self.members: list[dict[int, np.ndarray]] = [
            {
                int(code): indices
                for code, indices in pd.Series(np.arange(len(level)))
                .groupby(level)
                .apply(lambda s: s.to_numpy())
                .items()
            }
            for level in levels
        ]
        self.counts = dict.fromkeys(MATCH_LEVELS, 0)

    def draw(self, i: int, rng: np.random.Generator, *, tries: int = 8) -> int | None:
        for name, level, members in zip(MATCH_LEVELS, self.levels, self.members, strict=True):
            pool = members[int(level[i])]
            for j in rng.integers(0, len(pool), size=tries):
                candidate = int(pool[j])
                if self.mint[candidate] != self.mint[i]:
                    self.counts[name] += 1
                    return candidate
            others = pool[self.mint[pool] != self.mint[i]]
            if len(others):
                self.counts[name] += 1
                return int(others[rng.integers(0, len(others))])
        return None


# --------------------------------------------------------------------------- statistics


def bootstrap_means(
    values: np.ndarray, tokens: np.ndarray, *, samples: int, rng: np.random.Generator
) -> np.ndarray:
    """The means of ``samples`` draws of tokens. Empty under two tokens."""
    codes, uniques = pd.factorize(pd.Series(tokens, dtype=object))
    m = len(uniques)
    if m < 2:
        return np.array([], np.float64)
    totals = np.bincount(codes, weights=values, minlength=m)
    counts = np.bincount(codes, minlength=m).astype(np.float64)
    draws = rng.integers(0, m, size=(samples, m))
    return totals[draws].sum(axis=1) / counts[draws].sum(axis=1)


def difference_ci(
    model_means: np.ndarray, other_means: np.ndarray, *, ci: float, rng: np.random.Generator
) -> list[float | None]:
    """CI of model minus other, from each one's own distribution of means."""
    if not len(model_means) or not len(other_means):
        return [None, None]
    samples = max(len(model_means), len(other_means))
    a = rng.choice(model_means, size=samples)
    b = rng.choice(other_means, size=samples)
    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(a - b, [tail, 1.0 - tail])
    return [float(low), float(high)]


def _returns(entries: Sequence[tuple[Any, Outcome]]) -> tuple[np.ndarray, np.ndarray]:
    filled = _filled(entries)
    return (
        np.array([o.net_return for _, o in filled], np.float64),
        np.array([m for m, _ in filled], dtype=object),
    )


def draw_summary(
    means: np.ndarray, trades: np.ndarray, model_mean: float | None, ci: float
) -> dict[str, Any]:
    """Many random draws, against the model's mean. The p-value is one-sided: the share of draws
    that do at least as well as the model, with the draws' usual +1 so that it is never 0."""
    if not len(means):
        return {"draws": 0, "mean_return": None, "mean_return_range": [None, None], "p_value": None}
    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    p_value = None
    if model_mean is not None:
        p_value = float((1 + np.sum(means >= model_mean)) / (len(means) + 1))
    return {
        "draws": len(means),
        "mean_trades": float(trades.mean()),
        "mean_return": float(means.mean()),
        "mean_return_range": [float(low), float(high)],
        "p_value": p_value,
    }


def reading(report: Mapping[str, Any], *, ci_level: float) -> dict[str, Any]:
    """What this one period says, by the rule in docs/ML-MODEL.md."""
    model = report["model"]
    baselines = report["baselines"]
    matched = baselines["matched"]
    low = matched["difference_ci"][0]
    alpha = 1.0 - ci_level
    beats_matched = (
        low is not None
        and low > 0
        and matched["p_value"] is not None
        and matched["p_value"] < alpha
    )
    beaten_by = [
        name
        for name in PLAIN_RULES
        if baselines[name]["difference_ci"][1] is not None
        and baselines[name]["difference_ci"][1] < 0
    ]
    if beaten_by:
        verdict = "a plain rule beats the model"
    elif beats_matched:
        verdict = "the model adds value"
    else:
        verdict = "no evidence that the model adds value"
    return {
        "enough_trades": model["n_trades"] >= EDGE_MIN_TRADES,
        "beats_matched": beats_matched,
        "beaten_by": beaten_by,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- run


def compare(
    df: pd.DataFrame,
    paths: Paths,
    windows: Windows,
    *,
    pump_threshold: float = DEFAULT_PUMP_THRESHOLD,
    horizon_s: float = 900.0,
    rule: ExitRule = EXIT_RULE,
    execution: Execution | None = None,
    draws: int = DEFAULT_DRAWS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> dict[str, Any]:
    """The model's entries and every baseline, on the windows the read doesn't cut short."""
    check_settings(pump_threshold, draws, bootstrap_samples, ci)
    execution = Execution() if execution is None else execution
    rng = np.random.default_rng(seed)
    usable = uncensored(windows, paths, max(horizon_s, rule.timeout_s))

    selected = usable.take(usable.pump_prob >= pump_threshold)
    model_rows = entries_with_rows(paths, selected, rule, execution)
    model_entries = [(m, o) for _, m, o in model_rows]
    target = len(model_rows)
    metrics = {"execution": execution, "samples": bootstrap_samples, "ci": ci, "rng": rng}
    model_cell = cell_metrics(rule, model_entries, **metrics)
    values, tokens = _returns(model_entries)
    model_means = bootstrap_means(values, tokens, samples=bootstrap_samples, rng=rng)
    model_mean = float(values.mean()) if len(values) else None

    plain = without_model(usable)
    baselines: dict[str, Any] = {}
    signals = plain_signals(df, paths)
    for name in PLAIN_RULES:
        chosen = top_windows_for_trades(paths, plain, signals[name], target, rule, execution, rng)
        entries = simulate_rule(paths, chosen, rule, execution)
        arm = cell_metrics(rule, entries, **metrics)
        other, other_tokens = _returns(entries)
        means = bootstrap_means(other, other_tokens, samples=bootstrap_samples, rng=rng)
        arm["n_windows"] = len(chosen)
        arm["difference"] = (
            None if model_mean is None or not len(other) else model_mean - float(other.mean())
        )
        arm["difference_ci"] = difference_ci(model_means, means, ci=ci, rng=rng)
        baselines[name] = arm

    def run_draws(pick) -> dict[str, Any]:
        means, trades = [], []
        for _ in range(draws if target else 0):
            kept = pick()
            if not len(kept):
                continue
            result, _tokens = _returns(
                simulate_rule(paths, plain.take(np.array(kept, np.int64)), rule, execution)
            )
            if len(result):
                means.append(float(result.mean()))
                trades.append(len(result))
        means_a = np.array(means, np.float64)
        summary = draw_summary(means_a, np.array(trades, np.float64), model_mean, ci)
        summary["difference"] = (
            None if model_mean is None or not len(means_a) else model_mean - float(means_a.mean())
        )
        summary["difference_ci"] = difference_ci(model_means, means_a, ci=ci, rng=rng)
        return summary

    n = len(plain)
    baselines["random"] = run_draws(lambda: rng.integers(0, n, size=target) if n else [])

    # Each model entry among the usable windows: a mark belongs to one token, so its row finds it.
    by_row = np.argsort(plain.row, kind="stable")
    rows = np.array([row for row, _, _ in model_rows], np.int64)
    anchors = by_row[np.searchsorted(plain.row[by_row], rows)].tolist()
    if anchors:
        matcher = Matcher(strata(df, paths, plain), plain.mint)

        def pick_matched() -> list[int]:
            drawn = (matcher.draw(i, rng) for i in anchors)
            return [d for d in drawn if d is not None]

        baselines["matched"] = run_draws(pick_matched)
        baselines["matched"]["match_levels"] = dict(matcher.counts)
    else:
        baselines["matched"] = run_draws(lambda: [])
        baselines["matched"]["match_levels"] = dict.fromkeys(MATCH_LEVELS, 0)

    report = {
        "n_windows": len(windows),
        "n_censored": len(windows) - len(usable),
        "model": {**model_cell, "n_windows": len(selected)},
        "baselines": baselines,
    }
    report["reading"] = reading(report, ci_level=ci)
    return report


def check_settings(pump_threshold: float, draws: int, bootstrap_samples: int, ci: float) -> None:
    if not (isinstance(pump_threshold, int | float) and 0.0 < pump_threshold <= 1.0):
        raise ValueError(f"pump threshold must be in (0, 1], got {pump_threshold!r}")
    if draws < 1:
        raise ValueError(f"draws must be at least 1, got {draws!r}")
    if bootstrap_samples < 1:
        raise ValueError(f"bootstrap_samples must be at least 1, got {bootstrap_samples!r}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")


# --------------------------------------------------------------------------- report


def build_report(
    model_version: str, result: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    return json_safe(
        {
            "model_version": model_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "settings": dict(settings),
            **result,
        }
    )


def _band(values: Sequence[float | None]) -> str:
    low, high = values
    return "n/a" if low is None else f"[{low:+.3f},{high:+.3f}]"


def render(report: Mapping[str, Any]) -> str:
    settings = report["settings"]
    model = report["model"]
    lines = [
        f"Is the model useful? {report['model_version']}, entries at pump >= "
        f"{settings['pump_threshold']:g}, exits {settings['exit_rule']}, "
        f"generated {report['generated_at']}",
        f"{report['n_windows']} held-out windows, {report['n_censored']} censored by the read. "
        f"Differences are the model minus the baseline, with a {settings['ci'] * 100:g}% CI.",
        "",
        f"  {'arm':<14} {'trades':>7} {'tokens':>7} {'win':>7} {'mean':>8} {'median':>8} "
        f"{'mean CI / range':>17} {'difference':>10} {'its CI':>17} {'p':>6}",
        f"  {'model':<14} {model['n_trades']:>7} {model['n_tokens']:>7} "
        f"{fmt(model['win_rate']):>7} {fmt(model['mean_return']):>8} "
        f"{fmt(model['median_return']):>8} "
        f"{_band(model['mean_return_ci']):>17}",
    ]
    for name in PLAIN_RULES:
        a = report["baselines"][name]
        lines.append(
            f"  {name:<14} {a['n_trades']:>7} {a['n_tokens']:>7} {fmt(a['win_rate']):>7} "
            f"{fmt(a['mean_return']):>8} {fmt(a['median_return']):>8} "
            f"{_band(a['mean_return_ci']):>17} {fmt(a['difference']):>10} "
            f"{_band(a['difference_ci']):>17}"
        )
    for name in RANDOM_ARMS:
        a = report["baselines"][name]
        trades = f"{a['mean_trades']:.0f}" if a.get("mean_trades") is not None else "0"
        lines.append(
            f"  {name:<14} {trades:>7} {'':>7} {'':>7} {fmt(a['mean_return']):>8} {'':>8} "
            f"{_band(a['mean_return_range']):>17} {fmt(a['difference']):>10} "
            f"{_band(a['difference_ci']):>17} {fmt(a['p_value']):>6}"
        )
    levels = report["baselines"]["matched"]["match_levels"]
    read = report["reading"]
    lines += [
        "",
        "  matched at: " + ", ".join(f"{k} {v}" for k, v in levels.items()),
        f"  this period: {read['verdict']}"
        + ("" if read["enough_trades"] else f" (under {EDGE_MIN_TRADES} trades: too few to read)"),
    ]
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], path: str | Path) -> tuple[Path, Path]:
    """The JSON report at ``path`` and its rendered table next to it (``.txt``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = path.with_suffix(".txt")
    table.write_text(render(report) + "\n", encoding="utf-8")
    return path, table


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare a checkpoint's entries with entries that don't use it (read-only)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="path to a .pt checkpoint")
    source.add_argument("--champion", action="store_true", help="the registry's champion")
    parser.add_argument("--out", help="write the JSON report here (+ the table as .txt)")
    parser.add_argument("--horizon", default=DEFAULT_HORIZON, help="the entry rule's horizon")
    parser.add_argument("--pump-threshold", type=float, default=DEFAULT_PUMP_THRESHOLD)
    parser.add_argument("--lookback-days", type=float, help="override data.lookback_days")
    parser.add_argument("--max-rows", type=int, help="override data.max_rows")
    parser.add_argument(
        "--until", type=_utc, help="end of the read, ISO-8601 with an offset (default: newest row)"
    )
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        default=0.0,
        help="decisions per token: 0 = every snapshot; 180 is close to the live scoring cadence",
    )
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="random draws per arm")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--ci", type=float, default=DEFAULT_CI)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    try:  # fail on bad settings before reading or scoring anything
        check_settings(args.pump_threshold, args.draws, args.bootstrap_samples, args.ci)
        if args.window_stride_seconds < 0:
            raise ValueError(f"window stride must be >= 0, got {args.window_stride_seconds!r}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    from .config import horizon_seconds, load_config
    from .db import DataSettings, make_engine, read_training_features
    from .evaluate import holdout_dataset
    from .inference import Predictor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = make_engine()
    if args.champion:
        try:
            predictor = Predictor.from_champion(engine)
        except LookupError as exc:
            raise SystemExit(str(exc)) from None
    else:
        predictor = Predictor.from_checkpoint(args.checkpoint)
    horizons = list(predictor.horizons)
    if args.horizon not in horizons:
        raise SystemExit(f"horizon {args.horizon!r} is not one of the model's: {horizons}")
    h = horizons.index(args.horizon)

    config = load_config()
    config = {
        **config,
        "data": {**config.get("data", {}), "window_stride_seconds": args.window_stride_seconds},
    }
    settings_read = DataSettings.from_config(
        config, lookback_days=args.lookback_days, max_rows=args.max_rows
    )
    print("reading features")
    df, bounds = read_training_features(engine, settings_read, end=args.until)
    print(bounds.describe())
    print("building the held-out windows")
    dataset = holdout_dataset(df, config, predictor)
    print(f"scoring {len(dataset)} windows on {device}")
    predictions, truth = score_windows(predictor, dataset, device)
    print("comparing the entries")

    execution = Execution()
    paths = Paths.from_frame(df, floor=LiquidityFloor.from_config(config), execution=execution)
    samples, indices = dataset.samples, dataset.indices
    windows = Windows.align(
        paths,
        samples.mint[indices],
        samples.end_ts[indices],
        predictions["pump_prob"][:, h],
        predictions["expected_return"][:, h],
        truth["mask"][:, h] == 1,
    )
    result = compare(
        df,
        paths,
        windows,
        pump_threshold=args.pump_threshold,
        horizon_s=horizon_seconds(args.horizon),
        execution=execution,
        draws=args.draws,
        bootstrap_samples=args.bootstrap_samples,
        ci=args.ci,
        seed=args.seed,
    )
    settings = {
        "pump_threshold": args.pump_threshold,
        "exit_rule": EXIT_RULE.label,
        "window_stride_seconds": args.window_stride_seconds,
        "read": bounds.as_dict(),
        "draws": args.draws,
        "bootstrap_samples": args.bootstrap_samples,
        "ci": args.ci,
        "seed": args.seed,
        "execution": asdict(execution),
    }
    report = build_report(predictor.model_version, result, settings)
    print(render(report))
    if args.out:
        json_path, table_path = write_report(report, args.out)
        print(f"\nreport written to {json_path} and {table_path}")


if __name__ == "__main__":
    main()
