"""Gradient-boosted-trees benchmark (LightGBM) on the Transformer's own walk-forward windows.

**Measurement only** (#205): is the Transformer near what these features allow? A GBDT trained on
the same windows, labels and splits answers it — GBDT ≈ Transformer: the ceiling is the data;
clearly below: the sequence model earns its complexity; clearly above: the model side needs work.
Nothing is saved, registered, served or promoted.

* **Same data, same slices.** The windows and labels are :func:`solia_ai.dataset.config_samples`'s
  under the config, the slices :func:`solia_ai.ablation.walk_forward_windows`' (embargo included),
  so every GBDT run is paired with a Transformer run on the same (slice, seed). The Transformer side
  is :func:`solia_ai.ablation.train_and_score` itself.
* **Tabular inputs.** Per window, over the enabled groups' columns (``features.groups``): the last
  row, and the window's mean / min / max / last − first (padding excluded), plus its length — see
  :func:`tabular_features`.
* **One classifier per head × horizon**, trained on the slice's training windows labeled at that
  horizon, early-stopped (binary log-loss) on its validation windows — the Transformer's own time
  split of the training pool, never the scored slice — and scored on the slice with the ablation's
  ROC-AUC / PR-AUC / ECE (:func:`solia_ai.evaluate.classification_metrics`). Its probabilities are
  LightGBM's own (no extra calibration; the Transformer's are Platt-calibrated on the validation
  windows, as served).
* **Report** (JSON + text, like the ablation): each model's metrics (mean ± std over the runs), the
  paired GBDT − Transformer lift per metric with the ablation's verdict (``mean - std > margin``),
  the GBDT's best iterations and the top columns by gain importance (it also tells where feature
  work pays).

``lightgbm`` is an optional dependency (``pip install -e ".[benchmark]"``): the serving image and
the default install don't carry it, and the command fails with that hint when it is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ablation import (
    DEFAULT_EVAL_FRACTION,
    DEFAULT_MARGIN,
    DEFAULT_SEEDS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WINDOWS,
    Window,
    check_settings,
    decide,
    paired_lifts,
    train_and_score,
    walk_forward_windows,
)
from .dataset import Samples, SequenceDataset, config_samples, embargo_seconds
from .evaluate import (
    DEFAULT_ECE_BINS,
    ReferencePolicy,
    classification_metrics,
    default_policy_horizon,
    fmt,
    json_safe,
)
from .features import BASELINE_GROUPS, FEATURE_COLUMNS, group_columns
from .lift import PREDICTIVE_METRICS, summarize

HEADS = ("pump", "rug")
MODELS = ("transformer", "gbdt")
STATS = ("last", "mean", "min", "max", "delta")
INSTALL_HINT = 'pip install -e ".[benchmark]"'
DEFAULT_TOP_FEATURES = 20
_CHUNK = 4096  # windows widened per pass while building the tabular inputs

# Sensible defaults + early stopping; no search (out of scope for a benchmark).
DEFAULT_GBDT_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
DEFAULT_EARLY_STOPPING_ROUNDS = 100


class MissingDependencyError(ImportError):
    """``lightgbm`` isn't installed (it is an optional extra)."""


def require_lightgbm() -> Any:
    """The ``lightgbm`` module, or :class:`MissingDependencyError` with the install hint."""
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingDependencyError(
            f"the GBDT benchmark needs lightgbm, an optional dependency: {INSTALL_HINT} "
            f"(in services/ai-service; {exc})"
        ) from None
    return lightgbm


# --------------------------------------------------------------------------- tabular inputs


def tabular_columns(columns: Sequence[str]) -> list[str]:
    """Names of :func:`tabular_features`' columns: ``<stat>:<feature>`` then ``window_len``."""
    return [f"{stat}:{col}" for stat in STATS for col in columns] + ["window_len"]


def tabular_features(
    samples: Samples, columns: Sequence[str], *, indices: np.ndarray | None = None
) -> np.ndarray:
    """``[N, 5 * len(columns) + 1]`` float32 inputs of the windows at ``indices`` (default all).

    Per feature column: the window's last row (what the model's newest position sees), its mean,
    min and max over the window's real rows (padding excluded) and last − first (the trend); then
    the window's length in rows. The values are the windows' inputs as served
    (:meth:`Samples.padded_batch`), before the model's normalization — trees don't need it.
    """
    unknown = [c for c in columns if c not in FEATURE_COLUMNS]
    if unknown or not columns:
        raise ValueError(f"tabular columns must be feature columns, got {list(columns)}")
    col = np.array([FEATURE_COLUMNS.index(c) for c in columns], np.int64)
    idx = np.arange(len(samples)) if indices is None else np.asarray(indices, np.int64)
    out = np.empty((len(idx), len(STATS) * len(col) + 1), np.float32)
    for lo in range(0, len(idx), _CHUNK):
        part = idx[lo : lo + _CHUNK]
        x, pad = samples.padded_batch(part)
        x = x[:, :, col].astype(np.float64)
        real = ~pad[:, :, None]
        n = real.sum(axis=1)  # [B, 1]
        first = x[np.arange(len(part)), pad.sum(axis=1)]  # left padding: the first real row
        last = x[:, -1]
        stats = (
            last,
            np.where(real, x, 0.0).sum(axis=1) / n,
            np.where(real, x, np.inf).min(axis=1),
            np.where(real, x, -np.inf).max(axis=1),
            last - first,
        )
        out[lo : lo + len(part), :-1] = np.concatenate(stats, axis=1)
        out[lo : lo + len(part), -1] = n[:, 0]
    return out


# --------------------------------------------------------------------------- GBDT runs


@dataclass(frozen=True)
class GbdtSettings:
    params: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_GBDT_PARAMS))
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS
    n_jobs: int = 0  # 0: LightGBM's default (every core)
    device_type: str = "cpu"  # "gpu" with a GPU build of LightGBM


def fit_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    settings: GbdtSettings,
) -> Any:
    """One LightGBM binary classifier, early-stopped on ``(x_val, y_val)`` (binary log-loss)."""
    lgb = require_lightgbm()
    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=settings.n_jobs or None,
        device_type=settings.device_type,
        verbose=-1,
        **dict(settings.params),
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(settings.early_stopping_rounds, verbose=False)],
    )
    return model


def gbdt_run(
    tab: np.ndarray,
    samples: Samples,
    window: Window,
    *,
    seed: int,
    settings: GbdtSettings,
    n_bins: int = DEFAULT_ECE_BINS,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """GBDT metrics (``<head>/<horizon>/<metric>``) of one (slice, seed), and its diagnostics:
    best iteration and gain importance per head × horizon. A head × horizon whose training or
    validation windows hold a single class (nothing to learn or stop on) scores ``None``."""
    metrics: dict[str, float | None] = {}
    best: dict[str, int | None] = {}
    importance = np.zeros(tab.shape[1], np.float64)
    for h, horizon in enumerate(samples.horizons):
        labeled = samples.mask[:, h] == 1
        tr = window.train_idx[labeled[window.train_idx]]
        va = window.val_idx[labeled[window.val_idx]]
        ev = window.eval_idx[labeled[window.eval_idx]]
        for head in HEADS:
            y = getattr(samples, head)[:, h]
            key = f"{head}/{horizon}"
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 or len(ev) == 0:
                metrics.update({f"{key}/{m}": None for m in PREDICTIVE_METRICS})
                best[key] = None
                continue
            model = fit_head(tab[tr], y[tr], tab[va], y[va], seed=seed, settings=settings)
            prob = model.predict_proba(tab[ev])[:, 1]
            scored = classification_metrics(y[ev], prob, n_bins=n_bins)
            metrics.update({f"{key}/{m}": _finite(scored[m]) for m in PREDICTIVE_METRICS})
            best[key] = int(getattr(model, "best_iteration_", 0) or 0) or None
            gains = model.booster_.feature_importance(importance_type="gain")
            importance += np.asarray(gains, np.float64)
    return metrics, {"best_iteration": best, "importance": importance}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- report


@dataclass
class BenchmarkReport:
    generated_at: str
    settings: dict[str, Any]
    data: dict[str, Any] | None
    windows: list[dict[str, Any]]
    runs: list[dict[str, Any]]  # one per (model, window, seed): its metrics
    summary: dict[str, dict[str, dict[str, Any]]]  # model -> metric -> mean/std/n
    lift: dict[str, dict[str, Any]]  # metric -> GBDT − Transformer paired lift + verdict
    importance: list[dict[str, Any]]  # top columns by summed gain
    best_iterations: dict[str, dict[str, Any]]
    seconds: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    def write(self, path: str | Path) -> tuple[Path, Path]:
        """``<path>`` (JSON) and ``<path minus .json>.txt`` (:meth:`render`)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        text = path.with_suffix(".txt")
        text.write_text(self.render(), encoding="utf-8")
        return path, text

    def render(self) -> str:
        s = self.settings
        lines = [
            "GBDT (LightGBM) vs Transformer — same walk-forward slices, labels and seeds",
            f"generated {self.generated_at}; {s['n_windows']} slice(s) x seeds {s['seeds']}; "
            f"columns: {s['n_columns']} enabled x {list(STATS)} + window_len; "
            f"margin {s['margin']}",
            "",
            f"{'metric':<22} {'transformer':>18} {'gbdt':>18} {'lift (gbdt-tr)':>18}  verdict",
        ]
        for key, lifted in self.lift.items():
            cells = [_pm(self.summary.get(m, {}).get(key, {})) for m in MODELS]
            lines.append(
                f"{key:<22} {cells[0]:>18} {cells[1]:>18} {_pm(lifted):>18}  {lifted['verdict']}"
            )
        lines += ["", f"Top {len(self.importance)} GBDT columns by gain (all heads x horizons):"]
        for rank, item in enumerate(self.importance, 1):
            lines.append(f"  {rank:>2}. {item['column']:<40} {item['share']:.1%}")
        lines += ["", "seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in self.seconds.items())]
        return "\n".join(lines) + "\n"


def _pm(stats: Mapping[str, Any]) -> str:
    if not stats or stats.get("mean") is None:
        return "n/a"
    std = stats.get("std")
    return f"{fmt(stats['mean'])} ± {fmt(std)}" if std is not None else fmt(stats["mean"])


# --------------------------------------------------------------------------- the benchmark


def run_benchmark(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    n_windows: int = DEFAULT_WINDOWS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    margin: float = DEFAULT_MARGIN,
    policy: ReferencePolicy | None = None,
    gbdt: GbdtSettings | None = None,
    device: Any = None,
    top_features: int = DEFAULT_TOP_FEATURES,
    progress: Callable[[str], None] | None = None,
    data: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> BenchmarkReport:
    """Every (slice, seed): the Transformer (:func:`train_and_score`) and the GBDT (one classifier
    per head × horizon) on the same windows; see the module doc. Writes nothing.

    Raises:
        MissingDependencyError: lightgbm isn't installed (checked before any training).
        ValueError: bad settings, or an empty slice / split.
    """
    require_lightgbm()
    seeds = check_settings(
        n_windows=n_windows, seeds=seeds, eval_fraction=eval_fraction,
        val_fraction=val_fraction, margin=margin, pnl_margin=0.0,
    )  # fmt: skip
    if top_features < 1:
        raise ValueError(f"top_features must be >= 1, got {top_features}")
    gbdt = gbdt or GbdtSettings()
    samples = config_samples(df, config)
    horizons = list(samples.horizons)
    policy = policy or ReferencePolicy(horizon=default_policy_horizon(horizons))
    policy.horizon_index(horizons)
    windows = walk_forward_windows(
        samples.end_ts, n_windows=n_windows, eval_fraction=eval_fraction,
        val_fraction=val_fraction, embargo=embargo_seconds(horizons),
    )  # fmt: skip
    columns = group_columns(config.get("features", {}).get("groups", BASELINE_GROUPS))
    names = tabular_columns(columns)
    say = progress or (lambda _msg: None)

    started = clock()
    tab = tabular_features(samples, columns)
    seconds = {"tabular": clock() - started, "transformer": 0.0, "gbdt": 0.0}
    say(f"{len(samples)} windows, {tab.shape[1]} tabular columns, {len(windows)} slice(s)")

    runs: list[dict[str, Any]] = []
    importance = np.zeros(len(names), np.float64)
    best: dict[str, list[int | None]] = {}
    for window in windows:
        for seed in seeds:
            say(f"slice {window.index} seed {seed}: transformer")
            t0 = clock()
            _, tr_metrics = train_and_score(
                SequenceDataset(samples, window.train_idx),
                SequenceDataset(samples, window.val_idx),
                SequenceDataset(samples, window.eval_idx),
                config,
                seed=seed,
                policy=policy,
                device=device,
            )
            seconds["transformer"] += clock() - t0
            say(f"slice {window.index} seed {seed}: gbdt")
            t0 = clock()
            gb_metrics, diag = gbdt_run(tab, samples, window, seed=seed, settings=gbdt)
            seconds["gbdt"] += clock() - t0
            importance += diag["importance"]
            for key, it in diag["best_iteration"].items():
                best.setdefault(key, []).append(it)
            for model, metrics in (("transformer", tr_metrics), ("gbdt", gb_metrics)):
                runs.append({"model": model, "window": window.index, "seed": seed,
                             "metrics": {k: metrics.get(k) for k in _keys(horizons)}})  # fmt: skip

    summary = {
        model: {
            key: summarize(r["metrics"][key] for r in runs if r["model"] == model)
            for key in _keys(horizons)
        }
        for model in MODELS
    }
    lifted = {}
    for key in _keys(horizons):
        by_model = {
            m: [r["metrics"][key] for r in runs if r["model"] == m] for m in MODELS
        }  # same (window, seed) order for both: paired
        lifted[key] = decide(paired_lifts(by_model["gbdt"], by_model["transformer"], key), margin)
    total = importance.sum()
    order = np.argsort(-importance)[:top_features]
    top = [
        {"column": names[i], "gain": float(importance[i]),
         "share": float(importance[i] / total) if total > 0 else 0.0}
        for i in order if importance[i] > 0
    ]  # fmt: skip
    return BenchmarkReport(
        generated_at=datetime.now(UTC).isoformat(),
        settings={
            "n_windows": n_windows, "seeds": seeds, "eval_fraction": eval_fraction,
            "val_fraction": val_fraction, "margin": margin, "horizons": horizons,
            "policy_horizon": policy.horizon, "columns": columns, "n_columns": len(columns),
            "stats": list(STATS), "gbdt": {"params": dict(gbdt.params),
            "early_stopping_rounds": gbdt.early_stopping_rounds, "device_type": gbdt.device_type},
            "n_windows_total": len(samples),
        },  # fmt: skip
        data=None if data is None else dict(data),
        windows=[w.describe() for w in windows],
        runs=runs,
        summary=summary,
        lift=lifted,
        importance=top,
        best_iterations={k: summarize(v) for k, v in best.items()},
        seconds=seconds,
    )


def _keys(horizons: Sequence[str]) -> list[str]:
    return [f"{head}/{h}/{m}" for head in HEADS for h in horizons for m in PREDICTIVE_METRICS]


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    from .db import (
        DataSettings,
        add_data_bounds_arguments,
        add_prune_columns_argument,
        make_engine,
        read_training_features,
        training_columns,
        with_prune_columns,
    )

    parser = argparse.ArgumentParser(
        description="Benchmark a LightGBM model against the Transformer on the same walk-forward "
        "windows (measurement only: nothing is saved, registered or served)."
    )
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--policy-horizon", help="reference-policy horizon (Transformer runs)")
    parser.add_argument("--top-features", type=int, default=DEFAULT_TOP_FEATURES)
    parser.add_argument("--gbdt-device", choices=("cpu", "gpu"), default="cpu",
                        help="LightGBM device_type (gpu needs a GPU build)")  # fmt: skip
    parser.add_argument("--gbdt-jobs", type=int, default=os.cpu_count() or 0)
    add_data_bounds_arguments(parser)
    add_prune_columns_argument(parser)
    parser.add_argument("--out", help="write the JSON report (and a .txt summary) to this path")
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    try:
        require_lightgbm()
    except MissingDependencyError as exc:
        raise SystemExit(str(exc)) from None

    import torch

    from .config import load_config

    config = with_prune_columns(load_config(), args.prune_columns)
    try:
        check_settings(
            n_windows=args.windows, seeds=args.seeds, eval_fraction=args.eval_fraction,
            val_fraction=args.val_fraction, margin=args.margin, pnl_margin=0.0,
        )  # fmt: skip
        policy = ReferencePolicy(
            horizon=args.policy_horizon or default_policy_horizon(config["horizons"])
        )
        policy.horizon_index(config["horizons"])
        data = DataSettings.from_config(
            config, lookback_days=args.lookback_days, max_rows=args.max_rows
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    df, bounds = read_training_features(make_engine(), data, columns=training_columns(config))
    print(bounds.describe(), file=sys.stderr)
    try:
        report = run_benchmark(
            df, config, n_windows=args.windows, seeds=args.seeds,
            eval_fraction=args.eval_fraction, val_fraction=args.val_fraction, margin=args.margin,
            policy=policy, gbdt=GbdtSettings(n_jobs=args.gbdt_jobs, device_type=args.gbdt_device),
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            top_features=args.top_features, progress=lambda m: print(m, file=sys.stderr),
            data=bounds.as_dict(),
        )  # fmt: skip
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(report.render())
    if args.out:
        json_path, text_path = report.write(args.out)
        print(f"report written to {json_path} and {text_path}")


if __name__ == "__main__":
    main()
