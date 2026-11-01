"""Hyperparameter & labeling-threshold search, scored by the ablation harness.

Once the feature set is settled, the remaining lift is in the model shape, the optimizer and the
labels. This module tries a bounded number of configurations and reports the best one — with its
spread, never a single lucky run — as a *proposed* ``config.yaml`` diff. Nothing is applied,
saved, registered or promoted.

* **Search space** — ``seq_len``, ``d_model``, ``n_layers``, ``dropout``, ``learning_rate``,
  ``weight_decay``, the training-window stride (``window_stride_seconds``, ``data``) and the
  labeling thresholds, as one multiplicative scale per head over every horizon
  (``pump_threshold_scale`` / ``rug_threshold_scale``: the term structure across horizons is kept).
  Each parameter is either ``{choices: [...]}`` or a ``{low, high[, log]}`` range; parameters left
  out keep the base config's value. ``ff_dim`` follows ``d_model`` at the base ratio, and
  ``min_seq_len`` stays the base one, so every trial scores the same windows.
* **Strategy** — random (seeded) or grid (every combination of ``choices``, in a seeded shuffled
  order). The base config always runs first, as trial ``baseline``.
* **Budget** — at most ``n_trials`` trials (baseline included) and, optionally, ``max_seconds`` of
  wall time: no trial *starts* once it is spent (a started one finishes).
* **Scoring** — every trial goes through the ablation harness's walk-forward windows and seeds
  (:func:`solia_ai.ablation.train_and_score`): same windows, same seeds, paired run for run. The
  objective of a run is the mean ``roc_auc`` (or ``pr_auc``) over both heads and every horizon; a
  trial's score is ``mean - std`` of its runs, so a noisy config doesn't win on luck. The best
  trial is compared to the baseline with the ablation's paired-lift verdict.
* **Row budget** — ``max_rows`` (``data``) caps the rows a trial trains on the way the bounded
  read caps them (:func:`solia_ai.db.read_training_features`: whole tokens in ``md5(mint)`` order),
  out of one frame read at the largest budget of the search, so "stride + more tokens for the same
  rows" is compared with today's read on the same held-out windows. A trial without it uses the
  base config's ``data.max_rows``.
* **Stride** — a stride trial trains and early-stops on its own (thinned) windows but, like a
  threshold trial, is scored on the base config's windows: every trial is measured on the same
  held-out windows.
* **Fit diagnostics** — each trial reports its best epoch and the validation - train loss gap at it
  (overfitting shows as an early best epoch and a wide gap), and the rug metrics per walk-forward
  slice for the baseline and the best trial (drift across time).
* **Resume** — with a checkpoint (the CLI: ``--out`` gives ``<out>.trials.jsonl``), each finished
  trial is appended to a trials file as it ends. A rerun of the same search (same settings, base
  config and data window: the CLI re-reads the window the search started on) skips the trials
  already there, so a reboot loses at most the trial in progress; a different search refuses the
  file rather than mixing two searches.
* **Failures** — a candidate trial that raises (e.g. out of GPU memory) is recorded as failed, with
  its error, and the search goes on; the baseline failing still stops it (nothing to compare to).
* **Labels** — a threshold change alters the *target*. Training (and calibration) use the trial's
  labels, but every trial is **scored against the base config's labels** on the same windows, so
  the metrics keep one definition across trials: "how well does this model rank the reference
  events". ECE is reported but not an objective: a model trained on other labels is calibrated for
  another event.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import itertools
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import runlog
from .ablation import (
    DEFAULT_EVAL_FRACTION,
    DEFAULT_MARGIN,
    DEFAULT_SEEDS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WINDOWS,
    FIT_METRICS,
    HEADS,
    RunResult,
    check_settings,
    decide,
    metric_keys,
    paired_lifts,
    summarize,
    train_and_score,
    walk_forward_windows,
)
from .config import ConfigError, validate_config
from .dataset import SequenceDataset, config_samples, embargo_seconds, window_stride
from .entry_selection import entry_keys
from .evaluate import ReferencePolicy, default_policy_horizon, fmt, json_safe
from .lift import PREDICTIVE_METRICS

INT_PARAMS = ("seq_len", "d_model", "n_layers", "max_rows")
FLOAT_PARAMS = (
    "dropout",
    "learning_rate",
    "weight_decay",
    "window_stride_seconds",
    "pump_threshold_scale",
    "rug_threshold_scale",
)
PARAMS = INT_PARAMS + FLOAT_PARAMS
OBJECTIVES = ("roc_auc", "pr_auc")
STRATEGIES = ("random", "grid")
BASELINE = "baseline"

DEFAULT_SPACE: dict[str, dict[str, Any]] = {
    "seq_len": {"choices": [32, 64, 96, 128]},
    "d_model": {"choices": [64, 128, 192]},
    "n_layers": {"choices": [2, 3, 4, 6]},
    "dropout": {"low": 0.0, "high": 0.3},
    "learning_rate": {"low": 1e-4, "high": 1e-3, "log": True},
    "weight_decay": {"low": 1e-3, "high": 0.1, "log": True},
    "window_stride_seconds": {"choices": [0, 30, 60]},
    "pump_threshold_scale": {"choices": [0.75, 1.0, 1.25]},
    "rug_threshold_scale": {"choices": [0.85, 1.0, 1.15]},
}
DEFAULT_TRIALS = 20
DEFAULT_SEARCH_SEED = 0
_MAX_DRAWS_PER_TRIAL = 50  # random draws per wanted trial before the space counts as exhausted


# --------------------------------------------------------------------------- search space


def _min_seq_len(model: Mapping[str, Any]) -> int:
    return int(model.get("min_seq_len", model["seq_len"]))


def _value_error(name: str, value: Any, base: Mapping[str, Any]) -> str | None:
    """Why ``value`` is invalid for parameter ``name`` against the ``base`` config, or None."""
    if name in INT_PARAMS:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"must be an integer, got {value!r}"
    elif isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return f"must be a number, got {value!r}"
    model = base["model"]
    if name == "seq_len" and value < _min_seq_len(model):
        return f"must be >= the base min_seq_len ({_min_seq_len(model)}), got {value}"
    if name == "d_model" and (value <= 0 or value % model["n_heads"]):
        return f"must be a positive multiple of n_heads ({model['n_heads']}), got {value}"
    if name in ("n_layers", "max_rows") and value < 1:
        return f"must be >= 1, got {value}"
    if name == "dropout" and not 0 <= value < 1:
        return f"must be in [0, 1), got {value}"
    if name in ("learning_rate", "pump_threshold_scale", "rug_threshold_scale") and value <= 0:
        return f"must be > 0, got {value}"
    if name in ("weight_decay", "window_stride_seconds") and value < 0:
        return f"must be >= 0, got {value}"
    if name == "rug_threshold_scale":
        top = max(base["labeling"]["rug_threshold"].values())
        if value * top >= 1:
            return f"must keep every rug threshold below 1 (largest base {top}), got {value}"
    return None


def check_space(
    space: Mapping[str, Any], base: Mapping[str, Any], *, strategy: str = "random"
) -> dict[str, dict[str, Any]]:
    """Validate a search space against the ``base`` config; returns it normalized.

    Raises ``ValueError`` on an empty space, an unknown parameter, a spec that is neither
    ``choices`` nor ``low``/``high``, an empty or duplicated choice list, ``low > high``, a log
    range not above 0, a ``d_model`` range (it must divide ``n_heads``: list choices), any value
    the model or labels can't take, or a range under ``strategy='grid'``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}, got {strategy!r}")
    if not isinstance(space, Mapping) or not space:
        raise ValueError("the search space needs at least one parameter")
    unknown = [str(k) for k in space if k not in PARAMS]
    if unknown:
        raise ValueError(
            f"unknown search parameter(s): {', '.join(unknown)} (known: {', '.join(PARAMS)})"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, spec in space.items():

        def fail(problem: str, name: str = name) -> None:
            raise ValueError(f"search parameter {name!r}: {problem}")

        if not isinstance(spec, Mapping):
            fail("must be a mapping with 'choices' or 'low'/'high'")
        keys = set(spec)
        if keys == {"choices"}:
            choices = spec["choices"]
            if not isinstance(choices, list) or not choices:
                fail("'choices' must be a non-empty list")
            if len({json.dumps(c) for c in choices}) != len(choices):
                fail("'choices' has duplicates")
            for value in choices:
                if problem := _value_error(name, value, base):
                    fail(problem)
            normalized[name] = {"choices": list(choices)}
        elif keys in ({"low", "high"}, {"low", "high", "log"}):
            if strategy == "grid":
                fail("grid search needs 'choices', not a range")
            if name == "d_model":
                fail("needs explicit 'choices' (each must be a multiple of n_heads)")
            low, high, log = spec["low"], spec["high"], spec.get("log", False)
            if not isinstance(log, bool):
                fail(f"'log' must be a boolean, got {log!r}")
            for value in (low, high):
                if problem := _value_error(name, value, base):
                    fail(problem)
            if low > high:
                fail(f"low ({low}) must be <= high ({high})")
            if log and low <= 0:
                fail(f"a log range needs low > 0, got {low}")
            normalized[name] = {"low": low, "high": high, "log": log}
        else:
            fail("must have either 'choices' or 'low'/'high' (and optionally 'log')")
    return normalized


def sample_params(space: Mapping[str, Mapping[str, Any]], rng: np.random.Generator) -> dict:
    """One random point of a normalized ``space`` (parameters in space order)."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if "choices" in spec:
            params[name] = spec["choices"][int(rng.integers(len(spec["choices"])))]
        elif name in INT_PARAMS:
            params[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif spec["log"]:
            value = math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
            params[name] = float(f"{value:.3g}")
        else:
            params[name] = round(float(rng.uniform(spec["low"], spec["high"])), 4)
    return params


def candidate_params(
    space: Mapping[str, Mapping[str, Any]], *, strategy: str, seed: int
) -> Iterator[dict[str, Any]]:
    """Every point to try after the baseline, deterministic for ``seed``: a seeded shuffle of the
    grid, or random draws (duplicates skipped; it stops once draws stop finding new points)."""
    rng = np.random.default_rng(seed)
    if strategy == "grid":
        names = list(space)
        grid = [
            dict(zip(names, values, strict=True))
            for values in itertools.product(*(space[n]["choices"] for n in names))
        ]
        for i in rng.permutation(len(grid)):
            yield grid[int(i)]
        return
    seen: set[str] = set()
    misses = 0
    while misses < _MAX_DRAWS_PER_TRIAL:
        params = sample_params(space, rng)
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            misses += 1
            continue
        seen.add(key)
        misses = 0
        yield params


def apply_params(base: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """The ``base`` config with ``params`` applied (validated; ``base`` is not modified).

    ``ff_dim`` keeps the base ``ff_dim / d_model`` ratio, ``min_seq_len`` stays the base one, a
    threshold scale multiplies that head's threshold on every horizon, and the stride and row budget
    go to ``data.window_stride_seconds`` / ``data.max_rows``.
    """
    cfg = copy.deepcopy(dict(base))
    model, training, labeling = cfg["model"], cfg["training"], cfg["labeling"]
    model["min_seq_len"] = _min_seq_len(base["model"])
    for name in ("seq_len", "d_model", "n_layers", "dropout"):
        if name in params:
            model[name] = params[name]
    if "d_model" in params and "ff_dim" in base["model"]:
        ratio = base["model"]["ff_dim"] / base["model"]["d_model"]
        model["ff_dim"] = max(1, round(ratio * params["d_model"]))
    for name in ("learning_rate", "weight_decay"):
        if name in params:
            training[name] = params[name]
    for name in ("window_stride_seconds", "max_rows"):
        if name in params:
            cfg["data"] = {**cfg.get("data", {}), name: params[name]}
    for head in HEADS:
        scale = params.get(f"{head}_threshold_scale")
        if scale is not None:
            labeling[f"{head}_threshold"] = {
                h: round(float(v) * scale, 6)
                for h, v in base["labeling"][f"{head}_threshold"].items()
            }
    try:
        validate_config(cfg, f"trial {dict(params)}")
    except ConfigError as exc:
        raise ValueError(str(exc)) from None
    return cfg


def proposed_diff(base: Mapping[str, Any], proposed: Mapping[str, Any]) -> str:
    """A unified diff of the two configs as YAML (empty when they are equal). For review only:
    it is never applied to ``config.yaml``."""
    before = yaml.safe_dump(json_safe(base), sort_keys=False).splitlines(keepends=True)
    after = yaml.safe_dump(json_safe(proposed), sort_keys=False).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(before, after, fromfile="config.yaml", tofile="config.yaml (proposed)")
    )


def budget_tokens(mints: pd.Series, max_rows: int) -> set[str]:
    """The tokens a ``max_rows`` bounded read keeps out of ``mints`` (one entry per row): whole
    tokens in ``md5(mint)`` order while the running row total fits — the rule of
    :func:`solia_ai.db.read_training_features`, so a budget at most the one read gives the same
    tokens as reading with it."""
    counts = mints.dropna().value_counts(sort=False)
    order = sorted(counts.index, key=lambda m: hashlib.md5(str(m).encode("utf-8")).hexdigest())
    running = counts[order].cumsum()
    return set(running.index[running.to_numpy() <= max_rows])


def read_rows(space: Mapping[str, Mapping[str, Any]], max_rows: int) -> int:
    """The rows to read for a search: the base budget or the space's largest ``max_rows``."""
    spec = space.get("max_rows")
    if spec is None:
        return max_rows
    return max(max_rows, *(spec["choices"] if "choices" in spec else (spec["high"],)))


# --------------------------------------------------------------------------- objective


def run_objective(
    metrics: Mapping[str, float | None], objective: str, horizons: Sequence[str]
) -> float | None:
    """Mean ``objective`` over both heads and every horizon of one run (undefined ones skipped);
    ``None`` when none is defined."""
    values = [metrics.get(f"{head}/{h}/{objective}") for head in HEADS for h in horizons]
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def robust_score(stats: Mapping[str, Any]) -> float | None:
    """``mean - std`` of a trial's objective (``mean`` alone for a single run): the ranking key."""
    if stats["mean"] is None:
        return None
    return stats["mean"] - (stats["std"] or 0.0)


# --------------------------------------------------------------------------- report


@dataclass
class Trial:
    name: str
    params: dict[str, Any]  # what the trial overrides (empty for the baseline)
    runs: list[RunResult]
    objective: dict[str, Any]  # mean ± std of the per-run objective
    score: float | None
    summary: dict[str, dict[str, Any]]  # per metric: mean ± std over the runs
    seconds: float
    # Fit diagnostics, mean ± std over the runs: best_epoch (1-based) and fit/* (FIT_METRICS).
    fit: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per walk-forward slice (oldest first): the rug metrics, mean over the seeds — drift in time.
    drift: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trial:
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**{**fields, "runs": [RunResult(**r) for r in data["runs"]]})


@dataclass
class FailedTrial:
    """A candidate trial that raised: reported, never ranked."""

    name: str
    params: dict[str, Any]
    error: str
    seconds: float


@dataclass
class SearchReport:
    """Every trial, the ranking, the best trial vs the baseline, and the proposed config diff."""

    generated_at: str
    settings: dict[str, Any]
    horizons: list[str]
    windows: list[dict[str, Any]]
    trials: list[Trial] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)
    best: str = BASELINE
    stopped: str = "trials"  # what ended the search: trials | time | space
    failed: list[FailedTrial] = field(default_factory=list)
    resumed: list[str] = field(default_factory=list)  # taken from the checkpoint, not rerun
    elapsed_seconds: float = 0.0  # every trial's time, resumed ones included
    # Best vs baseline, paired run for run: the objective, then every metric (positive = better).
    comparison: dict[str, dict[str, Any]] = field(default_factory=dict)
    proposed_config: dict[str, Any] = field(default_factory=dict)
    proposed_diff: str = ""

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
        s = self.settings
        objective = s["objective"]
        lines = [
            f"Search ({s['strategy']}) — {len(self.trials)} trial(s) [stopped by {self.stopped}], "
            f"{len(self.windows)} window(s) x {len(s['seeds'])} seed(s); objective = mean "
            f"{objective} over heads x horizons, scored on the base labels; rank by mean - std",
            f"generated {self.generated_at}",
            "",
            f"{'trial':>10} | {objective + ' mean ± std':>20} | {'score':>8} | {'best ep':>7} | "
            f"{'val-train gap':>13} | params",
        ]
        by_name = {t.name: t for t in self.trials}
        for name in self.ranking:
            t = by_name[name]
            params = ", ".join(f"{k}={v}" for k, v in t.params.items()) or "(base config)"
            epoch = _mean(t.fit.get("best_epoch"))
            gap = _mean(t.fit.get("fit/gap"))
            lines.append(
                f"{name:>10} | {_pm(t.objective):>20} | {fmt(t.score):>8} | {epoch:>7} | "
                f"{gap:>13} | {params}"
            )
        if self.resumed:
            lines += ["", f"Resumed from the checkpoint (not rerun): {', '.join(self.resumed)}"]
        for f in self.failed:
            params = ", ".join(f"{k}={v}" for k, v in f.params.items())
            lines += ["", f"FAILED {f.name} ({params}) after {f.seconds:.0f} s: {f.error}"]
        for name in dict.fromkeys((BASELINE, self.best)):
            lines += ["", *render_drift(by_name[name], self.horizons)]
        if self.best == BASELINE:
            lines += ["", "Best: the baseline — no change proposed."]
            return "\n".join(lines)
        verdict = self.comparison["objective"]
        lines += [
            "",
            f"Best: {self.best} vs baseline — objective lift {_pm(verdict)} "
            f"(margin {s['margin']}): {verdict['verdict']}",
            f"{'metric':>22} | {'lift':>17} | verdict",
        ]
        for key, lift in self.comparison.items():
            if key != "objective":
                lines.append(f"{key:>22} | {_pm(lift):>17} | {lift['verdict']}")
        lines += ["", "Proposed config.yaml diff (NOT applied):", self.proposed_diff.rstrip()]
        return "\n".join(lines)


def _pm(stats: Mapping[str, Any]) -> str:
    if stats["mean"] is None:
        return "n/a"
    return f"{fmt(stats['mean'])} ± {fmt(stats['std'])}"


def _mean(stats: Mapping[str, Any] | None) -> str:
    return "n/a" if stats is None or stats["mean"] is None else fmt(stats["mean"])


DRIFT_METRICS = ("roc_auc", "pr_auc")


def render_drift(trial: Trial, horizons: Sequence[str]) -> list[str]:
    """The rug ROC-AUC / PR-AUC of ``trial`` per walk-forward slice (rows) and horizon."""
    lines = [
        f"Rug drift — {trial.name}: {' / '.join(DRIFT_METRICS)} per held-out slice",
        f"{'slice':>17} | " + " | ".join(f"{h:>13}" for h in horizons),
    ]
    for row in trial.drift:
        cells = ["/".join(fmt(row.get(f"rug/{h}/{m}")) for m in DRIFT_METRICS) for h in horizons]
        span = f"{row['eval_start'][5:16]}..{row['eval_end'][11:16]}"
        lines.append(f"{span:>17} | " + " | ".join(f"{c:>13}" for c in cells))
    return lines


def drift_by_window(
    runs: Sequence[RunResult], windows: Sequence[Mapping[str, Any]], horizons: Sequence[str]
) -> list[dict[str, Any]]:
    """Per walk-forward slice (``windows``: :meth:`Window.describe`), every rug predictive metric
    averaged over that slice's runs (one per seed)."""
    keys = [f"rug/{h}/{m}" for h in horizons for m in PREDICTIVE_METRICS]
    rows = []
    for window in windows:
        in_slice = [r for r in runs if r.window == window["index"]]
        row: dict[str, Any] = {
            "window": window["index"],
            "eval_start": window["eval_start"],
            "eval_end": window["eval_end"],
        }
        row.update({k: summarize(r.metrics.get(k) for r in in_slice)["mean"] for k in keys})
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- resume


def checkpoint_path(out: str | Path) -> Path:
    """The trials file of a search reported to ``out``: ``x.json`` -> ``x.trials.jsonl``."""
    out = Path(out)
    return out.with_name(f"{out.stem}.trials.jsonl")


def read_checkpoint_header(path: str | Path) -> dict[str, Any] | None:
    """The fingerprint a trials file was started with; None when there is no file yet."""
    return runlog.read_header(path, "a search trials file")


class TrialLog:
    """The finished trials of one search, one JSON line each, appended (and synced) as they end.

    The first line is the search's fingerprint (settings, base config, data window). Opening a
    file written by another search raises ``ValueError``; a torn last line (the process died
    mid-write) is ignored, so that trial simply runs again.
    """

    def __init__(
        self, path: Path, finished: dict[str, Trial | FailedTrial], windows: list[dict[str, Any]]
    ):
        self.path = path
        self.finished = finished
        self.windows = windows  # the baseline's described walk-forward windows

    @classmethod
    def open(cls, path: str | Path, fingerprint: Mapping[str, Any]) -> TrialLog:
        records = runlog.open_log(
            path, fingerprint, what="a search trials file", other="another search"
        )
        finished: dict[str, Trial | FailedTrial] = {}
        windows: list[dict[str, Any]] = []
        for record in records:
            kind = record.pop("kind")
            if kind == "failed":
                finished[record["name"]] = FailedTrial(**record)
            else:
                windows = record.pop("windows", None) or windows
                finished[record["name"]] = Trial.from_dict(record)
        return cls(Path(path), finished, windows)

    def record(
        self, trial: Trial | FailedTrial, windows: list[dict[str, Any]] | None = None
    ) -> None:
        record = {"kind": "failed" if isinstance(trial, FailedTrial) else "trial", **asdict(trial)}
        if windows is not None:
            record["windows"] = windows
        self.finished[trial.name] = trial
        runlog.append(self.path, json_safe(record))


# --------------------------------------------------------------------------- orchestration


def run_search(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    space: Mapping[str, Any] | None = None,
    strategy: str = "random",
    n_trials: int = DEFAULT_TRIALS,
    max_seconds: float | None = None,
    search_seed: int = DEFAULT_SEARCH_SEED,
    objective: str = "roc_auc",
    n_windows: int = DEFAULT_WINDOWS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    margin: float = DEFAULT_MARGIN,
    policy: ReferencePolicy | None = None,
    device: Any = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    checkpoint: str | Path | None = None,
    data: Mapping[str, Any] | None = None,
) -> SearchReport:
    """Run the baseline then up to ``n_trials - 1`` candidates within ``max_seconds``; see the
    module doc. ``checkpoint``: the trials file to resume from and append to; ``data`` (the read's
    bounds) is part of what must match to resume. Raises ``ValueError`` on bad settings, an invalid
    space or another search's checkpoint before any training."""
    seeds = check_settings(
        n_windows=n_windows,
        seeds=seeds,
        eval_fraction=eval_fraction,
        val_fraction=val_fraction,
        margin=margin,
        pnl_margin=0.0,
    )
    space = check_space(DEFAULT_SPACE if space is None else space, config, strategy=strategy)
    check_budget(n_trials=n_trials, max_seconds=max_seconds, objective=objective)

    reference = config_samples(df, config)  # the scored windows and their (base) labels
    horizons = list(reference.horizons)
    policy = policy or ReferencePolicy(horizon=default_policy_horizon(horizons))
    policy.horizon_index(horizons)
    embargo = embargo_seconds(horizons)

    def run_trial(name: str, params: Mapping[str, Any]) -> tuple[Trial, list[dict[str, Any]]]:
        started = clock()
        cfg = apply_params(config, params)
        train_samples = config_samples(df, cfg)
        # Scored on the base labels and the base stride: same windows, same target definition for
        # every trial.
        base_stride = window_stride(config)
        eval_samples = (
            train_samples
            if cfg["labeling"] == config["labeling"] and window_stride(cfg) == base_stride
            else config_samples(
                df,
                {
                    **cfg,
                    "labeling": config["labeling"],
                    "data": {**cfg.get("data", {}), "window_stride_seconds": base_stride},
                },
            )
        )
        if not np.array_equal(eval_samples.end_ts, reference.end_ts):
            raise ValueError(f"trial {name} would not score the reference windows")
        windows = walk_forward_windows(
            train_samples.end_ts,
            eval_end_ts=eval_samples.end_ts,
            n_windows=n_windows,
            eval_fraction=eval_fraction,
            val_fraction=val_fraction,
            embargo=embargo,
        )
        budget = cfg.get("data", {}).get("max_rows")
        trained = None  # windows of the tokens the trial's row budget keeps (None: every one)
        if budget is not None and budget < df["mint"].notna().sum():
            trained = np.isin(train_samples.mint, list(budget_tokens(df["mint"], budget)))
        runs = []
        for window in windows:
            train_idx, val_idx = window.train_idx, window.val_idx
            if trained is not None:
                train_idx, val_idx = train_idx[trained[train_idx]], val_idx[trained[val_idx]]
                if len(train_idx) == 0 or len(val_idx) == 0:
                    raise ValueError(
                        f"trial {name}: max_rows {budget} leaves window {window.index} an empty "
                        "train/validation split; raise it"
                    )
            train_ds = SequenceDataset(train_samples, train_idx)
            val_ds = SequenceDataset(train_samples, val_idx)
            eval_ds = SequenceDataset(eval_samples, window.eval_idx)
            for seed in seeds:
                if progress:
                    progress(f"{name} window {window.index} seed {seed}")
                best_epoch, metrics = train_and_score(
                    train_ds, val_ds, eval_ds, cfg, seed=seed, policy=policy, device=device
                )
                runs.append(RunResult(name, window.index, seed, best_epoch, metrics))
        stats = summarize(run_objective(r.metrics, objective, horizons) for r in runs)
        keys = metric_keys(horizons)
        described = [w.describe() for w in windows]
        fit = {"best_epoch": summarize(r.best_epoch + 1 for r in runs)}
        for m in FIT_METRICS:
            fit[f"fit/{m}"] = summarize(r.metrics.get(f"fit/{m}") for r in runs)
        trial = Trial(
            name=name,
            params=dict(params),
            runs=runs,
            objective=stats,
            score=robust_score(stats),
            summary={
                k: summarize(r.metrics.get(k) for r in runs) for k in [*keys, *entry_keys(horizons)]
            },
            seconds=clock() - started,
            fit=fit,
            drift=drift_by_window(runs, described, horizons),
        )
        return trial, described

    settings = {
        "strategy": strategy,
        "space": space,
        "n_trials": n_trials,
        "max_seconds": max_seconds,
        "search_seed": search_seed,
        "objective": objective,
        "n_windows": n_windows,
        "seeds": seeds,
        "eval_fraction": eval_fraction,
        "val_fraction": val_fraction,
        "margin": margin,
        "policy": {
            "horizon": policy.horizon,
            "pump_threshold": policy.pump_threshold,
            "rug_threshold": policy.rug_threshold,
        },
        "embargo_seconds": embargo,
        "eval_labels": {
            "pump_threshold": dict(config["labeling"]["pump_threshold"]),
            "rug_threshold": dict(config["labeling"]["rug_threshold"]),
        },
    }
    log = (
        None
        if checkpoint is None
        else TrialLog.open(checkpoint, {"settings": settings, "config": config, "data": data})
    )
    resumed: list[str] = []

    def from_log(name: str, params: Mapping[str, Any]) -> Trial | FailedTrial | None:
        done = None if log is None else log.finished.get(name)
        if done is None:
            return None
        if runlog.canonical(done.params) != runlog.canonical(params):
            raise ValueError(
                f"{log.path}: {name} ran with {done.params}, this search would run {dict(params)}; "
                "delete it (--restart) to start over"
            )
        resumed.append(name)
        if progress:
            progress(f"{name}: already finished, taken from {log.path}")
        return done

    start = clock()
    done = from_log(BASELINE, {})
    if isinstance(done, Trial):
        baseline, windows = done, log.windows
    else:
        baseline, windows = run_trial(BASELINE, {})
        if log is not None:
            log.record(baseline, windows)
    trials = [baseline]
    failed: list[FailedTrial] = []
    resumed_seconds = baseline.seconds if resumed else 0.0  # spent before this process started
    stopped = "space"
    candidates = candidate_params(space, strategy=strategy, seed=search_seed)
    while True:
        count = len(trials) + len(failed)
        if count >= n_trials:
            stopped = "trials"
            break
        if max_seconds is not None and clock() - start + resumed_seconds >= max_seconds:
            stopped = "time"
            break
        params = next(candidates, None)
        if params is None:
            break
        name = f"trial-{count:02d}"
        done = from_log(name, params)
        if done is not None:
            resumed_seconds += done.seconds
        else:
            if progress:
                progress(f"[{count + 1}/{n_trials}] {name}: {params}")
            started = clock()
            try:
                done = run_trial(name, params)[0]
            except (
                Exception
            ) as exc:  # one bad trial (e.g. GPU out of memory) must not end the night
                error = f"{type(exc).__name__}: {exc}"
                done = FailedTrial(name, dict(params), error, clock() - started)
                if progress:
                    progress(f"{name} FAILED: {error}")
            if log is not None:
                log.record(done)
        (failed if isinstance(done, FailedTrial) else trials).append(done)

    ranking = [
        t.name
        for t in sorted(
            trials, key=lambda t: (t.score is None, -(t.score or 0.0), t.name != BASELINE)
        )
    ]
    best = next(t for t in trials if t.name == ranking[0])
    if best.score is None:
        best = baseline
    comparison: dict[str, dict[str, Any]] = {}
    proposed = apply_params(config, best.params)
    if best is not baseline:
        objective_lifts = [
            None if a is None or b is None else a - b
            for a, b in zip(
                (run_objective(r.metrics, objective, horizons) for r in best.runs),
                (run_objective(r.metrics, objective, horizons) for r in baseline.runs),
                strict=True,
            )
        ]
        comparison["objective"] = decide(objective_lifts, margin)
        for key in metric_keys(horizons):
            comparison[key] = decide(
                paired_lifts(
                    [r.metrics.get(key) for r in best.runs],
                    [r.metrics.get(key) for r in baseline.runs],
                    key,
                ),
                0.0 if key.startswith("policy/") else margin,
            )

    return SearchReport(
        generated_at=datetime.now(UTC).isoformat(),
        settings=settings,
        horizons=horizons,
        windows=windows,
        trials=trials,
        ranking=ranking,
        best=best.name,
        stopped=stopped,
        failed=failed,
        resumed=resumed,
        elapsed_seconds=sum(t.seconds for t in [*trials, *failed]),
        comparison=comparison,
        proposed_config=proposed if best is not baseline else {},
        proposed_diff=proposed_diff(config, proposed) if best is not baseline else "",
    )


def check_budget(*, n_trials: int, max_seconds: float | None, objective: str) -> None:
    """Validate the budget and objective without touching any data."""
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"n_trials must be an integer >= 1, got {n_trials!r}")
    if max_seconds is not None and not max_seconds > 0:
        raise ValueError(f"max_seconds must be > 0, got {max_seconds!r}")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}, got {objective!r}")


def load_space(path: str | Path) -> dict[str, Any]:
    """A search space from a YAML file (``param: {choices: [...]}`` or ``{low, high, log}``): a
    path, else the name of a space shipped in ``solia_ai/sweeps`` (e.g. ``overfitting-data``)."""
    shipped = resources.files("solia_ai").joinpath("sweeps", f"{Path(path).stem}.yaml")
    if not Path(path).exists() and Path(path).parent == Path(".") and shipped.is_file():
        path = Path(str(shipped))
    try:
        space = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read search space {path}: {exc}") from exc
    if not isinstance(space, dict):
        raise ValueError(f"search space {path} must be a mapping of parameter -> spec")
    return space


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    from .db import DataSettings, add_data_bounds_arguments, make_engine, read_training_features

    parser = argparse.ArgumentParser(
        description="Search hyperparameters and labeling thresholds over walk-forward windows "
        "and seeds. Proposes a config.yaml diff; never applies it, saves or registers a model."
    )
    parser.add_argument(
        "--space",
        help="YAML search space, or a shipped one by name (solia_ai/sweeps: overfitting-data, "
        f"overfitting-regularization); default: {', '.join(DEFAULT_SPACE)}",
    )
    parser.add_argument("--strategy", choices=STRATEGIES, default="random")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="baseline included")
    parser.add_argument("--max-minutes", type=float, help="no trial starts after this")
    parser.add_argument("--search-seed", type=int, default=DEFAULT_SEARCH_SEED)
    parser.add_argument("--objective", choices=OBJECTIVES, default="roc_auc")
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument(
        "--policy-horizon", help="reference-policy horizon (default: 15m if served, else first)"
    )
    parser.add_argument("--pump-threshold", type=float, default=ReferencePolicy.pump_threshold)
    parser.add_argument("--rug-threshold", type=float, default=ReferencePolicy.rug_threshold)
    add_data_bounds_arguments(parser)  # the retrain's bounded read (config.yaml `data`)
    parser.add_argument(
        "--out",
        help="write the JSON report to this path; each finished trial also goes to "
        "<out>.trials.jsonl as it ends, and a rerun with the same --out and settings resumes",
    )
    parser.add_argument(
        "--restart", action="store_true", help="discard <out>.trials.jsonl and start over"
    )
    parser.add_argument("--diff-out", help="write the proposed config.yaml diff to this path")
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    import torch

    from .config import load_config

    config = load_config()
    max_seconds = None if args.max_minutes is None else args.max_minutes * 60
    try:
        space = load_space(args.space) if args.space else DEFAULT_SPACE
        # fail before reading anything
        normalized = check_space(space, config, strategy=args.strategy)
        check_budget(n_trials=args.trials, max_seconds=max_seconds, objective=args.objective)
        check_settings(
            n_windows=args.windows,
            seeds=args.seeds,
            eval_fraction=args.eval_fraction,
            val_fraction=args.val_fraction,
            margin=args.margin,
            pnl_margin=0.0,
        )
        policy = ReferencePolicy(
            horizon=args.policy_horizon or default_policy_horizon(config["horizons"]),
            pump_threshold=args.pump_threshold,
            rug_threshold=args.rug_threshold,
        )
        policy.horizon_index(config["horizons"])
        data = DataSettings.from_config(
            config, lookback_days=args.lookback_days, max_rows=args.max_rows
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    # The base config trains on --max-rows (else data.max_rows); a larger max_rows in the space
    # widens the read, and each trial trains on its own budget's tokens out of it.
    config = {**config, "data": {**config.get("data", {}), "max_rows": data.max_rows}}
    data = DataSettings(data.lookback_days, read_rows(normalized, data.max_rows))

    checkpoint, end = None, None
    if args.out:
        checkpoint = checkpoint_path(args.out)
        if args.restart:
            checkpoint.unlink(missing_ok=True)
        try:
            header = read_checkpoint_header(checkpoint)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if header is not None and isinstance(header.get("data"), dict):
            # Resume on the window the search started on, not today's: live rows keep arriving.
            end = datetime.fromisoformat(header["data"]["window_end"])
            print(f"resuming {checkpoint} (window ending {end.isoformat()})", file=sys.stderr)

    df, bounds = read_training_features(make_engine(), data, end=end)
    print(bounds.describe(), file=sys.stderr)
    try:
        report = run_search(
            df,
            config,
            space=space,
            strategy=args.strategy,
            n_trials=args.trials,
            max_seconds=max_seconds,
            search_seed=args.search_seed,
            objective=args.objective,
            n_windows=args.windows,
            seeds=args.seeds,
            eval_fraction=args.eval_fraction,
            val_fraction=args.val_fraction,
            margin=args.margin,
            policy=policy,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            progress=lambda msg: print(msg, file=sys.stderr),
            checkpoint=checkpoint,
            data=bounds.as_dict(),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(report.render())
    if args.out:
        print(f"\nreport written to {report.write(args.out)}")
    if args.diff_out:
        path = Path(args.diff_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.proposed_diff, encoding="utf-8")
        print(f"proposed diff written to {path}")


if __name__ == "__main__":
    main()
