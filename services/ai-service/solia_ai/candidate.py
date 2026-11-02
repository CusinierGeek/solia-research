"""Candidate training: the model a retrain proposes to replace the champion.

Each retrain of the continuous-training loop (docs/ML-MODEL.md "Continuous training loop") fits a
**candidate** on the latest ``features`` (live ingestion + replayed history). The promotion gate
(:mod:`solia_ai.promotion`) then judges it on the most recent held-out window(s), so:

* **Refreshed time split.** Labels are the forward-looking ones built from ``features`` (closed
  paper trades only *trigger* a retrain: a small, policy-biased sample, never labels). A window
  is only kept once a horizon is labeled, so the newest tail — whose forward horizons have not
  elapsed — is excluded by :func:`solia_ai.dataset.build_samples`. The held-out window is the
  most recent ``heldout_fraction`` of those labeled windows (by default the walk-forward eval
  share, :data:`solia_ai.config.DEFAULT_EVAL_FRACTION`, so the same cut as
  :func:`solia_ai.ablation.walk_forward_windows`; or an explicit start). The training range is
  every window ending at or before its start minus the label embargo, split by time into train /
  early-stopping validation: the candidate never sees — nor labels into — what it is judged on.
* **Warm start** (``candidate.warm_start``) begins from the champion's weights, with the
  normalizer refit on the refreshed training split, a fresh optimizer and scheduler, a shorter
  ``warm_start_epochs`` budget and a peak learning rate scaled by ``warm_start_lr_factor`` (the
  weights are already trained: don't re-heat them). It is cheaper and keeps what was learned;
  scratch stays available (e.g. after a feature-spec change). A champion whose spec — feature
  columns (``n_features``), enabled feature groups, horizons, model dims — differs from the config,
  whose training data reaches into the held-out window (its weights would carry what the
  candidate is judged on), or whose checkpoint can't be read falls back to scratch (logged and
  recorded in the result) or, with ``on_incompatible: error``, raises. Never silently.
* **Calibration** is fit as in :func:`solia_ai.train.train`, on the candidate's own validation
  split. The checkpoint's ``data_end_ts`` is the end of that split, so evaluation never scores a
  window the candidate trained, early-stopped or calibrated on.

The checkpoint is saved under a new file, but the candidate is **never registered** and never
touches ``is_champion``: promotion is the gate's decision. Reproducible for a given
``training.seed``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .checkpoints import model_directory
from .config import CANDIDATE_DEFAULTS, ConfigError, load_config, validate_config
from .dataset import (
    Samples,
    SequenceDataset,
    check_not_pruned,
    config_samples,
    embargo_seconds,
    temporal_split,
)
from .db import add_data_bounds_arguments
from .features import BASELINE_GROUPS, FEATURE_COLUMNS, group_mask
from .model import IncompatibleCheckpointError
from .train import (
    ProgressFn,
    build_checkpoint,
    default_version,
    fit,
    fit_calibration_window,
    hyperparams_snapshot,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# Model dims a warm start must share with the config (dropout may differ: it holds no weights).
ARCHITECTURE_KEYS = ("d_model", "n_heads", "n_layers", "ff_dim", "seq_len")


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class CandidateSettings:
    """The ``config.yaml`` ``candidate`` knobs (see the file for each one's meaning)."""

    warm_start: bool
    warm_start_epochs: int
    warm_start_lr_factor: float
    on_incompatible: str
    heldout_fraction: float
    val_fraction: float
    min_train_samples: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CandidateSettings:
        """Validated settings (:class:`solia_ai.config.ConfigError` on bad values); an omitted
        knob keeps its default."""
        validate_config(config)
        return cls(**{**CANDIDATE_DEFAULTS, **config.get("candidate", {})})


# --------------------------------------------------------------------------- time split


@dataclass(frozen=True)
class CandidateSplit:
    """The candidate's train / validation windows and the held-out ones it is judged on."""

    train: SequenceDataset
    val: SequenceDataset
    heldout: SequenceDataset
    heldout_start_ts: int  # held-out windows end strictly after this

    @property
    def train_end_ts(self) -> int:
        """End of the latest window the candidate took gradient steps on."""
        return int(self.train.samples.end_ts[self.train.indices].max())

    @property
    def data_end_ts(self) -> int:
        """End of the latest window the candidate saw at all (train or validation)."""
        return max(self.train_end_ts, int(self.val.samples.end_ts[self.val.indices].max()))

    def counts(self) -> dict[str, int]:
        """Window counts; ``n_samples`` (train + validation) is ``training_events.n_samples``."""
        n_train, n_val = len(self.train), len(self.val)
        return {
            "n_samples": n_train + n_val,
            "n_train": n_train,
            "n_val": n_val,
            "n_heldout": len(self.heldout),
        }


def candidate_split(
    samples: Samples,
    *,
    heldout_fraction: float = CANDIDATE_DEFAULTS["heldout_fraction"],
    val_fraction: float = CANDIDATE_DEFAULTS["val_fraction"],
    min_train_samples: int = CANDIDATE_DEFAULTS["min_train_samples"],
    heldout_start_ts: int | None = None,
) -> CandidateSplit:
    """Split labeled ``samples`` into the candidate's training range and the held-out window.

    Held-out: windows ending strictly after ``heldout_start_ts`` (default: the
    ``1 - heldout_fraction`` quantile of the window ends, as the walk-forward windows cut it).
    Training range: windows ending at or before ``heldout_start_ts - embargo`` (longest horizon),
    then split by time into train / validation with the same embargo.

    Raises:
        ValueError: no labeled window, an empty held-out window, fewer than ``min_train_samples``
            training windows, or an empty validation split.
    """
    if not 0 < heldout_fraction < 1:
        raise ValueError(f"heldout_fraction must be in (0, 1), got {heldout_fraction}")
    if min_train_samples < 1:
        raise ValueError(f"min_train_samples must be >= 1, got {min_train_samples}")
    end_ts = np.asarray(samples.end_ts, dtype=np.int64)
    if len(end_ts) == 0:
        raise ValueError("no labeled window to train a candidate on: the feature data is empty")
    if heldout_start_ts is None:
        cut = np.quantile(end_ts.astype(np.float64), 1.0 - heldout_fraction)
        heldout_start_ts = int(math.floor(cut))
    start = int(heldout_start_ts)
    embargo = embargo_seconds(samples.horizons)

    heldout_idx = np.where(end_ts > start)[0]
    if len(heldout_idx) == 0:
        raise ValueError(f"no labeled window ends after the held-out start {start}")
    pool = np.where(end_ts <= start - embargo)[0]
    train_rel, val_rel = temporal_split(
        end_ts[pool], val_fraction=val_fraction, embargo_seconds=embargo
    )
    if len(train_rel) < min_train_samples:
        raise ValueError(
            f"too few samples to train a candidate: {len(train_rel)} training window(s) "
            f"(min_train_samples={min_train_samples}) out of {len(end_ts)} labeled, "
            f"{len(pool)} before the held-out window and its {embargo}s embargo; "
            "ingest or replay more history"
        )
    if len(val_rel) == 0:
        raise ValueError(
            f"empty validation split of the {len(pool)}-window training range; "
            "ingest or replay more history"
        )
    return CandidateSplit(
        SequenceDataset(samples, pool[train_rel]),
        SequenceDataset(samples, pool[val_rel]),
        SequenceDataset(samples, heldout_idx),
        start,
    )


# --------------------------------------------------------------------------- warm start


def _architecture(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    dims = {key: model_cfg.get(key) for key in ARCHITECTURE_KEYS}
    if dims["ff_dim"] is None and dims["d_model"] is not None:
        dims["ff_dim"] = 4 * dims["d_model"]  # SoliaTransformer's default width
    return dims


def warm_start_state(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """The checkpoint's ``model_state``, with the all-enabled feature mask a checkpoint from
    before feature groups implies (as :func:`solia_ai.inference.build_model` loads it)."""
    state = dict(checkpoint["model_state"])
    n_features = len(checkpoint.get("feature_columns") or FEATURE_COLUMNS)
    state.setdefault("feature_mask", torch.ones(n_features, dtype=torch.bool))
    return state


def warm_start_problems(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Why ``checkpoint`` can't seed a model built from ``config`` (empty when it can).

    Compared: feature columns (the code's :data:`FEATURE_COLUMNS`, hence ``n_features``), the
    enabled feature groups (the stored feature mask), horizons, and the model dims. A spec the
    checkpoint does not record cannot be verified, so it counts as a mismatch.
    """
    if "model_state" not in checkpoint:
        return ["the checkpoint holds no model_state"]
    problems: list[str] = []
    columns = checkpoint.get("feature_columns")
    if columns is None:
        problems.append("the checkpoint does not record its feature columns")
    elif list(columns) != FEATURE_COLUMNS:
        problems.append(
            f"feature columns differ: the checkpoint has {len(columns)} (n_features), the code "
            f"{len(FEATURE_COLUMNS)}, or another order"
        )
    else:
        groups = config.get("features", {}).get("groups", BASELINE_GROUPS)
        wanted = group_mask(groups)
        mask = warm_start_state(checkpoint)["feature_mask"]
        if not np.array_equal(np.asarray(mask, dtype=bool), wanted):
            problems.append(f"enabled feature groups differ from the config's {list(groups)}")
    horizons = checkpoint.get("horizons")
    if horizons is None or list(horizons) != list(config["horizons"]):
        problems.append(f"horizons {horizons} differ from the config's {list(config['horizons'])}")
    trained = _architecture((checkpoint.get("hyperparams") or {}).get("model") or {})
    wanted_dims = _architecture(config["model"])
    for key in ARCHITECTURE_KEYS:
        if trained[key] != wanted_dims[key]:
            problems.append(f"model.{key} is {trained[key]!r}, the config's {wanted_dims[key]!r}")
    return problems


def champion_data_end_ts(checkpoint: Mapping[str, Any]) -> int | None:
    """End of the latest window ``checkpoint``'s model saw: its ``data_end_ts``, or for a checkpoint
    from before it the later of ``train_end_ts`` / ``calibration_end_ts`` (``None`` if neither)."""
    if checkpoint.get("data_end_ts") is not None:
        return int(checkpoint["data_end_ts"])
    ends = [
        int(checkpoint[key])
        for key in ("train_end_ts", "calibration_end_ts")
        if checkpoint.get(key) is not None
    ]
    return max(ends) if ends else None


def warm_start_leak(
    checkpoint: Mapping[str, Any], heldout_start_ts: int, embargo: int
) -> str | None:
    """Why seeding from ``checkpoint`` would leak the held-out window (``None`` when it doesn't).

    The candidate's own training range ends at ``heldout_start_ts - embargo``; the champion's
    data must end there too, or its weights carry windows (or labels) the candidate is judged on.
    A checkpoint that doesn't record its training data can't be cleared, so it counts as a leak.
    """
    seen = champion_data_end_ts(checkpoint)
    if seen is None:
        return "the checkpoint does not record its training data's end (data_end_ts)"
    if seen > int(heldout_start_ts) - int(embargo):
        return (
            f"its training data ends at {seen}, after the held-out start {heldout_start_ts} minus "
            f"the {embargo}s embargo: warm-starting would leak held-out windows into the candidate"
        )
    return None


@dataclass(frozen=True)
class WarmStart:
    """How the candidate was initialised: from the champion, or from scratch and why."""

    requested: bool
    used: bool
    champion_version: str | None
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "used": self.used,
            "champion_version": self.champion_version,
            "reasons": list(self.reasons),
        }


def resolve_warm_start(
    champion: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    settings: CandidateSettings,
    *,
    heldout_start_ts: int | None = None,
    embargo: int = 0,
    unavailable: str | None = None,
) -> tuple[WarmStart, dict[str, Any] | None]:
    """``(decision, init_state)``: the champion's state to start from, or ``None`` for scratch.

    Scratch when warm start is off or there is no champion. A champion that is incompatible, whose
    data reaches past ``heldout_start_ts - embargo`` (checked when ``heldout_start_ts`` is given),
    or that could not be loaded (``unavailable`` says why) falls back to scratch with a warning
    (``on_incompatible: scratch``) or raises.

    Raises:
        IncompatibleCheckpointError: such a champion with ``on_incompatible: error``.
    """
    if not settings.warm_start:
        return WarmStart(False, False, None), None
    if champion is None and unavailable is not None:
        if settings.on_incompatible == "error":
            raise IncompatibleCheckpointError(unavailable)
        logger.warning("candidate falls back to scratch: %s", unavailable)
        return WarmStart(True, False, None, (unavailable,)), None
    if champion is None:
        reason = "no champion checkpoint to warm-start from"
        logger.info("candidate trains from scratch: %s", reason)
        return WarmStart(True, False, None, (reason,)), None
    version = champion.get("version")
    problems = warm_start_problems(champion, config)
    if heldout_start_ts is not None and (
        leak := warm_start_leak(champion, heldout_start_ts, embargo)
    ):
        problems.append(leak)
    if problems:
        message = f"champion {version} can't seed the candidate: " + "; ".join(problems)
        if settings.on_incompatible == "error":
            raise IncompatibleCheckpointError(message)
        logger.warning("candidate falls back to scratch: %s", message)
        return WarmStart(True, False, version, tuple(problems)), None
    logger.info("candidate warm-starts from champion %s", version)
    return WarmStart(True, True, version), warm_start_state(champion)


# --------------------------------------------------------------------------- training


@dataclass
class CandidateResult:
    """A trained, saved but unregistered candidate."""

    version: str
    checkpoint_path: Path
    n_params: int
    best_epoch: int
    metrics: dict[str, float]
    hyperparams: dict[str, Any]
    warm_start: WarmStart
    counts: dict[str, int]
    heldout_start_ts: int
    train_end_ts: int
    data_end_ts: int
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.counts["n_samples"]


def train_candidate(
    df: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    champion: Mapping[str, Any] | None = None,
    heldout_start_ts: int | None = None,
    model_dir: str | os.PathLike[str] | None = None,
    version: str | None = None,
    device: str | torch.device | None = None,
    champion_unavailable: str | None = None,
    samples: Samples | None = None,
    progress: ProgressFn | None = None,
) -> CandidateResult:
    """Train a candidate on ``df`` (a ``features`` frame) and save — never register — it.

    ``champion`` is the champion's loaded checkpoint (:func:`solia_ai.inference.load_checkpoint`;
    ``None`` when there is none) for a warm start; ``champion_unavailable`` says why a registered
    champion could not be loaded. ``heldout_start_ts`` pins the held-out start (e.g. the promotion
    gate's first window); by default it is cut by ``heldout_fraction``. ``samples`` are ``df``'s
    windows under ``config`` when the caller already built them (built here otherwise);
    ``progress`` gets the fit's epoch events (:func:`solia_ai.train.fit`).

    The recorded ``hyperparams`` are the configured ones; a warm start's effective budget and
    learning rate are recorded beside them, under ``hyperparams["warm_start"]``.
    """
    cfg = load_config() if config is None else config
    check_not_pruned(cfg, "candidate")
    settings = CandidateSettings.from_config(cfg)
    samples = config_samples(df, cfg) if samples is None else samples
    split = candidate_split(
        samples,
        heldout_fraction=settings.heldout_fraction,
        val_fraction=settings.val_fraction,
        min_train_samples=settings.min_train_samples,
        heldout_start_ts=heldout_start_ts,
    )
    decision, init_state = resolve_warm_start(
        champion,
        cfg,
        settings,
        heldout_start_ts=split.heldout_start_ts,
        embargo=embargo_seconds(samples.horizons),
        unavailable=champion_unavailable,
    )
    fit_cfg: Mapping[str, Any] = cfg
    warm_hyperparams: dict[str, Any] | None = None
    if decision.used:
        warm_hyperparams = {
            "champion_version": decision.champion_version,
            "epochs": settings.warm_start_epochs,
            "learning_rate": float(cfg["training"]["learning_rate"])
            * settings.warm_start_lr_factor,
        }
        fit_cfg = copy.deepcopy(dict(cfg))
        fit_cfg["training"] = {
            **fit_cfg["training"],
            "epochs": warm_hyperparams["epochs"],
            "learning_rate": warm_hyperparams["learning_rate"],
        }
    fitted = fit(
        split.train, split.val, fit_cfg, device=device, init_state=init_state, progress=progress
    )
    model, metrics = fitted.model, fitted.metrics

    version = version or default_version()
    directory = model_directory(model_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{version}.pt"
    calibration, calibration_end_ts = fit_calibration_window(model, split.val, cfg)
    checkpoint = build_checkpoint(
        model,
        cfg,
        version=version,
        metrics=metrics,
        train_end_ts=split.train_end_ts,
        calibration=calibration,
        calibration_end_ts=calibration_end_ts,
        data_end_ts=split.data_end_ts,
    )
    hyperparams = hyperparams_snapshot(cfg)
    if warm_hyperparams is not None:
        hyperparams["warm_start"] = warm_hyperparams
    checkpoint["hyperparams"] = hyperparams
    checkpoint["candidate"] = {
        "warm_start": decision.as_dict(),
        "heldout_start_ts": split.heldout_start_ts,
        **split.counts(),
    }
    save_checkpoint(checkpoint, checkpoint_path)

    return CandidateResult(
        version=version,
        checkpoint_path=checkpoint_path,
        n_params=model.n_params,
        best_epoch=fitted.best_epoch,
        metrics=metrics,
        hyperparams=hyperparams,
        warm_start=decision,
        counts=split.counts(),
        heldout_start_ts=split.heldout_start_ts,
        train_end_ts=split.train_end_ts,
        data_end_ts=split.data_end_ts,
        history=fitted.history,
    )


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train an unregistered candidate model (warm start from the champion or not)."
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--warm-start", dest="warm_start", action="store_true", default=None,
        help="start from the champion's weights (default: candidate.warm_start)",
    )  # fmt: skip
    start.add_argument(
        "--scratch", dest="warm_start", action="store_false",
        help="train from scratch whatever the config says",
    )  # fmt: skip
    parser.add_argument("--version", help="candidate version (default: timestamp)")
    parser.add_argument(
        "--model-dir", help="checkpoint directory (default: $MODEL_DIR or ./models)"
    )
    add_data_bounds_arguments(parser)
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    from .db import DataSettings, make_engine, read_training_features
    from .inference import load_checkpoint, resolve_checkpoint_path
    from .registry import read_champion

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    try:
        data = DataSettings.from_config(
            cfg, lookback_days=args.lookback_days, max_rows=args.max_rows
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from None
    if args.warm_start is not None:
        cfg["candidate"] = {**cfg.get("candidate", {}), "warm_start": args.warm_start}
    engine = make_engine()
    record = read_champion(engine) if cfg.get("candidate", {}).get("warm_start", True) else None
    champion, unavailable = None, None
    if record is not None:
        try:
            champion = load_checkpoint(resolve_checkpoint_path(record.path))
        except Exception as exc:  # missing, unreadable or corrupt: on_incompatible decides
            unavailable = f"champion checkpoint {record.path} could not be loaded: {exc}"
    df, bounds = read_training_features(engine, data)
    print(bounds.describe())

    result = train_candidate(
        df,
        cfg,
        champion=champion,
        champion_unavailable=unavailable,
        model_dir=args.model_dir,
        version=args.version,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    origin = (
        f"warm start from {result.warm_start.champion_version}"
        if result.warm_start.used
        else "scratch"
    )
    print(
        f"candidate {result.version} ({origin}): {result.n_params} params, best epoch "
        f"{result.best_epoch}, val_total={result.metrics['val_total']:.4f}, "
        f"n_samples={result.n_samples} (held out {result.counts['n_heldout']}) "
        f"-> {result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
