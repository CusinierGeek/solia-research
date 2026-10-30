"""Initial training: fit the champion signal model and register its checkpoint.

Flow (one task per docs/ML-MODEL.md "Optimizer & training"):

1. Build train/val datasets by a **temporal** split (:func:`solia_ai.dataset.make_datasets`) — no
   look-ahead leaks into validation.
2. Fit the input normalizer on the **training rows only**, so validation statistics never leak in.
3. Train with AdamW + cosine-with-warmup and gradient clipping. Class imbalance is corrected
   **once**: class-balanced batches by default (``training.balanced_sampling``), otherwise the
   natural distribution with ``loss.pos_weight`` (``auto`` = measured per horizon on the training
   split, or fixed per head) — see :func:`resolve_pos_weight`.
4. Early-stop on the validation loss (the true masked mean over the validation set), keeping the
   best epoch's weights. A run whose validation loss is never finite fails instead of saving the
   untrained weights.
5. Calibrate the probabilities (:mod:`solia_ai.calibration`) on the earliest
   ``calibration.fraction`` of the validation windows; the checkpoint records the fit and the
   window's end (``calibration_end_ts``), so evaluation only scores windows after it.
6. Save the checkpoint (weights + the config needed to rebuild the architecture) under a new,
   never-overwritten file and, when a database engine is given, write the ``models`` registry row
   (with the checkpoint's absolute path). If registration fails, the new checkpoint is removed.

``train`` takes an in-memory ``features`` frame so it is unit-testable without a database; ``main``
wires it to PostgreSQL via :mod:`solia_ai.db`.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .calibration import Calibration
from .calibration import settings as calibration_settings
from .checkpoints import model_directory
from .config import ConfigError, load_config
from .dataset import (
    LoaderSettings,
    SequenceDataset,
    make_datasets,
    make_loaders,
    positive_weights,
)
from .db import add_data_bounds_arguments
from .eligibility import EligibilityRule
from .evaluate import calibrate
from .features import BASELINE_GROUPS
from .loss import SoliaLoss
from .model import SoliaTransformer

logger = logging.getLogger(__name__)

# progress(event) gets one "epoch" event per epoch from fit (describe_epoch); the retrain
# loop adds "candidate_start" / "candidate_end" events around each candidate's fit.
ProgressFn = Callable[[dict[str, Any]], None]


@dataclass
class TrainResult:
    """Outcome of a training run."""

    version: str
    checkpoint_path: Path
    n_params: int
    best_epoch: int
    metrics: dict[str, float]
    hyperparams: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)
    model_id: int | None = None


def default_version(now: datetime | None = None) -> str:
    """A sortable, unique-ish version like ``solia-20260420T120000123456Z`` (microseconds, so runs
    started in the same second don't collide)."""
    return (now or datetime.now(UTC)).strftime("solia-%Y%m%dT%H%M%S%fZ")


def hyperparams_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """The config knobs worth recording with a checkpoint (model + training + loss + calibration +
    horizons + the enabled feature groups, so the model is rebuilt with the mask it trained
    with, + the eligibility rule, so the bot trades the population it trained on)."""
    return {
        "model": dict(config["model"]),
        "training": dict(config["training"]),
        "loss": dict(config["loss"]),
        "calibration": calibration_settings(config),
        "horizons": list(config["horizons"]),
        "features": {"groups": list(config.get("features", {}).get("groups", BASELINE_GROUPS))},
        "eligibility": EligibilityRule.from_config(config).as_dict(),
    }


def resolve_pos_weight(
    train_ds: SequenceDataset, config: Mapping[str, Any]
) -> dict[str, torch.Tensor] | None:
    """The BCE ``pos_weight`` per head from ``loss.pos_weight`` (validated by ``config.py``).

    Default: ``none`` with balanced batches (they already put events at ~50%; weighting on top
    would correct the imbalance twice and bias the probabilities upward), ``auto`` with natural
    sampling. ``auto`` measures neg/pos per horizon on the training split; a number applies to
    both heads; a ``{pump: x, rug: y}`` mapping sets each head (an omitted head keeps 1.0). A fixed
    weight is applied as configured: validation already rejects one above ``MAX_POS_WEIGHT``.
    """
    balanced = bool(config["training"].get("balanced_sampling", True))
    spec = config["loss"].get("pos_weight", "none" if balanced else "auto")
    if spec == "none":
        return None
    if spec == "auto":
        return positive_weights(train_ds)
    n_h = len(train_ds.samples.horizons)
    per_head = spec if isinstance(spec, Mapping) else {"pump": spec, "rug": spec}
    return {head: torch.full((n_h,), float(per_head.get(head, 1.0))) for head in ("pump", "rug")}


def calibration_split(val_ds: SequenceDataset, fraction: float) -> SequenceDataset:
    """The earliest ``fraction`` of the validation windows (by end time): the calibration window.

    Always a strict, non-empty prefix: windows tied on the cutoff's end time are left out when
    including them would take the whole split, so some validation windows still end after it.
    Raises :class:`ValueError` when the split is too small (or its end times too few) for that.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"calibration fraction must be in (0, 1), got {fraction}")
    if len(val_ds) == 0:
        raise ValueError("cannot calibrate on an empty validation split")
    ends = val_ds.samples.end_ts[val_ds.indices]
    cutoff = np.quantile(ends.astype(np.float64), fraction)
    in_window = ends <= cutoff
    if in_window.all():
        in_window = ends < cutoff
    if not in_window.any():
        raise ValueError(
            f"the validation split ({len(val_ds)} windows, {len(np.unique(ends))} distinct end "
            "times) is too short to hold a calibration window and windows after it"
        )
    return SequenceDataset(val_ds.samples, val_ds.indices[in_window])


def fit_calibration_window(
    model: SoliaTransformer, val_ds: SequenceDataset, config: Mapping[str, Any]
) -> tuple[Calibration | None, int | None]:
    """``(calibration, calibration_end_ts)`` per ``config``'s ``calibration`` section, fit on
    :func:`calibration_split`; ``(None, None)`` (the identity) for ``method: none``, or — with a
    warning — when the validation split is too short to split. ``model`` is left untouched on
    its device: a model off the CPU is scored through a CPU copy."""
    cal = calibration_settings(config)
    if cal["method"] == "none":
        return None, None
    try:
        cal_ds = calibration_split(val_ds, cal["fraction"])
    except ValueError as exc:
        warnings.warn(f"calibration skipped (identity kept): {exc}", stacklevel=2)
        return None, None
    if any(p.device.type != "cpu" for p in model.parameters()):
        model = copy.deepcopy(model).cpu()
    calibration = calibrate(
        model,
        cal_ds,
        method=cal["method"],
        min_samples=cal["min_samples"],
        min_class_samples=cal["min_class_samples"],
    )
    return calibration, int(cal_ds.samples.end_ts[cal_ds.indices].max())


def training_feature_rows(dataset: SequenceDataset) -> np.ndarray:
    """The ``[R, F]`` feature rows covered by the training windows — what the normalizer fits on.

    Only rows reachable from a training window are kept, so validation-only rows (later in time)
    never contribute to the normalization statistics.
    """
    samples = dataset.samples
    keep = np.zeros(len(samples.features), dtype=bool)
    for i in dataset.indices:
        keep[int(samples.start_row[i]) : int(samples.end_row[i]) + 1] = True
    return samples.features[keep]


def _cosine_warmup(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    # The warmup never takes more than half the schedule: a budget shorter than warmup_steps (a
    # warm start, a small dataset) still decays instead of holding the peak learning rate.
    warmup = max(1, min(int(warmup_steps), int(total_steps) // 2))
    decay = max(1, total_steps - warmup)

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / decay)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def _to_device(
    batch: dict[str, torch.Tensor], device: torch.device, non_blocking: bool = False
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


def describe_epoch(event: Mapping[str, Any]) -> str:
    """One line for an ``epoch`` progress event."""
    return (
        f"epoch {event['epoch']}/{event['epochs']}: train_loss={event['train_loss']:.4f} "
        f"val_loss={event['val_loss']:.4f} ({event['seconds']:.1f}s, "
        f"{event['samples_per_s']:.0f} samples/s, ETA {event['eta_seconds']:.0f}s), "
        f"no improvement {event['since_improved']}/{event['patience']}"
    )


def _log_epoch(event: Mapping[str, Any]) -> None:
    logger.info(describe_epoch(event))


@torch.no_grad()
def _validate(
    model: SoliaTransformer,
    loss_fn: SoliaLoss,
    loader: DataLoader,
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, float]:
    """Mean loss components over the whole validation set.

    Each batch's components are masked means sharing one denominator (the batch's label weight),
    so weighting them by that denominator yields the exact masked mean over the set: batches with
    few or no labels don't skew it.
    """
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    total_weight = 0.0
    try:
        for x, padding_mask, targets in loader:
            targets = _to_device(targets, device, non_blocking)
            weight = float(loss_fn.label_weight(targets).sum())
            if weight <= 0:
                continue
            outputs = model(
                x.to(device, non_blocking=non_blocking),
                padding_mask.to(device, non_blocking=non_blocking),
            )
            components = loss_fn(outputs, targets)
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value) * weight
            total_weight += weight
    finally:
        model.train(was_training)
    if total_weight <= 0:
        raise ValueError("the validation split has no labeled horizon to score")
    return {key: value / total_weight for key, value in totals.items()}


def build_checkpoint(
    model: SoliaTransformer,
    config: Mapping[str, Any],
    *,
    version: str,
    metrics: Mapping[str, float] | None = None,
    train_end_ts: int | None = None,
    calibration: Calibration | None = None,
    calibration_end_ts: int | None = None,
    data_end_ts: int | None = None,
) -> dict[str, Any]:
    """The serialized checkpoint: weights plus everything needed to rebuild and identify it.

    ``train_end_ts`` is the epoch second of the latest training window's last row: evaluation only
    scores windows strictly after it (plus the label embargo), so held-out metrics never re-score
    data the model was fit on, whatever window the evaluation run reads. ``calibration`` (applied
    at inference; ``None`` = identity) was fit on windows ending by ``calibration_end_ts``, which
    pushes that boundary later still. ``data_end_ts`` is the end of the latest window the model saw
    in any way — gradient steps, early stopping or calibration (the whole validation split) — so
    evaluation and a warm start's leakage check never treat it as unseen.
    """
    return {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_state": model.state_dict(),
        "hyperparams": hyperparams_snapshot(config),
        "horizons": list(model.horizons),
        "feature_columns": list(model.feature_columns),
        "n_params": model.n_params,
        "metrics": dict(metrics or {}),
        "train_end_ts": train_end_ts,
        "calibration": None if calibration is None else calibration.as_dict(),
        "calibration_end_ts": calibration_end_ts,
        "data_end_ts": data_end_ts,
    }


def save_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> None:
    """Write ``checkpoint`` to a **new** file. An existing checkpoint (an already-registered
    version) is never overwritten: exclusive creation raises :class:`FileExistsError` instead, even
    when two runs race for the same name."""
    try:
        f = open(path, "xb")  # noqa: SIM115 - the file must exist before we own its cleanup
    except FileExistsError:
        raise FileExistsError(f"checkpoint {path} already exists; pick another version") from None
    try:
        with f:
            torch.save(dict(checkpoint), f)
    except BaseException:
        path.unlink(missing_ok=True)  # never leave a truncated checkpoint behind
        raise


@dataclass
class FitResult:
    """A fitted model (best epoch's weights) and its validation trace; nothing saved."""

    model: SoliaTransformer
    best_epoch: int
    metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


def fit(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    config: Mapping[str, Any],
    *,
    device: str | torch.device | None = None,
    init_state: Mapping[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> FitResult:
    """The training loop alone: fit a model on ``train_ds``, early-stopping on ``val_ds``.

    The model starts fresh or, with ``init_state`` (a champion's ``model_state``: a warm start),
    from those weights. Either way the normalizer is fit on ``train_ds`` — a warm start never
    keeps statistics of older data, so warm and scratch candidates share their preprocessing —
    and the optimizer and scheduler are fresh. The caller checks the state matches ``config``'s
    architecture (:func:`solia_ai.candidate.warm_start_problems`); loading still refuses other
    horizons or feature columns. Seeds torch from ``training.seed``. Writes nothing —
    :func:`train` saves and registers; the ablation harness only scores the in-memory model.

    Batches are fed per :class:`~solia_ai.dataset.LoaderSettings` (``training.num_workers`` ...);
    the fitted weights don't depend on them. After each epoch ``progress`` (default: the module
    logger) gets an ``epoch`` event: ``epoch`` (1-based) / ``epochs``, mean ``train_loss``,
    ``val_loss``, ``seconds``, training ``samples_per_s``, ``eta_seconds`` (the rest of the epoch
    budget at the mean pace so far — early stopping may end sooner) and ``since_improved`` /
    ``patience``.
    """
    cfg = config
    training = cfg["training"]
    device = torch.device(device) if device is not None else torch.device("cpu")
    torch.manual_seed(int(training["seed"]))

    balanced = bool(training.get("balanced_sampling", True))
    loader = LoaderSettings.from_config(cfg, device)
    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=int(training["batch_size"]),
        balanced=balanced,
        seed=int(training["seed"]),
        loader=loader,
    )
    report = progress or _log_epoch
    pin = loader.pin_memory

    model = SoliaTransformer.from_config(cfg).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.fit_normalizer(training_feature_rows(train_ds))
    pos_weight = resolve_pos_weight(train_ds, cfg)
    loss_fn = SoliaLoss.from_config(cfg, pos_weight=pos_weight).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training["epochs"])
    scheduler = _cosine_warmup(
        optimizer, int(training.get("warmup_steps", 0)), epochs * max(1, len(train_loader))
    )
    grad_clip = float(training.get("grad_clip", 0.0))
    patience = int(training.get("early_stopping_patience", epochs))

    history: list[dict[str, float]] = []
    best_val = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, float] = {}
    since_improved = 0
    started = time.monotonic()

    for epoch in range(epochs):
        epoch_start = time.monotonic()
        model.train()
        loss_sum = torch.zeros((), device=device)  # accumulated on the device: no sync per step
        n_batches = 0
        for x, padding_mask, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(
                x.to(device, non_blocking=pin), padding_mask.to(device, non_blocking=pin)
            )
            loss = loss_fn(outputs, _to_device(targets, device, pin))["total"]
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach()
            n_batches += 1
        train_seconds = time.monotonic() - epoch_start

        val = _validate(model, loss_fn, val_loader, device, pin)
        train_loss = float(loss_sum) / max(1, n_batches)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val.items()},
            }
        )
        if val["total"] < best_val - 1e-6:
            best_val, best_epoch, best_metrics = val["total"], epoch, val
            best_state = copy.deepcopy(model.state_dict())
            since_improved = 0
        else:
            since_improved += 1

        now = time.monotonic()
        report(
            {
                "event": "epoch",
                "epoch": epoch + 1,
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val["total"],
                "seconds": now - epoch_start,
                "samples_per_s": len(train_ds) / max(train_seconds, 1e-9),
                "eta_seconds": (now - started) / (epoch + 1) * (epochs - epoch - 1),
                "since_improved": since_improved,
                "patience": patience,
            }
        )
        if since_improved >= patience:
            break

    if not best_metrics:
        raise RuntimeError(
            f"training diverged: the validation loss was never finite over {len(history)} "
            "epoch(s); nothing was saved or registered"
        )
    model.load_state_dict(best_state)
    metrics = {f"val_{k}": v for k, v in best_metrics.items()}
    return FitResult(model=model, best_epoch=best_epoch, metrics=metrics, history=history)


def train(
    df: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    model_dir: str | os.PathLike[str] | None = None,
    engine: Any = None,
    version: str | None = None,
    register: bool = True,
    is_champion: bool = True,
    val_fraction: float = 0.2,
    device: str | torch.device | None = None,
    data_bounds: Mapping[str, Any] | None = None,
) -> TrainResult:
    """Train on a ``features`` frame, save a checkpoint, and optionally register it.

    ``register`` writes the ``models`` row; it needs an ``engine`` (skipped with a warning-free
    no-op otherwise, e.g. in tests). The checkpoint is always written. ``data_bounds`` (how the
    frame was read, :meth:`solia_ai.db.DataBounds.as_dict`) is recorded in the row's metrics.
    """
    cfg = load_config() if config is None else config
    train_ds, val_ds = make_datasets(df, cfg, val_fraction=val_fraction)
    fitted = fit(train_ds, val_ds, cfg, device=device)
    model, metrics = fitted.model, fitted.metrics

    version = version or default_version()
    directory = model_directory(model_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{version}.pt"
    train_end_ts = int(train_ds.samples.end_ts[train_ds.indices].max())
    data_end_ts = max(train_end_ts, int(val_ds.samples.end_ts[val_ds.indices].max()))
    calibration, calibration_end_ts = fit_calibration_window(model, val_ds, cfg)
    checkpoint = build_checkpoint(
        model,
        cfg,
        version=version,
        metrics=metrics,
        train_end_ts=train_end_ts,
        calibration=calibration,
        calibration_end_ts=calibration_end_ts,
        data_end_ts=data_end_ts,
    )
    save_checkpoint(checkpoint, checkpoint_path)

    model_id = None
    if register and engine is not None:
        from .registry import register_model

        try:
            record = register_model(
                engine,
                version=version,
                path=str(checkpoint_path),
                metrics=metrics if data_bounds is None else {**metrics, "data_bounds": data_bounds},
                hyperparams=hyperparams_snapshot(cfg),
                n_params=model.n_params,
                is_champion=is_champion,
            )
        except Exception:
            checkpoint_path.unlink(missing_ok=True)  # don't leave an unregistered orphan behind
            raise
        model_id = record.id

    return TrainResult(
        version=version,
        checkpoint_path=checkpoint_path,
        n_params=model.n_params,
        best_epoch=fitted.best_epoch,
        metrics=metrics,
        hyperparams=hyperparams_snapshot(cfg),
        history=fitted.history,
        model_id=model_id,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Solia signal model.")
    parser.add_argument("--version", help="model version (default: timestamp)")
    parser.add_argument(
        "--model-dir", help="checkpoint directory (default: $MODEL_DIR or ./models)"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    add_data_bounds_arguments(parser)
    parser.add_argument("--no-register", action="store_true", help="skip the models registry row")
    parser.add_argument(
        "--not-champion", action="store_true", help="register without promoting to champion"
    )
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    from .db import DataSettings, make_engine, read_training_features

    try:
        data = DataSettings.from_config(
            load_config(), lookback_days=args.lookback_days, max_rows=args.max_rows
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from None
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = make_engine()
    df, bounds = read_training_features(engine, data)
    print(bounds.describe())

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = train(
        df,
        model_dir=args.model_dir,
        engine=engine,
        version=args.version,
        register=not args.no_register,
        is_champion=not args.not_champion,
        val_fraction=args.val_fraction,
        device=device,
        data_bounds=bounds.as_dict(),
    )
    print(
        f"trained {result.version}: {result.n_params} params, best epoch {result.best_epoch}, "
        f"val_total={result.metrics['val_total']:.4f} -> {result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
