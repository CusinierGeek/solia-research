"""Initial training: fit the champion signal model and register its checkpoint.

Flow (one task per docs/ML-MODEL.md "Optimizer & training"):

1. Build train/val datasets by a **temporal** split (:func:`solia_ai.dataset.make_datasets`) — no
   look-ahead leaks into validation.
2. Fit the input normalizer on the **training rows only**, so validation statistics never leak in.
3. Train with AdamW + cosine-with-warmup and gradient clipping. Class imbalance is corrected
   **once**: class-balanced batches by default (``training.balanced_sampling``), otherwise the
   natural distribution with a per-horizon ``pos_weight`` measured on the training split.
4. Early-stop on the validation loss (the true masked mean over the validation set), keeping the
   best epoch's weights. A run whose validation loss is never finite fails instead of saving the
   untrained weights.
5. Save the checkpoint (weights + the config needed to rebuild the architecture) under a new,
   never-overwritten file and, when a database engine is given, write the ``models`` registry row
   (with the checkpoint's absolute path). If registration fails, the new checkpoint is removed.

``train`` takes an in-memory ``features`` frame so it is unit-testable without a database; ``main``
wires it to PostgreSQL via :mod:`solia_ai.db`.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from collections.abc import Mapping
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

from .config import load_config
from .dataset import SequenceDataset, make_datasets, make_loaders, positive_weights
from .loss import SoliaLoss
from .model import SoliaTransformer

DEFAULT_MODEL_DIR = "models"


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
    """The config knobs worth recording with a checkpoint (model + training + loss + horizons)."""
    return {
        "model": dict(config["model"]),
        "training": dict(config["training"]),
        "loss": dict(config["loss"]),
        "horizons": list(config["horizons"]),
    }


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
    warmup = max(1, int(warmup_steps))
    decay = max(1, total_steps - warmup)

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / decay)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _validate(
    model: SoliaTransformer, loss_fn: SoliaLoss, loader: DataLoader, device: torch.device
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
            targets = _to_device(targets, device)
            weight = float(loss_fn.label_weight(targets).sum())
            if weight <= 0:
                continue
            outputs = model(x.to(device), padding_mask.to(device))
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
) -> dict[str, Any]:
    """The serialized checkpoint: weights plus everything needed to rebuild and identify it."""
    return {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_state": model.state_dict(),
        "hyperparams": hyperparams_snapshot(config),
        "horizons": list(model.horizons),
        "feature_columns": list(model.feature_columns),
        "n_params": model.n_params,
        "metrics": dict(metrics or {}),
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
) -> TrainResult:
    """Train on a ``features`` frame, save a checkpoint, and optionally register it.

    ``register`` writes the ``models`` row; it needs an ``engine`` (skipped with a warning-free
    no-op otherwise, e.g. in tests). The checkpoint is always written.
    """
    cfg = load_config() if config is None else config
    training = cfg["training"]
    device = torch.device(device) if device is not None else torch.device("cpu")
    torch.manual_seed(int(training["seed"]))

    train_ds, val_ds = make_datasets(df, cfg, val_fraction=val_fraction)
    balanced = bool(training.get("balanced_sampling", True))
    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=int(training["batch_size"]),
        balanced=balanced,
        seed=int(training["seed"]),
    )

    model = SoliaTransformer.from_config(cfg).to(device)
    model.fit_normalizer(training_feature_rows(train_ds))
    # Balanced batches already put events at ~50%; adding pos_weight on top would correct the
    # imbalance twice and bias the probabilities upward, so it only applies to natural sampling.
    pos_weight = None if balanced else positive_weights(train_ds)
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

    for epoch in range(epochs):
        model.train()
        for x, padding_mask, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(x.to(device), padding_mask.to(device))
            loss = loss_fn(outputs, _to_device(targets, device))["total"]
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

        val = _validate(model, loss_fn, val_loader, device)
        history.append({"epoch": float(epoch), **{f"val_{k}": v for k, v in val.items()}})
        if val["total"] < best_val - 1e-6:
            best_val, best_epoch, best_metrics = val["total"], epoch, val
            best_state = copy.deepcopy(model.state_dict())
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    if not best_metrics:
        raise RuntimeError(
            f"training diverged: the validation loss was never finite over {len(history)} "
            "epoch(s); nothing was saved or registered"
        )
    model.load_state_dict(best_state)
    metrics = {f"val_{k}": v for k, v in best_metrics.items()}

    version = version or default_version()
    directory = Path(
        model_dir if model_dir is not None else os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"{version}.pt"
    save_checkpoint(build_checkpoint(model, cfg, version=version, metrics=metrics), checkpoint_path)

    model_id = None
    if register and engine is not None:
        from .registry import register_model

        try:
            record = register_model(
                engine,
                version=version,
                path=str(checkpoint_path),
                metrics=metrics,
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
        best_epoch=best_epoch,
        metrics=metrics,
        hyperparams=hyperparams_snapshot(cfg),
        history=history,
        model_id=model_id,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Solia signal model.")
    parser.add_argument("--version", help="model version (default: timestamp)")
    parser.add_argument(
        "--model-dir", help="checkpoint directory (default: $MODEL_DIR or ./models)"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, help="cap the number of feature rows read")
    parser.add_argument("--no-register", action="store_true", help="skip the models registry row")
    parser.add_argument(
        "--not-champion", action="store_true", help="register without promoting to champion"
    )
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    from .db import make_engine, read_features

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = make_engine()
    df = read_features(engine, limit=args.limit)

    result = train(
        df,
        model_dir=args.model_dir,
        engine=engine,
        version=args.version,
        register=not args.no_register,
        is_champion=not args.not_champion,
        val_fraction=args.val_fraction,
        device=device,
    )
    print(
        f"trained {result.version}: {result.n_params} params, best epoch {result.best_epoch}, "
        f"val_total={result.metrics['val_total']:.4f} -> {result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
