"""Measure whether the signal has an edge, on a held-out validation window.

This is the *does it actually work?* side of the model (docs/ML-MODEL.md "Evaluation metrics").
It scores a trained model over the validation split and reports, per horizon:

* **pump / rug** (classification): ROC-AUC, PR-AUC and calibration error (ECE). AUCs are ``None``
  when a horizon has only one class in the window (undefined, not zero).
* **expected_return** (regression): information coefficient (Spearman rank IC) and Pearson
  correlation — do higher predictions line up with higher realized returns? — plus the MAE.
* **economic**: a simulated win-rate and P&L via a **simple reference policy** (enter when the
  pump probability is high and the rug probability low, equal-sized, realized at the horizon).
  This is a *yardstick* for the signal only — the real, tunable policy lives in the bot
  (Sprints 3–4), not here.

Metrics are computed on labeled samples only (``mask == 1``): a horizon a window can't see the
outcome of never pollutes the score. The report is written to a file (JSON) and rendered to
stdout. ``main`` wires it to the registry champion and PostgreSQL and scores only windows after the
checkpoint's recorded training data (``train_end_ts`` + embargo), so the held-out metrics never
re-score training windows; ``evaluate`` works on an in-memory model + dataset so it is
unit-testable without either.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import SequenceDataset, make_datasets, make_holdout_dataset
from .inference import contract_outputs
from .model import SoliaTransformer

# Finance convention: the information coefficient is a rank correlation, so a monotonic but
# non-linear signal still scores well. Pearson is reported alongside for the linear view.
DEFAULT_ECE_BINS = 10


# --------------------------------------------------------------------------- scoring


def score_dataset(
    model: SoliaTransformer,
    dataset: SequenceDataset,
    *,
    temperature: float = 1.0,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Score every window in ``dataset`` into the serving contract.

    Returns ``(predictions, truth)``: ``predictions`` holds ``pump_prob`` / ``rug_prob`` /
    ``expected_return`` / ``confidence`` and ``truth`` holds the labels ``pump`` / ``rug`` /
    ``expected_return`` / ``mask`` — each a ``[N, H]`` array aligned row-for-row, over the
    dataset's own order (unshuffled), so a metric can mask by ``truth['mask']`` per horizon.
    """
    n_h = len(dataset.samples.horizons)
    pred_keys = ("pump_prob", "rug_prob", "expected_return", "confidence")
    true_keys = ("pump", "rug", "expected_return", "mask")
    preds: dict[str, list[np.ndarray]] = {k: [] for k in pred_keys}
    trues: dict[str, list[np.ndarray]] = {k: [] for k in true_keys}

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for x, padding_mask, targets in DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            ):
                out = contract_outputs(model(x, padding_mask), temperature)
                for key in pred_keys:
                    preds[key].append(out[key].numpy())
                for key in true_keys:
                    trues[key].append(targets[key].numpy())
    finally:
        model.train(was_training)

    def stack(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty((0, n_h), np.float32)

    return ({k: stack(v) for k, v in preds.items()}, {k: stack(v) for k, v in trues.items()})


# --------------------------------------------------------------------------- metrics


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = DEFAULT_ECE_BINS
) -> float:
    """Expected Calibration Error: the probability-weighted gap between confidence and accuracy.

    Probabilities are bucketed into ``n_bins`` equal-width bins over ``[0, 1]``; each bin
    contributes ``|mean(y_true) - mean(y_prob)|`` weighted by its share of samples. 0 is perfect
    calibration. Reported so a strong AUC with badly scaled probabilities is still caught — the
    policy reads probabilities, not just their ranking.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Interior edges only, so probabilities of exactly 0 and 1 land in the first/last bin.
    bucket = np.digitize(y_prob, edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        in_bin = bucket == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        ece += (count / y_true.size) * abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
    return float(ece)


def _auc(metric: Any, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """A ranking AUC, or ``None`` when the window holds a single class (the metric is undefined)."""
    positives = int(y_true.sum())
    if positives == 0 or positives == y_true.size:
        return None
    return float(metric(y_true, y_score))


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = DEFAULT_ECE_BINS
) -> dict[str, Any]:
    """ROC-AUC, PR-AUC, ECE and the base rate for one binary head on one horizon."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = int(y_true.size)
    positives = int(y_true.sum())
    return {
        "n": n,
        "positives": positives,
        "base_rate": (positives / n) if n else None,
        "roc_auc": _auc(roc_auc_score, y_true, y_prob),
        "pr_auc": _auc(average_precision_score, y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins) if n else None,
    }


def _correlation(a: np.ndarray, b: np.ndarray, *, rank: bool) -> float | None:
    """Pearson (``rank=False``) or Spearman (``rank=True``) correlation, or ``None`` when it is
    undefined (fewer than two points, or either series constant).

    Spearman ranks ties by their **average** rank: realized returns tie often (many windows at
    exactly 0, clipped rugs), and ordinal ranks would make the IC depend on the input order.
    """
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    x = rankdata(a) if rank else a.astype(np.float64)
    y = rankdata(b) if rank else b.astype(np.float64)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Information coefficient (Spearman), Pearson correlation and MAE for ``expected_return``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = int(y_true.size)
    return {
        "n": n,
        "mae": float(np.abs(y_pred - y_true).mean()) if n else None,
        "ic": _correlation(y_pred, y_true, rank=True),
        "corr": _correlation(y_pred, y_true, rank=False),
    }


# --------------------------------------------------------------------------- reference policy


@dataclass(frozen=True)
class ReferencePolicy:
    """A deliberately simple entry rule, used **only** to put a P&L number on the signal.

    Enter a (unit-sized) position on a window when, at ``horizon``, ``pump_prob >= pump_threshold``
    and ``rug_prob <= rug_threshold``; the realized outcome is that horizon's labeled return. The
    real policy — sizing, TP/SL, dynamic thresholds — lives in the bot (docs/ML-MODEL.md "Policy
    layer"); this one is a fixed yardstick, not a tunable strategy.
    """

    horizon: str
    pump_threshold: float = 0.6
    rug_threshold: float = 0.3

    def horizon_index(self, horizons: Any) -> int:
        """The policy horizon's column in ``horizons``; raises when the model doesn't serve it."""
        horizons = tuple(horizons)
        if self.horizon not in horizons:
            raise ValueError(
                f"policy horizon {self.horizon!r} is not one of the model's {list(horizons)}"
            )
        return horizons.index(self.horizon)

    def simulate(
        self, predictions: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray], horizons: Any
    ) -> dict[str, Any]:
        """Run the rule over the scored window and return win-rate / P&L (see the class doc)."""
        h = self.horizon_index(horizons)
        labeled = truth["mask"][:, h] == 1
        realized = truth["expected_return"][:, h][labeled]
        enter = (
            labeled
            & (predictions["pump_prob"][:, h] >= self.pump_threshold)
            & (predictions["rug_prob"][:, h] <= self.rug_threshold)
        )
        taken = truth["expected_return"][:, h][enter]
        n_trades = int(taken.size)
        return {
            "horizon": self.horizon,
            "pump_threshold": self.pump_threshold,
            "rug_threshold": self.rug_threshold,
            "n_candidates": int(labeled.sum()),
            "n_trades": n_trades,
            "win_rate": float((taken > 0).mean()) if n_trades else None,
            "total_pnl": float(taken.sum()) if n_trades else 0.0,
            "avg_pnl": float(taken.mean()) if n_trades else None,
            # Buy-everything baseline: the policy only earns its keep if it beats this.
            "baseline_avg_return": float(realized.mean()) if realized.size else None,
        }


# --------------------------------------------------------------------------- report


@dataclass
class EvaluationReport:
    """The full evaluation: per-horizon classification + regression metrics and the policy P&L."""

    model_version: str
    generated_at: str
    n_samples: int
    horizons: list[str]
    pump: dict[str, dict[str, Any]]
    rug: dict[str, dict[str, Any]]
    expected_return: dict[str, dict[str, Any]]
    policy: dict[str, Any]
    temperature: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe dict (non-finite metrics, which JSON can't hold, become ``null``)."""
        return json_safe(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=False)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    def render(self) -> str:
        """A compact text summary for stdout / logs."""
        lines = [
            f"Evaluation — model {self.model_version} ({self.n_samples} validation windows)",
            f"generated {self.generated_at}",
            "",
            f"{'horizon':>8} | {'pump AUC':>8} {'pump PR':>8} {'pump ECE':>8} | "
            f"{'rug AUC':>8} {'rug PR':>8} {'rug ECE':>8} | {'ret IC':>7} {'ret MAE':>8}",
        ]
        for horizon in self.horizons:
            p, r, e = self.pump[horizon], self.rug[horizon], self.expected_return[horizon]
            lines.append(
                f"{horizon:>8} | {fmt(p['roc_auc']):>8} {fmt(p['pr_auc']):>8} "
                f"{fmt(p['ece']):>8} | {fmt(r['roc_auc']):>8} {fmt(r['pr_auc']):>8} "
                f"{fmt(r['ece']):>8} | {fmt(e['ic']):>7} {fmt(e['mae']):>8}"
            )
        pol = self.policy
        lines += [
            "",
            f"Reference policy @ {pol['horizon']} "
            f"(pump>={pol['pump_threshold']}, rug<={pol['rug_threshold']}): "
            f"{pol['n_trades']}/{pol['n_candidates']} trades, "
            f"win-rate {fmt(pol['win_rate'])}, avg P&L {fmt(pol['avg_pnl'])}, "
            f"total P&L {fmt(pol['total_pnl'])} "
            f"(buy-all avg {fmt(pol['baseline_avg_return'])})",
        ]
        return "\n".join(lines)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- orchestration


def evaluate(
    model: SoliaTransformer,
    dataset: SequenceDataset,
    *,
    model_version: str = "unknown",
    policy: ReferencePolicy | None = None,
    temperature: float = 1.0,
    n_bins: int = DEFAULT_ECE_BINS,
    batch_size: int = 256,
) -> EvaluationReport:
    """Score ``dataset`` with ``model`` and build the full :class:`EvaluationReport`.

    The ``policy`` defaults to :class:`ReferencePolicy` on the model's first horizon. Each horizon
    is scored on its labeled samples only. The dataset must be labeled for exactly the model's
    horizons, in the model's order — prediction column ``h`` is only comparable to label column
    ``h`` then — and the policy horizon is checked up front, before the inference pass.
    """
    horizons = list(dataset.samples.horizons)
    if not horizons:
        raise ValueError("the dataset has no horizons to evaluate")
    if tuple(model.horizons) != tuple(horizons):
        raise ValueError(
            f"the dataset is labeled for horizons {horizons} but the model predicts "
            f"{list(model.horizons)}; build the dataset with the model's horizons"
        )
    policy = policy or ReferencePolicy(horizon=horizons[0])
    policy.horizon_index(horizons)

    predictions, truth = score_dataset(
        model, dataset, temperature=temperature, batch_size=batch_size
    )
    pump: dict[str, dict[str, Any]] = {}
    rug: dict[str, dict[str, Any]] = {}
    ret: dict[str, dict[str, Any]] = {}
    for h, horizon in enumerate(horizons):
        labeled = truth["mask"][:, h] == 1
        pump[horizon] = classification_metrics(
            truth["pump"][:, h][labeled], predictions["pump_prob"][:, h][labeled], n_bins=n_bins
        )
        rug[horizon] = classification_metrics(
            truth["rug"][:, h][labeled], predictions["rug_prob"][:, h][labeled], n_bins=n_bins
        )
        ret[horizon] = regression_metrics(
            truth["expected_return"][:, h][labeled],
            predictions["expected_return"][:, h][labeled],
        )

    return EvaluationReport(
        model_version=model_version,
        generated_at=datetime.now(UTC).isoformat(),
        n_samples=len(dataset),
        horizons=horizons,
        pump=pump,
        rug=rug,
        expected_return=ret,
        policy=policy.simulate(predictions, truth, horizons),
        temperature=temperature,
    )


# --------------------------------------------------------------------------- CLI


def holdout_dataset(
    df: Any, config: Mapping[str, Any], predictor: Any, *, val_fraction: float = 0.2
) -> SequenceDataset:
    """The windows of ``df`` the ``predictor``'s model never trained on, labeled for its horizons.

    Labels follow the **checkpoint's** horizons (not whatever ``config.yaml`` lists today), so
    predictions and truth stay column-aligned. Windows are kept only after the checkpoint's
    ``train_end_ts`` + embargo; a legacy checkpoint without it falls back to the time split's
    validation share, with a warning that overlap with its training data can't be ruled out.
    """
    cfg = {**config, "horizons": list(predictor.horizons)}
    train_end_ts = predictor.metadata.get("train_end_ts")
    if train_end_ts is not None:
        return make_holdout_dataset(df, cfg, train_end_ts=int(train_end_ts))
    print(
        "warning: the checkpoint records no train_end_ts; evaluating the val_fraction split, "
        "which may overlap the model's training data",
        file=sys.stderr,
    )
    _, val_ds = make_datasets(df, cfg, val_fraction=val_fraction)
    return val_ds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Solia checkpoint on the val window.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="path to a .pt checkpoint to evaluate")
    source.add_argument(
        "--champion", action="store_true", help="evaluate the registry's champion checkpoint"
    )
    parser.add_argument("--out", help="write the JSON report to this path")
    parser.add_argument("--limit", type=int, help="cap the number of feature rows read")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="validation share, only for legacy checkpoints that don't record train_end_ts",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=DEFAULT_ECE_BINS, help="ECE bin count")
    parser.add_argument("--policy-horizon", help="reference-policy horizon (default: first)")
    parser.add_argument("--pump-threshold", type=float, default=ReferencePolicy.pump_threshold)
    parser.add_argument("--rug-threshold", type=float, default=ReferencePolicy.rug_threshold)
    args = parser.parse_args(argv)

    from .config import load_config
    from .db import make_engine, read_features
    from .inference import Predictor

    engine = make_engine()
    if args.champion:
        try:
            predictor = Predictor.from_champion(engine, temperature=args.temperature)
        except LookupError as exc:
            raise SystemExit(str(exc)) from None
    else:
        predictor = Predictor.from_checkpoint(args.checkpoint, temperature=args.temperature)

    policy = ReferencePolicy(
        horizon=args.policy_horizon or predictor.horizons[0],
        pump_threshold=args.pump_threshold,
        rug_threshold=args.rug_threshold,
    )
    try:
        policy.horizon_index(predictor.horizons)  # fail before reading or scoring anything
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    df = read_features(engine, limit=args.limit)
    val_ds = holdout_dataset(df, load_config(), predictor, val_fraction=args.val_fraction)
    report = evaluate(
        predictor.model,
        val_ds,
        model_version=predictor.model_version,
        policy=policy,
        temperature=predictor.temperature,
        n_bins=args.bins,
    )
    print(report.render())
    if args.out:
        path = report.write(args.out)
        print(f"\nreport written to {path}")


if __name__ == "__main__":
    main()
