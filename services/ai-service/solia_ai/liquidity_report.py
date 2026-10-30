"""Where does a model's signal live? Its held-out metrics split by the anchor pool's liquidity.

Prices read off near-empty pools are artifacts (:mod:`solia_ai.liquidity`); a model can score well
by recognizing a dust pool that "jumps". This read-only diagnostic scores a checkpoint once over
its held-out windows (the retrain's bounded read, windows after the checkpoint's data) and reports,
per liquidity bucket of the window's **anchor** (:func:`solia_ai.liquidity.liquidity_buckets`:
``amm`` / ``curve`` x ``<1`` / ``1-10`` / ``>=10`` SOL / ``unknown``) and over all windows:

* per horizon, pump and rug ROC-AUC / PR-AUC and base rates (:func:`classification_metrics`);
* the entry rule (:class:`ReferencePolicy`, the bot's defaults): trades, precision, win rate, mean
  / median return and a bootstrap CI (:func:`selection_metrics`), against buy-everything;
* the entry-rule sweep (:func:`solia_ai.entry_sweep.sweep_horizon`) at the rule's horizon, inside
  the bucket: which rule it would recommend there, and how precise it is;
* each bucket's share of the windows, of the pumps and of the rule's trades / P&L.

``--labels none`` (default) labels as before the floor — every window, every price — which is how a
champion trained without the floor was scored; ``--labels config`` applies the config's floor
(windows below it dropped, dust prices ignored by pump / return), the population a floored model is
trained and promoted on.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .entry_sweep import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CI,
    DEFAULT_MIN_TRADES,
    _rule_label,
    recommend,
    selection_metrics,
    sweep_horizon,
)
from .evaluate import ReferencePolicy, classification_metrics, fmt, json_safe

ALL = "all"
LABEL_MODES = ("none", "config")


def bucket_report(
    predictions: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    buckets: np.ndarray,
    horizons: Sequence[str],
    policy: ReferencePolicy,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
    sweep: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The metrics of every bucket of ``buckets`` (``[N]``, aligned with the ``[N, H]`` arrays)
    and of all windows (module doc). Buckets are sorted by name, ``all`` first. ``sweep``: the
    :func:`sweep_horizon` grid keywords (``None`` = no sweep)."""
    horizons = list(horizons)
    buckets = np.asarray(buckets, dtype=object)
    n = len(buckets)
    if any(np.asarray(truth[k]).shape[0] != n for k in ("pump", "rug", "mask")):
        raise ValueError("buckets and labels must have one row per window")
    p = policy.horizon_index(horizons)
    labeled_p = truth["mask"][:, p] == 1
    enter = (
        labeled_p
        & (predictions["pump_prob"][:, p] >= policy.pump_threshold)
        & (predictions["rug_prob"][:, p] <= policy.rug_threshold)
    )
    returns = truth["expected_return"][:, p]
    pumps = truth["pump"][:, p]
    totals = {
        "windows": n,
        "pumps": float(pumps[labeled_p].sum()),
        "trades": int(enter.sum()),
        "pnl": float(returns[enter].sum()),
    }

    def share(part: float, whole: float) -> float | None:
        return part / whole if whole else None

    out: dict[str, Any] = {}
    for name in [ALL, *sorted(set(buckets.tolist()))]:
        rows = np.ones(n, bool) if name == ALL else buckets == name
        rng = np.random.default_rng(seed)
        heads: dict[str, dict[str, Any]] = {"pump": {}, "rug": {}}
        for h, horizon in enumerate(horizons):
            labeled = rows & (truth["mask"][:, h] == 1)
            for head in heads:
                heads[head][horizon] = classification_metrics(
                    truth[head][:, h][labeled], predictions[f"{head}_prob"][:, h][labeled]
                )
        base = rows & labeled_p
        base_rate = float(pumps[base].mean()) if base.any() else None
        taken = rows & enter
        rule = selection_metrics(
            returns[taken], pumps[taken], base_rate=base_rate, samples=bootstrap_samples, ci=ci,
            rng=rng,
        )  # fmt: skip
        baseline = selection_metrics(
            returns[base], pumps[base], base_rate=base_rate, samples=bootstrap_samples, ci=ci,
            rng=rng,
        )  # fmt: skip
        out[name] = {
            "n_windows": int(rows.sum()),
            "share": {
                "windows": share(float(rows.sum()), totals["windows"]),
                "pumps": share(float(pumps[base].sum()), totals["pumps"]),
                "trades": share(float(taken.sum()), totals["trades"]),
                "pnl": share(float(returns[taken].sum()), totals["pnl"]),
            },
            "pump": heads["pump"],
            "rug": heads["rug"],
            "rule": rule,
            "baseline": baseline,
        }
        if sweep is not None:
            swept = sweep_horizon(
                predictions["pump_prob"][:, p][base],
                predictions["rug_prob"][:, p][base],
                predictions["confidence"][:, p][base],
                pumps[base],
                returns[base],
                bootstrap_samples=bootstrap_samples, ci=ci, seed=seed, **sweep,
            )  # fmt: skip
            out[name]["sweep"] = {**swept, "recommended": recommend(swept)}
    return out


def build_report(
    model_version: str,
    labels: str,
    floor: Mapping[str, float],
    policy: ReferencePolicy,
    by_bucket: Mapping[str, Any],
    horizons: Sequence[str],
) -> dict[str, Any]:
    return json_safe(
        {
            "model_version": model_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "labels": labels,
            "liquidity_floor": dict(floor),
            "horizons": list(horizons),
            "policy": {
                "horizon": policy.horizon,
                "pump_threshold": policy.pump_threshold,
                "rug_threshold": policy.rug_threshold,
            },
            "buckets": dict(by_bucket),
        }
    )


def render(report: Mapping[str, Any]) -> str:
    """Text tables: per bucket, its shares, pump / rug PR-AUC per horizon, and the entry rule."""
    pol = report["policy"]
    horizons = report["horizons"]
    lines = [
        f"Liquidity diagnostic — model {report['model_version']}, labels: {report['labels']} "
        f"(floor {report['liquidity_floor'] or 'disabled'}), generated {report['generated_at']}",
        "",
        f"{'bucket':<14} {'windows':>9} {'%win':>6} {'%pump':>6} {'%trade':>6} {'%pnl':>6}"
        + "".join(f" {'pumpPR ' + h:>11}" for h in horizons),
    ]
    for name, b in report["buckets"].items():
        s = b["share"]
        lines.append(
            f"{name:<14} {b['n_windows']:>9} {_pct(s['windows']):>6} {_pct(s['pumps']):>6} "
            f"{_pct(s['trades']):>6} {_pct(s['pnl']):>6}"
            + "".join(f" {fmt(b['pump'][h]['pr_auc']):>11}" for h in horizons)
        )
    lines += [
        "",
        f"{'bucket':<14}" + "".join(f" {'base ' + h:>11}" for h in horizons)
        + "".join(f" {'rugPR ' + h:>11}" for h in horizons),
    ]  # fmt: skip
    for name, b in report["buckets"].items():
        lines.append(
            f"{name:<14}"
            + "".join(f" {fmt(b['pump'][h]['base_rate']):>11}" for h in horizons)
            + "".join(f" {fmt(b['rug'][h]['pr_auc']):>11}" for h in horizons)
        )
    lines += [
        "",
        f"Entry rule @ {pol['horizon']} (pump>={pol['pump_threshold']}, "
        f"rug<={pol['rug_threshold']}) vs buy-everything",
        f"{'bucket':<14} {'trades':>7} {'prec':>7} {'lift':>6} {'win':>6} {'mean':>8} "
        f"{'median':>8} {'mean CI':>19} {'buy-all mean':>12}",
    ]
    for name, b in report["buckets"].items():
        r = b["rule"]
        low, high = r["mean_return_ci"]
        ci = "n/a" if low is None else f"[{low:+.3f},{high:+.3f}]"
        lift = "n/a" if r["precision_lift"] is None else f"{r['precision_lift']:.1f}x"
        lines.append(
            f"{name:<14} {r['n_trades']:>7} {fmt(r['precision']):>7} {lift:>6} "
            f"{fmt(r['win_rate']):>6} {fmt(r['mean_return']):>8} {fmt(r['median_return']):>8} "
            f"{ci:>19} {fmt(b['baseline']['mean_return']):>12}"
        )
    swept = {
        name: b["sweep"]["recommended"] for name, b in report["buckets"].items() if "sweep" in b
    }
    if swept:
        lines += [
            "",
            f"Entry-rule sweep @ {pol['horizon']}: the recommended rule inside each bucket",
        ]
        for name, rec in swept.items():
            lines.append(
                f"{name:<14} none (no rule with enough trades)"
                if rec is None
                else f"{name:<14} {_rule_label(rec):<34} {rec['n_trades']:>7} trades, "
                f"precision {fmt(rec['precision'])}, mean {fmt(rec['mean_return'])}"
                + ("" if rec["beats_baseline"] else " (CI does not clear buy-everything)")
            )
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}"


def write_report(report: Mapping[str, Any], path: str | Path) -> tuple[Path, Path]:
    """The JSON report at ``path`` and its rendered tables next to it (``.txt``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = path.with_suffix(".txt")
    table.write_text(render(report) + "\n", encoding="utf-8")
    return path, table


def labeling_config(config: Mapping[str, Any], labels: str) -> dict[str, Any]:
    """``config`` with the floor the report labels under: ``none`` disables it, ``config`` keeps
    the config's."""
    if labels not in LABEL_MODES:
        raise ValueError(f"labels must be one of {', '.join(LABEL_MODES)}, got {labels!r}")
    if labels == "config":
        return dict(config)
    labeling = {k: v for k, v in config["labeling"].items() if k != "liquidity_floor"}
    return {**config, "labeling": labeling}


def heldout_after(metadata: Mapping[str, Any]) -> int:
    """The last epoch second of a checkpoint's data (training, early-stopping validation or
    calibration — the same keys as :func:`solia_ai.evaluate.holdout_dataset`). Raises for a legacy
    checkpoint that records none: its held-out windows can't be told apart."""
    ends = [
        int(metadata[key])
        for key in ("train_end_ts", "calibration_end_ts", "data_end_ts")
        if metadata.get(key) is not None
    ]
    if not ends:
        raise ValueError("the checkpoint records no train_end_ts: its held-out windows are unknown")
    return max(ends)


def report_path(out: str | Path, version: str, several: bool) -> Path:
    """``out`` itself for one model, ``<stem>-<version><suffix>`` beside it for several."""
    out = Path(out)
    return out.with_name(f"{out.stem}-{version}{out.suffix}") if several else out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split checkpoints' held-out metrics by anchor pool liquidity (read-only). "
        "Several models are scored on the same windows: those after every model's data."
    )
    parser.add_argument("--checkpoint", action="append", default=[], help=".pt path (repeatable)")
    parser.add_argument("--champion", action="store_true", help="the registry's champion")
    parser.add_argument("--out", help="write the JSON report here (+ the tables as .txt)")
    parser.add_argument("--labels", choices=LABEL_MODES, default="none")
    parser.add_argument(
        "--lookback-days", type=float, help="override data.lookback_days (the bounded read)"
    )
    parser.add_argument("--max-rows", type=int, help="override data.max_rows (the bounded read)")
    parser.add_argument("--policy-horizon", help="entry-rule horizon (default: 15m if served)")
    parser.add_argument("--pump-threshold", type=float, default=ReferencePolicy.pump_threshold)
    parser.add_argument("--rug-threshold", type=float, default=ReferencePolicy.rug_threshold)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--no-sweep", action="store_true", help="skip the per-bucket rule sweep")
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)
    if not args.champion and not args.checkpoint:
        parser.error("give --champion and/or --checkpoint")

    import torch

    from .config import load_config
    from .dataset import SequenceDataset, embargo_seconds, make_holdout_dataset
    from .db import DataSettings, make_engine, read_training_features
    from .evaluate import apply_calibration, default_policy_horizon, score_logits
    from .inference import Predictor
    from .liquidity import LiquidityFloor, liquidity_buckets

    engine = make_engine()
    predictors = [Predictor.from_checkpoint(path) for path in args.checkpoint]
    if args.champion:
        try:
            predictors.insert(0, Predictor.from_champion(engine))
        except LookupError as exc:
            raise SystemExit(str(exc)) from None
    horizons = list(predictors[0].horizons)
    try:
        if any(list(p.horizons) != horizons for p in predictors):
            raise ValueError("the models must serve the same horizons to share held-out windows")
        after = max(heldout_after(p.metadata) for p in predictors)
        policy = ReferencePolicy(
            horizon=args.policy_horizon or default_policy_horizon(horizons),
            pump_threshold=args.pump_threshold,
            rug_threshold=args.rug_threshold,
        )
        policy.horizon_index(horizons)  # fail before reading or scoring anything
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    config = {**labeling_config(load_config(), args.labels), "horizons": horizons}
    data_settings = DataSettings.from_config(
        config, lookback_days=args.lookback_days, max_rows=args.max_rows
    )
    df, _ = read_training_features(engine, data_settings)
    dataset = make_holdout_dataset(df, config, train_end_ts=after)
    del df
    samples, idx = dataset.samples, dataset.indices
    buckets = liquidity_buckets(samples.anchor_liquidity[idx], samples.anchor_venue[idx])
    floor = LiquidityFloor.from_config(config)
    print(
        f"{len(idx)} held-out windows ending after {after} + {embargo_seconds(horizons)}s embargo"
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for predictor in predictors:
        model = predictor.model.to(device)
        logits, truth = score_logits(model, SequenceDataset(samples, idx), batch_size=1024)
        predictions = apply_calibration(logits, predictor.calibration)
        by_bucket = bucket_report(
            predictions, truth, buckets, horizons, policy,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
            sweep=None if args.no_sweep else {"min_trades": args.min_trades},
        )  # fmt: skip
        report = build_report(
            predictor.model_version,
            args.labels,
            floor.as_dict() if floor.enabled else {},
            policy,
            by_bucket,
            horizons,
        )
        print("\n" + render(report))
        if args.out:
            path = report_path(args.out, predictor.model_version, len(predictors) > 1)
            json_path, table_path = write_report(report, path)
            print(f"\nreport written to {json_path} and {table_path}")


if __name__ == "__main__":
    main()
