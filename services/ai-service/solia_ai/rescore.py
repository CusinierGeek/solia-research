"""Re-score stored live predictions offline: do live and offline scoring agree? (#239)

The bot's feed stores each prediction with the ``ts`` of the feature row it scored. This read-only
command rebuilds the window the feed would load today (the token's latest rows up to that ``ts``),
scores it through the same checkpoint and :class:`~solia_ai.inference.Predictor` path as
``/predict`` and compares the probabilities. Nothing is written but the report.

* **Same probabilities**: the model sees the same thing live and offline, so a gap between paper
  trading and the backtests comes from elsewhere (population, period, exits).
* **Different probabilities**: backtests don't describe what the bot saw.

**What a difference can't tell apart.** The window is read from today's ``features``, and replays
wrote part of that table after these predictions. A difference may be a live vs replay skew, or a
row rewritten since. ``trades.features_snapshot`` keeps the anchor row the bot saw, so
:func:`snapshot_differences` compares it with today's row, per column, on the trades. ``exact_row``
tells whether a row still exists at the prediction's ``ts``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import fmt, json_safe
from .features import DB_FEATURE_COLUMNS, build_feature_matrix

DEFAULT_HORIZON = "15m"
DEFAULT_ENTRY_THRESHOLD = 0.4
DEFAULT_SELECTED_FROM = 0.2  # every prediction at or above it is re-scored ...
DEFAULT_OTHERS = 5000  # ... and a random sample of the others
LARGE_DIFFERENCE = 0.05
HISTORY_MARGIN = pd.Timedelta(hours=6)  # rows read before the first prediction, for its window
SNAPSHOT_TOLERANCE = 1e-6
BUCKET_EDGES = (0.0, 0.05, 0.2, 0.3, 0.4, 1.0000001)
BATCH_SIZE = 256
MINTS_PER_READ = 200


def check_settings(
    entry_threshold: float, selected_from: float, others: int, batch_size: int = BATCH_SIZE
) -> None:
    for name, value in (("entry threshold", entry_threshold), ("selected-from", selected_from)):
        if not (math.isfinite(value) and 0.0 < value <= 1.0):
            raise ValueError(f"{name} must be in (0, 1], got {value!r}")
    if others < 0:
        raise ValueError(f"others must be >= 0, got {others!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")


def select_predictions(
    predictions: pd.DataFrame, *, selected_from: float, others: int, seed: int = 0
) -> pd.DataFrame:
    """Every prediction at or above ``selected_from``, plus ``others`` sampled below it."""
    high = predictions[predictions["pump_prob"] >= selected_from]
    low = predictions[predictions["pump_prob"] < selected_from]
    if len(low) > others:
        low = low.sample(others, random_state=seed)
    return pd.concat([high, low]).sort_values(["mint", "ts"], ignore_index=True)


def mint_blocks(predictions: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    """``predictions`` cut into blocks of at most ``size`` tokens, each token whole."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size!r}")
    codes = pd.factorize(predictions["mint"])[0]
    for k in range(int(codes.max()) // size + 1 if len(codes) else 0):
        yield predictions[codes // size == k]


def rescore(
    predictor: Any,
    predictions: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    horizon: str,
    horizon_seconds: float,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """``predictions`` (``mint``, ``ts``, ``pump_prob``, ``rug_prob``) with their offline scores.

    Adds ``exact_row`` (a row exists at ``ts``), ``window_rows``, ``too_short`` (under the model's
    minimum length, which ``/predict`` refuses), ``flat`` (no row of the horizon moved from the
    anchor), ``offline_pump`` and ``offline_rug`` (NaN when there is no window to score).
    """
    rows = rows.sort_values(["mint", "ts"], kind="stable", ignore_index=True)
    matrix = build_feature_matrix(rows)
    price = pd.to_numeric(rows["price"], errors="coerce").to_numpy(np.float64)
    ts = _epoch(rows["ts"])
    spans = {
        mint: (int(i[0]), int(i[-1]) + 1)
        for mint, i in rows.groupby("mint", sort=False).indices.items()
    }

    n = len(predictions)
    exact, flat, too_short = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    lengths = np.zeros(n, np.int64)
    windows: list[tuple[int, np.ndarray]] = []
    wanted = _epoch(predictions["ts"])
    for i, (mint, t) in enumerate(zip(predictions["mint"], wanted, strict=True)):
        start, stop = spans.get(mint, (0, 0))
        end = start + int(np.searchsorted(ts[start:stop], t, side="right"))
        if end == start:
            continue
        exact[i] = ts[end - 1] == t
        first = max(start, end - predictor.seq_len)
        lengths[i] = end - first
        after = start + int(
            np.searchsorted(ts[start:stop], t + horizon_seconds * 1000, side="right")
        )
        flat[i] = bool(np.all(price[end:after] == price[end - 1]))
        if lengths[i] < predictor.min_seq_len:
            too_short[i] = True
            continue
        windows.append((i, matrix[first:end]))

    pump, rug = np.full(n, np.nan), np.full(n, np.nan)
    for b in range(0, len(windows), batch_size):
        batch = windows[b : b + batch_size]
        scored = predictor.predict([w for _, w in batch], [horizon])
        for (i, _), prediction in zip(batch, scored, strict=True):
            pump[i] = prediction.predictions[0].pump_prob
            rug[i] = prediction.predictions[0].rug_prob
    return predictions.assign(
        exact_row=exact,
        window_rows=lengths,
        too_short=too_short,
        flat=flat,
        offline_pump=pump,
        offline_rug=rug,
    )


def _epoch(values: pd.Series) -> np.ndarray:
    """Epoch milliseconds: the exact ``ts`` comparison must not go through float seconds."""
    stamps = pd.to_datetime(values, utc=True).dt.tz_localize(None)
    return stamps.to_numpy("datetime64[ms]").astype(np.int64)


def _group(frame: pd.DataFrame, *, entry_threshold: float) -> dict[str, Any]:
    """The comparison over one group of predictions; ``n_scored`` of ``n`` have a window."""
    scored = frame[frame["offline_pump"].notna()]
    out: dict[str, Any] = {
        "n": len(frame),
        "n_scored": len(scored),
        "n_tokens": frame["mint"].nunique(),
    }
    if scored.empty:
        return out
    live, offline = scored["pump_prob"], scored["offline_pump"]
    difference = (offline - live).abs()
    rank_correlation = live.rank().corr(offline.rank()) if len(scored) > 1 else None
    return out | {
        "live_mean": float(live.mean()),
        "offline_mean": float(offline.mean()),
        "median_abs_difference": float(difference.median()),
        "large_difference_share": float((difference > LARGE_DIFFERENCE).mean()),
        "rug_median_abs_difference": float(
            (scored["offline_rug"] - scored["rug_prob"]).abs().median()
        ),
        "rank_correlation": None if rank_correlation is None else float(rank_correlation),
        "live_only_entries": int(((live >= entry_threshold) & (offline < entry_threshold)).sum()),
        "offline_only_entries": int(
            ((live < entry_threshold) & (offline >= entry_threshold)).sum()
        ),
        "both_entries": int(((live >= entry_threshold) & (offline >= entry_threshold)).sum()),
    }


def compare(results: pd.DataFrame, *, entry_threshold: float) -> dict[str, Any]:
    """Coverage, then the comparison overall, per stored ``pump_prob`` bucket and, on the
    predictions the entry rule selected, by whether the price stayed flat."""
    buckets = {}
    for low, high in zip(BUCKET_EDGES, BUCKET_EDGES[1:], strict=False):
        name = f"{low:g}-{min(high, 1.0):g}"
        part = results[(results["pump_prob"] >= low) & (results["pump_prob"] < high)]
        buckets[name] = _group(part, entry_threshold=entry_threshold)
    selected = results[results["pump_prob"] >= entry_threshold]
    return {
        "coverage": {
            "n": len(results),
            "exact_row_share": float(results["exact_row"].mean()) if len(results) else None,
            "no_window": int((results["window_rows"] == 0).sum()),
            "too_short": int(results["too_short"].sum()),
        },
        "overall": _group(results, entry_threshold=entry_threshold),
        "buckets": buckets,
        "selected_flat": _group(selected[selected["flat"]], entry_threshold=entry_threshold),
        "selected_moving": _group(selected[~selected["flat"]], entry_threshold=entry_threshold),
    }


def _number(value: Any) -> float | None:
    """A snapshot or row value as a number (booleans too); ``None`` when it is NULL or not one."""
    if value is None or isinstance(value, str):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def snapshot_differences(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any] | None]],
    *,
    columns: Sequence[str] = DB_FEATURE_COLUMNS,
    tolerance: float = SNAPSHOT_TOLERANCE,
) -> dict[str, Any]:
    """Per column, how the live ``features_snapshot`` differs from today's row at the same
    ``(mint, ts)``. ``pairs`` holds ``(snapshot, row)``; a ``None`` row no longer exists. A value
    against a NULL counts as a difference. Only columns differing at least once are listed."""
    compared = [(s, r) for s, r in pairs if r is not None]
    different: dict[str, dict[str, Any]] = {}
    for column in columns:
        n, examples = 0, []
        for snapshot, row in compared:
            if column not in snapshot:
                continue
            a, b = _number(snapshot[column]), _number(row.get(column))
            same = (a is None and b is None) or (
                a is not None
                and b is not None
                and math.isclose(a, b, rel_tol=tolerance, abs_tol=1e-12)
            )
            n += 1
            if not same:
                examples.append((a, b))
        if examples:
            different[column] = {
                "n_compared": n,
                "n_different": len(examples),
                "examples": examples[:3],
            }
    return {
        "n_trades": len(pairs),
        "n_missing_rows": len(pairs) - len(compared),
        "n_identical_trades": sum(
            all(
                c not in s
                or (_number(s[c]) is None and _number(r.get(c)) is None)
                or (
                    _number(s[c]) is not None
                    and _number(r.get(c)) is not None
                    and math.isclose(
                        _number(s[c]), _number(r.get(c)), rel_tol=tolerance, abs_tol=1e-12
                    )
                )
                for c in columns
            )
            for s, r in compared
        ),
        "columns": dict(sorted(different.items(), key=lambda kv: -kv[1]["n_different"])),
    }


# --------------------------------------------------------------------------- report


def build_report(
    model_version: str, horizon: str, comparison: Mapping[str, Any], snapshots: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    return json_safe(
        {
            "model_version": model_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "horizon": horizon,
            "settings": dict(settings),
            "comparison": dict(comparison),
            "snapshots": dict(snapshots),
        }
    )


def _row(name: str, g: Mapping[str, Any]) -> str:
    if not g.get("n_scored"):
        return f"  {name:<12} {g['n']:>7} {0:>7}"
    return (
        f"  {name:<12} {g['n']:>7} {g['n_scored']:>7} {g['n_tokens']:>6} {fmt(g['live_mean']):>7} "
        f"{fmt(g['offline_mean']):>7} {fmt(g['median_abs_difference']):>7} "
        f"{g['large_difference_share'] * 100:>5.0f}% {fmt(g['rank_correlation']):>7} "
        f"{g['both_entries']:>5} {g['live_only_entries']:>5} {g['offline_only_entries']:>5}"
    )


def render(report: Mapping[str, Any]) -> str:
    c, s = report["comparison"], report["snapshots"]
    coverage = c["coverage"]
    lines = [
        f"Live vs offline scoring — model {report['model_version']}, horizon {report['horizon']}, "
        f"generated {report['generated_at']}",
        f"{coverage['n']} predictions re-scored: exact row {fmt(coverage['exact_row_share'])}, "
        f"no window {coverage['no_window']}, too short {coverage['too_short']}",
        "",
        f"  {'stored pump':<12} {'n':>7} {'scored':>7} {'tokens':>6} {'live':>7} {'offline':>7} "
        f"{'med|d|':>7} {'|d|>.05':>6} {'rank r':>7} {'both':>5} {'live':>5} {'off':>5}",
        _row("overall", c["overall"]),
    ]
    lines += [_row(name, g) for name, g in c["buckets"].items()]
    lines += [
        "  entry rule's selection, by what the price did over the horizon",
        _row("flat", c["selected_flat"]),
        _row("moving", c["selected_moving"]),
        "",
        f"Trade snapshots vs today's rows: {s['n_trades']} trades, "
        f"{s['n_missing_rows']} rows gone, {s['n_identical_trades']} identical",
    ]
    lines += [
        f"  {name}: {v['n_different']} of {v['n_compared']} differ, e.g. {v['examples'][0]}"
        for name, v in s["columns"].items()
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


PREDICTIONS_SQL = """
SELECT p.mint, p.ts, p.pump_prob, p.rug_prob
FROM predictions p JOIN models m ON m.id = p.model_id
WHERE m.version = :version AND p.horizon = :horizon AND p.pump_prob IS NOT NULL
"""
SNAPSHOTS_SQL = """
SELECT t.features_snapshot AS snapshot, to_jsonb(f) AS row
FROM trades t
JOIN predictions p ON p.id = t.prediction_id
JOIN models m ON m.id = p.model_id
LEFT JOIN features f ON f.mint = p.mint AND f.ts = p.ts
WHERE m.version = :version AND t.features_snapshot IS NOT NULL
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-score a model's stored predictions and compare (read-only)."
    )
    parser.add_argument("--model-version", required=True, help="models.version")
    parser.add_argument("--checkpoint", help="default: the registry's path for that version")
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--entry-threshold", type=float, default=DEFAULT_ENTRY_THRESHOLD)
    parser.add_argument("--selected-from", type=float, default=DEFAULT_SELECTED_FROM)
    parser.add_argument("--others", type=int, default=DEFAULT_OTHERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the JSON report here (+ the table as .txt)")
    args = parser.parse_args(argv)
    try:
        check_settings(args.entry_threshold, args.selected_from, args.others)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    from sqlalchemy import text

    from .config import horizon_seconds
    from .db import make_engine
    from .inference import Predictor

    engine = make_engine()
    params = {"version": args.model_version}
    path = args.checkpoint
    if path is None:
        found = pd.read_sql(
            text("SELECT path FROM models WHERE version = :version"), engine, params=params
        )
        if found.empty:
            raise SystemExit(f"no model {args.model_version!r} in the registry")
        path = found["path"].iloc[0]
    predictor = Predictor.from_checkpoint(path)
    if args.horizon not in predictor.horizons:
        raise SystemExit(f"horizon {args.horizon!r} is not one of {list(predictor.horizons)}")

    stored = pd.read_sql(text(PREDICTIONS_SQL), engine, params={**params, "horizon": args.horizon})
    if stored.empty:
        raise SystemExit(f"no stored {args.horizon} prediction for {args.model_version!r}")
    chosen = select_predictions(
        stored, selected_from=args.selected_from, others=args.others, seed=args.seed
    )
    seconds = horizon_seconds(args.horizon)
    columns = ", ".join(["mint", "ts", "venue", *DB_FEATURE_COLUMNS])
    query = text(
        f"SELECT {columns} FROM features WHERE mint = ANY(:mints) AND ts >= :start AND ts <= :end"
    )
    start = pd.to_datetime(chosen["ts"], utc=True).min() - HISTORY_MARGIN
    end = pd.to_datetime(chosen["ts"], utc=True).max() + pd.Timedelta(seconds=seconds)
    parts = []
    # Token by token block: the rows of every chosen token together don't fit in memory.
    for block in mint_blocks(chosen, MINTS_PER_READ):
        rows = pd.read_sql(
            query,
            engine,
            params={"mints": block["mint"].unique().tolist(), "start": start, "end": end},
        )
        parts.append(rescore(predictor, block, rows, horizon=args.horizon, horizon_seconds=seconds))
        print(f"re-scored {sum(len(p) for p in parts)} of {len(chosen)}", flush=True)
    results = pd.concat(parts, ignore_index=True)
    pairs = pd.read_sql(text(SNAPSHOTS_SQL), engine, params=params)
    report = build_report(
        args.model_version,
        args.horizon,
        compare(results, entry_threshold=args.entry_threshold),
        snapshot_differences(list(zip(pairs["snapshot"], pairs["row"], strict=True))),
        {
            "entry_threshold": args.entry_threshold,
            "selected_from": args.selected_from,
            "others": args.others,
            "seed": args.seed,
            "n_stored": len(stored),
        },
    )
    print(render(report))
    if args.out:
        json_path, table_path = write_report(report, args.out)
        print(f"\nreport written to {json_path} and {table_path}")


if __name__ == "__main__":
    main()
