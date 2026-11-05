"""Live vs offline scoring: window selection, coverage, the comparison, the trade snapshots, the
report and bad settings, with a fake predictor."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from solia_ai.features import DB_FEATURE_COLUMNS, FEATURE_COLUMNS
from solia_ai.rescore import (
    build_report,
    check_settings,
    compare,
    main,
    mint_blocks,
    render,
    rescore,
    select_predictions,
    snapshot_differences,
    write_report,
)

T0 = pd.Timestamp("2026-11-02T18:00:00.250Z")
PRICE = FEATURE_COLUMNS.index("price")


class FakePredictor:
    """Scores a window by its last price, and records what it was given."""

    def __init__(self, seq_len=3, min_seq_len=1):
        self.seq_len, self.min_seq_len = seq_len, min_seq_len
        self.calls = []

    def predict(self, windows, horizons):
        self.calls.append(horizons)
        self.windows = getattr(self, "windows", []) + list(windows)
        return [
            SimpleNamespace(
                predictions=[SimpleNamespace(pump_prob=float(w[-1, PRICE]), rug_prob=0.1)]
            )
            for w in windows
        ]


def rows_of(prices, *, mint="A", step=10.0, t0=T0):
    frame = pd.DataFrame(0.0, index=range(len(prices)), columns=DB_FEATURE_COLUMNS)
    frame["price"] = prices
    frame["mint"] = mint
    frame["venue"] = "pumpswap"
    frame["ts"] = [t0 + pd.Timedelta(seconds=step * i) for i in range(len(prices))]
    return frame


def stored(*items):
    return pd.DataFrame(
        [
            {"mint": m, "ts": T0 + pd.Timedelta(seconds=s), "pump_prob": p, "rug_prob": 0.1}
            for m, s, p in items
        ]
    )


def run(predictor, predictions, rows, **kwargs):
    settings = {"horizon": "15m", "horizon_seconds": 30, **kwargs}
    return rescore(predictor, predictions, rows, **settings)


# --------------------------------------------------------------------------- windows


def test_the_window_is_the_latest_rows_up_to_the_prediction():
    predictor = FakePredictor(seq_len=3)
    out = run(predictor, stored(("A", 30, 0.5)), rows_of([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert predictor.windows[0][:, PRICE].tolist() == pytest.approx([0.2, 0.3, 0.4])
    assert out["offline_pump"].tolist() == pytest.approx([0.4])
    assert out["exact_row"].tolist() == [True]
    assert out["window_rows"].tolist() == [3]
    assert predictor.calls == [["15m"]]


def test_a_row_rows_later_than_the_prediction_never_enters_the_window():
    predictor = FakePredictor(seq_len=5)
    run(predictor, stored(("A", 15, 0.5)), rows_of([0.1, 0.2, 0.9, 0.9]))
    assert predictor.windows[0][:, PRICE].tolist() == pytest.approx([0.1, 0.2])


def test_no_exact_row_is_reported():
    out = run(FakePredictor(), stored(("A", 15, 0.5)), rows_of([0.1, 0.2, 0.3]))
    assert out["exact_row"].tolist() == [False]
    assert out["offline_pump"].tolist() == pytest.approx([0.2])  # scored on the rows before it


def test_a_token_without_rows_has_no_window():
    out = run(FakePredictor(), stored(("Z", 0, 0.5)), rows_of([0.1]))
    assert out["window_rows"].tolist() == [0]
    assert out["offline_pump"].isna().all()


def test_a_prediction_before_the_tokens_rows_has_no_window():
    out = run(FakePredictor(), stored(("A", -5, 0.5)), rows_of([0.1]))
    assert out["window_rows"].tolist() == [0]


def test_a_window_under_the_minimum_length_is_not_scored():
    predictor = FakePredictor(seq_len=3, min_seq_len=2)
    out = run(predictor, stored(("A", 0, 0.5), ("A", 10, 0.5)), rows_of([0.1, 0.2]))
    assert out["too_short"].tolist() == [True, False]
    assert np.isnan(out["offline_pump"][0])
    assert out["offline_pump"][1] == pytest.approx(0.2)


def test_tokens_stay_apart_in_any_row_order():
    rows = pd.concat([rows_of([0.3, 0.4], mint="B"), rows_of([0.1, 0.2], mint="A")])
    predictor = FakePredictor()
    out = run(
        predictor, stored(("A", 10, 0.5), ("B", 10, 0.5)), rows.sample(frac=1, random_state=1)
    )
    assert out["offline_pump"].tolist() == pytest.approx([0.2, 0.4])


def test_flat_looks_only_at_the_horizon():
    rows = rows_of([0.5, 0.5, 0.5, 0.5, 0.9])  # the jump comes 40 s later, the horizon is 30 s
    out = run(FakePredictor(), stored(("A", 0, 0.5), ("A", 10, 0.5)), rows)
    assert out["flat"].tolist() == [True, False]


def test_a_silent_token_is_flat():
    out = run(FakePredictor(), stored(("A", 10, 0.5)), rows_of([0.5, 0.5]))
    assert out["flat"].tolist() == [True]


def test_batches_cover_every_window():
    rows = rows_of([0.1 * k for k in range(1, 8)])
    predictions = stored(*(("A", 10 * k, 0.5) for k in range(7)))
    predictor = FakePredictor()
    out = run(predictor, predictions, rows, batch_size=3)
    assert len(predictor.calls) == 3
    assert out["offline_pump"].notna().all()


def test_selection_keeps_high_predictions_and_samples_the_others():
    predictions = stored(*(("A", k, 0.01) for k in range(10)), ("B", 0, 0.5), ("C", 0, 0.2))
    out = select_predictions(predictions, selected_from=0.2, others=3)
    assert (out["pump_prob"] >= 0.2).sum() == 2
    assert (out["pump_prob"] < 0.2).sum() == 3
    assert len(select_predictions(predictions, selected_from=0.2, others=99)) == 12


def test_blocks_keep_tokens_whole_and_cover_every_prediction():
    predictions = stored(("A", 0, 0.5), ("B", 0, 0.5), ("A", 10, 0.5), ("C", 0, 0.5))
    blocks = list(mint_blocks(predictions, 2))
    assert [sorted(set(b["mint"])) for b in blocks] == [["A", "B"], ["C"]]
    assert sum(len(b) for b in blocks) == 4
    assert list(mint_blocks(predictions.iloc[:0], 2)) == []
    with pytest.raises(ValueError):
        list(mint_blocks(predictions, 0))


# --------------------------------------------------------------------------- comparison


def results(rows):
    """``rows``: (stored pump, offline pump, flat)."""
    return pd.DataFrame(
        [
            {
                "mint": f"M{i}",
                "pump_prob": live,
                "rug_prob": 0.2,
                "offline_pump": offline,
                "offline_rug": 0.3,
                "exact_row": True,
                "window_rows": 3 if offline == offline else 0,
                "too_short": False,
                "flat": flat,
            }
            for i, (live, offline, flat) in enumerate(rows)
        ]
    )


def test_identical_scores_show_no_difference():
    report = compare(
        results([(0.01, 0.01, True), (0.1, 0.1, False), (0.5, 0.5, False)]), entry_threshold=0.4
    )
    overall = report["overall"]
    assert overall["median_abs_difference"] == 0
    assert overall["large_difference_share"] == 0
    assert overall["rank_correlation"] == pytest.approx(1.0)
    assert overall["both_entries"] == 1
    assert overall["live_only_entries"] == overall["offline_only_entries"] == 0


def test_differences_and_threshold_crossings_are_counted():
    frame = results([(0.5, 0.1, True), (0.45, 0.42, True), (0.1, 0.6, False), (0.1, 0.11, False)])
    report = compare(frame, entry_threshold=0.4)
    overall = report["overall"]
    assert overall["live_only_entries"] == 1
    assert overall["offline_only_entries"] == 1
    assert overall["both_entries"] == 1
    assert overall["large_difference_share"] == pytest.approx(0.5)
    assert overall["rug_median_abs_difference"] == pytest.approx(0.1)


def test_buckets_and_flat_groups():
    frame = results([(0.01, 0.01, True), (0.5, 0.1, True), (0.45, 0.45, False), (1.0, 1.0, False)])
    report = compare(frame, entry_threshold=0.4)
    assert report["buckets"]["0-0.05"]["n"] == 1
    assert report["buckets"]["0.4-1"]["n"] == 3  # a stored 1.0 included
    assert report["selected_flat"]["n"] == 1
    assert report["selected_flat"]["live_only_entries"] == 1
    assert report["selected_moving"]["n"] == 2


def test_coverage_counts_what_could_not_be_scored():
    frame = results([(0.5, float("nan"), False), (0.5, 0.5, False)])
    report = compare(frame, entry_threshold=0.4)
    assert report["coverage"]["no_window"] == 1
    assert report["overall"]["n"] == 2
    assert report["overall"]["n_scored"] == 1


def test_a_group_without_scores_stays_reportable():
    report = compare(results([(0.01, 0.01, False)]), entry_threshold=0.4)
    assert report["selected_flat"] == {"n": 0, "n_scored": 0, "n_tokens": 0}


# --------------------------------------------------------------------------- snapshots


def test_identical_snapshots_list_no_column():
    row = {"price": 1.0, "holders": 3, "is_migrated": True, "ema": None}
    out = snapshot_differences([(dict(row, is_migrated=1), row)], columns=list(row))
    assert out["n_identical_trades"] == 1
    assert out["columns"] == {}


def test_snapshot_differences_per_column():
    pairs = [
        ({"price": 1.0, "holders": 3}, {"price": 1.0, "holders": 4}),
        ({"price": 1.0, "holders": 3}, {"price": 1.0 + 1e-9, "holders": 3}),
        ({"price": 2.0, "holders": None}, {"price": 2.0, "holders": 1}),
    ]
    out = snapshot_differences(pairs, columns=["price", "holders"])
    assert out["n_identical_trades"] == 1
    assert list(out["columns"]) == ["holders"]
    assert out["columns"]["holders"]["n_different"] == 2
    assert out["columns"]["holders"]["examples"][0] == (3.0, 4.0)


def test_a_missing_row_and_an_unsnapshotted_column():
    pairs = [({"price": 1.0}, None), ({"price": 1.0}, {"price": 1.0, "ema": 7.0})]
    out = snapshot_differences(pairs, columns=["price", "ema"])
    assert out["n_missing_rows"] == 1
    assert out["n_identical_trades"] == 1
    assert out["columns"] == {}


# --------------------------------------------------------------------------- report


def _report():
    frame = results([(0.5, 0.1, True), (0.45, 0.45, False), (0.01, 0.01, False)])
    snapshots = snapshot_differences([({"price": 1.0}, {"price": 2.0})], columns=["price"])
    return build_report(
        "model-x", "15m", compare(frame, entry_threshold=0.4), snapshots, {"entry_threshold": 0.4}
    )


def test_report_renders_and_is_written(tmp_path):
    report = _report()
    text = render(report)
    assert "model-x" in text and "overall" in text and "flat" in text
    assert "price: 1 of 1 differ" in text
    json_path, table_path = write_report(report, tmp_path / "sub" / "rescore.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["horizon"] == "15m"
    assert "moving" in table_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- bad settings


@pytest.mark.parametrize(
    "args",
    [(0.0, 0.2, 1), (1.5, 0.2, 1), (float("nan"), 0.2, 1), (0.4, 0.0, 1), (0.4, 0.2, -1)],
)
def test_bad_settings_are_rejected(args):
    with pytest.raises(ValueError):
        check_settings(*args)


def test_bad_batch_size_is_rejected():
    with pytest.raises(ValueError):
        check_settings(0.4, 0.2, 1, batch_size=0)


@pytest.mark.parametrize(
    "args",
    [
        ["--model-version", "x", "--entry-threshold", "0"],
        ["--model-version", "x", "--others", "-1"],
        [],
    ],
)
def test_cli_rejects_bad_settings_before_reading_anything(args):
    with pytest.raises(SystemExit):
        main(args)
