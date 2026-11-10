"""Rug guard (#260): split, pool seeds, dataset assembly, criterion, reasons, CLI."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from solia_ai import rug_guard as g
from solia_ai.features import VENUES

DAY = 24 * 3600


def t(value: str):
    return g._utc(value)


# --------------------------------------------------------------------------- split


def small_split() -> g.Split:
    return g.Split(
        train=(t("2026-01-02"), t("2026-01-04")),
        val=(t("2026-01-05"), t("2026-01-06")),
        p1=(t("2026-01-07"), t("2026-01-07 12:00")),
        p2=(t("2026-01-08 12:00"), t("2026-01-09")),
    )


def test_default_split_respects_the_24h_embargo():
    g.DEFAULT_SPLIT.check(DAY)
    assert g.DATA_SINCE < g.DEFAULT_SPLIT.train[0]


def test_split_rejects_a_segment_inside_the_previous_labels_future():
    split = small_split()
    late = g.Split(split.train, split.val, (t("2026-01-06 18:00"), t("2026-01-07")), split.p2)
    with pytest.raises(ValueError, match="p1 starts"):
        late.check(DAY)


def test_split_rejects_an_empty_or_reversed_segment():
    split = small_split()
    with pytest.raises(ValueError, match="val is empty"):
        g.Split(split.train, (t("2026-01-06"), t("2026-01-05")), split.p1, split.p2).check(DAY)


def test_segment_of_leaves_the_embargo_gaps_unassigned():
    ts = pd.Series(
        pd.to_datetime(
            [
                "2026-01-02 00:00",
                "2026-01-03 23:59",
                "2026-01-04 00:00",
                "2026-01-05 10:00",
                "2026-01-07 11:59",
                "2026-01-08 00:00",
                "2026-01-08 12:00",
            ],
            utc=True,
        )
    )
    assert list(small_split().segment_of(ts)) == ["train", "train", "", "val", "p1", "", "p2"]


def test_no_label_of_an_earlier_segment_reads_a_later_segment():
    split = small_split()
    split.check(DAY)
    for (_, (_, end)), (_, (start, _)) in zip(split.segments(), split.segments()[1:], strict=False):
        assert end + timedelta(seconds=DAY) <= start


def test_chunks_cut_at_midnights_and_segment_edges():
    out = g.chunks(small_split(), t("2026-01-01"))
    assert out[0] == (t("2026-01-01"), t("2026-01-02"), None)
    assert (t("2026-01-02"), t("2026-01-03"), "train") in out
    assert (t("2026-01-07"), t("2026-01-07 12:00"), "p1") in out
    assert (t("2026-01-07 12:00"), t("2026-01-08"), None) in out
    assert (t("2026-01-08"), t("2026-01-08 12:00"), None) in out
    assert out[-1] == (t("2026-01-08 12:00"), t("2026-01-09"), "p2")
    # contiguous, covering [since, end of p2)
    assert all(a[1] == b[0] for a, b in zip(out, out[1:], strict=False))


def test_chunks_reject_a_start_after_the_split():
    with pytest.raises(ValueError, match="before the end of p2"):
        g.chunks(small_split(), t("2026-02-01"))


# --------------------------------------------------------------------------- pool seeds


def firsts(rows):
    return pd.DataFrame(rows, columns=["key", "ts", "liquidity"]).assign(
        ts=lambda d: pd.to_datetime(d["ts"], utc=True)
    )


def test_seed_is_the_first_observation_and_later_ones_are_ignored():
    tracker = g.SeedTracker(known_after=t("2026-01-01 01:00"))
    tracker.update(firsts([("A", "2026-01-01 02:00", 85.0), ("A", "2026-01-01 02:01", 80.0)]))
    tracker.update(firsts([("A", "2026-01-01 03:00", 1.0), ("B", "2026-01-01 03:00", 78.9)]))
    assert list(tracker.lookup(np.array(["A", "B"], object))) == [85.0, 78.9]
    assert np.isnan(tracker.lookup(np.array(["C"], object))[0])


def test_seed_of_a_pool_first_seen_in_the_warmup_is_unknown():
    tracker = g.SeedTracker(known_after=t("2026-01-01 01:00"))
    tracker.update(firsts([("A", "2026-01-01 00:30", 85.0)]))
    tracker.update(firsts([("A", "2026-01-01 02:00", 85.0)]))  # seen again: still unknown
    assert np.isnan(tracker.lookup(np.array(["A"], object))[0])


def test_pool_keys_fall_back_to_the_venue_curve():
    keys = g.pool_keys(pd.Series(["P1", None, None]), pd.Series(["pumpswap", "pumpfun", None]))
    assert list(keys) == ["P1", "venue:pumpfun", "venue:"]


# --------------------------------------------------------------------------- dataset


def wide_rows(n=3) -> pd.DataFrame:
    rows = {c: np.full(n, 1.0, np.float32) for c in g.INPUT_COLUMNS}
    rows.update(
        mint=["m1", "m1", "m2"][:n],
        ts=pd.to_datetime(
            ["2026-01-02 00:00", "2026-01-02 00:15", "2026-01-02 00:00"][:n], utc=True
        ),
        venue=["pumpswap", "pumpswap", "orca"][:n],
        pool_address=["P1", "P1", None][:n],
    )
    return pd.DataFrame(rows)


def label_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mint": pd.Categorical(["m1", "m1", "m2"]),
            "ts": pd.to_datetime(
                ["2026-01-02 00:00", "2026-01-02 00:15", "2026-01-02 00:00"], utc=True
            ).astype("datetime64[ns, UTC]"),
            "venue": ["pumpswap", "pumpswap", "orca"],
            "token_age_sec": [10.0, 910.0, 5.0],
            "liquidity": [80.0, 80.0, 3.0],
            "next_sudden_sec": [np.nan, np.nan, 30.0],
            "sudden_1h": np.array([0.0, np.nan, 1.0], np.float32),
            "gradual_1h": np.array([0.0, np.nan, 0.0], np.float32),
            "drain_1h": np.array([np.nan, np.nan, 1.0], np.float32),
        }
    )


def test_assemble_joins_labels_on_mint_and_ts_with_the_seed():
    tracker = g.SeedTracker(known_after=t("2026-01-01"))
    tracker.update(firsts([("P1", "2026-01-01 05:00", 85.0)]))
    wide = wide_rows()
    wide["ts"] = wide["ts"].astype("datetime64[us, UTC]")  # the database's precision
    out = g.assemble(label_rows(), wide, tracker, "train")
    assert len(out) == 3
    out = out.sort_values(["mint", "ts"]).reset_index(drop=True)
    assert list(out["sudden_1h"].fillna(-1)) == [0.0, -1.0, 1.0]
    assert list(out["pool_seed_sol"].fillna(-1)) == [85.0, 85.0, -1.0]  # m2: venue:orca unseen
    assert set(out["segment"]) == {"train"}
    assert "next_sudden_sec" not in out.columns


def test_assemble_drops_anchors_missing_on_either_side():
    tracker = g.SeedTracker(known_after=t("2026-01-01"))
    out = g.assemble(label_rows().iloc[:2], wide_rows().iloc[1:], tracker, "p1")
    assert list(out["mint"]) == ["m1"]


def test_model_inputs_derive_seed_wash_and_venue_columns():
    frame = wide_rows()
    frame["pool_seed_sol"] = [85.0, 78.9, np.nan]
    frame["liquidity"] = [42.5, 78.9, 3.0]
    frame["buy_events"] = [6.0, 1.0, 0.0]
    frame["sell_events"] = [2.0, 1.0, 0.0]
    frame["unique_traders"] = [2.0, 1.0, 0.0]
    frame["venue"] = ["pumpswap", "orca", None]
    frame["rep_creator_rug_share"] = [np.nan, 0.5, np.nan]
    x = g.model_inputs(frame)
    assert list(x.columns) == list(g.MODEL_COLUMNS)
    assert x.dtypes.eq(np.float32).all()
    assert x["pool_seed_frac"].iloc[0] == 0.0
    assert x["pool_seed_frac"].iloc[1] == pytest.approx(0.1, abs=1e-5)
    assert x["liquidity_vs_seed"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(x["liquidity_vs_seed"].iloc[2])
    assert x["events_per_trader"].iloc[0] == 4.0
    assert np.isnan(x["events_per_trader"].iloc[2])  # no trader: unknown, not 0
    assert x["venue_code"].iloc[0] == VENUES.index("pumpswap")
    assert x["venue_code"].iloc[1] == len(VENUES)  # orca: other
    assert x["venue_code"].iloc[2] == len(VENUES)  # NULL: other
    assert np.isnan(x["rep_creator_rug_share"].iloc[0])  # NULL stays NaN (no 0 imputation)


def test_load_dataset_requires_files(tmp_path):
    with pytest.raises(ValueError, match="no parquet"):
        g.load_dataset(tmp_path)


# --------------------------------------------------------------------------- criterion


def test_pick_threshold_keeps_the_most_rows_under_the_rate():
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = np.array([0, 0, 0, 1, 0, 1])
    assert g.pick_threshold(score, y, 0.0) == 0.3
    assert g.pick_threshold(score, y, 0.2) == 0.5  # 1/5
    assert g.pick_threshold(score, y, 1.0) == 0.6


def test_pick_threshold_never_splits_tied_scores():
    score = np.array([0.1, 0.2, 0.2, 0.3])
    y = np.array([0, 0, 1, 0])
    assert g.pick_threshold(score, y, 0.0) == 0.1


def test_pick_threshold_none_when_even_the_safest_group_is_too_risky():
    assert g.pick_threshold(np.array([0.1, 0.2]), np.array([1, 0]), 0.03) is None
    assert g.pick_threshold(np.array([]), np.array([]), 0.03) is None


def test_wilson_upper_bound():
    assert g.wilson_upper(0, 0) is None
    assert g.wilson_upper(0, 100) == pytest.approx(0.037, abs=1e-3)
    assert g.wilson_upper(3, 100) > 0.03


def test_kept_stats():
    s = g.kept_stats(np.array([True, True, False, False]), np.array([0, 1, 1, 1]))
    assert (s["n"], s["kept"], s["coverage"], s["positives"], s["rate"]) == (4, 2, 0.5, 1, 0.5)
    assert g.kept_stats(np.zeros(2, bool), np.array([0, 1]))["rate"] is None


def test_beats_rule_on_a_lower_rate_at_the_rules_coverage():
    score = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 0, 1, 1])
    rule = np.array([False, False, True, True])  # the rule keeps the two rugs
    out = g.beats_rule(score, y, rule, score <= 0.2)
    assert out["beats"] and "lower rate" in out["why"]


def test_beats_rule_on_more_coverage_at_the_same_rate():
    score = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 0, 0, 1])
    rule = np.array([True, False, False, False])  # 0 % at 25 %
    out = g.beats_rule(score, y, rule, score <= 0.3)  # 0 % at 75 %
    assert out["beats"] and "more coverage" in out["why"]


def test_beats_rule_fails_when_the_rule_is_safer():
    score = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([1, 0, 0, 0])
    rule = np.array([False, True, False, False])
    assert not g.beats_rule(score, y, rule, score <= 0.1)["beats"]


def test_beats_rule_when_the_rule_keeps_nothing():
    out = g.beats_rule(np.array([0.1, 0.2]), np.array([0, 1]), np.zeros(2, bool), np.ones(2, bool))
    assert out["beats"]


def survivors(score, y, burned):
    return pd.DataFrame({"score": score, "sudden_24h": y, "burned_liquidity": burned})


def test_judge_passes_a_model_that_separates_both_periods():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(80), np.ones(20)]
    score = np.r_[rng.uniform(0, 0.5, 80), rng.uniform(0.6, 1, 20)]
    burned = np.r_[np.full(10, 100.0), np.full(70, np.nan), np.full(20, 100.0)]
    out = g.judge(survivors(score, y, burned), survivors(score, y, burned), g.Criterion())
    assert out["passed"], out["why"]
    # the cut goes past the 80 safe rows while the rate stays <= 3 % (2 rugs in 82)
    assert out["p2"]["coverage"] == 0.82 and out["p2"]["rate"] == pytest.approx(2 / 82)


def test_judge_fails_on_period2_rate_and_records_why():
    y1 = np.r_[np.zeros(80), np.ones(20)]
    score = np.linspace(0, 1, 100)
    y2 = np.r_[np.ones(20), np.zeros(80)]  # inverted in period 2
    out = g.judge(survivors(score, y1, np.nan), survivors(score, y2, np.nan), g.Criterion())
    assert not out["passed"]
    assert not out["checks"]["rate"]
    assert "rate" in out["why"]


def test_judge_fails_without_a_period1_threshold():
    rows = survivors(np.array([0.1, 0.2]), np.array([1.0, 1.0]), np.nan)
    out = g.judge(rows, rows, g.Criterion())
    assert not out["passed"] and out["threshold"] is None


def test_first_anchors_one_per_token_with_a_known_label():
    frame = pd.DataFrame(
        {
            "mint": ["a", "a", "b", "b", "c"],
            "ts": pd.to_datetime(
                [
                    "2026-01-01 00:15",
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 00:15",
                    "2026-01-01 00:00",
                ],
                utc=True,
            ),
            "token_age_sec": [30000.0, 100.0, 22000.0, 23000.0, 100.0],
            "sudden_24h": [0.0, 1.0, np.nan, 0.0, 1.0],
        }
    )
    every = g.first_anchors(frame, "sudden_24h")
    assert sorted(every["mint"]) == ["a", "c"]  # b's first anchor is unknown: dropped
    assert every.set_index("mint").loc["a", "sudden_24h"] == 1.0  # the earliest one
    old = g.first_anchors(frame, "sudden_24h", min_age_seconds=g.SURVIVOR_AGE_SECONDS)
    assert list(old["mint"]) == ["a"] and old["sudden_24h"].iloc[0] == 0.0


# --------------------------------------------------------------------------- models & reasons


def synthetic_frame(n_per_segment=600, seed=0) -> pd.DataFrame:
    """Anchors whose sudden rug comes from burned_liquidity == 0 and a creator rug history."""
    rng = np.random.default_rng(seed)
    starts = {
        "train": "2026-01-02",
        "val": "2026-01-05",
        "p1": "2026-01-07",
        "p2": "2026-01-08 12:00",
    }
    parts = []
    for segment, start in starts.items():
        n = n_per_segment
        rows = {c: rng.normal(size=n).astype(np.float32) for c in g.INPUT_COLUMNS}
        burned = rng.choice([0.0, 100.0], size=n)
        creator = np.where(rng.random(n) < 0.5, np.nan, rng.random(n))
        risk = (burned == 0) & (np.nan_to_num(creator, nan=0.0) > 0.3)
        rows.update(
            burned_liquidity=burned,
            rep_creator_rug_share=creator,
            token_age_sec=rng.uniform(0, 2 * g.SURVIVOR_AGE_SECONDS, n),
            mint=[f"{segment}-{i}" for i in range(n)],
            ts=pd.Timestamp(start, tz="UTC") + pd.to_timedelta(rng.uniform(0, 3600, n), "s"),
            venue=rng.choice(["pumpswap", "meteora", "orca"], n),
            pool_address=None,
            pool_seed_sol=rng.choice([85.0, 78.9, np.nan], n),
            segment=segment,
        )
        for h in ("1h", "6h", "24h"):
            rows[f"sudden_{h}"] = (risk & (rng.random(n) < 0.9)).astype(float)
            rows[f"gradual_{h}"] = (rng.random(n) < 0.2).astype(float)
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)


SMALL = {"n_estimators": 60, "learning_rate": 0.2, "num_leaves": 7, "min_child_samples": 10}


def test_reasons_name_the_driving_features():
    pytest.importorskip("lightgbm")
    frame = synthetic_frame()
    x = g.model_inputs(frame)
    tr, va = frame["segment"] == "train", frame["segment"] == "val"
    model = g.fit_target(
        x[tr],
        frame["sudden_24h"][tr].to_numpy(),
        x[va],
        frame["sudden_24h"][va].to_numpy(),
        params=SMALL,
        early_stopping_rounds=10,
    )
    risky = frame[(frame["burned_liquidity"] == 0) & (frame["rep_creator_rug_share"] > 0.6)]
    why = g.reasons(model, g.model_inputs(risky.head(20)), top=2)
    top = {item["feature"] for items in why for item in items[:1]}
    assert top <= {"burned_liquidity", "rep_creator_rug_share"}
    assert all(item["contribution"] > 0 for items in why for item in items)
    assert "burned_liquidity=0" in g.format_reasons(why[0]) or "rep_creator" in g.format_reasons(
        why[0]
    )


def test_reasons_validate_top():
    with pytest.raises(ValueError, match="top"):
        g.reasons(None, pd.DataFrame(), top=0)


def test_format_reasons_marks_null():
    text = g.format_reasons([{"feature": "f", "value": None, "contribution": 0.5}])
    assert text == "f=NULL (+0.50)"


def test_fit_target_needs_both_classes():
    pytest.importorskip("lightgbm")
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="both classes"):
        g.fit_target(x, np.zeros(3), x, np.array([0, 1, 0]))


def test_evaluate_end_to_end_on_synthetic_anchors():
    pytest.importorskip("lightgbm")
    out = g.evaluate(
        synthetic_frame(), ("1h", "6h", "24h"), params=SMALL, early_stopping_rounds=10, examples=3
    )
    report = out.report
    assert set(report["targets"]) == {f"{k}_{h}" for k in g.KINDS for h in ("1h", "6h", "24h")}
    entry = report["targets"]["sudden_24h"]
    assert entry["p2/all"]["roc_auc"] > 0.85
    assert entry["importance"][0]["feature"] in {"burned_liquidity", "rep_creator_rug_share"}
    crit = report["criterion"]
    assert crit["threshold"] is not None and "checks" in crit
    assert report["p2_survivors_seen_in_train"] == 0.0
    assert len(report["examples"]) == 6 and "reasons" in report["examples"][0]
    text = g.render(report)
    assert "VERDICT" in text and "sudden_24h" in text


def test_evaluate_rejects_an_empty_segment_and_an_unknown_target():
    frame = synthetic_frame(n_per_segment=20)
    with pytest.raises(ValueError, match="segment p2"):
        g.evaluate(frame[frame["segment"] != "p2"], ("1h", "6h", "24h"))
    with pytest.raises(ValueError, match="criterion target"):
        g.evaluate(frame, ("1h",))


# --------------------------------------------------------------------------- CLI


def test_cli_defaults_to_the_preregistered_split():
    args = g.parser().parse_args(["evaluate", "--dataset", "d"])
    assert g.split_from(args) == g.DEFAULT_SPLIT
    args = g.parser().parse_args(["build", "--out-dir", "o", "--resume"])
    assert args.resume and args.anchor_seconds == g.DEFAULT_ANCHOR_SECONDS
    assert args.since == g.DATA_SINCE


def test_cli_parses_a_window_override():
    args = g.parser().parse_args(["evaluate", "--dataset", "d", "--p2", "2026-11-07/2026-11-08"])
    assert args.p2 == (t("2026-11-07"), t("2026-11-08"))


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["evaluate"],
        ["build"],
        ["evaluate", "--dataset", "d", "--p1", "2026-11-05"],
    ],
)
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        g.parser().parse_args(argv)


def test_main_rejects_a_split_without_embargo(tmp_path):
    with pytest.raises(SystemExit, match="p1 starts"):
        g.main(["evaluate", "--dataset", str(tmp_path), "--p1", "2026-11-04 06:00/2026-11-06"])
