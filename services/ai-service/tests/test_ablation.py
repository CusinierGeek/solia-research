"""The ablation harness: variants, walk-forward windows, lift decisions and an end-to-end run."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_train import make_features_df, small_config

import solia_ai.ablation as ablation
import solia_ai.config
import solia_ai.db
import solia_ai.registry
from solia_ai.ablation import (
    AblationReport,
    Variant,
    decide,
    paired_lifts,
    plan_variants,
    run_ablation,
    run_metrics,
    summarize,
    walk_forward_windows,
)
from solia_ai.evaluate import EvaluationReport, ReferencePolicy
from solia_ai.features import BASELINE_GROUPS, FEATURE_COLUMNS, FEATURE_GROUPS

# --- variants & masking ---


def test_default_plan_is_baseline_then_one_variant_per_declared_group():
    variants = plan_variants()

    assert variants[0] == Variant("baseline", BASELINE_GROUPS)
    assert [v.group for v in variants[1:]] == list(FEATURE_GROUPS)
    for v in variants[1:]:
        if v.group in BASELINE_GROUPS:  # baseline group: dropped
            assert (v.name, v.mode) == (f"-{v.group}", "drop")
            assert set(v.groups) == set(BASELINE_GROUPS) - {v.group}
        else:  # candidate group (e.g. buy_pressure): added
            assert (v.name, v.mode) == (f"+{v.group}", "add")
            assert v.groups == (*BASELINE_GROUPS, v.group)


def test_a_group_outside_the_baseline_is_added_to_it():
    variants = plan_variants(["flow"], baseline=("price",))

    assert variants[1] == Variant("+flow", ("price", "flow"), "flow", "add")


def test_duplicate_candidates_are_planned_once():
    assert [v.name for v in plan_variants(["flow", "flow"])] == ["baseline", "-flow"]


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError, match="unknown feature group.*bogus"):
        plan_variants(["price", "bogus"])
    with pytest.raises(ValueError, match="unknown feature group"):
        plan_variants(baseline=("bogus",))


def test_dropping_the_only_group_is_rejected():
    with pytest.raises(ValueError, match="no feature"):
        plan_variants(["price"], baseline=("price",))


def test_an_empty_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline needs at least one"):
        plan_variants(["flow"], baseline=())


def test_duplicate_baseline_groups_are_kept_once():
    variants = plan_variants(["flow"], baseline=("price", "price"))

    assert variants[0].groups == ("price",)
    assert variants[1].groups == ("price", "flow")


def test_rows_predating_a_tested_group_are_refused_before_any_training(monkeypatch):
    df = make_features_df()
    df.loc[df.index[:5], "net_sol_flow"] = None  # rows written before migration 0005

    def no_fit(*args, **kwargs):
        raise AssertionError("nothing should train")

    monkeypatch.setattr(ablation, "fit", no_fit)
    with pytest.raises(ValueError, match="5 of .* predate the 'buy_pressure' group"):
        run_ablation(df, small_config(), variants=plan_variants(["buy_pressure"]))


def test_rows_predating_an_untested_group_dont_block_the_ablation(no_registry):
    df = make_features_df()
    df["net_sol_flow"] = None  # buy_pressure isn't in any variant

    report = run_ablation(
        df, small_config(), variants=plan_variants(["context"]), n_windows=1, seeds=[0]
    )

    assert len(report.runs) == 2  # baseline and -context both trained


def test_walk_forward_windows_score_the_most_recent_slices_without_leakage():
    end_ts = np.arange(0, 1000, dtype=np.int64)  # one sample per second
    embargo = 20

    windows = walk_forward_windows(
        end_ts, n_windows=3, eval_fraction=0.3, val_fraction=0.2, embargo=embargo
    )

    assert len(windows) == 3
    scored = np.concatenate([w.eval_idx for w in windows])
    assert len(scored) == len(set(scored.tolist()))  # disjoint slices
    assert end_ts[scored].min() > 690 and end_ts[scored].max() == 999  # the latest 30%
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert end_ts[prev.eval_idx].max() < end_ts[nxt.eval_idx].min()  # time-ordered
    for w in windows:
        first_scored = end_ts[w.eval_idx].min()
        assert end_ts[w.train_idx].max() <= w.eval_start_ts - embargo < first_scored
        assert end_ts[w.val_idx].max() <= w.eval_start_ts - embargo
        assert end_ts[w.train_idx].max() < end_ts[w.val_idx].min()  # early-stop val is later
        assert not set(w.train_idx.tolist()) & set(w.eval_idx.tolist())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_windows": 0}, "n_windows"),
        ({"eval_fraction": 1.0}, "eval_fraction"),
        ({"eval_fraction": 0.0}, "eval_fraction"),
    ],
)
def test_invalid_window_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        walk_forward_windows(np.arange(100), **kwargs)


def test_no_sample_is_an_empty_window_error():
    with pytest.raises(ValueError, match="no labeled window"):
        walk_forward_windows(np.array([], dtype=np.int64))


def test_a_window_with_nothing_to_train_on_is_an_error():
    # Every sample ends within the embargo of the eval slice: the training pool is empty.
    with pytest.raises(ValueError, match="window 0 is empty"):
        walk_forward_windows(np.arange(10), n_windows=1, eval_fraction=0.5, embargo=100)


def test_too_many_windows_for_the_data_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        walk_forward_windows(np.array([0, 0, 0, 5]), n_windows=4, eval_fraction=0.5)


# --- lift & decision on synthetic metrics ---


def test_summarize_skips_undefined_values():
    assert summarize([1.0, None, 3.0]) == {"n": 2, "mean": 2.0, "std": pytest.approx(2**0.5)}
    assert summarize([0.5]) == {"n": 1, "mean": 0.5, "std": None}  # no spread from one value
    assert summarize([None, None]) == {"n": 0, "mean": None, "std": None}


def test_paired_lifts_orient_positive_as_better():
    assert paired_lifts([0.7, 0.8], [0.6, 0.6], "pump/30s/roc_auc") == pytest.approx([0.1, 0.2])
    # Lower ECE is better: a drop in ECE is a positive lift.
    assert paired_lifts([0.05], [0.10], "rug/1m/ece") == pytest.approx([0.05])


def test_paired_lifts_are_undefined_when_either_run_is():
    assert paired_lifts([None, 0.7], [0.6, None], "pump/30s/roc_auc") == [None, None]


def test_a_consistent_lift_above_the_margin_is_significant():
    result = decide([0.05, 0.06, 0.04], margin=0.01)

    assert result["verdict"] == "better" and result["significant"] is True
    assert result["mean"] == pytest.approx(0.05)


def test_a_consistent_loss_beyond_the_margin_is_flagged_worse():
    result = decide([-0.05, -0.06, -0.04], margin=0.01)

    assert result["verdict"] == "worse" and result["significant"] is True


def test_a_lift_within_the_margin_is_not_significant():
    assert decide([0.005, 0.006, 0.004], margin=0.01)["significant"] is False


def test_a_noisy_lift_is_not_significant_even_with_a_good_mean():
    # Mean 0.03 clears the 0.01 margin, but one lucky run carries it: spread swamps the edge.
    result = decide([0.15, -0.03, -0.03], margin=0.01)

    assert result["mean"] == pytest.approx(0.03)
    assert result["verdict"] == "not significant"


def test_a_single_comparable_run_is_never_significant():
    # One run has no spread estimate: a big lift from a single (window, seed) proves nothing.
    result = decide([0.2], margin=0.01)

    assert result["verdict"] == "not significant" and result["significant"] is False
    assert result["n"] == 1 and result["std"] is None
    assert decide([None, -0.2, None], margin=0.01)["verdict"] == "not significant"


def test_no_comparable_run_is_not_applicable_not_a_crash():
    result = decide([None, None], margin=0.01)

    assert result["verdict"] == "n/a" and result["significant"] is None


def test_negative_margin_is_rejected():
    with pytest.raises(ValueError, match="margin"):
        decide([0.1], margin=-0.1)


def test_single_class_horizon_metrics_are_none_not_a_crash():
    single_class = {"roc_auc": None, "pr_auc": None, "ece": 0.02}
    report = EvaluationReport(
        model_version="v",
        generated_at="now",
        n_samples=3,
        horizons=["30s"],
        pump={"30s": single_class},
        rug={"30s": {"roc_auc": 0.7, "pr_auc": 0.4, "ece": float("nan")}},
        expected_return={"30s": {}},
        policy={"total_pnl": 0.0, "avg_pnl": None},
    )

    metrics = run_metrics(report)

    assert metrics["pump/30s/roc_auc"] is None and metrics["pump/30s/pr_auc"] is None
    assert metrics["rug/30s/roc_auc"] == 0.7
    assert metrics["rug/30s/ece"] is None  # non-finite -> undefined
    assert metrics["policy/avg_pnl"] is None and metrics["policy/total_pnl"] == 0.0


# --- end to end ---


@pytest.fixture
def no_registry(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an ablation run must never register a model")

    monkeypatch.setattr(solia_ai.registry, "register_model", forbidden)


def test_run_ablation_scores_every_variant_on_the_same_windows(no_registry, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # any checkpoint written would land here
    variants = plan_variants(["holders"])

    report = run_ablation(
        make_features_df(), small_config(), variants=variants, n_windows=2, seeds=[0, 1]
    )

    assert isinstance(report, AblationReport)
    assert len(report.runs) == 2 * 2 * 2  # variants x windows x seeds
    assert {(r.window, r.seed) for r in report.runs if r.variant == "baseline"} == {
        (r.window, r.seed) for r in report.runs if r.variant == "-holders"
    }
    assert [w["index"] for w in report.windows] == [0, 1]
    assert all(w["n_eval"] > 0 for w in report.windows)
    assert set(report.summary) == {"baseline", "-holders"}
    assert set(report.lifts) == {"-holders"}
    lifts = report.lifts["-holders"]
    assert set(lifts) == set(ablation.metric_keys(["30s", "1m"]))
    assert all(
        v["verdict"] in {"better", "worse", "not significant", "n/a"} for v in lifts.values()
    )
    assert json.loads(report.to_json())["settings"]["seeds"] == [0, 1]
    assert "-holders" in report.render() and "pump/30s/roc_auc" in report.render()
    assert not list(tmp_path.iterdir())  # no checkpoint saved


def test_each_variant_trains_a_model_masking_its_own_groups(no_registry, monkeypatch):
    masks: dict[tuple[str, ...], list[float]] = {}
    real_fit = ablation.fit

    def spy(train_ds, val_ds, cfg, **kwargs):
        fitted = real_fit(train_ds, val_ds, cfg, **kwargs)
        masks[tuple(cfg["features"]["groups"])] = fitted.model.feature_mask.tolist()
        return fitted

    monkeypatch.setattr(ablation, "fit", spy)
    run_ablation(
        make_features_df(),
        small_config(),
        variants=plan_variants(["buy_pressure"]),
        n_windows=1,
        seeds=[0],
    )

    with_group = (*BASELINE_GROUPS, "buy_pressure")
    assert set(masks) == {BASELINE_GROUPS, with_group}
    for groups, mask in masks.items():
        enabled = {c for g in groups for c in FEATURE_GROUPS[g]}
        assert mask == [float(c in enabled) for c in FEATURE_COLUMNS]


def test_run_ablation_is_reproducible_for_the_same_seeds(no_registry):
    df, cfg = make_features_df(), small_config()
    variants = plan_variants(["context"])

    first = run_ablation(df, cfg, variants=variants, n_windows=1, seeds=[3])
    second = run_ablation(df, cfg, variants=variants, n_windows=1, seeds=[3])

    assert [r.metrics for r in first.runs] == [r.metrics for r in second.runs]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seeds": []}, "seed"),
        ({"variants": [Variant("-flow", ("price",), "flow", "drop")]}, "baseline"),
        ({"margin": -1.0}, "margin"),
        ({"pnl_margin": -1.0}, "pnl_margin"),
        ({"n_windows": 0}, "n_windows"),
        ({"eval_fraction": 1.0}, "eval_fraction"),
        ({"val_fraction": 0.0}, "val_fraction"),
        (
            {
                "variants": [
                    Variant("baseline", BASELINE_GROUPS),
                    Variant("-flow", ("price",), "flow", "drop"),
                    Variant("-flow", ("holders",), "flow", "drop"),
                ]
            },
            "duplicate variant name.*-flow",
        ),
        ({"policy": ReferencePolicy(horizon="5m")}, "policy horizon"),
        ({"n_windows": 50}, "empty"),
    ],
)
def test_run_ablation_rejects_bad_setups(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_ablation(make_features_df(), small_config(), **kwargs)


def test_run_ablation_accepts_a_baseline_with_a_custom_name(no_registry):
    variants = [Variant("base", BASELINE_GROUPS), *plan_variants(["holders"])[1:]]

    report = run_ablation(
        make_features_df(), small_config(), variants=variants, n_windows=1, seeds=[0]
    )

    assert set(report.summary) == {"base", "-holders"}
    assert set(report.lifts) == {"-holders"}
    assert "vs base" in report.render()


def test_run_ablation_on_empty_data_is_an_error():
    empty = make_features_df().iloc[0:0]

    with pytest.raises(ValueError, match="no labeled window"):
        run_ablation(empty, small_config(), n_windows=1, seeds=[0])


# --- CLI ---


@pytest.fixture
def offline_cli(monkeypatch):
    df = make_features_df()
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda *a, **k: object())
    monkeypatch.setattr(solia_ai.db, "read_features", lambda *a, **k: df)
    return df


def test_main_writes_the_json_report(offline_cli, no_registry, tmp_path, capsys):
    out = tmp_path / "ablation.json"

    ablation.main(
        [
            "--groups",
            "holders",
            "--windows",
            "1",
            "--seeds",
            "0",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [v["name"] for v in payload["variants"]] == ["baseline", "-holders"]
    assert "-holders" in capsys.readouterr().out


def test_main_rejects_an_unknown_group_before_reading_data(monkeypatch):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())

    def no_db(*_a, **_k):
        raise AssertionError("must fail before touching the database")

    monkeypatch.setattr(solia_ai.db, "make_engine", no_db)

    with pytest.raises(SystemExit, match="unknown feature group"):
        ablation.main(["--groups", "bogus"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--windows", "0"], "n_windows"),
        (["--margin", "-1"], "margin"),
        (["--eval-fraction", "1.5"], "eval_fraction"),
        (["--val-fraction", "0"], "val_fraction"),
    ],
)
def test_main_rejects_bad_settings_before_reading_data(monkeypatch, argv, message):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())

    def no_db(*_a, **_k):
        raise AssertionError("must fail before touching the database")

    monkeypatch.setattr(solia_ai.db, "make_engine", no_db)
    monkeypatch.setattr(solia_ai.db, "read_features", no_db)

    with pytest.raises(SystemExit, match=message):
        ablation.main(argv)


def test_main_reports_an_empty_window_cleanly(offline_cli, capsys):
    with pytest.raises(SystemExit, match="empty"):
        ablation.main(["--windows", "50", "--seeds", "0", "--device", "cpu"])
