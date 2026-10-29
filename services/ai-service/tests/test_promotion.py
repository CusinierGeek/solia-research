"""Promotion gate (solia_ai.promotion) on synthetic metrics: decisions, guard rails, rollback."""

import copy
import json
import math
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from solia_ai import ablation
from solia_ai.config import ConfigError, load_config
from solia_ai.features import FEATURE_COLUMNS
from solia_ai.lift import LOWER_IS_BETTER, lift
from solia_ai.promotion import (
    PNL_METRIC,
    PromotionSettings,
    bootstrap_ci,
    check_rollback,
    decide_promotion,
)
from solia_ai.registry import TRAINING_DECISIONS, ModelRecord

CONFIG = load_config()
HORIZONS = CONFIG["horizons"]


def settings(**promotion) -> PromotionSettings:
    config = copy.deepcopy(CONFIG)
    config["promotion"] = {**config.get("promotion", {}), **promotion}
    return PromotionSettings.from_config(config)


SETTINGS = settings(windows=3, seeds=1)
PRIMARY = SETTINGS.primary_key


def metrics(pr_auc=0.5, *, pnl=0.0, **overrides):
    """One run's ablation-shaped metrics: every predictive metric at a flat value."""
    out = {}
    for key in ablation.metric_keys(HORIZONS):
        metric = key.rsplit("/", 1)[-1]
        out[key] = {"roc_auc": 0.6, "pr_auc": 0.3, "ece": 0.05}.get(metric, 0.0)
    out[PRIMARY] = pr_auc
    out[PNL_METRIC] = pnl
    out.update(overrides)
    return out


def runs(*per_run, seed=0):
    return {(w, seed): m for w, m in enumerate(per_run)}


CHAMPION = runs(metrics(0.40), metrics(0.42), metrics(0.38))


def test_lift_is_oriented_so_positive_is_better():
    assert "ece" in LOWER_IS_BETTER
    assert lift(0.6, 0.5, "pump/5m/pr_auc") == pytest.approx(0.1)
    assert lift(0.04, 0.05, "rug/1m/ece") == pytest.approx(0.01)  # lower ECE is a gain
    assert PNL_METRIC in ablation.metric_keys(HORIZONS)


def test_guard_keys_are_predictive_only():
    keys = SETTINGS.guard_keys()
    assert PRIMARY not in keys and f"rug/{HORIZONS[0]}/ece" in keys
    assert not any(k.startswith(("policy/", "raw/")) for k in keys)


def test_settings_default_from_packaged_config():
    s = PromotionSettings.from_config(CONFIG)
    assert s.metric == "pr_auc" and s.head == "pump" and s.horizon in HORIZONS
    assert s.expected_runs == s.windows * s.seeds


def test_settings_default_horizon_is_the_middle_one():
    config = copy.deepcopy(CONFIG)
    config.pop("promotion")
    assert PromotionSettings.from_config(config).horizon == HORIZONS[len(HORIZONS) // 2]


def test_explicit_null_horizon_is_the_middle_one():
    assert settings(horizon=None).horizon == HORIZONS[len(HORIZONS) // 2]


def test_clearly_better_is_promoted():
    candidate = runs(metrics(0.50), metrics(0.52), metrics(0.49))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert decision.promote and not decision.initial
    assert decision.decision == "promote" and decision.event_decision == "promoted"
    assert decision.summary["primary_lift"]["win_share"] == 1.0
    assert decision.summary["primary_lift"]["mean"] == pytest.approx(0.1033, abs=1e-3)


def test_lift_within_margin_is_rejected():
    candidate = runs(metrics(0.405), metrics(0.425), metrics(0.385))  # +0.005 < margin 0.01

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote and decision.event_decision == "rejected"
    assert any("does not clear the margin" in r for r in decision.reasons)


def test_better_on_one_window_only_is_rejected():
    # Huge lift on window 0 drags the mean over the margin, but 1/3 runs < min_win_share.
    candidate = runs(metrics(0.90), metrics(0.42), metrics(0.38))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote
    assert decision.summary["primary_lift"]["mean"] > SETTINGS.margin
    assert any("1/3 runs" in r for r in decision.reasons)


def grid(lifts, *, windows=3, seeds=3):
    """windows x seeds paired runs, the candidate beating a 0.4 champion by ``lifts`` in order."""
    keys = [(w, seed) for w in range(windows) for seed in range(seeds)]
    champion = {k: metrics(0.4) for k in keys}
    candidate = {k: metrics(0.4 + d) for k, d in zip(keys, lifts, strict=True)}
    return candidate, champion


def test_two_thirds_of_runs_is_enough_with_the_default_share():
    s = settings(windows=3, seeds=3)  # packaged min_win_share: two thirds
    candidate, champion = grid([0.1] * 6 + [0.0] * 3)

    decision = decide_promotion(candidate, champion, s)

    assert decision.promote, decision.reasons
    assert decision.summary["primary_lift"]["wins"] == 6
    assert decide_promotion(
        candidate, champion, settings(windows=3, seeds=3, min_win_share=2 / 3)
    ).promote


def test_under_two_thirds_of_runs_is_rejected():
    candidate, champion = grid([0.2] * 5 + [0.0] * 4)

    decision = decide_promotion(candidate, champion, settings(windows=3, seeds=3))

    assert not decision.promote and any("5/9 runs" in r for r in decision.reasons)


def test_better_pnl_but_worse_pr_auc_is_rejected():
    candidate = runs(metrics(0.30, pnl=50.0), metrics(0.32, pnl=60.0), metrics(0.28, pnl=55.0))
    champion = runs(metrics(0.40), metrics(0.42), metrics(0.38))

    decision = decide_promotion(candidate, champion, SETTINGS)

    assert not decision.promote
    assert decision.summary["pnl_lift"]["verdict"] == "better"


@pytest.mark.parametrize("bad", [math.nan, math.inf, None])
def test_missing_or_nan_primary_is_rejected(bad):
    candidate = runs(metrics(0.50), metrics(bad), metrics(0.49))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote


def test_nan_secondary_metric_is_rejected():
    candidate = runs(metrics(0.5), metrics(0.5, **{"rug/1m/ece": math.nan}), metrics(0.5))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote and "rug/1m/ece" in decision.reasons[0]


def test_nan_candidate_is_rejected_even_without_champion():
    decision = decide_promotion(runs(metrics(math.nan), metrics(0.5), metrics(0.5)), None, SETTINGS)

    assert not decision.promote and not decision.initial


def test_no_champion_promotes_first_valid_candidate():
    decision = decide_promotion(runs(metrics(0.2), metrics(0.2), metrics(0.2)), None, SETTINGS)

    assert decision.promote and decision.initial


def test_too_few_runs_is_rejected():
    decision = decide_promotion(runs(metrics(0.9)), None, settings(windows=3, seeds=3))

    assert not decision.promote and "windows x seeds" in decision.reasons[0]


def test_seeds_on_one_window_are_not_a_multi_window_comparison():
    one_window = {(0, seed): metrics(0.9) for seed in range(3)}  # 3 runs, 1 window

    decision = decide_promotion(one_window, None, SETTINGS)  # windows=3, seeds=1

    assert not decision.promote and "1 window(s)" in decision.reasons[0]


def test_more_runs_than_configured_is_rejected():
    extra = runs(metrics(0.6), metrics(0.6), metrics(0.6), metrics(0.6))

    decision = decide_promotion(extra, None, SETTINGS)

    assert not decision.promote and "windows x seeds" in decision.reasons[0]


def test_candidate_missing_a_metric_the_champion_has_is_rejected():
    candidate = runs(*(metrics(0.6, **{"rug/1h/roc_auc": None}) for _ in range(3)))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote and "rug/1h/roc_auc" in decision.reasons[0]


def test_candidate_without_policy_trades_is_not_rejected_for_it():
    # ReferencePolicy leaves avg_pnl undefined when it makes no trade: P&L can only veto via its CI.
    candidate = runs(*(metrics(0.6, **{"policy/avg_pnl": None}) for _ in range(3)))
    champion = runs(*(metrics(0.4, **{"policy/avg_pnl": 0.03}) for _ in range(3)))

    decision = decide_promotion(candidate, champion, settings(windows=3, seeds=1, pnl_veto=False))

    assert decision.promote, decision.reasons


def test_uncalibrated_twins_do_not_decide():
    raw = {ablation.raw_key(PRIMARY): 0.9}
    champion = runs(*(metrics(0.4, **raw) for _ in range(3)))
    candidate = runs(metrics(0.5), metrics(0.52), metrics(0.49))  # no raw/ keys at all

    decision = decide_promotion(candidate, champion, SETTINGS)

    assert decision.promote, decision.reasons


def test_non_finite_champion_metric_rejects_the_comparison():
    champion = runs(*(metrics(0.4, **{"rug/1m/ece": math.nan}) for _ in range(3)))
    candidate = runs(*(metrics(0.6, **{"rug/1m/ece": 0.5}) for _ in range(3)))

    decision = decide_promotion(candidate, champion, SETTINGS)

    assert not decision.promote and "champion rug/1m/ece is not finite" in decision.reasons[0]


def test_undefined_secondary_on_both_sides_is_skipped():
    undefined = {"rug/1h/roc_auc": None}
    candidate = runs(*(metrics(0.6, **undefined) for _ in range(3)))
    champion = runs(*(metrics(0.4, **undefined) for _ in range(3)))

    decision = decide_promotion(candidate, champion, SETTINGS)

    assert decision.promote
    assert "rug/1h/roc_auc" not in decision.summary["guard_lifts"]


def test_ece_regression_beyond_tolerance_is_rejected():
    worse = {"pump/1m/ece": 0.05 + 0.03}  # tolerance 0.01; ECE is lower-is-better
    candidate = runs(*(metrics(0.6, **worse) for _ in range(3)))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote
    assert any("pump/1m/ece regresses" in r for r in decision.reasons)


def test_ece_improvement_is_not_a_regression():
    candidate = runs(*(metrics(0.6, **{"pump/1m/ece": 0.0}) for _ in range(3)))

    assert decide_promotion(candidate, CHAMPION, SETTINGS).promote


def test_secondary_metric_regression_is_rejected():
    candidate = runs(*(metrics(0.6, **{"rug/5m/roc_auc": 0.5}) for _ in range(3)))  # -0.1

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote
    assert any("rug/5m/roc_auc regresses" in r for r in decision.reasons)


def test_significantly_worse_pnl_vetoes_only_when_enabled():
    candidate = runs(metrics(0.6, pnl=-5.0), metrics(0.6, pnl=-6.0), metrics(0.6, pnl=-5.5))

    assert not decide_promotion(candidate, CHAMPION, SETTINGS).promote
    assert decide_promotion(
        candidate, CHAMPION, settings(windows=3, seeds=1, pnl_veto=False)
    ).promote


def test_mismatched_runs_raise():
    with pytest.raises(ValueError, match="same"):
        decide_promotion(runs(metrics(0.6), metrics(0.6)), CHAMPION, SETTINGS)
    with pytest.raises(ValueError, match="no candidate"):
        decide_promotion({}, None, SETTINGS)


def test_metrics_are_json_ready_for_training_events():
    candidate = runs(metrics(0.50), metrics(0.52), metrics(0.49))
    windows = [{"index": 0, "eval_start": "2026-10-01T00:00:00+00:00"}]

    payload = decide_promotion(candidate, CHAMPION, SETTINGS, windows=windows).metrics()

    json.dumps(payload, allow_nan=False)  # registry.to_json rejects NaN
    assert payload["decision"] == "promote" and payload["windows"] == windows
    assert payload["primary"] == PRIMARY and len(payload["primary_lift"]["runs"]) == 3
    assert {"promoted", "rejected"} <= set(TRAINING_DECISIONS)


def test_multiple_seeds_are_paired_per_window_and_seed():
    s = settings(windows=2, seeds=2)
    champion = {(w, seed): metrics(0.4) for w in range(2) for seed in range(2)}
    candidate = {(w, seed): metrics(0.5) for w in range(2) for seed in range(2)}

    decision = decide_promotion(candidate, champion, s)

    assert decision.promote and decision.summary["n_runs"] == 4


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [1.0, 2.0, 3.0, 4.0]
    a = bootstrap_ci(values, samples=500, confidence=0.9, seed=7)

    assert a == bootstrap_ci(values, samples=500, confidence=0.9, seed=7)
    assert a["low"] <= a["mean"] == 2.5 <= a["high"] and a["verdict"] == "better"


def test_bootstrap_ci_needs_two_values():
    assert bootstrap_ci([1.0], samples=10, confidence=0.9, seed=0)["low"] is None
    assert bootstrap_ci([], samples=10, confidence=0.9, seed=0)["verdict"] == "inconclusive"


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "promotion, message",
    [
        ({"metric": "ece"}, "promotion.metric"),
        ({"head": "moon"}, "promotion.head"),
        ({"horizon": "2h"}, "promotion.horizon"),
        ({"margin": -0.1}, "promotion.margin"),
        ({"windows": 0}, "promotion.windows"),
        ({"seeds": 1.5}, "promotion.seeds"),
        ({"min_win_share": 0}, "promotion.min_win_share"),
        ({"min_win_share": 1.2}, "promotion.min_win_share"),
        ({"ece_tolerance": "x"}, "promotion.ece_tolerance"),
        ({"confidence": 1.0}, "promotion.confidence"),
        ({"bootstrap_samples": 0}, "promotion.bootstrap_samples"),
        ({"pnl_veto": "yes"}, "promotion.pnl_veto"),
        ({"marign": 0.1}, "unknown key"),
    ],
)
def test_bad_promotion_config_fails_fast(promotion, message):
    with pytest.raises(ConfigError, match=message):
        settings(**promotion)


def test_promotion_must_be_a_mapping():
    config = {**CONFIG, "promotion": [1]}
    with pytest.raises(ConfigError, match="'promotion' must be a mapping"):
        PromotionSettings.from_config(config)


# --------------------------------------------------------------------------- rollback


def record(id_, version, champion=False):
    return ModelRecord(id_, version, f"/models/{version}.pt", champion, 1000, datetime.now(UTC))


CURRENT = record(2, "v2", champion=True)
PREVIOUS = record(1, "v1")
CHECKPOINT = {"version": "v1", "horizons": HORIZONS, "feature_columns": FEATURE_COLUMNS[:19]}


def test_rollback_possible():
    check = check_rollback(CURRENT, PREVIOUS, checkpoint=CHECKPOINT, horizons=HORIZONS)

    assert check.possible and check.target == PREVIOUS and not check.reasons


@pytest.mark.parametrize(
    "current, previous, checkpoint, reason",
    [
        (CURRENT, None, CHECKPOINT, "no previous champion"),
        (None, PREVIOUS, CHECKPOINT, "no current champion"),
        (CURRENT, CURRENT, {**CHECKPOINT, "version": "v2"}, "already the champion"),
        (CURRENT, PREVIOUS, None, "not found"),
        (CURRENT, PREVIOUS, {**CHECKPOINT, "version": "v9"}, "is v9, not v1"),
        (
            CURRENT,
            PREVIOUS,
            {**CHECKPOINT, "feature_columns": list(reversed(FEATURE_COLUMNS))},
            "feature spec",
        ),
        (CURRENT, PREVIOUS, {**CHECKPOINT, "feature_columns": ["gone"]}, "feature spec"),
        (CURRENT, PREVIOUS, {**CHECKPOINT, "horizons": ["30s"]}, "serves horizons"),
        (CURRENT, PREVIOUS, {"horizons": HORIZONS, "feature_columns": FEATURE_COLUMNS}, "version"),
        (CURRENT, PREVIOUS, {"version": "v1", "feature_columns": FEATURE_COLUMNS}, "horizons"),
        (CURRENT, PREVIOUS, {"version": "v1", "horizons": HORIZONS}, "feature_columns"),
        (CURRENT, PREVIOUS, {}, "does not record"),
    ],
)
def test_rollback_refused(current, previous, checkpoint, reason):
    check = check_rollback(current, previous, checkpoint=checkpoint, horizons=HORIZONS)

    assert not check.possible and check.target is None
    assert any(reason in r for r in check.reasons)


@pytest.mark.parametrize("checkpoint", [CHECKPOINT, None])
def test_rollback_to_a_pruned_model_is_refused(checkpoint):
    pruned = replace(PREVIOUS, pruned_at=datetime(2026, 10, 1, tzinfo=UTC))
    check = check_rollback(CURRENT, pruned, checkpoint=checkpoint, horizons=HORIZONS)

    assert not check.possible and check.target is None
    assert check.reasons == (
        "checkpoint of v1 was pruned by the retention policy at 2026-10-01T00:00:00+00:00",
    )
