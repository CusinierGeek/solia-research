"""Entry selection (solia_ai.entry_selection), its config and its promotion-gate guard (#180)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from solia_ai import ablation
from solia_ai.config import ConfigError, load_config
from solia_ai.entry_selection import (
    ENTRY_METRICS,
    GATED,
    PUMP_ONLY,
    RUG_THRESHOLD_KEY,
    entry_key,
    entry_keys,
    flatten,
    select_entries,
    summarize_selection,
)
from solia_ai.promotion import PNL_METRIC, PromotionSettings, decide_promotion

CONFIG = load_config()
HORIZONS = CONFIG["horizons"]
H = HORIZONS[-1]


# --------------------------------------------------------------------------- select_entries


def test_selection_takes_the_top_fraction_of_the_rug_gated_pool():
    pump = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    rug = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    pumped = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    returns = np.linspace(-1, 1, 10)
    mints = np.array(["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"], dtype=object)

    gated = select_entries(pump, rug, pumped, returns, mints, top_fraction=0.25, rug_threshold=0.2)
    every = select_entries(pump, rug, pumped, returns, mints, top_fraction=0.25, rug_threshold=None)

    # 9 windows pass the gate: ceil(0.25 x 9) = 3 -> windows 1, 2, 3 (window 0 is gated out).
    assert gated["pool_windows"] == 9 and gated["pool_tokens"] == 3
    assert gated["windows"] == 3 and gated["tokens"] == 2
    assert gated["top_token_share"] == pytest.approx(2 / 3)
    assert gated["precision"] == pytest.approx(1 / 3)
    assert gated["base_rate"] == pytest.approx(0.2)
    assert gated["median_return"] == pytest.approx(returns[2])
    # Without the gate: ceil(2.5) = 3 -> windows 0, 1, 2.
    assert (
        every["windows"] == 3
        and every["tokens"] == 2
        and every["precision"] == pytest.approx(2 / 3)
    )


def test_selection_rounding_does_not_overshoot():
    n = 300
    out = select_entries(
        np.arange(n) / n,
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.array([str(i) for i in range(n)], dtype=object),
        top_fraction=0.01,
        rug_threshold=0.2,
    )
    assert out["windows"] == 3


def test_a_rug_gate_that_empties_the_pool_selects_nothing():
    out = select_entries(
        np.ones(4), np.ones(4), np.ones(4), np.zeros(4), np.array(list("abcd"), dtype=object),
        top_fraction=0.5, rug_threshold=0.2,
    )  # fmt: skip
    assert out["windows"] == 0 and out["pool_tokens"] == 0
    assert out["precision"] is None and out["top_token_share"] is None
    assert out["base_rate"] == 1.0


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_selection_rejects_an_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="top_fraction"):
        select_entries(
            np.ones(1), np.ones(1), np.ones(1), np.ones(1), np.array(["a"], dtype=object),
            top_fraction=fraction, rug_threshold=None,
        )  # fmt: skip


def test_flatten_names_every_entry_key():
    sel = {m: 1.0 for m in ENTRY_METRICS}
    out = flatten({h: {"gated": sel, "pump_only": sel} for h in HORIZONS})
    assert sorted(out) == sorted(entry_keys(HORIZONS))


# --------------------------------------------------------------------------- gate


def settings(entry=None, **promotion) -> PromotionSettings:
    config = copy.deepcopy(CONFIG)
    config["promotion"] = {**config["promotion"], "windows": 3, "seeds": 1, **promotion}
    if entry is not None:
        config["promotion"]["entry_selection"] = entry
    return PromotionSettings.from_config(config)


SETTINGS = settings()


def selection(prefix=GATED, horizon=H, **values):
    """Entry metrics of one horizon (a broad, healthy selection unless overridden)."""
    base = {
        "windows": 400.0,
        "tokens": 120.0,
        "top_token_share": 0.05,
        "precision": 0.25,
        "base_rate": 0.04,
        "median_return": 0.1,
        "pool_tokens": 3000.0,
    }
    return {entry_key(prefix, horizon, m): v for m, v in {**base, **values}.items()}


def run(pr_auc=0.5, *, gated=None, pump_only=None, **overrides):
    out = {k: {"roc_auc": 0.6, "pr_auc": 0.3, "ece": 0.05}.get(k.rsplit("/", 1)[-1], 0.0)
           for k in ablation.metric_keys(HORIZONS)}  # fmt: skip
    out[SETTINGS.primary_key] = pr_auc
    out[PNL_METRIC] = 0.0
    out[RUG_THRESHOLD_KEY] = 0.2
    for h in HORIZONS:
        out.update(selection(GATED, h))
        out.update(selection(PUMP_ONLY, h))
    out.update(selection(GATED, H, **(gated or {})))
    out.update(selection(PUMP_ONLY, H, **(pump_only or {})))
    out.update(overrides)
    return out


def runs(*per_run):
    return {(w, 0): m for w, m in enumerate(per_run)}


CHAMPION = runs(run(0.40), run(0.40), run(0.40))
HEALTHY = runs(run(0.50), run(0.50), run(0.50))


def test_default_config_enables_the_guard():
    entry = SETTINGS.entry_selection
    assert entry is not None and entry.enabled
    assert (entry.top_fraction, entry.min_top_lift, entry.min_tokens, entry.max_token_share) == (
        0.01, 1.0, 20, 0.5
    )  # fmt: skip
    assert SETTINGS.entry_top_fraction == 0.01


def test_a_healthy_broad_selection_passes():
    decision = decide_promotion(HEALTHY, CHAMPION, SETTINGS)

    assert decision.promote, decision.reasons
    entry = decision.metrics()["entry_selection"]
    assert entry["rug_threshold"] == 0.2
    assert entry["horizons"][H]["candidate"]["status"] == "pass"
    assert entry["horizons"][H]["champion"]["precision"] == pytest.approx(0.25)
    assert entry["horizons"][H]["pump_only"]["defined"]


def test_the_179_shape_is_rejected_for_its_breadth():
    # 417 windows from 11 tokens, 85 % one token, precision 0 — while the pump-only top-k is fine.
    shape = {"windows": 417.0, "tokens": 11.0, "top_token_share": 0.85, "precision": 0.0,
             "base_rate": 0.038, "pool_tokens": 5000.0}  # fmt: skip
    candidate = runs(*(run(0.50, gated=shape, pump_only={"precision": 0.268}) for _ in range(3)))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote
    reasons = [r for r in decision.reasons if r.startswith(f"pump/{H} entry")]
    assert f"pump/{H} entry top-1% (rug<=0.2): precision 0.000 < base rate 0.038" in reasons
    assert (
        f"pump/{H} entry top-1% (rug<=0.2): 417 windows from 11 tokens (top token 85%) "
        "< min_tokens 20"
    ) in reasons
    assert any("> max_token_share 50%" in r for r in reasons)
    horizon = decision.metrics()["entry_selection"]["horizons"][H]
    assert horizon["candidate"]["status"] == "fail"
    assert horizon["pump_only"]["precision"] == pytest.approx(0.268)  # reported, not blocking


def test_high_precision_on_one_token_is_a_lucky_false_champion():
    lucky = {"tokens": 30.0, "top_token_share": 0.9, "precision": 0.8}
    candidate = runs(*(run(0.50, gated=lucky) for _ in range(3)))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert not decision.promote
    assert decision.reasons == (
        f"pump/{H} entry top-1% (rug<=0.2): 400 windows from 30 tokens (top token 90%) "
        "> max_token_share 50%",
    )


def test_precision_below_a_custom_lift_names_the_multiplier():
    candidate = runs(*(run(0.50, gated={"precision": 0.06}) for _ in range(3)))

    decision = decide_promotion(candidate, CHAMPION, settings({"min_top_lift": 2.0}))

    assert decision.reasons == (
        f"pump/{H} entry top-1% (rug<=0.2): precision 0.060 < 2 x base rate 0.040",
    )


def test_the_guard_averages_over_runs():
    # One bad run among three healthy-enough ones: the mean share 0.4 stays under 0.5.
    candidate = runs(
        run(0.50, gated={"top_token_share": 0.9}),
        run(0.50, gated={"top_token_share": 0.15}),
        run(0.50, gated={"top_token_share": 0.15}),
    )
    assert decide_promotion(candidate, CHAMPION, SETTINGS).promote


@pytest.mark.parametrize("empty", [{"windows": 0.0, "pool_tokens": 0.0}, {"pool_tokens": 5.0}])
def test_a_rug_gate_that_empties_the_selection_is_undefined_not_passed(empty):
    bad = {"tokens": 1.0, "top_token_share": 1.0, "precision": 0.0, **empty}
    candidate = runs(*(run(0.50, gated=bad) for _ in range(3)))

    decision = decide_promotion(candidate, CHAMPION, SETTINGS)

    assert decision.promote, decision.reasons  # undefined never rejects ...
    sel = decision.metrics()["entry_selection"]["horizons"][H]["candidate"]
    assert sel["status"] == "undefined" and not sel["defined"]  # ... and never reads as a pass


def test_runs_without_entry_metrics_are_undefined():
    legacy = {(w, 0): {k: v for k, v in m.items() if not k.startswith("entry")}
              for (w, _), m in HEALTHY.items()}  # fmt: skip
    decision = decide_promotion(legacy, CHAMPION, SETTINGS)

    assert decision.promote
    statuses = {
        h["candidate"]["status"] for h in decision.summary["entry_selection"]["horizons"].values()
    }
    assert statuses == {"undefined"}


def test_the_guard_also_checks_the_first_candidate():
    shape = {"tokens": 3.0, "top_token_share": 0.9}
    decision = decide_promotion({(0, 0): run(gated=shape), (1, 0): run(gated=shape),
                                 (2, 0): run(gated=shape)}, None, SETTINGS)  # fmt: skip

    assert not decision.promote and not decision.initial
    assert any("< min_tokens 20" in r for r in decision.reasons)


def test_min_top_lift_zero_reports_without_blocking():
    shape = {"tokens": 3.0, "top_token_share": 0.9, "precision": 0.0}
    candidate = runs(*(run(0.50, gated=shape) for _ in range(3)))

    decision = decide_promotion(candidate, CHAMPION, settings({"min_top_lift": 0}))

    assert decision.promote, decision.reasons
    sel = decision.metrics()["entry_selection"]["horizons"][H]["candidate"]
    assert sel["status"] == "report-only" and sel["tokens"] == 3.0


def test_omitting_the_block_disables_the_guard():
    config = copy.deepcopy(CONFIG)
    config["promotion"] = {**config["promotion"], "windows": 3, "seeds": 1}
    config["promotion"].pop("entry_selection")
    gate = PromotionSettings.from_config(config)
    shape = {"tokens": 3.0, "top_token_share": 0.9, "precision": 0.0}

    decision = decide_promotion(runs(*(run(0.5, gated=shape) for _ in range(3))), CHAMPION, gate)

    assert gate.entry_selection is None and gate.entry_top_fraction == 0.01
    assert decision.promote and "entry_selection" not in decision.metrics()


def test_summarize_skips_unmeasurable_runs():
    good, small = selection(tokens=40.0), selection(pool_tokens=3.0, tokens=1.0)
    out = summarize_selection([good, small], GATED, H, min_tokens=20)
    assert out["defined"] and out["n_runs"] == 1 and out["tokens"] == 40.0


# --------------------------------------------------------------------------- config


def write(tmp_path: Path, entry) -> Path:
    config = copy.deepcopy(CONFIG)
    config["promotion"]["entry_selection"] = entry
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ([0.01], "'promotion.entry_selection' must be a mapping"),
        ({"top_k": 5}, r"'promotion.entry_selection' has unknown key\(s\): top_k"),
        ({"top_fraction": 0}, r"top_fraction' must be a number in \(0, 0.5\]"),
        ({"top_fraction": 0.6}, r"top_fraction' must be a number in \(0, 0.5\]"),
        ({"top_fraction": "1%"}, "top_fraction' must be a number"),
        ({"min_top_lift": -1}, "min_top_lift' must be a number >= 0"),
        ({"min_top_lift": True}, "min_top_lift' must be a number"),
        ({"max_token_share": 0}, r"max_token_share' must be a number in \(0, 1\]"),
        ({"max_token_share": 1.5}, r"max_token_share' must be a number in \(0, 1\]"),
        ({"min_tokens": 0}, "min_tokens' must be an integer >= 1"),
        ({"min_tokens": 2.5}, "min_tokens' must be an integer >= 1"),
    ],
)
def test_invalid_entry_selection_settings_are_rejected(tmp_path, entry, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, entry))


def test_null_or_partial_block_is_valid(tmp_path):
    assert load_config(write(tmp_path, None))["promotion"]["entry_selection"] is None
    config = load_config(write(tmp_path, {"min_tokens": 5}))
    gate = PromotionSettings.from_config(config).entry_selection
    assert gate.min_tokens == 5 and gate.top_fraction == 0.01
