"""Is the model useful: entries without the model on synthetic paths, their selectivity, the
matching, the absence of look-ahead, the reading, the report and bad settings."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from solia_ai.exit_sim import FILLED, IDEAL, Execution, Paths, Windows
from solia_ai.liquidity import LiquidityFloor
from solia_ai.model_value import (
    BREAKOUT_MIN_MARKS,
    EXIT_RULE,
    MATCH_LEVELS,
    PLAIN_RULES,
    Matcher,
    bootstrap_means,
    breakout_signal,
    build_report,
    compare,
    difference_ci,
    draw_summary,
    entries_with_rows,
    main,
    plain_signals,
    reading,
    render,
    strata,
    top_windows_for_trades,
    without_model,
    write_report,
)

T0 = pd.Timestamp("2026-11-01T00:00:00Z")
FLOOR = LiquidityFloor.from_mapping({"default": 1.0})
LONG = 120  # marks: 20 minutes, more than the rule's timeout


def token(prices, mint, *, venue="pumpswap", age=200.0, volatility=0.1, pressure=0.5):
    n = len(prices)
    return pd.DataFrame(
        {
            "mint": [mint] * n,
            "ts": [T0 + pd.Timedelta(seconds=10.0 * i) for i in range(n)],
            "price": prices,
            "liquidity": [100.0] * n,
            "venue": [venue] * n,
            "volatility": [volatility] * n,
            "token_age_sec": [age + 10.0 * i for i in range(n)],
            "buy_volume_ratio": [pressure] * n,
        }
    )


def winner(mint, **kwargs):
    return token([1.0] * 10 + [3.5] * (LONG - 10), mint, **kwargs)


def flat(mint, **kwargs):
    return token([1.0] * LONG, mint, **kwargs)


def build(frames, pump):
    """Paths and one window per token at row 8 (no stride), with the given pump probability."""
    df = pd.concat(frames, ignore_index=True)
    paths = Paths.from_frame(df, floor=FLOOR, execution=IDEAL)
    rows = [(m, paths.spans[m][0] + 8) for m in paths.spans]
    n = len(rows)
    windows = Windows(
        np.array([m for m, _ in rows], dtype=object),
        np.array([r for _, r in rows], np.int64),
        np.array([pump(m) for m, _ in rows], np.float64),
        np.ones(n),
        np.ones(n, bool),
    )
    return df, paths, windows


def half_winners(n=12):
    winners = [winner(f"w{i}") for i in range(n)]
    losers = [flat(f"l{i}") for i in range(n)]
    return build(winners + losers, lambda m: 0.9 if m.startswith("w") else 0.1)


def run_compare(df, paths, windows, **kwargs):
    defaults = {"execution": IDEAL, "draws": 40, "bootstrap_samples": 200, "seed": 1}
    return compare(df, paths, windows, **{**defaults, **kwargs})


# --------------------------------------------------------------------------- signals


def test_breakout_reads_only_earlier_marks():
    prices = [1.0] * 10 + [2.0] + [1.0] * 20
    paths = Paths.from_frame(token(prices, "A"), floor=FLOOR, execution=IDEAL)
    before = breakout_signal(paths)[:11].copy()

    later = prices[:11] + [50.0] * 20
    changed = Paths.from_frame(token(later, "A"), floor=FLOOR, execution=IDEAL)
    # Changing what comes after a mark never changes its signal.
    np.testing.assert_array_equal(breakout_signal(changed)[:11], before)
    assert before[10] == pytest.approx(1.0)  # double the previous high
    assert before[9] == pytest.approx(0.0)


def test_breakout_needs_some_history_and_stays_within_a_token():
    frames = [token([5.0] * 8, "A"), token([1.0] * 8, "B")]
    paths = Paths.from_frame(pd.concat(frames, ignore_index=True), floor=FLOOR, execution=IDEAL)
    signal = breakout_signal(paths)
    start = paths.spans["B"][0]
    assert np.isnan(signal[start : start + BREAKOUT_MIN_MARKS]).all()
    assert signal[start + BREAKOUT_MIN_MARKS] == pytest.approx(0.0)  # A's 5.0 doesn't leak into B


def test_plain_signals_follow_the_paths_order_even_on_a_shuffled_frame():
    frames = [token([1.0] * 8, "A", pressure=0.2), token([1.0] * 8, "B", pressure=0.8)]
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=3)
    paths = Paths.from_frame(df, floor=FLOOR, execution=IDEAL)
    pressure = plain_signals(df, paths)["buy_pressure"]
    a, b = paths.spans["A"], paths.spans["B"]
    assert set(pressure[a[0] : a[1]]) == {0.2} and set(pressure[b[0] : b[1]]) == {0.8}


def test_plain_signals_need_their_column():
    df = token([1.0] * 8, "A").drop(columns="buy_volume_ratio")
    paths = Paths.from_frame(df, floor=FLOOR, execution=IDEAL)
    with pytest.raises(ValueError, match="buy_volume_ratio"):
        plain_signals(df, paths)


# --------------------------------------------------------------------------- entries


def test_without_model_opens_the_upside_gate():
    _, paths, windows = half_winners(2)
    gated = Windows(
        windows.mint, windows.row, windows.pump_prob, np.full(len(windows), -1.0), windows.labeled
    )
    assert not [e for e in entries_with_rows(paths, gated, EXIT_RULE, IDEAL)]
    assert len(entries_with_rows(paths, without_model(gated), EXIT_RULE, IDEAL)) == len(windows)


def test_entries_keep_their_decision_row_and_skip_a_held_mint():
    df = winner("A")
    paths = Paths.from_frame(df, floor=FLOOR, execution=IDEAL)
    windows = Windows(
        np.array(["A", "A"], dtype=object),
        np.array([8, 9], np.int64),
        np.array([0.9, 0.9]),
        np.ones(2),
        np.ones(2, bool),
    )
    out = entries_with_rows(paths, windows, EXIT_RULE, IDEAL)
    assert [row for row, _, _ in out] == [8]
    assert out[0][2].status == FILLED


def test_top_windows_match_the_models_number_of_trades():
    _, paths, windows = half_winners()
    plain = without_model(windows)
    signal = np.arange(len(paths.price), dtype=np.float64)
    rng = np.random.default_rng(0)
    chosen = top_windows_for_trades(paths, plain, signal, 5, EXIT_RULE, IDEAL, rng)
    assert len(entries_with_rows(paths, chosen, EXIT_RULE, IDEAL)) == 5
    # The highest scores: the tokens last in the paths' order.
    assert chosen.row.min() >= np.sort(plain.row)[-5]


def test_top_windows_ignore_unknown_scores():
    _, paths, windows = half_winners(2)
    signal = np.full(len(paths.price), np.nan)
    rng = np.random.default_rng(0)
    plain = without_model(windows)
    assert not len(top_windows_for_trades(paths, plain, signal, 3, EXIT_RULE, IDEAL, rng))
    signal[windows.row[0]] = 1.0
    assert len(top_windows_for_trades(paths, plain, signal, 3, EXIT_RULE, IDEAL, rng)) == 1


def test_top_windows_break_ties_at_random():
    _, paths, windows = half_winners()
    plain = without_model(windows)
    signal = np.ones(len(paths.price))
    picks = {
        tuple(
            top_windows_for_trades(
                paths, plain, signal, 3, EXIT_RULE, IDEAL, np.random.default_rng(seed)
            ).row
        )
        for seed in range(6)
    }
    assert len(picks) > 1


# --------------------------------------------------------------------------- matching


def matching_setup():
    frames = [
        flat("a1", venue="meteora", age=200.0, volatility=0.1),
        flat("a2", venue="meteora", age=200.0, volatility=0.1),
        flat("b1", venue="pumpswap", age=200.0, volatility=0.1),
        flat("c1", venue="meteora", age=5000.0, volatility=0.1),
    ]
    df, paths, windows = build(frames, lambda m: 0.5)
    return df, paths, windows, list(paths.spans)


def test_matcher_stays_in_the_stratum_and_on_another_token():
    df, paths, windows, _ = matching_setup()
    matcher = Matcher(strata(df, paths, windows), windows.mint)
    i = int(np.flatnonzero(windows.mint == "a1")[0])
    drawn = {matcher.draw(i, np.random.default_rng(s)) for s in range(10)}
    assert {windows.mint[d] for d in drawn} == {"a2"}
    assert matcher.counts[MATCH_LEVELS[0]] == 10


def test_matcher_falls_back_to_a_coarser_stratum():
    df, paths, windows, _ = matching_setup()
    matcher = Matcher(strata(df, paths, windows), windows.mint)
    i = int(np.flatnonzero(windows.mint == "b1")[0])  # alone on its venue
    assert windows.mint[matcher.draw(i, np.random.default_rng(0))] != "b1"
    assert matcher.counts["any"] == 1


def test_matcher_with_no_other_token_gives_nothing():
    df, paths, windows = build([flat("A")], lambda m: 0.5)
    matcher = Matcher(strata(df, paths, windows), windows.mint)
    assert matcher.draw(0, np.random.default_rng(0)) is None


def test_strata_separate_venue_age_and_volatility():
    frames = [
        flat("v1", venue="meteora"),
        flat("v2", venue="pumpswap"),
        flat("a1", age=5000.0),
        flat("h1", volatility=0.9),
        flat("h2", volatility=0.8),
        flat("n1", volatility=np.nan),
    ]
    df, paths, windows = build(frames, lambda m: 0.5)
    finest, by_age, venue, anything = strata(df, paths, windows)
    code = {m: i for i, m in enumerate(windows.mint)}
    assert finest[code["v1"]] != finest[code["v2"]]
    assert by_age[code["a1"]] != by_age[code["v2"]]
    assert venue[code["a1"]] == venue[code["v2"]]
    assert finest[code["h1"]] != finest[code["v2"]]
    assert finest[code["n1"]] != finest[code["v2"]]  # unknown volatility has its own bin
    assert len(set(anything.tolist())) == 1


def test_strata_need_their_columns():
    df, paths, windows = half_winners(2)
    with pytest.raises(ValueError, match="token_age_sec"):
        strata(df.drop(columns="token_age_sec"), paths, windows)


# --------------------------------------------------------------------------- statistics


def test_bootstrap_draws_tokens():
    values = np.array([1.0, 1.0, 1.0, 0.0])
    tokens = np.array(["A", "A", "A", "B"], dtype=object)
    means = bootstrap_means(values, tokens, samples=200, rng=np.random.default_rng(0))
    assert set(np.round(means, 6)) <= {0.0, 0.75, 1.0}
    assert not len(bootstrap_means(values[:1], tokens[:1], samples=5, rng=np.random.default_rng(0)))


def test_difference_ci_and_its_empty_sides():
    rng = np.random.default_rng(0)
    low, high = difference_ci(np.full(50, 1.0), np.full(50, 0.25), ci=0.95, rng=rng)
    assert low == pytest.approx(0.75) and high == pytest.approx(0.75)
    assert difference_ci(np.array([]), np.ones(3), ci=0.95, rng=rng) == [None, None]


def test_draw_summary_p_value_is_never_zero_and_is_one_sided():
    means = np.array([0.0, 0.1, 0.2, 0.3])
    assert draw_summary(means, np.ones(4), 9.0, 0.95)["p_value"] == pytest.approx(1 / 5)
    assert draw_summary(means, np.ones(4), -9.0, 0.95)["p_value"] == pytest.approx(1.0)
    assert draw_summary(np.array([]), np.array([]), 1.0, 0.95)["p_value"] is None


def fake_report(matched_low=0.1, p=0.01, plain_high=0.5, trades=300):
    arm = {"difference_ci": [-1.0, plain_high]}
    return {
        "model": {"n_trades": trades},
        "baselines": {
            "matched": {"difference_ci": [matched_low, 1.0], "p_value": p},
            **{name: dict(arm) for name in PLAIN_RULES},
        },
    }


def test_reading_says_the_model_adds_value_only_with_both_tests():
    assert reading(fake_report(), ci_level=0.95)["verdict"] == "the model adds value"
    weak = reading(fake_report(matched_low=-0.1), ci_level=0.95)
    assert weak["verdict"] == "no evidence that the model adds value"
    assert reading(fake_report(p=0.2), ci_level=0.95)["beats_matched"] is False


def test_reading_reports_a_plain_rule_that_beats_the_model():
    out = reading(fake_report(plain_high=-0.01), ci_level=0.95)
    assert out["verdict"] == "a plain rule beats the model"
    assert out["beaten_by"] == list(PLAIN_RULES)


def test_reading_flags_too_few_trades():
    assert reading(fake_report(trades=299), ci_level=0.95)["enough_trades"] is False


# --------------------------------------------------------------------------- compare


def test_a_model_that_picks_the_winners_beats_random_entries():
    df, paths, windows = half_winners()
    out = run_compare(df, paths, windows)
    assert out["model"]["n_trades"] == 12
    assert out["model"]["mean_return"] == pytest.approx(2.5)
    random_arm = out["baselines"]["random"]
    assert random_arm["difference"] > 0.5
    assert random_arm["p_value"] < 0.1
    assert out["reading"]["beats_matched"] is True
    assert out["reading"]["verdict"] == "the model adds value"


def test_a_model_scoring_at_random_shows_no_value():
    rng = np.random.default_rng(5)
    frames = [winner(f"w{i}") for i in range(12)] + [flat(f"l{i}") for i in range(12)]
    pump = {f"{k}{i}": float(rng.random()) for k in "wl" for i in range(12)}
    df, paths, windows = build(frames, pump.get)
    out = run_compare(df, paths, windows, pump_threshold=0.5)
    assert out["reading"]["verdict"] != "the model adds value"


def test_a_plain_rule_that_sees_the_winners_beats_a_bad_model():
    winners = [winner(f"w{i}", pressure=0.9) for i in range(12)]
    losers = [flat(f"l{i}", pressure=0.1) for i in range(12)]
    df, paths, windows = build(winners + losers, lambda m: 0.9 if m.startswith("l") else 0.1)
    out = run_compare(df, paths, windows)
    assert out["baselines"]["buy_pressure"]["mean_return"] == pytest.approx(2.5)
    assert "buy_pressure" in out["reading"]["beaten_by"]
    assert out["reading"]["verdict"] == "a plain rule beats the model"


def test_plain_rules_take_as_many_trades_as_the_model():
    df, paths, windows = half_winners()
    out = run_compare(df, paths, windows)
    for name in PLAIN_RULES:
        assert out["baselines"][name]["n_trades"] == out["model"]["n_trades"]


def test_baselines_ignore_the_models_upside_gate():
    df, paths, windows = half_winners()
    gated = Windows(
        windows.mint, windows.row, windows.pump_prob, np.full(len(windows), -1.0), windows.labeled
    )
    out = run_compare(df, paths, gated, pump_threshold=0.05)
    assert out["model"]["n_trades"] == 0
    assert out["reading"]["beats_matched"] is False


def test_windows_too_close_to_the_end_are_censored():
    # 95 marks end the read at 940 s: a decision at 80 s can't be held for 900 s.
    df, paths, windows = build([token([1.0] * 95, "A"), token([1.0] * 95, "B")], lambda m: 0.9)
    out = run_compare(df, paths, windows)
    assert out["n_censored"] == 2 and out["model"]["n_trades"] == 0


def test_only_a_small_part_of_the_windows_is_simulated_to_reach_the_target(monkeypatch):
    import solia_ai.model_value as module

    _, paths, windows = half_winners()
    plain = without_model(windows)
    seen = []
    real = module.simulate_rule

    def spy(paths_, windows_, rule, execution):
        seen.append(len(windows_))
        return real(paths_, windows_, rule, execution)

    monkeypatch.setattr(module, "simulate_rule", spy)
    signal = np.arange(len(paths.price), dtype=np.float64)
    top_windows_for_trades(paths, plain, signal, 2, EXIT_RULE, IDEAL, np.random.default_rng(0))
    assert max(seen) <= 4


def test_the_same_seed_gives_the_same_result():
    df, paths, windows = half_winners()
    a = run_compare(df, paths, windows, seed=3)
    b = run_compare(df, paths, windows, seed=3)
    assert a["baselines"]["matched"]["p_value"] == b["baselines"]["matched"]["p_value"]
    assert a["baselines"]["random"]["difference_ci"] == b["baselines"]["random"]["difference_ci"]


def test_bot_execution_runs_too():
    df, _, windows = half_winners(4)
    execution = Execution()
    paths = Paths.from_frame(df, floor=FLOOR, execution=execution)
    out = compare(df, paths, windows, execution=execution, draws=5, bootstrap_samples=20)
    assert out["model"]["n_trades"] == 4
    assert out["model"]["mean_return"] < 2.5  # costs and impact


# --------------------------------------------------------------------------- report


def _report():
    df, paths, windows = half_winners(4)
    result = run_compare(df, paths, windows, draws=5, bootstrap_samples=20)
    settings = {
        "pump_threshold": 0.5,
        "exit_rule": EXIT_RULE.label,
        "ci": 0.95,
    }
    return build_report("model-x", result, settings)


def test_report_is_written_as_json_and_table(tmp_path):
    report = _report()
    json_path, table_path = write_report(report, tmp_path / "sub" / "value.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["model_version"] == "model-x"
    table = table_path.read_text(encoding="utf-8")
    for arm in ("model", "breakout", "buy_pressure", "volatility", "random", "matched"):
        assert arm in table
    assert "this period:" in table and "matched at:" in table
    assert "too few to read" in table  # 4 trades


def test_report_renders_with_no_model_trade():
    df, paths, windows = half_winners(2)
    gated = Windows(
        windows.mint, windows.row, windows.pump_prob, np.full(len(windows), -1.0), windows.labeled
    )
    result = run_compare(df, paths, gated, draws=2, bootstrap_samples=5)
    settings = {"pump_threshold": 0.5, "exit_rule": EXIT_RULE.label, "ci": 0.95}
    assert "no evidence" in render(build_report("m", result, settings))


# --------------------------------------------------------------------------- bad settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pump_threshold": 0.0},
        {"pump_threshold": 1.5},
        {"draws": 0},
        {"bootstrap_samples": 0},
        {"ci": 1.0},
    ],
)
def test_compare_rejects_bad_settings(kwargs):
    df, paths, windows = half_winners(1)
    with pytest.raises(ValueError):
        compare(df, paths, windows, **kwargs)


@pytest.mark.parametrize(
    "args",
    [
        ["--champion", "--pump-threshold", "0"],
        ["--champion", "--draws", "0"],
        ["--champion", "--ci", "2"],
        ["--champion", "--window-stride-seconds", "-1"],
        ["--champion", "--until", "yesterday"],
        [],
    ],
)
def test_cli_rejects_bad_settings_before_reading_anything(args):
    with pytest.raises(SystemExit):
        main(args)
