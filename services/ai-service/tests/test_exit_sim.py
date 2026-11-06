"""Exit-rule simulation: exits on synthetic price paths, the bot's formulas and defaults, one
position per mint, the sweep-to-entries steps, the report and bad settings."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from solia_ai import exit_sim
from solia_ai.dataset import build_samples
from solia_ai.exit_sim import (
    ABANDONED,
    BOT_RULE,
    DECLINED,
    FAILED,
    FILLED,
    IDEAL,
    Execution,
    ExitRule,
    Paths,
    Windows,
    bootstrap_by_token,
    build_report,
    default_grid,
    first_per_mint,
    main,
    one_per_mint,
    render,
    score_windows,
    simulate_position,
    simulate_rule,
    simulate_rules,
    stop_loss_level,
    take_profit_level,
    uncensored,
    window_outcomes,
    write_report,
)
from solia_ai.features import FEATURE_COLUMNS
from solia_ai.liquidity import LiquidityFloor

T0 = pd.Timestamp("2026-11-01T00:00:00Z")
FLOOR = LiquidityFloor.from_mapping({"default": 1.0, "pumpfun": 0.0})
BOT_OPTIONS = Path(__file__).parents[3] / "services/backend/Solia.Bot/Configuration/BotOptions.cs"


def frame(
    prices, *, mint="A", step=10.0, liquidity=100.0, venue="pumpswap", volatility=0.1, t0=0.0
):
    n = len(prices)
    return pd.DataFrame(
        {
            "mint": [mint] * n,
            "ts": [T0 + pd.Timedelta(seconds=t0 + step * i) for i in range(n)],
            "price": prices,
            "liquidity": liquidity if np.ndim(liquidity) else [liquidity] * n,
            "venue": [venue] * n,
            "volatility": [volatility] * n,
        }
    )


def paths_of(*frames, execution=IDEAL):
    return Paths.from_frame(pd.concat(frames, ignore_index=True), floor=FLOOR, execution=execution)


def run(prices, rule, *, execution=IDEAL, expected_return=1.0, row=0, **kwargs):
    paths = paths_of(frame(prices, **kwargs), execution=execution)
    return simulate_position(paths, row, len(paths.ts), expected_return, rule, execution)


FIXED = ExitRule(take_profit=0.5, stop_loss=0.2, timeout_s=300)


# --------------------------------------------------------------------------- exit reasons


def test_take_profit_fills_at_the_tick_that_triggers_it():
    out = run([1.0, 1.1, 1.8, 1.0], FIXED)
    assert out.status == FILLED and out.reason == "tp"
    assert out.net_return == pytest.approx(0.8)  # the tick, not the level
    assert out.hold_s == pytest.approx(20.0)


def test_stop_loss_realizes_the_gap():
    out = run([1.0, 0.95, 0.4, 2.0], FIXED)
    assert out.reason == "sl" and out.net_return == pytest.approx(-0.6)


def test_timeout_closes_at_the_last_mark_before_the_deadline():
    prices = [1.0] * 31 + [1.2] + [1.0] * 10  # ts 310 is at or after the 300 s deadline
    prices[30] = 1.1  # ts 300 == deadline: swept before this mark can act
    out = run(prices, FIXED)
    assert out.reason == "timeout"
    assert out.net_return == pytest.approx(0.0)  # the mark before the deadline (ts 290)
    assert out.hold_s == pytest.approx(300.0)


def test_a_mark_at_the_deadline_never_triggers_an_exit():
    prices = [1.0] * 30 + [5.0, 5.0]  # ts 300 would be a take-profit, but the deadline comes first
    assert run(prices, FIXED).reason == "timeout"


def test_first_level_reached_wins_whatever_comes_next():
    assert run([1.0, 1.6, 0.1], FIXED).reason == "tp"
    assert run([1.0, 0.7, 9.0], FIXED).reason == "sl"


def test_stop_loss_is_checked_before_take_profit_within_one_step():
    # Levels can't cross in the bot; a mark satisfying both must still resolve to the stop.
    rule = ExitRule(take_profit=0.5, stop_loss=0.2, timeout_s=300)
    paths = paths_of(frame([1.0, 1.0]))
    object.__setattr__(paths, "price", np.array([1.0, 0.8]))
    out = simulate_position(paths, 0, 2, 1.0, rule, IDEAL)
    assert out.reason == "sl"

    class Crossed(ExitRule):
        def levels(self, expected_return, volatility):
            return -0.5, 0.2  # TP below the stop: 0.5 and 0.8

    out = simulate_position(paths, 0, 2, 1.0, Crossed(), IDEAL)
    assert out.reason == "sl"


def test_no_take_profit_and_no_stop_only_time_out():
    rule = ExitRule(take_profit=None, stop_loss=None, timeout_s=30)
    out = run([1.0, 100.0, 3.0, 9.0, 9.0], rule)
    assert out.reason == "timeout" and out.net_return == pytest.approx(2.0)


def test_a_token_going_silent_exits_at_its_last_price_after_the_timeout():
    out = run([1.0, 1.1, 1.2], FIXED)
    assert out.reason == "timeout"
    assert out.net_return == pytest.approx(0.2)
    assert out.hold_s == pytest.approx(300.0)


def test_non_positive_prices_are_not_marks():
    frame_ = frame([1.0, 0.0, -1.0, np.nan, 1.1])
    paths = Paths.from_frame(frame_, floor=FLOOR, execution=IDEAL)
    assert paths.price.tolist() == [1.0, 1.1]


# --------------------------------------------------------------------------- the bot's formulas


@pytest.mark.parametrize(
    ("multiple", "volatility", "expected"),
    [
        (3.0, 0.04, 0.12),  # multiple x volatility
        (3.0, 0.001, 0.05),  # floored at MinStopLossPct
        (3.0, 0.5, 0.3),  # capped at StopLossPct
        (8.0, 0.02, 0.16),
        (3.0, float("nan"), 0.3),  # unknown: the cap
        (3.0, 0.0, 0.3),
        (3.0, -0.1, 0.3),
    ],
)
def test_volatility_stop_matches_exit_levels(multiple, volatility, expected):
    assert stop_loss_level(f"vol{multiple:g}", volatility) == pytest.approx(expected)


def test_no_stop_never_fires_and_a_fixed_stop_ignores_volatility():
    assert stop_loss_level(None, 0.1) == 1.0
    assert stop_loss_level(0.2, 0.9) == 0.2


def test_take_profit_cap_is_the_sizers_gain():
    assert take_profit_level("cap", 0.2) == pytest.approx(0.2)
    assert take_profit_level("cap", 3.0) == pytest.approx(0.5)  # TakeProfitPct
    assert take_profit_level(2.0, 0.2) == pytest.approx(2.0)
    assert take_profit_level(None, 0.2) == float("inf")


def test_an_entry_without_upside_is_declined_in_every_cell():
    assert run([1.0, 2.0], FIXED, expected_return=0.04).status == DECLINED
    assert run([1.0, 2.0], FIXED, expected_return=float("nan")).status == DECLINED


def test_realistic_fill_prices_match_fills_cs():
    # fee 1 %, R = 100, S = 1: buy = P (R + n) / (R keep); sell = P R keep / (R + value).
    execution = Execution(
        entry_latency_s=0.0,
        exit_latency_s=0.0,
        network_fee=0.0,
        account_rent=0.0,
        max_slippage=0.0,
        fee_pct={"default": 0.01},
        virtual_sol={},
    )
    out = run([1.0, 2.0], ExitRule(0.5, 0.2, 300), execution=execution)
    keep = 0.99
    entry = (100 + keep) / (100 * keep)
    value = 2.0 / entry
    sell = 2.0 * 100 * keep / (100 + value)
    assert out.net_return == pytest.approx(sell / entry - 1.0)


def test_virtual_reserve_and_venue_fee_apply_on_curves():
    paths = paths_of(frame([1.0], venue="PumpFun", liquidity=2.0), execution=Execution())
    assert paths.reserve[0] == pytest.approx(32.0)
    assert paths.fee[0] == pytest.approx(0.0125)
    other = paths_of(frame([1.0], venue=None, liquidity=2.0), execution=Execution())
    assert other.reserve[0] == pytest.approx(2.0)
    assert other.fee[0] == pytest.approx(0.0025)


def test_position_is_capped_by_the_pool_and_dust_is_declined():
    execution = Execution(
        entry_latency_s=0.0, exit_latency_s=0.0, max_slippage=0.0, account_rent=0.0, network_fee=0.0
    )
    capped = run([1.0, 2.0], FIXED, execution=execution, liquidity=10.0)
    assert capped.status == FILLED  # size 0.2 = 2 % of 10 SOL
    assert capped.pnl_sol == pytest.approx(capped.net_return * 0.2)
    dust = run([1.0, 2.0], FIXED, execution=execution, liquidity=0.4)  # 0.008 < MinPositionSize
    assert dust.status == DECLINED


def test_unknown_liquidity_is_declined():
    out = run([1.0, 2.0], FIXED, execution=Execution(), liquidity=float("nan"))
    assert out.status == DECLINED


def test_costs_are_charged_on_the_size():
    execution = Execution(
        realistic_fills=False, entry_latency_s=0.0, exit_latency_s=0.0, max_slippage=0.0
    )
    out = run([1.0, 1.0, 1.0], ExitRule(None, None, 10), execution=execution)
    cost = 0.0002 + 0.00203928 + 0.0002
    assert out.pnl_sol == pytest.approx(-cost)
    assert out.net_return == pytest.approx(-cost / 1.0)


# --------------------------------------------------------------------------- latency


def latency(entry=0.0, exit_=0.0, slippage=0.0, retries=3, age=60.0):
    return Execution(
        realistic_fills=False,
        entry_latency_s=entry,
        exit_latency_s=exit_,
        network_fee=0.0,
        account_rent=0.0,
        max_slippage=slippage,
        max_exit_retries=retries,
        max_mark_age_s=age,
    )


def test_entry_fills_at_the_first_mark_after_the_latency():
    out = run([1.0, 2.0, 3.0], ExitRule(0.4, None, 300), execution=latency(entry=3.0))
    # fills at the 10 s mark (2.0), levels from it: TP 2.8, hit by 3.0
    assert out.reason == "tp" and out.net_return == pytest.approx(0.5)


def test_entry_without_a_mark_in_time_is_abandoned_and_frees_the_mint():
    out = run([1.0, 2.0], FIXED, execution=latency(entry=3.0, age=5.0), step=100.0)
    assert out.status == ABANDONED
    assert out.free_at == pytest.approx((T0 + pd.Timedelta(seconds=8)).timestamp())


def test_the_fill_tick_does_not_trigger_an_exit():
    out = run([1.0, 5.0, 5.0, 5.0], ExitRule(0.5, None, 25), execution=latency(entry=3.0))
    assert out.reason == "timeout" and out.net_return == pytest.approx(0.0)


def test_exit_fills_at_the_first_mark_after_its_latency():
    out = run([1.0, 1.6, 1.3, 9.0], FIXED, execution=latency(exit_=2.0))
    assert out.reason == "tp" and out.net_return == pytest.approx(0.3)  # the trigger's reason
    assert out.hold_s == pytest.approx(20.0)


def test_exit_without_a_mark_in_time_fills_at_the_last_mark():
    out = run([1.0, 1.6], FIXED, execution=latency(exit_=2.0))
    assert out.reason == "tp" and out.net_return == pytest.approx(0.6)
    assert out.hold_s == pytest.approx(10.0 + 2.0 + 60.0)


def test_entry_above_the_slippage_tolerance_fails():
    out = run([1.0, 1.2, 1.2], FIXED, execution=latency(entry=3.0, slippage=0.15))
    assert out.status == FAILED
    assert run([1.0, 1.1, 1.1], FIXED, execution=latency(entry=3.0, slippage=0.15)).status == FILLED


def test_entry_into_an_empty_pool_fails():
    execution = Execution(entry_latency_s=3.0, max_slippage=0.0)
    out = run([1.0, 1.1], FIXED, execution=execution, liquidity=[10.0, 0.0])
    assert out.status == FAILED


def test_exit_below_the_tolerance_is_retried_then_fills_at_any_price():
    # stop at 0.8 trigger (quote 0.7), then each attempt falls more than 15 % below the last quote
    prices = [1.0, 0.7, 0.5, 0.3, 0.1, 0.05]
    rule = ExitRule(None, 0.2, 300)
    out = run(prices, rule, execution=latency(exit_=2.0, slippage=0.15, retries=2))
    assert out.reason == "sl"
    assert out.net_return == pytest.approx(0.1 - 1.0)  # 2 retries (0.5, 0.3), then 0.1 at any price


def test_each_exit_attempt_pays_its_network_fee():
    execution = Execution(
        realistic_fills=False,
        entry_latency_s=0.0,
        exit_latency_s=2.0,
        account_rent=0.0,
        max_slippage=0.15,
        max_exit_retries=2,
    )
    out = run([1.0, 0.7, 0.5, 0.45], ExitRule(None, 0.2, 300), execution=execution)
    # 0.5 breaches (attempt 1), 0.45 is within 15 % of 0.5 (attempt 2): entry + 2 exits
    assert out.net_return == pytest.approx(0.45 - 1.0 - 3 * 0.0002)


# --------------------------------------------------------------------------- one position per mint


def windows_of(paths, mints_rows, *, pump=0.5, expected_return=1.0, labeled=True):
    n = len(mints_rows)
    return Windows(
        np.array([m for m, _ in mints_rows], dtype=object),
        np.array([r for _, r in mints_rows], np.int64),
        np.full(n, pump, np.float64),
        np.full(n, expected_return, np.float64),
        np.full(n, labeled, bool),
    )


def test_overlapping_windows_of_a_held_mint_are_skipped():
    paths = paths_of(frame([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0]))
    windows = windows_of(paths, [("A", 0), ("A", 1), ("A", 2), ("A", 4)])
    entries = simulate_rule(paths, windows, FIXED, IDEAL)
    assert len(entries) == 2  # windows 1 and 2 fall inside the first position, closed at row 3
    assert [o.reason for _, o in entries] == ["tp", "timeout"]


def test_positions_of_different_mints_do_not_block_each_other():
    paths = paths_of(frame([1.0, 2.0], mint="A"), frame([1.0, 2.0], mint="B"))
    windows = windows_of(paths, [("A", 0), ("B", paths.spans["B"][0])])
    assert len(simulate_rule(paths, windows, FIXED, IDEAL)) == 2


def test_failed_entry_frees_the_mint_for_the_next_signal():
    paths = paths_of(frame([1.0, 1.5, 1.5, 3.0]), execution=latency(entry=3.0, slippage=0.15))
    windows = windows_of(paths, [("A", 0), ("A", 1)])
    entries = simulate_rule(paths, windows, FIXED, latency(entry=3.0, slippage=0.15))
    assert [o.status for _, o in entries] == [FAILED, FILLED]


def test_tokens_stay_apart_even_if_the_frame_is_shuffled():
    df = pd.concat([frame([1.0, 2.0, 3.0], mint="B"), frame([5.0, 6.0], mint="A")])
    paths = Paths.from_frame(df.sample(frac=1, random_state=3), floor=FLOOR, execution=IDEAL)
    a0, a1 = paths.spans["A"]
    b0, b1 = paths.spans["B"]
    assert paths.price[a0:a1].tolist() == [5.0, 6.0]
    assert paths.price[b0:b1].tolist() == [1.0, 2.0, 3.0]


def test_windows_align_on_their_mark_and_drop_an_invalid_anchor():
    paths = paths_of(frame([1.0, 0.0, 1.0]))  # the 10 s row is not a mark
    t0 = int(T0.timestamp())
    windows = Windows.align(
        paths,
        np.array(["A", "A", "A", "Z"], dtype=object),
        np.array([t0, t0 + 10, t0 + 20, t0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([True, True, False, True]),
    )
    assert windows.row.tolist() == [0, 1]  # marks 0 and 1 (the third row, now the second mark)
    assert windows.pump_prob.tolist() == [0.1, 0.3]
    assert windows.labeled.tolist() == [True, False]


def test_windows_align_with_the_samples_built_from_the_same_frame():
    """The real builder's mint / end_ts find each window's anchor, tokens and order whatever."""
    rows = []
    for mint, prices in (("B", [1.0 + k for k in range(9)]), ("A", [50.0 + k for k in range(8)])):
        for k, p in enumerate(prices):
            row = dict.fromkeys((c for c in FEATURE_COLUMNS if not c.startswith("venue_")), 0.0)
            row.update(
                is_migrated=False,
                price=p,
                liquidity=50.0,
                venue="pumpswap",
                volatility=0.1,
                mint=mint,
                ts=T0 + pd.Timedelta(seconds=10 * k + 0.25),  # sub-second, as the pipeline writes
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    samples = build_samples(
        df,
        seq_len=1,
        horizons=("1m",),
        pump_thresholds={"1m": 0.1},
        rug_thresholds={"1m": 0.4},
        min_gap_seconds=1000,
    )
    paths = Paths.from_frame(df, floor=FLOOR, execution=IDEAL)
    n = len(samples)
    assert n > 0
    windows = Windows.align(
        paths, samples.mint, samples.end_ts, np.zeros(n), np.zeros(n), samples.mask[:, 0] == 1
    )
    assert len(windows) == n
    t0 = T0.timestamp()
    for mint, row, end_ts in zip(windows.mint, windows.row, samples.end_ts, strict=True):
        start, stop = paths.spans[mint]
        assert start <= row < stop
        assert paths.ts[row] == pytest.approx(end_ts + 0.25)
        base = {"B": 1.0, "A": 50.0}[mint]
        assert paths.price[row] == base + round((end_ts - t0) / 10)


def test_windows_too_close_to_the_end_of_the_read_are_censored():
    paths = paths_of(frame([1.0] * 11))  # 100 s of data
    windows = windows_of(paths, [("A", 0), ("A", 5), ("A", 10)])
    assert uncensored(windows, paths, 50).row.tolist() == [0, 5]


# --------------------------------------------------------------------------- from sweep to entries


def test_window_outcomes_read_quiet_tokens_the_labels_would_mask():
    quiet = frame([1.0, 1.0, 1.0], mint="Q")  # goes silent, flat: never labeled at 15m
    pumping = frame([1.0, 1.5, 2.5], mint="P")
    dust = frame([1.0, 9.0], mint="D", liquidity=[5.0, 0.1])  # the jump is read off a dust pool
    paths = paths_of(quiet, pumping, dust)
    windows = windows_of(paths, [(m, paths.spans[m][0]) for m in ("Q", "P", "D")], labeled=False)
    out = window_outcomes(paths, windows, horizon_s=900, pump_threshold=1.0)
    assert out["touch"].tolist() == [False, True, False]
    assert out["hold"].tolist() == pytest.approx([0.0, 1.5, 0.0])
    assert out["flat"].tolist() == [True, False, False]


def test_window_outcomes_stop_at_the_horizon_and_cap_the_return():
    paths = paths_of(frame([1.0, 1.1, 50.0], step=100.0), frame([1.0, 500.0], mint="B"))
    windows = windows_of(paths, [("A", 0), ("B", paths.spans["B"][0])])
    out = window_outcomes(paths, windows, horizon_s=100, pump_threshold=1.0, max_return=10.0)
    assert out["touch"].tolist() == [False, True]
    assert out["hold"].tolist() == pytest.approx([0.1, 10.0])


def test_one_per_mint_and_first_per_mint():
    paths = paths_of(frame([1.0] * 5, step=100.0))
    windows = windows_of(paths, [("A", i) for i in range(5)])
    assert one_per_mint(windows, paths, 200).tolist() == [0, 2, 4]
    assert first_per_mint(windows).tolist() == [0]


def test_selection_steps_expose_the_labeled_windows_bias():
    survivor = frame([1.0, 2.5, 2.5, 2.5, 2.5], mint="S")
    quiet = frame([1.0, 1.0], mint="Q")
    tail = frame([1.0] * 200, mint="T", step=10.0)
    paths = paths_of(survivor, quiet, tail)
    rows = [("S", paths.spans["S"][0]), ("S", paths.spans["S"][0] + 1), ("Q", paths.spans["Q"][0])]
    windows = windows_of(paths, rows)
    object.__setattr__(windows, "labeled", np.array([True, True, False]))

    rules = simulate_rules(
        paths,
        windows,
        pump_thresholds=[0.4],
        horizon_s=900,
        pump_threshold=1.0,
        grid=[FIXED],
        execution=IDEAL,
        bootstrap_samples=50,
    )
    steps = rules[0]["selection"]
    assert steps["labeled_windows"]["n"] == 2
    assert steps["labeled_windows"]["precision"] == pytest.approx(0.5)
    assert steps["all_windows"]["n"] == 3
    assert steps["all_windows"]["labeled_share"] == pytest.approx(2 / 3)
    assert steps["all_windows"]["flat_share"] == pytest.approx(2 / 3)
    assert steps["first_per_mint"]["n"] == 2
    assert steps["first_per_mint"]["precision"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- report


def test_bootstrap_draws_tokens_not_trades():
    rng = np.random.default_rng(0)
    values = np.array([1.0] * 100 + [0.0])
    tokens = np.array(["A"] * 100 + ["B"], dtype=object)
    low, high = bootstrap_by_token(values, tokens, samples=500, ci=0.95, rng=rng)
    assert low == pytest.approx(0.0)  # a draw of only B
    assert high == pytest.approx(1.0)  # ... of only A: one token's 100 trades count as one draw
    assert bootstrap_by_token(
        values, np.array(["A"] * 101, dtype=object), samples=5, ci=0.9, rng=rng
    ) == (
        None,
        None,
    )


def test_default_grid_covers_the_issues_cells():
    grid = default_grid()
    assert len(grid) == 8 * 5 * 3
    assert BOT_RULE in grid
    assert {r.timeout_s for r in grid} == {300, 900, 1800}
    assert {r.stop_loss for r in grid} == {None, 0.1, 0.2, 0.3, 0.5, "vol3", "vol5", "vol8"}
    assert {r.take_profit for r in grid} == {"cap", 0.5, 1.0, 2.0, None}


def _report():
    paths = paths_of(
        frame([1.0, 1.6, 1.0, 1.0], mint="A"), frame([1.0, 0.5] + [1.0] * 200, mint="B")
    )
    rows = [("A", paths.spans["A"][0]), ("B", paths.spans["B"][0])]
    windows = windows_of(paths, rows)
    grid = [BOT_RULE, FIXED, ExitRule(take_profit=None, stop_loss=None, timeout_s=60)]
    rules = simulate_rules(
        paths,
        windows,
        pump_thresholds=[0.4, 0.9],
        horizon_s=900,
        pump_threshold=1.0,
        grid=grid,
        execution=IDEAL,
        min_trades=1,
        bootstrap_samples=50,
    )
    settings = {"min_trades": 1, "ci": 0.95, "execution": {"realistic_fills": False}}
    return build_report("model-x", 2, "15m", rules, settings)


def test_cell_reports_what_the_issue_asks_for():
    report = _report()
    cell = report["rules"][0]["cells"][1]
    assert cell["n_trades"] == 2
    assert cell["n_tokens"] == 2
    assert cell["win_rate"] == pytest.approx(0.5)
    assert cell["mean_return"] == pytest.approx((0.6 - 0.5) / 2)
    assert cell["median_return"] == pytest.approx((0.6 - 0.5) / 2)
    assert cell["exit_share"] == {"tp": 0.5, "sl": 0.5, "timeout": 0.0}
    assert cell["median_hold_s"] == pytest.approx(10.0)
    assert None not in cell["mean_return_ci"]
    assert report["rules"][0]["cells"][0]["is_bot_default"] is True
    assert cell["is_bot_default"] is False


def test_a_rule_selecting_nothing_still_reports():
    rule = _report()["rules"][1]  # pump >= 0.9 selects nothing
    assert rule["selection"]["all_windows"] == {"n": 0, "n_tokens": 0}
    assert rule["cells"][0]["n_trades"] == 0
    assert rule["cells"][0]["mean_return"] is None
    assert "[pump >= 0.9]" in render(_report())


def test_report_is_written_as_json_and_table(tmp_path):
    report = _report()
    json_path, table_path = write_report(report, tmp_path / "sub" / "exit.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["model_version"] == "model-x"
    table = table_path.read_text(encoding="utf-8")
    assert "<- bot" in table
    assert "labeled_windows" in table and "first_per_mint" in table
    assert "marginals" in table


def test_marginals_skip_cells_under_min_trades():
    paths = paths_of(frame([1.0, 1.6, 1.0] + [1.0] * 200))
    windows = windows_of(paths, [("A", 0)])
    rules = simulate_rules(
        paths,
        windows,
        pump_thresholds=[0.4],
        horizon_s=900,
        pump_threshold=1.0,
        grid=[FIXED],
        execution=IDEAL,
        min_trades=2,
        bootstrap_samples=10,
    )
    assert rules[0]["marginals"]["stop_loss"][0] == {
        "value": 0.2,
        "cells": 0,
        "mean_return": None,
        "median_return": None,
    }


def test_read_only_mirror_of_bot_options_defaults():
    """The constants mirror BotOptions.cs: a changed bot default must change them too."""
    import solia_ai.exit_sim as sim

    text = BOT_OPTIONS.read_text(encoding="utf-8")

    def default(name):
        match = re.search(rf"public \w+\??\s+{name} \{{ get; set; \}} = ([0-9_.]+)m?;", text)
        assert match, name
        return float(match.group(1).replace("_", ""))

    assert default("TakeProfitPct") == sim.TAKE_PROFIT_PCT
    assert default("MinTakeProfitPct") == sim.MIN_TAKE_PROFIT_PCT
    assert default("StopLossPct") == sim.STOP_LOSS_PCT
    assert default("StopLossVolatilityMultiple") == sim.STOP_LOSS_VOLATILITY_MULTIPLE
    assert default("MinStopLossPct") == sim.MIN_STOP_LOSS_PCT
    assert default("PositionTimeoutSeconds") == sim.POSITION_TIMEOUT_SECONDS
    assert default("StartingBalance") == sim.STARTING_BALANCE
    assert default("FixedPositionFraction") == sim.FIXED_POSITION_FRACTION
    assert default("MinPositionSize") == sim.MIN_POSITION_SIZE
    assert default("MaxPoolFraction") == sim.MAX_POOL_FRACTION
    assert default("EntryLatencyMs") / 1000 == sim.ENTRY_LATENCY_SECONDS
    assert default("ExitLatencyMs") / 1000 == sim.EXIT_LATENCY_SECONDS
    assert default("MaxMarkAgeSeconds") == sim.MAX_MARK_AGE_SECONDS
    assert default("NetworkFeeSol") == sim.NETWORK_FEE_SOL
    assert default("AccountRentSol") == sim.ACCOUNT_RENT_SOL
    assert default("MaxSlippagePct") == sim.MAX_SLIPPAGE_PCT
    assert default("MaxExitRetries") == sim.MAX_EXIT_RETRIES

    def table(name):
        block = re.search(rf"> {name} \{{ get; set; \}} = new\(.*?\{{(.*?)\n    \}};", text, re.S)
        assert block, name
        out = {}
        for key, value in re.findall(
            r'\[(?:"([\w-]+)"|Trading\.Fills\.DefaultVenue)\] = ([0-9.]+)', block.group(1)
        ):
            out[key or "default"] = float(value)
        return out

    assert table("FeePct") == sim.FEE_PCT
    assert table("VirtualSol") == sim.VIRTUAL_SOL


# --------------------------------------------------------------------------- bad settings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pump_thresholds": []},
        {"pump_thresholds": [0.0]},
        {"pump_thresholds": [1.5]},
        {"pump_thresholds": [float("nan")]},
        {"min_trades": 0},
        {"bootstrap_samples": 0},
        {"ci": 1.0},
    ],
)
def test_simulate_rules_rejects_bad_settings(kwargs):
    paths = paths_of(frame([1.0, 1.0]))
    settings = {"pump_thresholds": [0.4], "horizon_s": 10, "pump_threshold": 1.0, **kwargs}
    with pytest.raises(ValueError):
        simulate_rules(paths, windows_of(paths, [("A", 0)]), **settings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"take_profit": 0},
        {"take_profit": "x"},
        {"stop_loss": 0},
        {"stop_loss": 1.0},
        {"stop_loss": "volx"},
        {"stop_loss": "vol0"},
        {"timeout_s": 0},
    ],
)
def test_exit_rule_rejects_bad_levels(kwargs):
    with pytest.raises(ValueError):
        ExitRule(**kwargs)


def test_paths_need_their_columns():
    with pytest.raises(ValueError, match="volatility"):
        Paths.from_frame(frame([1.0]).drop(columns="volatility"), floor=FLOOR)


@pytest.mark.parametrize(
    "args",
    [
        ["--champion", "--pump-thresholds", "0"],
        ["--champion", "--min-trades", "0"],
        ["--champion", "--ci", "2"],
        ["--champion", "--window-stride-seconds", "-1"],
        ["--champion", "--until", "yesterday"],
        ["--champion", "--until", "2026-11-04T02:11:00"],
    ],
)
def test_cli_rejects_bad_settings_before_reading_anything(args):
    with pytest.raises(SystemExit):
        main(args)


def test_scoring_moves_the_model_to_the_requested_device(monkeypatch):
    # The "meta" device stands in for the GPU, so this also runs on a machine without one.
    class FakePredictor:
        model = torch.nn.Linear(1, 1)
        calibration = 1.0

    seen = []

    def fake_score(model, dataset):
        seen.append(next(model.parameters()).device.type)
        return {"pump_logits": dataset}, {"mask": dataset}

    monkeypatch.setattr(exit_sim, "score_logits", fake_score)
    monkeypatch.setattr(exit_sim, "apply_calibration", lambda logits, calibration: logits)

    predictions, truth = score_windows(FakePredictor(), "windows", "meta")

    assert seen == ["meta"]
    assert predictions == {"pump_logits": "windows"}
    assert truth == {"mask": "windows"}


def test_until_ends_the_read_there(monkeypatch):
    class Stop(Exception):
        pass

    class FakePredictor:
        horizons = ("15m",)

    seen = {}

    def fake_read(engine, settings, *, end=None):
        seen["end"] = end
        raise Stop

    monkeypatch.setattr("solia_ai.db.make_engine", lambda: None)
    monkeypatch.setattr("solia_ai.db.read_training_features", fake_read)
    monkeypatch.setattr(
        "solia_ai.inference.Predictor.from_checkpoint", lambda path: FakePredictor()
    )

    with pytest.raises(Stop):
        main(["--checkpoint", "model.pt", "--until", "2026-11-03T21:11:00-05:00"])

    assert seen["end"] == pd.Timestamp("2026-11-04T02:11:00Z")
