"""Rug-guard labels (solia_ai.rug_labels): synthetic paths for each case."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from solia_ai.config import ConfigError, load_config, validate_config
from solia_ai.liquidity import LiquidityFloor
from solia_ai.rug_labels import (
    RugLabelSettings,
    age_bucket,
    concat_categorical,
    label_columns,
    label_in_groups,
    label_rug_events,
    rates,
    session_agreement,
)

T0 = pd.Timestamp("2026-11-06 00:00:00", tz="UTC")
FLOOR = LiquidityFloor.from_mapping({"default": 1.0, "pumpfun": 0.0})
S = RugLabelSettings(horizons=("5m",), floor=FLOOR)


def token(mint, prices, liqs=None, *, step=10, venue="pumpswap", times=None):
    """A token's rows, one every ``step`` seconds (or at ``times`` seconds), liquidity 100 SOL."""
    n = len(prices)
    times = np.arange(n) * step if times is None else np.asarray(times)
    liqs = [100.0] * n if liqs is None else liqs
    return pd.DataFrame(
        {
            "mint": mint,
            "ts": T0 + pd.to_timedelta(times, unit="s"),
            "price": prices,
            "liquidity": liqs,
            "venue": venue,
            "token_age_sec": times,
        }
    )


def first(labels, col):
    return labels[col].iloc[0]


def test_a_flat_covered_token_is_neither():
    labels = label_rug_events(token("FLAT", [1.0] * 60), S)  # 10 min of rows

    assert first(labels, "sudden_5m") == 0
    assert first(labels, "gradual_5m") == 0
    assert math.isnan(first(labels, "drain_5m"))
    assert labels["sudden_5m"].iloc[-1] != labels["sudden_5m"].iloc[-1]  # last row: no future


def test_a_drained_pool_is_a_sudden_rug_even_without_coverage():
    # 10 rows then the pool is emptied in one step, and the token stops trading.
    labels = label_rug_events(token("DRAIN", [1.0] * 11, [100.0] * 10 + [0.0]), S)

    assert first(labels, "sudden_5m") == 1
    assert first(labels, "drain_5m") == 1
    assert first(labels, "gradual_5m") == 0  # the fall was sudden: decided
    assert first(labels, "next_sudden_sec") == 100
    assert first(labels, "next_sudden_drain") == 1


def test_a_one_step_price_crash_on_a_pool_left_above_the_floor_is_not_drained():
    prices = [1.0] * 10 + [0.5] * 50
    labels = label_rug_events(token("CRASH", prices, [100.0] * 10 + [70.0] * 50), S)

    assert first(labels, "sudden_5m") == 1
    assert first(labels, "drain_5m") == 0


@pytest.mark.parametrize(("after", "expected"), [(0.5, 1.0), (0.51, 0.0)])
def test_the_sudden_threshold_is_inclusive(after, expected):
    labels = label_rug_events(token("EDGE", [1.0] * 10 + [after] * 50), S)

    assert first(labels, "sudden_5m") == expected


def test_a_slow_decline_is_a_gradual_dump_not_a_sudden_rug():
    prices = list(np.geomspace(1.0, 0.1, 60))  # -3.8 % per step, -68 % within 5 min
    labels = label_rug_events(token("SLOW", prices), S)

    assert first(labels, "gradual_5m") == 1
    assert first(labels, "sudden_5m") == 0


def test_a_gradual_dump_then_a_drain_is_both():
    prices = list(np.geomspace(1.0, 0.4, 20))
    labels = label_rug_events(token("BOTH", prices + [0.4], [100.0] * 20 + [0.0]), S)

    assert first(labels, "gradual_5m") == 1
    assert first(labels, "sudden_5m") == 1
    assert first(labels, "drain_5m") == 1


def test_a_drop_across_a_gap_decides_nothing():
    # 10 rows, nothing for 2 min, then the price is halved: one transaction or many?
    times = list(range(0, 100, 10)) + [220 + 10 * k for k in range(40)]
    labels = label_rug_events(token("GAP", [1.0] * 10 + [0.5] * 40, times=times), S)

    assert math.isnan(first(labels, "sudden_5m"))
    assert math.isnan(first(labels, "gradual_5m"))
    assert labels["next_sudden_sec"].isna().all()


def test_a_sudden_drop_before_an_ambiguous_one_is_labeled():
    times = list(range(0, 100, 10)) + [220 + 10 * k for k in range(40)]
    prices = [1.0] * 5 + [0.4] * 5 + [0.1] * 40  # sudden at row 5, ambiguous after the gap
    labels = label_rug_events(token("ORDER", prices, times=times), S)

    assert first(labels, "sudden_5m") == 1


def test_a_horizon_the_rows_dont_cover_is_unknown():
    labels = label_rug_events(token("SHORT", [1.0] * 12), S)  # 2 min of rows, horizon 5 min

    assert math.isnan(first(labels, "sudden_5m"))
    assert math.isnan(first(labels, "gradual_5m"))


def test_a_stale_last_row_is_not_a_covered_horizon():
    # Rows to 2 min, then one row at 10 min: the 5-min horizon's last row is 3 min stale.
    times = list(range(0, 130, 10)) + [600]
    labels = label_rug_events(token("STALE", [1.0] * 14, times=times), S)

    assert math.isnan(first(labels, "sudden_5m"))


def test_steps_off_a_dust_pool_are_not_drops():
    # Below the 1 SOL AMM floor, prices jump on micro-trades: no drop, and no anchor.
    labels = label_rug_events(token("DUST", [1.0] * 10 + [0.2] * 50, [0.5] * 60), S)

    assert labels["sudden_5m"].isna().all()  # untradeable anchors aren't labeled
    assert labels["next_sudden_sec"].isna().all()


def test_a_curve_token_is_dumped_never_drained():
    labels = label_rug_events(
        token("CURVE", [1.0] * 10 + [0.3] * 50, [5.0] * 10 + [0.2] * 50, venue="pumpfun"), S
    )

    assert first(labels, "sudden_5m") == 1
    assert first(labels, "drain_5m") == 0  # floor 0: curves can't be "drained"


def test_a_pool_falling_under_its_floor_by_less_than_the_drop_is_drained():
    labels = label_rug_events(token("FLOOR", [1.0] * 60, [1.5] * 10 + [0.9] * 50), S)

    assert first(labels, "sudden_5m") == 1
    assert first(labels, "drain_5m") == 1


def test_switching_between_pools_is_not_a_liquidity_drop():
    # A multi-pool token: each snapshot carries the liquidity of the pool traded last.
    pools = ["BIG", "SMALL"] * 30
    df = token("MULTI", [1.0] * 60, [1000.0 if p == "BIG" else 20.0 for p in pools])
    df["pool_address"] = pools

    labels = label_rug_events(df, S)

    assert first(labels, "sudden_5m") == 0


def test_a_side_pool_drained_is_not_the_tokens_rug_but_the_main_pool_is():
    pools = ["BIG", "SIDE"] * 30
    side = [1000.0 if p == "BIG" else (20.0 if k < 20 else 0.0) for k, p in enumerate(pools)]
    df = token("SIDE", [1.0] * 60, side)
    df["pool_address"] = pools
    main = token("MAIN", [1.0] * 60, [1000.0] * 20 + [0.0] * 40)
    main["pool_address"] = "BIG"

    labels = label_rug_events(pd.concat([df, main]), S).groupby("mint").first()

    assert labels.loc["SIDE", "sudden_5m"] == 0
    assert labels.loc["MAIN", "sudden_5m"] == 1
    assert labels.loc["MAIN", "drain_5m"] == 1


def test_a_main_pool_drained_while_trading_goes_on_on_a_dust_pool_is_a_drain():
    # Seen live: the 200 SOL pool is emptied; the next snapshots come from a 0.01 SOL side pool.
    liqs = [200.0] * 12 + [0.01] * 48
    prices = [1.0] * 13 + [0.02] * 47  # the dust pool's price, read on untradeable rows
    labels = label_rug_events(token("BVIH", prices, liqs), S)

    assert first(labels, "sudden_5m") == 1
    assert first(labels, "drain_5m") == 1
    assert first(labels, "next_sudden_sec") == 120


def test_a_wick_that_recovers_within_a_minute_is_no_drop():
    prices = [1.0] * 10 + [0.4] + [1.0] * 49
    labels = label_rug_events(token("WICK", prices, [100.0] * 10 + [40.0] + [100.0] * 49), S)

    assert first(labels, "sudden_5m") == 0


def test_null_pools_and_venues_are_handled():
    curve = token("CURVE2", [1.0] * 60, [5.0] * 10 + [0.2] * 50, venue="pumpfun")
    curve["pool_address"] = None  # a bonding curve: no pool address, the venue is the pool
    nameless = token("NOVENUE", [np.nan] * 5, [np.nan] * 5, venue=None)
    nameless["pool_address"] = None
    df = pd.concat([curve, nameless]).astype(
        {"mint": "category", "venue": "category", "pool_address": "category"}
    )

    labels = label_rug_events(df, S).groupby("mint", observed=True).first()

    assert labels.loc["CURVE2", "sudden_5m"] == 1
    assert math.isnan(labels.loc["NOVENUE", "sudden_5m"])  # no price: never an anchor


def test_switching_to_smaller_pools_for_over_a_minute_is_not_a_drop():
    # A token on many pools: its 1000 SOL pool doesn't trade for 2 minutes, smaller ones do.
    pools = ["BIG"] * 10 + ["P20", "P30", "P40"] * 12 + ["BIG"] * 14
    liqs = [1000.0] * 10 + [20.0, 30.0, 40.0] * 12 + [1000.0] * 14
    df = token("MANY", [1.0] * 60, liqs)
    df["pool_address"] = pools

    assert first(label_rug_events(df, S), "sudden_5m") == 0


@pytest.mark.parametrize(("main_back_after_min", "expected"), [(20, 0.0), (None, 1.0)])
def test_a_dust_pool_after_the_main_pool_is_a_drain_only_if_the_main_never_returns(
    main_back_after_min, expected
):
    # One row a minute: the 500 SOL main pool, then only a 0.2 SOL dust pool...
    n_dust = 40 if main_back_after_min is None else main_back_after_min
    pools = ["MAIN"] * 5 + ["DUST"] * n_dust
    liqs = [500.0] * 5 + [0.2] * n_dust
    if main_back_after_min is not None:  # ... until the main pool trades again
        pools += ["MAIN"] * 60  # rows past the 1 h horizon: covered
        liqs += [500.0] * 60
    df = token("XPOOL", [1.0] * len(pools), liqs, step=60)
    df["pool_address"] = pools
    s = RugLabelSettings(horizons=("1h",), floor=FLOOR, max_step_seconds=90)

    labels = label_rug_events(df, s)

    assert first(labels, "sudden_1h") == expected


def test_the_main_pool_window_never_reaches_into_the_previous_token():
    big = token("AAA", [1.0] * 60, [1e6] * 60)  # sorts first, a huge pool
    small = token("BBB", [1.0] * 60, [100.0] * 3 + [0.0] * 57)  # drained right after it starts
    small["ts"] = big["ts"]

    labels = label_rug_events(pd.concat([big, small]), S).groupby("mint").first()

    assert labels.loc["BBB", "sudden_5m"] == 1


def test_tokens_never_mix_and_input_order_does_not_matter():
    a = token("A", [1.0] * 11, [100.0] * 10 + [0.0])
    b = token("B", [1.0] * 60)
    shuffled = pd.concat([a, b]).sample(frac=1.0, random_state=0)

    labels = label_rug_events(shuffled, S)

    assert labels.groupby("mint")["sudden_5m"].first().to_dict() == {"A": 1.0, "B": 0.0}
    assert labels.loc[labels["mint"] == "A", "ts"].is_monotonic_increasing


def test_anchor_range_limits_the_anchors_not_the_future():
    df = token("RANGE", [1.0] * 11, [100.0] * 10 + [0.0])

    labels = label_rug_events(df, S, anchor_from=T0, anchor_to=T0 + pd.Timedelta(seconds=30))

    assert len(labels) == 3
    assert (labels["sudden_5m"] == 1).all()  # the drain at 100 s is still seen


def test_label_columns_and_several_horizons():
    s = RugLabelSettings(horizons=("1m", "5m"), floor=FLOOR)
    labels = label_rug_events(token("H2", [1.0] * 10 + [1.0] * 50, [100.0] * 9 + [0.0] * 51), s)

    assert set(label_columns(s.horizons)) <= set(labels.columns)
    assert first(labels, "sudden_1m") == 0  # drain at 90 s: after 1 min
    assert first(labels, "sudden_5m") == 1


def test_labeling_in_groups_matches_one_pass_and_thins_anchors():
    df = pd.concat(
        [
            token("G1", [1.0] * 11, [100.0] * 10 + [0.0]),
            token("G2", list(np.geomspace(1.0, 0.1, 60))),
            token("G3", [1.0] * 60),
        ]
    )
    df["mint"] = df["mint"].astype("category")  # as read_label_frame returns it
    whole = label_rug_events(df, S)

    grouped = label_in_groups(df, S, rows_per_group=20)  # several groups, tokens never split
    thinned = label_in_groups(df, S, anchor_every_seconds=60)

    cols = ["sudden_5m", "gradual_5m", "drain_5m"]
    pd.testing.assert_frame_equal(
        grouped.sort_values(["mint", "ts"])[cols].reset_index(drop=True),
        whole.sort_values(["mint", "ts"])[cols].astype(np.float32).reset_index(drop=True),
    )
    assert len(thinned) == 2 + 10 + 10  # one anchor per token per minute
    with pytest.raises(ValueError):
        label_in_groups(df, S, rows_per_group=0)


def test_slices_with_all_null_venues_concatenate():
    a = token("C1", [1.0] * 3).astype({"mint": "category", "venue": "category"})
    b = token("C2", [1.0] * 3, venue=None).astype({"mint": "category", "venue": "category"})

    frame = concat_categorical([a, b], ("mint", "venue"))

    assert list(frame["mint"].cat.categories) == ["C1", "C2"]
    assert frame["venue"].isna().sum() == 3
    assert len(label_rug_events(frame, S)) == 6


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="liquidity"):
        label_rug_events(token("X", [1.0] * 3).drop(columns="liquidity"), S)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sudden_drop": 0.0},
        {"gradual_drop": 1.0},
        {"max_step_seconds": 0},
        {"horizons": ()},
        {"horizons": ("1h", "1h")},
        {"horizons": ("soon",)},
        {"max_gap_fraction": 0.0},
        {"min_gap_seconds": -1},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RugLabelSettings(**kwargs)


def test_settings_come_from_the_config():
    s = RugLabelSettings.from_config(load_config())

    assert s.horizons == ("1h", "6h", "24h")
    assert s.sudden_drop == 0.5 and s.gradual_drop == 0.5 and s.max_step_seconds == 20
    assert s.floor.floor_of("pumpswap") == 1.0 and s.floor.floor_of("pumpfun") == 0.0


@pytest.mark.parametrize(
    "section", [{"sudden_drop": 2}, {"bogus": 1}, {"horizons": "1h"}, {"max_step_seconds": -5}]
)
def test_an_invalid_rug_labels_section_fails_the_config(section):
    config = load_config()
    config["rug_labels"] = section

    with pytest.raises(ConfigError, match="rug_labels"):
        validate_config(config)


def test_age_buckets():
    assert list(age_bucket(np.array([0, 299, 300, 3600, 86_400, np.nan]))) == [
        "<5m", "<5m", "5-30m", "1-6h", ">=24h", "unknown",
    ]  # fmt: skip


def test_rates_report_labeled_share_and_rates():
    labels = pd.concat(
        [
            label_rug_events(token("R1", [1.0] * 11, [100.0] * 10 + [0.0]), S),
            label_rug_events(token("R2", [1.0] * 60), S),
        ]
    )

    table = rates(labels, "mint", ("5m",)).set_index("mint")

    assert table.loc["R1", "sudden"] == 1.0
    assert table.loc["R1", "drained_of_sudden"] == 1.0
    assert table.loc["R2", "sudden"] == 0.0
    assert 0 < table.loc["R2", "labeled"] < 1  # the last 5 min have no covered future


def test_session_agreement_takes_the_labels_at_entry():
    labels = label_rug_events(token("TRADE", [1.0] * 11, [100.0] * 10 + [0.0]), S)
    trades = pd.DataFrame({"mint": ["TRADE"], "opened_at": [T0 + pd.Timedelta(seconds=45)]})

    agree = session_agreement(labels, trades, ("5m",))

    assert agree["ts"].iloc[0] == T0 + pd.Timedelta(seconds=40)
    assert agree["sudden_5m"].iloc[0] == 1
    assert agree["next_sudden_sec"].iloc[0] == 60
