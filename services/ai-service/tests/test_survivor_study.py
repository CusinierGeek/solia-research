"""Survivor study (#261): costs, entries, outcomes, loss causes, CIs, warning signs, CLI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solia_ai import survivor_study as s
from solia_ai.liquidity import LiquidityFloor

T0 = pd.Timestamp("2026-01-01", tz="UTC")
FLOOR = LiquidityFloor.from_mapping({"default": 1.0, "pumpfun": 0.0})


def rows(mint, seconds, price, liquidity=100.0, age0=7 * 3600, venue="pumpswap", **extra):
    seconds = np.asarray(seconds, np.int64)
    n = len(seconds)
    data = {
        "mint": mint,
        "ts": T0 + pd.to_timedelta(seconds, "s"),
        "price": np.broadcast_to(np.asarray(price, np.float64), n).copy(),
        "liquidity": np.broadcast_to(np.asarray(liquidity, np.float64), n).copy(),
        "venue": venue,
        "pool_address": f"pool-{mint}",
        "token_age_sec": age0 + seconds,
        "burned_liquidity": 100.0,
        "is_migrated": 1.0,
    }
    data.update(extra)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- costs


def one(x):
    return np.array([x], np.float64)


def test_net_return_without_costs_is_the_price_return():
    out = s.net_return(one(1.0), one(1e9), one(0.0), one(1.5), one(1e9), one(0.0), s.FREE)
    assert out[0] == pytest.approx(0.5, abs=1e-6)


def test_net_return_pays_fees_impact_and_fixed_costs():
    costs = s.Costs()
    deep = s.net_return(one(1.0), one(1e7), one(0.003), one(1.0), one(1e7), one(0.003), costs)
    fixed = (2 * costs.network_fee + costs.account_rent) / costs.size
    assert deep[0] == pytest.approx(-(1 - 0.997**2) - fixed, abs=1e-4)
    shallow = s.net_return(one(1.0), one(20.0), one(0.003), one(1.0), one(20.0), one(0.003), costs)
    assert shallow[0] < deep[0] - 0.05  # 1 SOL in a 20 SOL pool: ~10 % impact round trip


def test_net_return_of_an_emptied_pool_is_a_total_loss_whatever_its_price():
    costs = s.Costs()
    for mark in (0.0, 1.0, 1e7):
        out = s.net_return(one(1.0), one(80.0), one(0.003), one(mark), one(0.0), one(0.003), costs)
        assert out[0] <= -1.0
    # a 0.5 SOL leftover pool pays at most its reserve, even on a broken price
    out = s.net_return(one(1.0), one(80.0), one(0.003), one(1e7), one(0.5), one(0.003), costs)
    assert out[0] < -0.5


def test_costs_per_venue():
    c = s.Costs()
    venue = np.array(["pumpfun", "PumpSwap", None, "orca"], object)
    assert list(c.fee(venue)) == [0.0125, 0.003, 0.0025, 0.0025]
    assert list(c.reserve(np.array([0.5, 10.0, np.nan, -1.0]), venue)[:2]) == [30.5, 10.0]
    assert np.isnan(c.reserve(np.array([np.nan]), np.array([None], object))[0])


# --------------------------------------------------------------------------- entries


def test_entry_is_the_first_tradeable_row_past_the_age():
    frame = pd.concat(
        [
            rows(
                "a",
                [0, 10, 20, 30],
                [1.0, 1.0, 2.0, 3.0],
                age0=6 * 3600 - 15,
                liquidity=[50.0, 50.0, 0.5, 50.0],
            ),
            rows("b", [0, 10], [1.0, 1.0], age0=100),
        ]
    )
    paths = s.Paths.of(frame)
    e = s.entries(paths, 6 * 3600, FLOOR)
    assert list(e["mint"]) == ["a"]  # b never reaches 6 h
    # 10 s is still under 6 h, 20 s is a dust pool (under the floor): 30 s is the entry
    assert e["ts"].iloc[0] == T0.value // 10**9 + 30


def test_entry_skips_a_row_without_price():
    frame = rows("a", [0, 10], [np.nan, 2.0])
    e = s.entries(s.Paths.of(frame), 3600, FLOOR)
    assert e["price"].iloc[0] == 2.0


def test_guard_groups():
    assert list(s.guard_group(np.array([100.0, 90.0, 89.9, np.nan]))) == [
        "guard-safe",
        "guard-safe",
        "burned<90",
        "burn-unknown",
    ]


# --------------------------------------------------------------------------- outcomes


def outcome(frame, horizons=("1h",), data_end=None, costs=s.FREE, **kw):
    paths = s.Paths.of(frame)
    e = s.entries(paths, 3600, FLOOR)
    end = int(paths.ts.max()) if data_end is None else data_end
    return e.join(s.outcomes(paths, e, horizons, costs, end, **kw))


def test_return_is_taken_at_the_last_row_within_the_horizon():
    out = outcome(rows("a", [0, 1800, 3600, 3700], [1.0, 1.2, 1.5, 9.0]), data_end=10**10)
    assert out["net_1h"].iloc[0] == pytest.approx(0.5, abs=1e-6)
    assert out["gross_1h"].iloc[0] == pytest.approx(0.5)


def test_a_token_that_stopped_trading_is_sold_at_its_last_mark():
    out = outcome(rows("a", [0, 600], [1.0, 0.8]), data_end=10**10)
    assert out["net_1h"].iloc[0] == pytest.approx(-0.2, abs=1e-6)


def test_horizon_past_the_data_is_censored():
    frame = rows("a", [0, 600], [1.0, 2.0])
    out = outcome(frame, data_end=T0.value // 10**9 + 1800)
    assert not out["covered_1h"].iloc[0]
    assert np.isnan(out["net_1h"].iloc[0]) and np.isnan(out["up50_1h"].iloc[0])


def test_target_reached_before_the_stop():
    frame = rows("a", [0, 60, 120, 180], [1.0, 1.6, 0.5, 2.5])
    out = outcome(frame, data_end=10**10)
    assert out["up50_1h"].iloc[0] == 1.0  # +60 % at 60 s, before the -50 % at 120 s
    assert out["up100_1h"].iloc[0] == 0.0  # +150 % only after the stop


def test_stop_before_target_and_targets_after_the_horizon():
    frame = rows("a", [0, 60, 120, 7200], [1.0, 0.6, 2.0, 5.0])
    out = outcome(frame, data_end=10**10)
    assert out["up50_1h"].iloc[0] == 0.0
    late = outcome(rows("b", [0, 60, 7200], [1.0, 1.1, 5.0]), data_end=10**10)
    assert late["up50_1h"].iloc[0] == 0.0  # the doubling comes after the hour


def test_a_drain_hits_the_stop_through_the_pool():
    frame = rows("a", [0, 60, 120], [1.0, 1e6, 1e6], liquidity=[80.0, 0.0, 0.0])
    out = outcome(frame, data_end=10**10, costs=s.Costs())
    assert out["up50_1h"].iloc[0] == 0.0
    assert out["net_1h"].iloc[0] <= -1.0


def test_outcomes_keep_tokens_apart():
    frame = pd.concat([rows("a", [0, 60], [1.0, 1.0]), rows("b", [0, 60], [1.0, 3.0])])
    out = outcome(frame, data_end=10**10).set_index("mint")
    assert out.loc["a", "up50_1h"] == 0.0 and out.loc["b", "up50_1h"] == 1.0
    assert out.loc["a", "net_1h"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- losses & CIs


def test_loss_causes_count_only_losses_and_prefer_the_gradual_first():
    net = np.array([-0.9, -0.5, -0.4, -0.35, 0.2, -0.95])
    sudden = np.array([1.0, 1.0, 0.0, np.nan, 1.0, 0.0])
    gradual = np.array([1.0, 0.0, 0.0, np.nan, 0.0, 1.0])
    out = s.loss_causes(net, sudden, gradual)
    assert out.to_dict() == {"gradual": 2, "sudden": 1, "neither": 1, "unknown": 1}


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1.0, 500)
    lo, hi = s.bootstrap_ci(x, s._mean, samples=500, seed=3)
    assert lo < x.mean() < hi
    assert (lo, hi) == s.bootstrap_ci(x, s._mean, samples=500, seed=3)
    assert hi - lo == pytest.approx(2 * 1.96 / np.sqrt(500), rel=0.25)
    assert s.bootstrap_ci(np.array([1.0]), s._mean) is None
    assert s.bootstrap_ci(np.full(10, 2.0), s._median) == (2.0, 2.0)


def test_summarize_counts_covered_entries_only():
    df = pd.DataFrame(
        {
            "covered_4h": [True, True, True, False],
            "net_4h": [0.5, -0.2, 0.1, np.nan],
            "gross_4h": [0.6, -0.1, 0.2, np.nan],
            "up50_4h": [1.0, 0.0, 0.0, np.nan],
            "up100_4h": [0.0, 0.0, 0.0, np.nan],
        }
    )
    out = s.summarize(df, "4h", samples=200)
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(0.4 / 3)
    assert out["median"] == pytest.approx(0.1)
    assert out["positive"] == pytest.approx(2 / 3) and out["up50"] == pytest.approx(1 / 3)
    assert s.summarize(df.iloc[3:], "4h") == {"n": 0}


# --------------------------------------------------------------------------- warning signs


def test_dump_onset_is_the_first_row_30pc_under_the_running_peak():
    frame = rows("a", [0, 60, 120, 180, 240], [1.0, 2.0, 1.5, 1.39, 0.1])
    paths = s.Paths.of(frame)
    e = s.entries(paths, 3600, FLOOR)
    assert s.dump_onsets(paths, e, "1h")[0] == T0.value // 10**9 + 180


def test_signs_over_the_hour_before():
    secs = np.arange(0, 3601, 300)
    frame = rows(
        "a",
        secs,
        1.0,
        liquidity=np.linspace(100, 50, len(secs)),
        holders=np.linspace(200, 150, len(secs)),
        volume=np.r_[np.full(6, 10.0), np.full(7, 2.0)],
        buy_sell_ratio=0.3,
        hs_top10_net_flow=-5.0,
    )
    paths = s.Paths.of(frame)
    got = s.signs_at(paths, 0, T0.value // 10**9 + 3600)
    assert got["liquidity_trend"] == pytest.approx(-0.5)
    assert got["holder_change"] == pytest.approx(-0.25)
    assert got["volume_decay"] == pytest.approx(0.2)
    assert got["buy_sell_ratio"] == pytest.approx(0.3)
    assert got["top10_net_flow"] == pytest.approx(-5.0)


def test_signs_unknown_without_rows():
    paths = s.Paths.of(rows("a", [0], 1.0))
    assert all(np.isnan(v) for v in s.signs_at(paths, 0, T0.value // 10**9 + 99999).values())


def test_warning_signs_compare_dumps_with_quiet_tokens():
    frames = []
    for i in range(6):  # dumpers: liquidity drains out before the onset
        secs = np.arange(0, 7201, 300)
        price = np.where(secs < 3600, 1.0, 0.5)
        frames.append(
            rows(
                f"d{i}",
                secs,
                price,
                liquidity=np.linspace(100, 40, len(secs)),
                holders=100.0,
                volume=1.0,
                buy_sell_ratio=0.5,
                hs_top10_net_flow=-1.0,
            )
        )
    for i in range(6):  # quiet: flat
        secs = np.arange(0, 7201, 300)
        frames.append(
            rows(
                f"q{i}",
                secs,
                1.0,
                liquidity=100.0,
                holders=100.0,
                volume=1.0,
                buy_sell_ratio=0.5,
                hs_top10_net_flow=1.0,
            )
        )
    paths = s.Paths.of(pd.concat(frames))
    e = s.entries(paths, 3600, FLOOR)
    e = e.join(s.outcomes(paths, e, ("24h",), s.FREE, 10**10))
    e["gradual_24h"] = np.where(e["mint"].str.startswith("d"), 1.0, 0.0)
    e["sudden_24h"] = 0.0
    out = s.warning_signs(paths, e, "24h")
    assert out["dumps"] == 6 and out["controls"] == 6
    assert out["onset_minutes_median"] == pytest.approx(60)
    liq = out["signs"]["liquidity_trend"]
    assert liq["auc"] == 0.0  # dumps' liquidity trend is lower than every quiet token's
    assert out["signs"]["top10_net_flow"]["auc"] == 0.0


# --------------------------------------------------------------------------- study & CLI


def test_study_end_to_end_on_synthetic_paths():
    secs = np.arange(0, 80 * 3600, 600)
    rug = np.sort(np.r_[secs, 3 * 3600 - 10])  # the drain is one 10 s step: sudden
    frames = [
        rows("up", secs, np.linspace(1, 3, len(secs)), age0=3600),
        rows("flat", secs, 1.0, age0=3600, burned_liquidity=np.nan),
        rows(
            "rug",
            rug,
            np.where(rug < 3 * 3600, 1.0, 1e6),
            age0=3600,
            liquidity=np.where(rug < 3 * 3600, 80.0, 0.0),
            burned_liquidity=0.0,
        ),
    ]
    frame = pd.concat(frames)
    for col in s.SIGN_COLUMNS:
        frame[col] = 1.0
    report = s.study(frame, {"labeling": {"liquidity_floor": {"default": 1.0}}}, samples=50)
    one_h = report["ages"]["1h"]
    assert one_h["entries"] == 3
    assert one_h["groups"]["all"]["4h"]["n"] == 3
    assert one_h["groups"]["guard-safe"]["4h"]["n"] == 1
    assert one_h["groups"]["burn-unknown"]["4h"]["n"] == 1
    assert one_h["loss_causes"]["4h"]["sudden"] == 1  # the drain
    assert "VERDICT" not in s.render(report) and "entry at age >= 6h" in s.render(report)


def population() -> pd.DataFrame:
    """Tokens past 6 h: four gradual dumps (price and liquidity sliding over 4 h), four quiet."""
    secs = np.arange(0, 30 * 3600, 60)
    frames = []
    for i in range(4):
        price = np.interp(secs, [0, 2 * 3600, 6 * 3600, 30 * 3600], [1.0, 1.0, 0.3, 0.3])
        liq = np.interp(secs, [0, 2 * 3600, 6 * 3600, 30 * 3600], [100.0, 100.0, 40.0, 40.0])
        frames.append(rows(f"dump{i}", secs, price, liquidity=liq, holders=100.0 - i))
    for i in range(4):
        frames.append(rows(f"quiet{i}", secs, 1.0 + 0.01 * i, liquidity=100.0, holders=100.0))
    frame = pd.concat(frames, ignore_index=True)
    for col in ("volume", "buy_sell_ratio", "hs_top10_net_flow"):
        frame[col] = 1.0
    return frame


CONFIG = {"labeling": {"liquidity_floor": {"default": 1.0}}}


def test_study_finds_the_gradual_dumps_and_their_signs():
    report = s.study(population(), CONFIG, samples=50)
    w = report["warning_signs"]
    assert (w["dumps"], w["controls"]) == (4, 4)
    assert w["signs"]["liquidity_trend"]["auc"] == 0.0
    assert report["ages"]["6h"]["loss_causes"]["24h"]["gradual"] == 4


def test_partitioned_study_matches_the_whole_study(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = population()
    files = []
    for k in range(2):
        part = frame[s.partition_of(frame["mint"], 2) == k]
        path = tmp_path / f"part-{k:02d}.parquet"
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False), path)
        files.append(path)
    end = s.data_end_of(files)
    assert end == int(frame["ts"].max().value // 10**9)
    whole = s.study(frame, CONFIG, samples=50, data_end=end)
    parted = s.study_partitions(files, CONFIG, end, samples=50)
    for age in s.AGES:
        a, b = whole["ages"][age], parted["ages"][age]
        assert a["entries"] == b["entries"]
        assert a["loss_causes"] == b["loss_causes"]
        for h in s.HORIZONS:
            assert a["groups"]["all"][h].get("mean") == pytest.approx(
                b["groups"]["all"][h].get("mean")
            )
    assert whole["warning_signs"] == parted["warning_signs"]
    assert (whole["rows"], whole["tokens"]) == (parted["rows"], parted["tokens"])


def test_partition_of_is_stable_and_bounded():
    mints = pd.Series(["a", "b", "c", "a"])
    k = s.partition_of(mints, 3)
    assert k[0] == k[3] and k.min() >= 0 and k.max() < 3
    assert list(k) == list(s.partition_of(pd.Categorical(mints).astype(str), 3))


def test_control_offsets_are_dealt_in_mint_order():
    out = s.control_offsets(pd.Series(["q2", "q1", "q3"]), np.array([300, 100]))
    assert out.to_dict() == {"q1": 100, "q2": 300, "q3": 100}


def test_cli_arguments():
    args = s.parser().parse_args(["--out", "r.json", "--samples", "10", "--cache-dir", "c"])
    assert args.samples == 10 and args.parts == 8 and args.start is None
    with pytest.raises(SystemExit):
        s.parser().parse_args(["--out", "r.json"])  # --cache-dir is required
    with pytest.raises(SystemExit):
        s.main(["--samples", "0", "--cache-dir", "c"])
    with pytest.raises(SystemExit):
        s.main(["--parts", "0", "--cache-dir", "c"])
    with pytest.raises(SystemExit):
        s.parser().parse_args(["--start", "not-a-date", "--cache-dir", "c"])
