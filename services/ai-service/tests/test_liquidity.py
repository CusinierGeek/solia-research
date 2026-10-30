"""The liquidity floor (#166): per-venue floor, labels ignoring dust prices, the diagnostic."""

import numpy as np
import pandas as pd
import pytest
import yaml

from solia_ai.config import ConfigError, load_config
from solia_ai.dataset import build_samples, config_samples
from solia_ai.evaluate import ReferencePolicy
from solia_ai.features import FEATURE_COLUMNS, VENUE_COLUMNS
from solia_ai.liquidity import CURVE_VENUES, LiquidityFloor, liquidity_buckets
from solia_ai.liquidity_report import bucket_report, build_report, labeling_config, render
from solia_ai.train import hyperparams_snapshot

VALID = load_config()  # the packaged config
PUMP = {"1m": 0.25}
RUG = {"1m": 0.4}
TS = list(range(0, 130, 10))  # a snapshot every 10 s, like the pipeline


def frame(prices, ts_seconds, liquidity=None) -> pd.DataFrame:
    rows = []
    for k, (t, p) in enumerate(zip(ts_seconds, prices, strict=True)):
        row = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        row["is_migrated"] = False
        row["price"] = p
        row["liquidity"] = None if liquidity is None else liquidity[k]
        row["mint"] = "M"
        row["ts"] = pd.Timestamp(t, unit="s", tz="UTC")
        rows.append(row)
    return pd.DataFrame(rows)


def label_at(s, t, name="pump", h=0):
    """The ``name`` label (and mask) of the window ending at epoch second ``t``."""
    (i,) = np.where(s.end_ts == t)[0]
    return getattr(s, name)[i, h], s.mask[i, h]


FLOOR = LiquidityFloor.from_mapping({"default": 1.0, "pumpfun": 0.0})


def samples(df, floor=FLOOR, **kwargs):
    return build_samples(
        df,
        seq_len=1,
        horizons=("1m",),
        pump_thresholds=PUMP,
        rug_thresholds=RUG,
        liquidity_floor=floor,
        **kwargs,
    )


def with_venue(df, venue):
    """``df`` with a raw ``venue`` column in place of its one-hot encoding."""
    return df.drop(columns=list(VENUE_COLUMNS)).assign(venue=venue)


# --- the floor per venue ---


def test_a_row_is_tradeable_at_or_above_its_venue_s_floor():
    floor = LiquidityFloor.from_mapping({"default": 1.0, "pumpfun": 0.0, "Raydium ": 5.0})
    liquidity = np.array([0.5, 1.0, 0.0, np.nan, 4.9, 5.0, 0.5, np.nan])
    venue = ["pumpswap", "pumpswap", "pumpfun", "pumpfun", "raydium", "RAYDIUM", None, "meteora"]

    ok = floor.tradeable(liquidity, venue)

    # pumpfun has floor 0: any liquidity, even unknown; NULL / unlisted venues take the default.
    assert ok.tolist() == [False, True, True, True, False, True, False, False]
    assert floor.floor_of("raydium") == 5.0 and floor.floor_of(None) == 1.0
    assert floor.floor_of("orca") == 1.0 and floor.floor_of(" PumpFun") == 0.0


def test_without_a_venue_column_every_row_takes_the_default():
    ok = FLOOR.tradeable(np.array([0.5, 2.0]), None)

    assert ok.tolist() == [False, True]


def test_a_disabled_floor_accepts_everything_and_is_the_default_of_old_models():
    disabled = LiquidityFloor()

    assert not disabled.enabled and disabled.describe() == "disabled (every pool)"
    assert disabled.tradeable(np.array([np.nan, 0.0]), ["raydium", None]).all()
    assert LiquidityFloor.of(None) == disabled
    assert LiquidityFloor.of({"eligibility": {}}) == disabled  # a model from before the floor
    assert LiquidityFloor.from_mapping({"default": 0.0, "pumpfun": 0.0}).enabled is False


def test_the_floor_round_trips_through_the_recorded_hyperparams():
    cfg = {**VALID, "labeling": {**VALID["labeling"], "liquidity_floor": {"default": 2, "x": 0}}}

    recorded = hyperparams_snapshot(cfg)["liquidity_floor"]

    assert recorded == {"default": 2.0, "x": 0.0}
    assert LiquidityFloor.of({"liquidity_floor": recorded}) == LiquidityFloor.from_config(cfg)
    assert "2 SOL (x 0)" in LiquidityFloor.from_config(cfg).describe()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([1.0], "must be a mapping"),
        ({"default": -1.0}, r"'labeling.liquidity_floor.default' must be a finite number >= 0"),
        ({"default": float("nan")}, "must be a finite number >= 0"),
        ({"default": float("inf")}, "must be a finite number >= 0"),
        ({"raydium": "1"}, r"'labeling.liquidity_floor.raydium' must be a number"),
        ({"raydium": True}, "must be a number"),
        ({"raydium": 1.0, "Raydium": 2.0}, "lists venue 'raydium' twice"),
        ({1: 1.0}, "keys must be venue names"),
    ],
)
def test_an_invalid_floor_is_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        LiquidityFloor.from_mapping(values)


def test_an_invalid_floor_fails_the_config_load(tmp_path):
    cfg = {**VALID, "labeling": {**VALID["labeling"], "liquidity_floor": {"default": -1}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ConfigError, match="liquidity_floor.default"):
        load_config(path)


def test_the_shipped_floor_exempts_the_bonding_curves_only():
    floor = LiquidityFloor.from_config(VALID)

    assert floor.default == 1.0
    assert all(floor.floor_of(v) == 0.0 for v in CURVE_VENUES)
    assert all(floor.floor_of(v) == 1.0 for v in ("pumpswap", "raydium", "meteora", "orca", None))


# --- labels and windows ---


def test_a_window_anchored_below_the_floor_is_dropped():
    liquidity = [0.5, 0.5, 2.0] + [2.0] * 10
    df = frame([1.0] * 13, TS, liquidity=liquidity)

    s = samples(df)

    assert TS[0] not in s.end_ts and TS[1] not in s.end_ts and TS[2] in s.end_ts
    assert len(samples(df, floor=LiquidityFloor())) == len(s) + 2  # disabled: every window


def test_an_unknown_anchor_liquidity_is_dropped_unless_the_venue_is_exempt():
    df = frame([1.0] * 13, TS, liquidity=[None] * 13)

    assert len(samples(df)) == 0
    assert len(samples(with_venue(df, "pumpfun"))) == len(samples(df, floor=LiquidityFloor()))


def test_a_curve_token_with_little_real_sol_is_not_excluded():
    prices = [1.0, 1.0, 3.0] + [3.0] * 10  # a real pump on the curve, priced off virtual reserves
    df = with_venue(frame(prices, TS, liquidity=[0.01] * 13), "pumpfun")

    s = samples(df)

    assert label_at(s, 0, "pump") == (1.0, 1.0)
    assert label_at(s, 0, "expected_return")[0] == pytest.approx(2.0)


def test_a_pump_read_off_a_dust_pool_does_not_count():
    # The pool is drained to dust; its last micro-trade prints 6x the anchor.
    prices = [1.0, 1.0, 6.0, 1.1] + [1.1] * 9
    liquidity = [5.0, 5.0, 0.02, 5.0] + [5.0] * 9
    df = with_venue(frame(prices, TS, liquidity=liquidity), "raydium")

    assert label_at(samples(df), 0, "pump") == (0.0, 1.0)
    assert label_at(samples(df, floor=LiquidityFloor()), 0, "pump") == (1.0, 1.0)  # before


def test_the_return_skips_dust_prices_back_to_the_last_tradeable_one():
    prices = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 9.0] + [9.0] * 6  # horizon end read off dust
    liquidity = [5.0] * 6 + [0.1] * 7
    df = with_venue(frame(prices, TS, liquidity=liquidity), "pumpswap")

    s = samples(df)

    assert label_at(s, 0, "expected_return") == (pytest.approx(0.6), 1.0)  # price at t=50
    assert label_at(s, 0, "pump")[0] == 1.0  # 1.6 at t=50 is a tradeable +60 % pump
    without = samples(df, floor=LiquidityFloor())
    assert label_at(without, 0, "expected_return")[0] == pytest.approx(8.0)


def test_the_anchor_is_the_return_fallback_when_the_whole_horizon_is_dust():
    prices = [1.0] + [7.0] * 12
    liquidity = [5.0] + [0.1] * 12
    df = frame(prices, TS, liquidity=liquidity)

    assert label_at(samples(df), 0, "expected_return") == (pytest.approx(0.0), 1.0)
    assert label_at(samples(df), 0, "pump")[0] == 0.0


def test_a_liquidity_collapse_into_dust_is_still_a_rug():
    prices = [1.0] * 13
    liquidity = [100.0, 100.0, 0.01] + [0.01] * 10
    df = with_venue(frame(prices, TS, liquidity=liquidity), "pumpswap")

    s = samples(df)

    assert label_at(s, 0, "rug") == (1.0, 1.0)
    assert label_at(samples(df, floor=LiquidityFloor()), 0, "rug") == (1.0, 1.0)


def test_a_price_crash_read_off_dust_is_still_a_rug():
    prices = [1.0, 1.0, 0.1] + [0.1] * 10
    liquidity = [5.0, 5.0, 0.5] + [0.5] * 10

    assert label_at(samples(frame(prices, TS, liquidity=liquidity)), 0, "rug") == (1.0, 1.0)


def test_samples_record_the_anchor_liquidity_and_venue():
    df = with_venue(frame([1.0] * 13, TS, liquidity=[3.0] * 13), "meteora")

    s = samples(df)

    assert np.all(s.anchor_liquidity == 3.0) and set(s.anchor_venue) == {"meteora"}


def test_the_floor_needs_a_liquidity_column():
    df = frame([1.0] * 13, TS).drop(columns="liquidity")

    with pytest.raises(ValueError, match="'liquidity' column"):
        samples(df)


def test_config_samples_reads_the_floor_from_labeling():
    df = frame([1.0] * 13, TS, liquidity=[0.5, 0.5] + [2.0] * 11)
    labeling = {"pump_threshold": PUMP, "rug_threshold": RUG}
    cfg = {"model": {"seq_len": 1}, "horizons": ["1m"], "labeling": labeling}

    floored = config_samples(
        df, {**cfg, "labeling": {**labeling, "liquidity_floor": {"default": 1}}}
    )

    assert len(config_samples(df, cfg)) == len(floored) + 2


# --- diagnostic by liquidity bucket ---


def test_buckets_split_curve_and_amm_pools_by_liquidity():
    liquidity = np.array([0.5, 5.0, 50.0, np.nan, 0.5, 10.0, 1.0])
    venue = ["raydium", "pumpswap", None, "meteora", "pumpfun", "Bonk", "raydium-launchpad"]

    buckets = liquidity_buckets(liquidity, venue)

    assert buckets.tolist() == [
        "amm <1", "amm 1-10", "amm >=10", "amm unknown", "curve <1", "curve >=10", "curve 1-10",
    ]  # fmt: skip
    assert liquidity_buckets(np.array([0.5]), None).tolist() == ["amm <1"]


def report_fixture():
    # 6 windows, 1 horizon: two dust windows hold the only pump and the only trade.
    truth = {
        "pump": np.array([[1], [0], [0], [0], [1], [0]], np.float32),
        "rug": np.array([[0], [1], [0], [1], [0], [0]], np.float32),
        "expected_return": np.array([[2.0], [-0.5], [0.1], [-0.2], [0.3], [0.0]], np.float32),
        "mask": np.ones((6, 1), np.float32),
    }
    predictions = {
        "pump_prob": np.array([[0.9], [0.1], [0.2], [0.1], [0.05], [0.3]]),
        "rug_prob": np.array([[0.1], [0.8], [0.1], [0.7], [0.1], [0.1]]),
        "expected_return": np.zeros((6, 1)),
        "confidence": np.full((6, 1), 80.0),
    }
    buckets = np.array(["amm <1", "amm <1", "amm >=10", "amm >=10", "amm >=10", "amm >=10"])
    return predictions, truth, buckets


def test_the_bucket_report_shows_where_the_signal_lives():
    predictions, truth, buckets = report_fixture()
    policy = ReferencePolicy(horizon="1m", pump_threshold=0.5, rug_threshold=0.2)

    out = bucket_report(
        predictions, truth, buckets, ["1m"], policy, bootstrap_samples=50, sweep={"min_trades": 1}
    )

    assert list(out) == ["all", "amm <1", "amm >=10"]
    dust, deep = out["amm <1"], out["amm >=10"]
    assert dust["share"] == {
        "windows": pytest.approx(1 / 3),
        "pumps": 0.5,
        "trades": 1.0,
        "pnl": 1.0,
    }
    assert dust["rule"]["n_trades"] == 1 and dust["rule"]["precision"] == 1.0
    assert deep["rule"]["n_trades"] == 0 and deep["share"]["trades"] == 0.0
    assert deep["pump"]["1m"]["base_rate"] == 0.25 and dust["pump"]["1m"]["base_rate"] == 0.5
    assert dust["pump"]["1m"]["pr_auc"] == 1.0  # the dust pump is ranked first
    assert deep["pump"]["1m"]["roc_auc"] == pytest.approx(0.0)  # the real pump is ranked last
    assert out["all"]["n_windows"] == 6 and out["all"]["rule"]["n_trades"] == 1
    assert deep["sweep"]["recommended"] is not None

    report = build_report("v1", "none", {}, policy, out, ["1m"])
    text = render(report)
    assert "amm <1" in text and "Entry rule @ 1m" in text and "floor disabled" in text


def test_the_bucket_report_rejects_misaligned_inputs():
    predictions, truth, buckets = report_fixture()
    policy = ReferencePolicy(horizon="1m")

    with pytest.raises(ValueError, match="one row per window"):
        bucket_report(predictions, truth, buckets[:3], ["1m"], policy)
    with pytest.raises(ValueError, match="not one of the model's"):
        bucket_report(predictions, truth, buckets, ["5m"], policy)


def test_the_report_labels_with_or_without_the_config_floor():
    assert "liquidity_floor" in labeling_config(VALID, "config")["labeling"]
    stripped = labeling_config(VALID, "none")
    assert "liquidity_floor" not in stripped["labeling"]
    assert "liquidity_floor" in VALID["labeling"]  # the config itself is untouched
    assert not LiquidityFloor.from_config(stripped).enabled
    with pytest.raises(ValueError, match="labels must be one of"):
        labeling_config(VALID, "model")


def test_a_frame_with_categorical_venues_is_floored_like_strings():
    df = with_venue(frame([1.0] * 13, TS, liquidity=[0.5] * 13), pd.Categorical(["pumpfun"] * 13))

    assert len(samples(df)) == len(samples(df, floor=LiquidityFloor()))


def test_a_retrain_notes_a_champion_trained_under_another_floor():
    from solia_ai.retrain import liquidity_floor_note

    checkpoint = {"version": "v0", "hyperparams": {"eligibility": {}}}  # before the floor

    note = liquidity_floor_note(checkpoint, VALID)

    assert "v0" in note and "disabled (every pool)" in note and "1 SOL" in note
    recorded = {**checkpoint, "hyperparams": hyperparams_snapshot(VALID)}
    assert liquidity_floor_note(recorded, VALID) is None


def test_several_models_share_the_windows_after_all_their_data():
    from solia_ai.liquidity_report import heldout_after, report_path

    assert heldout_after({"train_end_ts": 10, "calibration_end_ts": 30, "data_end_ts": 20}) == 30
    assert heldout_after({"train_end_ts": 10, "data_end_ts": None}) == 10
    with pytest.raises(ValueError, match="no train_end_ts"):
        heldout_after({})
    assert report_path("r/liq.json", "v1", several=False).as_posix() == "r/liq.json"
    assert report_path("r/liq.json", "v1", several=True).as_posix() == "r/liq-v1.json"
