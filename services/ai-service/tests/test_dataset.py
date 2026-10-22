import numpy as np
import pandas as pd
import pytest
import torch

from solia_ai.dataset import (
    MAX_POS_WEIGHT,
    Samples,
    SequenceDataset,
    balanced_sampler,
    build_samples,
    horizon_seconds,
    make_datasets,
    make_loaders,
    pad_window,
    positive_weights,
    temporal_split,
)
from solia_ai.features import FEATURE_COLUMNS

PUMP = {"1m": 0.25}
RUG = {"1m": 0.4}
PRICE = FEATURE_COLUMNS.index("price")


def frame(prices, ts_seconds, mint="M", liquidity=None) -> pd.DataFrame:
    rows = []
    for k, (t, p) in enumerate(zip(ts_seconds, prices, strict=True)):
        row = dict.fromkeys(FEATURE_COLUMNS, 0.0)
        row["is_migrated"] = False
        row["price"] = p
        row["liquidity"] = None if liquidity is None else liquidity[k]
        row["mint"] = mint
        row["ts"] = pd.Timestamp(t, unit="s", tz="UTC")
        rows.append(row)
    return pd.DataFrame(rows)


def samples(df, horizons=("1m",), pump=PUMP, rug=RUG, seq_len=1, **kwargs):
    return build_samples(
        df, seq_len=seq_len, horizons=horizons, pump_thresholds=pump, rug_thresholds=rug, **kwargs
    )


def label_at(s, t, name="pump", h=0):
    """The ``name`` label (and mask) of the window ending at epoch second ``t``."""
    (i,) = np.where(s.end_ts == t)[0]
    return getattr(s, name)[i, h], s.mask[i, h]


TS = list(range(0, 130, 10))  # a snapshot every 10 s, like the pipeline


def test_horizon_seconds():
    assert horizon_seconds("30s") == 30
    assert horizon_seconds("1m") == 60
    assert horizon_seconds("5m") == 300
    assert horizon_seconds("1h") == 3600
    for bad in ("1x", "m", "", "abc", "0s"):
        with pytest.raises(ValueError):
            horizon_seconds(bad)


# --- labels ---


def test_pump_uses_the_highest_price_in_the_horizon_not_just_the_last():
    prices = [
        1.0,
        1.0,
        3.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]  # spike that reverses
    s = samples(frame(prices, TS))

    assert label_at(s, 0) == (1.0, 1.0)  # +200% at t=20 counts as a pump
    assert label_at(s, 0, "expected_return") == (pytest.approx(0.0), 1.0)  # but it ended flat


def test_pump_and_rug_can_both_happen_in_one_horizon():
    prices = [1.0, 0.5, 1.3] + [1.3] * 10
    s = samples(frame(prices, TS))

    assert label_at(s, 0, "pump")[0] == 1.0  # reached +30%
    assert label_at(s, 0, "rug")[0] == 1.0  # after a -50% drawdown


def test_liquidity_pull_is_a_rug_even_when_price_barely_moves():
    liquidity = [100.0, 100.0, 5.0] + [5.0] * 10
    s = samples(frame([1.0] * 13, TS, liquidity=liquidity))

    assert label_at(s, 0, "rug") == (1.0, 1.0)


def test_unusable_liquidity_anchor_falls_back_to_price_only():
    liquidity = [0.0, 100.0, 5.0] + [5.0] * 10  # anchor 0 -> liquidity ignored
    s = samples(frame([1.0] * 13, TS, liquidity=liquidity))

    assert label_at(s, 0, "rug") == (0.0, 1.0)


def test_expected_return_is_the_price_at_the_horizon_end():
    prices = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6] + [1.6] * 6
    s = samples(frame(prices, TS))

    assert label_at(s, 0, "expected_return")[0] == pytest.approx(0.6)  # price at t=60


def test_a_token_that_collapses_then_stops_trading_is_labeled_rug():
    # Last snapshot is the crash; nothing after it (the pipeline stops tracking dead tokens).
    s = samples(frame([1.0, 1.0, 1.0, 0.2], [0, 10, 20, 30]))

    assert label_at(s, 0, "rug") == (1.0, 1.0)  # rug already happened inside the horizon
    assert label_at(s, 0, "expected_return")[0] == pytest.approx(-0.8)
    assert label_at(s, 20, "rug") == (1.0, 1.0)


def test_a_series_that_ends_early_without_a_crash_is_masked():
    s = samples(frame([1.0, 1.1, 1.2, 1.3], [0, 10, 20, 30]))

    assert len(s) == 0  # outcome unknown for every window


def test_a_gap_never_passes_a_stale_price_off_as_the_outcome():
    # Rows at 0 and 5, then nothing until 200: the last row inside (0, 60] is 55 s stale.
    s = samples(frame([1.0, 1.0, 9.0], [0, 5, 200]))

    assert len(s) == 0


def test_a_crash_inside_a_gap_is_still_labeled():
    s = samples(frame([1.0, 0.3, 1.0], [0, 5, 200]))

    assert label_at(s, 0, "rug") == (1.0, 1.0)


def test_normal_snapshot_spacing_is_not_a_gap_for_short_horizons():
    pump = {"30s": 0.25}
    rug = {"30s": 0.4}
    # 10 s spacing: the last row in (0, 30] is at 30, but for t=5-ish offsets it can be 10 s old.
    s = samples(frame([1.0] * 13, [t + (5 if t else 0) for t in TS]), ("30s",), pump, rug)

    assert s.mask[:, 0].all()


def test_partial_horizon_is_masked_per_horizon():
    pump = {"1m": 0.25, "5m": 0.5}
    rug = {"1m": 0.4, "5m": 0.5}
    s = samples(frame([1.0] * 13, TS), ("1m", "5m"), pump, rug)

    assert s.mask[:, 0].sum() > 0  # 1m labelable early on
    assert s.mask[:, 1].sum() == 0  # 5m never complete over a 120 s span


@pytest.mark.parametrize("anchor", [float("nan"), 0.0, -1.0])
def test_unusable_anchor_price_is_skipped(anchor):
    s = samples(frame([anchor] + [1.0] * 12, TS))

    assert 0 not in s.end_ts.tolist()


def test_missing_threshold_for_a_horizon_is_an_error():
    with pytest.raises(ValueError, match="no rug threshold for horizon\\(s\\): 5m"):
        samples(frame([1.0] * 13, TS), ("1m", "5m"), {"1m": 0.2, "5m": 0.3}, {"1m": 0.4})


@pytest.mark.parametrize(
    "kwargs", [{"seq_len": 0}, {"max_gap_fraction": 0}, {"min_gap_seconds": -1}, {"horizons": ()}]
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        samples(frame([1.0] * 13, TS), **kwargs)


def test_frame_without_mint_or_ts_is_rejected():
    with pytest.raises(ValueError, match="'mint' and 'ts'"):
        samples(frame([1.0] * 13, TS).drop(columns=["mint"]))


# --- windows & memory ---


def test_interleaved_mints_never_share_a_window():
    a = frame([1.0] * 13, TS, mint="A")
    b = frame([5.0] * 13, TS, mint="B")
    mixed = pd.concat([a, b]).sample(frac=1.0, random_state=1)  # shuffled together

    s = samples(mixed, seq_len=3)

    for i in range(len(s)):
        window_prices = set(s.window(i)[:, PRICE].tolist())
        assert window_prices == {1.0 if s.mint[i] == "A" else 5.0}


def test_windows_are_views_over_rows_stored_once():
    s = samples(frame([1.0 + i / 100 for i in range(13)], TS), seq_len=4)

    assert s.features.shape == (13, len(FEATURE_COLUMNS))  # not N * seq_len rows
    assert s.x.shape == (len(s), 4, len(FEATURE_COLUMNS))
    assert np.shares_memory(s.window(0), s.features)
    np.testing.assert_array_equal(s.window(0)[:, PRICE], np.float32([1.0, 1.01, 1.02, 1.03]))


def test_young_tokens_get_short_windows_down_to_min_seq_len():
    prices = [1.0 + i / 100 for i in range(13)]
    s = samples(frame(prices, TS), seq_len=4, min_seq_len=2)

    lengths = [len(s.window(i)) for i in range(len(s))]
    assert lengths[:3] == [2, 3, 4]  # grows with the token's history, then capped at seq_len
    assert max(lengths) == 4
    x, mask = s.padded(0)
    assert mask.tolist() == [True, True, False, False]  # left padding
    np.testing.assert_array_equal(x[2:, PRICE], np.float32([1.0, 1.01]))
    assert (x[:2] == 0).all()


def test_tokens_shorter_than_min_seq_len_are_skipped():
    s = samples(frame([1.0] * 13, TS), seq_len=20, min_seq_len=14)

    assert len(s) == 0


def test_min_seq_len_defaults_to_full_windows_and_is_validated():
    assert min(len(w) for w in map(samples(frame([1.0] * 13, TS), seq_len=4).window, range(3))) == 4
    with pytest.raises(ValueError, match="min_seq_len"):
        samples(frame([1.0] * 13, TS), seq_len=4, min_seq_len=5)


def test_pad_window_left_pads_and_truncates_to_the_most_recent_rows():
    window = np.arange(6, dtype=np.float32).reshape(3, 2)

    x, mask = pad_window(window, 5)
    assert mask.tolist() == [True, True, False, False, False]
    np.testing.assert_array_equal(x[2:], window)

    x, mask = pad_window(window, 2)
    assert not mask.any()
    np.testing.assert_array_equal(x, window[-2:])

    with pytest.raises(ValueError, match="non-empty"):
        pad_window(np.empty((0, 2), np.float32), 4)


def test_make_datasets_uses_min_seq_len_from_config():
    ts = list(range(0, 1200, 10))
    cfg = {**CFG, "model": {"seq_len": 6, "min_seq_len": 2}}

    train, _ = make_datasets(frame([1.0] * len(ts), ts), cfg, val_fraction=0.3)

    assert min(len(train.samples.window(i)) for i in train.indices) == 2


def test_duplicate_timestamps_give_the_same_samples_in_any_input_order():
    df = frame([1.0, 2.0, 1.5] + [1.5] * 11, [0, 10, 10] + TS[2:])
    df["id"] = range(len(df))

    first = samples(df, seq_len=2)
    second = samples(df.iloc[::-1], seq_len=2)

    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.pump, second.pump)


def test_empty_result_has_the_right_shapes():
    s = samples(frame([1.0, 1.0], [0, 1]), seq_len=5)

    assert len(s) == 0
    assert s.x.shape == (0, 5, len(FEATURE_COLUMNS))
    assert s.pump.shape == (0, 1)


# --- split ---


def test_temporal_split_with_embargo():
    end_ts = np.arange(0, 100, 10, dtype=np.int64)  # 0,10,...,90

    train, val = temporal_split(end_ts, val_fraction=0.2, embargo_seconds=15)

    # cutoff = 0.8-quantile = 72; val > 72; train <= 57; the 60/70 samples are embargoed
    assert set(val.tolist()) == {8, 9}
    assert set(train.tolist()) == {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_temporal_split_rejects_a_bad_fraction(fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        temporal_split(np.arange(10), val_fraction=fraction)


def test_temporal_split_rejects_a_negative_embargo():
    with pytest.raises(ValueError, match="embargo"):
        temporal_split(np.arange(10), embargo_seconds=-1)


CFG = {
    "model": {"seq_len": 2},
    "horizons": ["1m"],
    "labeling": {"pump_threshold": {"1m": 0.25}, "rug_threshold": {"1m": 0.4}},
}


def test_make_datasets_applies_the_longest_horizon_as_embargo():
    ts = list(range(0, 1200, 10))
    train, val = make_datasets(frame([1.0] * len(ts), ts), CFG, val_fraction=0.3)

    train_end = train.samples.end_ts[train.indices]
    val_end = val.samples.end_ts[val.indices]
    assert len(train) > 0 and len(val) > 0
    assert val_end.min() - train_end.max() > 60  # no train look-ahead reaches validation


def test_make_datasets_fails_clearly_when_the_span_is_too_short():
    ts = list(range(0, 130, 10))  # 2 minutes of data vs a 1 h embargo
    cfg = {
        "model": {"seq_len": 2},
        "horizons": ["1m", "1h"],
        "labeling": {
            "pump_threshold": {"1m": 0.25, "1h": 2.0},
            "rug_threshold": {"1m": 0.4, "1h": 0.7},
        },
    }

    with pytest.raises(ValueError, match="empty split.*3600s embargo"):
        make_datasets(frame([1.0] * len(ts), ts), cfg)


# --- torch dataset & class imbalance ---


def labeled(pump, rug, mask=None) -> SequenceDataset:
    """A dataset over synthetic labels (one horizon), for weighting/sampling tests."""
    n = len(pump)
    synthetic = Samples(
        features=np.zeros((n, len(FEATURE_COLUMNS)), np.float32),
        start_row=np.arange(n, dtype=np.int64),
        end_row=np.arange(n, dtype=np.int64),
        seq_len=1,
        pump=np.asarray(pump, np.float32).reshape(n, 1),
        rug=np.asarray(rug, np.float32).reshape(n, 1),
        expected_return=np.zeros((n, 1), np.float32),
        mask=(
            np.ones((n, 1), np.float32)
            if mask is None
            else np.asarray(mask, np.float32).reshape(n, 1)
        ),
        end_ts=np.arange(n, dtype=np.int64),
        mint=np.full(n, "M", dtype=object),
        horizons=("1m",),
    )
    return SequenceDataset(synthetic)


def test_sequence_dataset_yields_tensors():
    s = samples(frame([1.0] * 13, TS), seq_len=2)

    ds = SequenceDataset(s)
    x, padding_mask, targets = ds[0]

    assert x.shape == (2, len(FEATURE_COLUMNS))
    assert x.dtype == torch.float32
    assert padding_mask.dtype == torch.bool and not padding_mask.any()
    assert set(targets) == {"pump", "rug", "expected_return", "mask"}
    assert targets["pump"].shape == (1,)


def test_positive_weights_are_negatives_over_positives_on_labeled_samples():
    ds = labeled(pump=[1, 0, 0, 0, 0], rug=[1, 1, 0, 0, 0], mask=[1, 1, 1, 1, 0])

    weights = positive_weights(ds)

    assert weights["pump"].tolist() == [3.0]  # 3 labeled negatives / 1 positive
    assert weights["rug"].tolist() == [1.0]  # 2 / 2


def test_positive_weights_default_to_one_without_positives_and_are_capped():
    assert positive_weights(labeled(pump=[0] * 5, rug=[0] * 5))["pump"].tolist() == [1.0]
    big = labeled(pump=[1] + [0] * 5000, rug=[0] * 5001)
    assert positive_weights(big)["pump"].tolist() == [MAX_POS_WEIGHT]


def test_balanced_sampler_draws_rare_events_about_half_the_time():
    ds = labeled(pump=[1] * 10 + [0] * 990, rug=[0] * 1000)

    draws = list(balanced_sampler(ds, seed=7))
    event_share = np.mean([i < 10 for i in draws])

    assert 0.4 < event_share < 0.6  # vs 1% with uniform sampling
    assert draws == list(balanced_sampler(ds, seed=7))  # reproducible


def test_balanced_sampler_is_uniform_for_a_single_class_and_rejects_empty():
    sampler = balanced_sampler(labeled(pump=[0] * 4, rug=[0] * 4))
    assert torch.equal(sampler.weights, torch.ones(4, dtype=torch.double))

    with pytest.raises(ValueError, match="empty"):
        balanced_sampler(labeled(pump=[], rug=[]))


def test_make_loaders_batches_train_balanced_and_val_in_order():
    ts = list(range(0, 1200, 10))
    train, val = make_datasets(frame([1.0] * len(ts), ts), CFG, val_fraction=0.3)

    train_loader, val_loader = make_loaders(train, val, batch_size=8)
    x, padding_mask, targets = next(iter(train_loader))

    assert x.shape == (8, 2, len(FEATURE_COLUMNS))
    assert padding_mask.shape == (8, 2)
    assert targets["mask"].shape == (8, 1)
    assert isinstance(train_loader.sampler, torch.utils.data.WeightedRandomSampler)
    assert list(val_loader.sampler) == list(range(len(val)))  # unshuffled


def test_make_loaders_can_shuffle_unbalanced_and_validates_batch_size():
    ts = list(range(0, 1200, 10))
    train, val = make_datasets(frame([1.0] * len(ts), ts), CFG, val_fraction=0.3)

    train_loader, _ = make_loaders(train, val, batch_size=4, balanced=False)
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)

    with pytest.raises(ValueError, match="batch_size"):
        make_loaders(train, val, batch_size=0)
