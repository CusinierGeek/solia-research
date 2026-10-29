"""The vectorized dataset build against the per-token loop it replaced (#152): same samples; and
the batched loading that feeds the model.

``legacy_build_samples`` is the pre-#152 implementation, kept verbatim (minus the argument checks)
as the reference: a Python loop over tokens, then over every window and horizon.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import SequentialSampler

from solia_ai.config import horizon_seconds
from solia_ai.dataset import (
    LoaderSettings,
    SequenceDataset,
    auto_num_workers,
    batch_loader,
    build_samples,
    make_loaders,
)
from solia_ai.eligibility import INPUT_COLUMNS, EligibilityRule, frame_inputs
from solia_ai.features import FEATURE_COLUMNS, build_feature_matrix

HORIZONS = ("30s", "1m", "5m")
PUMP = {"30s": 0.1, "1m": 0.25, "5m": 0.5}
RUG = {"30s": 0.2, "1m": 0.4, "5m": 0.5}


def legacy_build_samples(
    df, *, seq_len, horizons, pump_thresholds, rug_thresholds, max_gap_fraction=0.25,
    min_gap_seconds=20, min_seq_len=None, max_return=10.0, eligibility=None,
):  # fmt: skip
    rule = EligibilityRule() if eligibility is None else eligibility
    min_len = seq_len if min_seq_len is None else min_seq_len
    horizon_s = np.array([horizon_seconds(h) for h in horizons], np.int64)
    pump_thr = np.array([pump_thresholds[h] for h in horizons], np.float64)
    rug_thr = np.array([rug_thresholds[h] for h in horizons], np.float64)
    max_gap = np.maximum(horizon_s * max_gap_fraction, min_gap_seconds)
    blocks, starts, ends, labels, end_ts, mints, anchors = [], [], [], [], [], [], []
    offset = 0
    for mint, sub in df.groupby("mint", sort=True):
        if len(sub) < min_len:
            continue
        sort_keys = ["ts", "id"] if "id" in sub.columns else ["ts"]
        sub = sub.sort_values(sort_keys, kind="stable")
        matrix = build_feature_matrix(sub)
        price = pd.to_numeric(sub["price"], errors="coerce").to_numpy(dtype=np.float64)
        liquidity = pd.to_numeric(sub["liquidity"], errors="coerce").to_numpy(dtype=np.float64)
        ts_utc = pd.to_datetime(sub["ts"], utc=True).dt.tz_localize(None)
        ts_sec = ts_utc.to_numpy().astype("datetime64[s]").astype(np.int64)
        window_ends = np.arange(min_len - 1, len(sub))
        pump, rug, ret, mask = legacy_label(
            window_ends, ts_sec, price, liquidity, horizon_s, pump_thr, rug_thr, max_gap, max_return
        )
        keep = mask.any(axis=1)
        anchor_inputs = frame_inputs(sub)[window_ends]
        keep &= rule.accepts(anchor_inputs)
        if not keep.any():
            continue
        blocks.append(matrix)
        starts.append(np.maximum(window_ends[keep] - seq_len + 1, 0) + offset)
        ends.append(window_ends[keep] + offset)
        labels.append((pump[keep], rug[keep], ret[keep], mask[keep]))
        end_ts.append(ts_sec[window_ends[keep]])
        mints.append(np.full(int(keep.sum()), mint, dtype=object))
        anchors.append(anchor_inputs[keep])
        offset += len(matrix)
    return {
        "features": np.concatenate(blocks),
        "start_row": np.concatenate(starts).astype(np.int64),
        "end_row": np.concatenate(ends).astype(np.int64),
        **{
            name: np.concatenate([lab[k] for lab in labels]).astype(np.float32)
            for k, name in enumerate(("pump", "rug", "expected_return", "mask"))
        },
        "end_ts": np.concatenate(end_ts).astype(np.int64),
        "mint": np.concatenate(mints),
        "eligibility_inputs": np.concatenate(anchors).astype(np.float32),
    }


def legacy_label(window_ends, ts_sec, price, liquidity, horizon_s, pump_thr, rug_thr, max_gap,
                 max_return):  # fmt: skip
    n, n_h = len(window_ends), len(horizon_s)
    pump, rug, ret, mask = (np.zeros((n, n_h), np.float32) for _ in range(4))
    last_sec = ts_sec[-1]
    for k, i in enumerate(window_ends):
        p_i, l_i, t_i = price[i], liquidity[i], ts_sec[i]
        if not np.isfinite(p_i) or p_i <= 0:
            continue
        lo = np.searchsorted(ts_sec, t_i, side="right")
        for h in range(n_h):
            horizon_end = t_i + horizon_s[h]
            hi = np.searchsorted(ts_sec, horizon_end, side="right")
            if hi <= lo:
                continue
            future_p = price[lo:hi]
            if not np.isfinite(future_p).any():
                continue
            p_min = np.nanmin(future_p)
            crashed = (p_i - p_min) / p_i >= rug_thr[h]
            if np.isfinite(l_i) and l_i > 0:
                future_l = liquidity[lo:hi]
                if np.isfinite(future_l).any():
                    crashed |= (l_i - np.nanmin(future_l)) / l_i >= rug_thr[h]
            covered = last_sec >= horizon_end and ts_sec[hi - 1] >= horizon_end - max_gap[h]
            if not covered and not crashed:
                continue
            finite = future_p[np.isfinite(future_p)]
            pump[k, h] = float(np.max(finite) / p_i - 1.0 >= pump_thr[h])
            rug[k, h] = float(crashed)
            ret[k, h] = min(finite[-1] / p_i - 1.0, max_return)
            mask[k, h] = 1.0
    return pump, rug, ret, mask


def random_frame(seed: int, n_tokens: int = 40) -> pd.DataFrame:
    """Tokens of random length and cadence: gaps, duplicate timestamps, NaN / zero / huge prices,
    liquidity pulls, NaN eligibility inputs, short tokens; rows shuffled across tokens."""
    rng = np.random.default_rng(seed)
    parts = []
    next_id = 0
    for t in range(n_tokens):
        n = int(rng.integers(1, 120))
        steps = rng.choice([0, 5, 10, 10, 10, 30, 200], size=n)  # 0 = duplicate timestamp
        ts = 1_700_000_000 + int(rng.integers(0, 5000)) + np.cumsum(steps)
        price = np.exp(np.cumsum(rng.normal(0, 0.15, n))) * 1e-6
        price[rng.random(n) < 0.05] = np.nan
        price[rng.random(n) < 0.01] = 0.0
        price[rng.random(n) < 0.01] = 1e3  # a dust-pool artifact
        liquidity = 50 * np.exp(np.cumsum(rng.normal(0, 0.1, n)))
        liquidity[rng.random(n) < 0.1] = np.nan
        if rng.random() < 0.3:  # a pull
            liquidity[int(rng.integers(0, n)) :] *= 0.1
        frame = pd.DataFrame(
            {
                col: rng.normal(size=n).astype(np.float64)
                for col in FEATURE_COLUMNS
                if col not in ("price", "is_migrated")
            }
        )
        frame["is_migrated"] = rng.random(n) < 0.1
        frame["price"] = price
        frame["liquidity"] = liquidity
        for col in INPUT_COLUMNS:
            values = np.cumsum(rng.random(n) * 3)
            values[rng.random(n) < 0.3] = np.nan
            frame[col] = values
        frame["mint"] = f"mint{rng.integers(0, 10**6):06d}-{t}"
        frame["ts"] = pd.to_datetime(ts, unit="s", utc=True)
        frame["id"] = np.arange(next_id, next_id + n)[rng.permutation(n)]
        next_id += n
        parts.append(frame)
    df = pd.concat(parts, ignore_index=True)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def assert_same(new, legacy) -> None:
    for name, expected in legacy.items():
        np.testing.assert_array_equal(getattr(new, name), expected, err_msg=name, strict=True)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    ("seq_len", "min_seq_len", "rule"),
    [
        (16, None, None),
        (16, 4, None),
        (8, 1, EligibilityRule(min_unique_traders=5, min_volume_sol=2.0, min_token_age_sec=20)),
    ],
)
def test_vectorized_build_matches_the_per_token_loop(seed, seq_len, min_seq_len, rule):
    df = random_frame(seed)
    kwargs = dict(
        seq_len=seq_len, horizons=HORIZONS, pump_thresholds=PUMP, rug_thresholds=RUG,
        min_seq_len=min_seq_len, eligibility=rule, max_return=5.0,
    )  # fmt: skip
    new = build_samples(df, **kwargs)
    assert len(new) > 0
    assert_same(new, legacy_build_samples(df, **kwargs))


def test_without_ids_ties_keep_the_input_order():
    df = random_frame(7).drop(columns="id")
    kwargs = dict(seq_len=8, horizons=HORIZONS, pump_thresholds=PUMP, rug_thresholds=RUG,
                  min_seq_len=2)  # fmt: skip
    assert_same(build_samples(df, **kwargs), legacy_build_samples(df, **kwargs))


def test_rows_without_a_mint_are_ignored_like_groupby():
    df = random_frame(3, n_tokens=5)
    df.loc[df.index[:7], "mint"] = None
    kwargs = dict(seq_len=8, horizons=HORIZONS, pump_thresholds=PUMP, rug_thresholds=RUG,
                  min_seq_len=2)  # fmt: skip
    assert_same(build_samples(df, **kwargs), legacy_build_samples(df, **kwargs))


def test_a_too_long_time_span_is_refused():
    df = random_frame(1, n_tokens=2)
    df.loc[df.index[0], "ts"] = pd.Timestamp("1850-01-01", tz="UTC")
    with pytest.raises(ValueError, match="too long a time range"):
        build_samples(df, seq_len=4, horizons=HORIZONS, pump_thresholds=PUMP,
                      rug_thresholds=RUG, min_seq_len=1)  # fmt: skip


# --------------------------------------------------------------------------- batched loading


def test_a_gathered_batch_is_the_default_collation_of_its_items():
    from torch.utils.data import default_collate

    s = build_samples(random_frame(2), seq_len=16, min_seq_len=1, horizons=HORIZONS,
                      pump_thresholds=PUMP, rug_thresholds=RUG)  # fmt: skip
    assert (s.end_row - s.start_row + 1 < 16).any()  # some windows are padded
    ds = SequenceDataset(s, np.arange(len(s))[::-1].copy())
    ks = [0, 5, 3, len(ds) - 1, 5]

    x, padding_mask, targets = ds.batch(ks)
    ex, emask, etargets = default_collate([ds[k] for k in ks])

    assert torch.equal(x, ex) and torch.equal(padding_mask, emask)
    assert targets.keys() == etargets.keys()
    assert all(torch.equal(targets[k], etargets[k]) for k in targets)


def test_loaders_yield_every_item_once_per_epoch_in_sampler_order():
    s = build_samples(random_frame(4), seq_len=8, min_seq_len=1, horizons=HORIZONS,
                      pump_thresholds=PUMP, rug_thresholds=RUG)  # fmt: skip
    ds = SequenceDataset(s)
    _, val_loader = make_loaders(ds, ds, batch_size=7)
    assert len(val_loader) == -(-len(ds) // 7)
    ends = torch.cat([b[0][:, -1, :] for b in val_loader])  # each window's last row, in order
    assert torch.equal(ends, torch.from_numpy(s.features[s.end_row]))


def test_loader_settings_resolve_auto(monkeypatch):
    import solia_ai.dataset as dataset_module

    cfg = {"training": {"num_workers": "auto", "pin_memory": "auto"}}
    monkeypatch.setattr(dataset_module.multiprocessing, "get_start_method", lambda: "spawn")
    assert LoaderSettings.from_config(cfg, "cpu") == LoaderSettings(0, False, 2)

    monkeypatch.setattr(dataset_module.multiprocessing, "get_start_method", lambda: "fork")
    monkeypatch.setattr(dataset_module.os, "sched_getaffinity", lambda _: set(range(6)),
                        raising=False)  # fmt: skip
    assert LoaderSettings.from_config(cfg, "cuda") == LoaderSettings(3, True, 2)
    monkeypatch.setattr(dataset_module.os, "sched_getaffinity", lambda _: set(range(32)),
                        raising=False)  # fmt: skip
    assert auto_num_workers() == 4  # capped
    monkeypatch.setattr(dataset_module.os, "sched_getaffinity", lambda _: {0}, raising=False)
    assert auto_num_workers() == 1

    explicit = {"training": {"num_workers": 2, "pin_memory": False, "prefetch_factor": 4}}
    assert LoaderSettings.from_config(explicit, "cuda") == LoaderSettings(2, False, 4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_workers": -1}, "num_workers must be >= 0"),
        ({"num_workers": True}, "num_workers must be an integer"),
        ({"prefetch_factor": 0}, "prefetch_factor must be >= 1"),
        ({"prefetch_factor": 2.5}, "prefetch_factor must be an integer"),
    ],
)
def test_loader_settings_reject_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LoaderSettings(**kwargs)


def test_batch_loader_rejects_a_non_positive_batch_size():
    ds = SequenceDataset(build_samples(random_frame(0, n_tokens=3), seq_len=4, min_seq_len=1,
                                       horizons=HORIZONS, pump_thresholds=PUMP,
                                       rug_thresholds=RUG))  # fmt: skip
    with pytest.raises(ValueError, match="batch_size"):
        batch_loader(ds, SequentialSampler(ds), batch_size=0)
