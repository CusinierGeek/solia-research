"""#202: a narrower training read (enabled groups' columns only, a narrow feature matrix) and a
searchable lookback — same windows, same model inputs, the existing behaviour unchanged."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch
from test_train import make_features_df, small_config

import solia_ai.ablation as ablation
import solia_ai.search as search
from solia_ai.config import ConfigError, validate_config
from solia_ai.dataset import (
    SequenceDataset,
    build_samples,
    check_not_pruned,
    config_samples,
    temporal_split,
)
from solia_ai.db import REQUIRED_COLUMNS, _check_columns, training_columns, with_prune_columns
from solia_ai.features import (
    BASELINE_GROUPS,
    DB_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    VENUE_COLUMNS,
    build_feature_matrix,
    group_columns,
)
from solia_ai.model import SoliaTransformer
from solia_ai.search import LookbackTokens, apply_params, read_lookback, run_search
from solia_ai.train import fit

GROUPS = ["price", "liquidity", "flow", "holders", "context", "momentum", "concentration"]


def pruned(cfg: dict, groups=GROUPS) -> dict:
    return with_prune_columns({**cfg, "features": {"groups": list(groups)}}, True)


def full(cfg: dict, groups=GROUPS) -> dict:
    return {**cfg, "features": {"groups": list(groups)}}


# --- the feature matrix ---


def test_a_narrow_matrix_holds_the_same_values_as_those_columns_of_the_full_one():
    df = make_features_df(n_tokens=2, n_rows=20)
    columns = group_columns(GROUPS)
    rows = np.arange(len(df))[::-1]

    narrow = build_feature_matrix(df, rows=rows, columns=columns)

    whole = build_feature_matrix(df, rows=rows)
    index = [FEATURE_COLUMNS.index(c) for c in columns]
    assert narrow.dtype == np.float32 and narrow.shape == (len(df), len(columns))
    np.testing.assert_array_equal(narrow, whole[:, index])


def test_a_narrow_matrix_needs_only_its_own_columns():
    df = make_features_df(n_tokens=1, n_rows=10)
    columns = group_columns(["price", "liquidity"])
    slim = df[["mint", "ts", *columns]]

    assert build_feature_matrix(slim, columns=columns).shape == (10, len(columns))
    with pytest.raises(ValueError, match="missing column"):
        build_feature_matrix(slim)  # the full projection still needs every column


def test_a_narrow_matrix_encodes_the_venue_only_when_asked():
    df = make_features_df(n_tokens=1, n_rows=4).drop(columns=list(VENUE_COLUMNS))
    df["venue"] = "pump"
    columns = [*group_columns(["price"]), *VENUE_COLUMNS]

    with_venue = build_feature_matrix(df, columns=columns)
    without = build_feature_matrix(df, columns=group_columns(["price"]))

    np.testing.assert_array_equal(with_venue, build_feature_matrix(df)[:, _index(columns)])
    assert without.shape == (4, len(group_columns(["price"])))


@pytest.mark.parametrize("columns", [[], ["not_a_column"], ["volume", "price"]])
def test_matrix_columns_must_be_an_ordered_subset_of_the_spec(columns):
    with pytest.raises(ValueError, match="ordered subset"):
        build_feature_matrix(make_features_df(n_tokens=1, n_rows=4), columns=columns)


def _index(columns):
    return [FEATURE_COLUMNS.index(c) for c in columns]


# --- the windows ---


def test_narrow_windows_are_served_like_the_full_ones_with_the_other_columns_zero():
    df = make_features_df()
    columns = group_columns(GROUPS)
    kwargs = dict(seq_len=6, min_seq_len=2, horizons=("30s", "1m"),
                  pump_thresholds={"30s": 0.1, "1m": 0.2}, rug_thresholds={"30s": 0.3, "1m": 0.4},
                  window_stride_seconds=10)  # fmt: skip

    narrow = build_samples(df, feature_columns=columns, **kwargs)
    whole = build_samples(df, **kwargs)

    assert narrow.columns == tuple(columns) and whole.columns is None
    assert narrow.features.shape == (len(whole.features), len(columns))
    for name in ("start_row", "end_row", "pump", "rug", "expected_return", "mask", "end_ts"):
        np.testing.assert_array_equal(getattr(narrow, name), getattr(whole, name))
    enabled = np.zeros(len(FEATURE_COLUMNS), bool)
    enabled[_index(columns)] = True
    masked = whole.features * enabled  # what the model's feature mask makes of the full rows
    picks = np.arange(len(whole))
    x_n, pad_n = narrow.padded_batch(picks)
    x_w, pad_w = whole.padded_batch(picks)
    np.testing.assert_array_equal(pad_n, pad_w)
    np.testing.assert_array_equal(x_n, x_w * enabled)
    np.testing.assert_array_equal(
        narrow.window(3), masked[whole.start_row[3] : whole.end_row[3] + 1]
    )
    assert narrow.n_features == whole.n_features == len(FEATURE_COLUMNS)
    assert narrow.x.shape == whole.x.shape


def test_no_window_still_records_the_narrow_columns():
    df = make_features_df(n_tokens=1, n_rows=3)
    s = build_samples(
        df,
        seq_len=6,
        horizons=("30s",),
        pump_thresholds={"30s": 0.1},
        rug_thresholds={"30s": 0.3},
        feature_columns=group_columns(["price"]),
    )
    assert len(s) == 0 and s.features.shape == (0, len(group_columns(["price"])))
    assert s.x.shape == (0, 6, len(FEATURE_COLUMNS))


def test_prune_columns_narrows_the_config_windows_to_the_enabled_groups():
    df = make_features_df()

    narrow = config_samples(df, pruned(small_config()))
    default = config_samples(df, full(small_config()))

    assert narrow.columns == tuple(group_columns(GROUPS))
    assert default.columns is None and default.features.shape[1] == len(FEATURE_COLUMNS)
    np.testing.assert_array_equal(narrow.end_row, default.end_row)


# --- the normalizer & training ---


def test_a_narrow_normalizer_fit_matches_the_full_one():
    df = make_features_df()
    cfg = full(small_config())
    whole = config_samples(df, cfg)
    narrow = config_samples(df, pruned(small_config()))

    a = SoliaTransformer.from_config(cfg).fit_normalizer(whole.features)
    b = SoliaTransformer.from_config(cfg).fit_normalizer(narrow.features, columns=narrow.columns)

    torch.testing.assert_close(a.normalizer.mean, b.normalizer.mean)
    torch.testing.assert_close(a.normalizer.std, b.normalizer.std)
    assert bool(b.normalizer.fitted)


def test_a_narrow_normalizer_fit_rejects_bad_rows_or_columns():
    model = SoliaTransformer.from_config(full(small_config()))
    columns = group_columns(["price"])
    with pytest.raises(ValueError, match="expected non-empty"):
        model.fit_normalizer(np.zeros((4, len(columns) + 1), np.float32), columns=columns)
    with pytest.raises(ValueError, match="not among the model's features"):
        model.fit_normalizer(np.zeros((4, 1), np.float32), columns=["nope"])
    with pytest.raises(ValueError, match="duplicates"):
        model.normalizer.fit(np.zeros((4, 2), np.float32), index=[0, 0])
    with pytest.raises(ValueError, match="out of range"):
        model.normalizer.fit(np.zeros((4, 1), np.float32), index=[len(FEATURE_COLUMNS)])


def test_the_model_sees_the_same_inputs_from_a_narrow_matrix():
    df = make_features_df()
    cfg = full(small_config())
    whole, narrow = config_samples(df, cfg), config_samples(df, pruned(small_config()))
    model = SoliaTransformer.from_config(cfg).fit_normalizer(whole.features).eval()
    picks = np.arange(len(whole))

    with torch.no_grad():
        outputs = [model(*map(torch.as_tensor, s.padded_batch(picks))) for s in (whole, narrow)]

    for key, value in outputs[0].items():
        torch.testing.assert_close(value, outputs[1][key], rtol=0, atol=0, msg=key)


def test_training_runs_on_a_narrow_matrix_and_fits_the_normalizer_on_its_columns():
    df = make_features_df()
    cfg = pruned(small_config())
    samples = config_samples(df, cfg)
    train_idx, val_idx = temporal_split(samples.end_ts, val_fraction=0.25, embargo_seconds=60)

    result = fit(SequenceDataset(samples, train_idx), SequenceDataset(samples, val_idx), cfg,
                 device="cpu")  # fmt: skip

    stats = result.model.normalizer
    off = np.setdiff1d(np.arange(len(FEATURE_COLUMNS)), _index(samples.columns))
    assert bool(stats.fitted) and np.all(stats.mean.numpy()[off] == 0)
    assert (
        np.all(stats.std.numpy()[off] == 1) and stats.std.numpy()[_index(samples.columns)].min() > 0
    )


# --- the read's columns & the guards ---


def test_the_read_keeps_every_column_unless_pruning():
    assert training_columns(full(small_config())) is None
    assert training_columns(with_prune_columns(full(small_config()), False)) is None


def test_a_pruned_read_holds_the_enabled_groups_and_what_labels_need_in_table_order():
    cols = training_columns(pruned(small_config(), groups=["price", "momentum"]))

    expected = set(group_columns(["price", "momentum"])) | set(REQUIRED_COLUMNS)
    assert set(cols) == expected
    assert cols == [c for c in DB_FEATURE_COLUMNS if c in expected]
    assert not set(cols) & set(VENUE_COLUMNS)  # derived from `venue`, always read


def test_a_pruned_read_defaults_to_the_baseline_groups():
    cfg = with_prune_columns(small_config(), True)
    assert set(training_columns(cfg)) == set(group_columns(BASELINE_GROUPS)) | set(REQUIRED_COLUMNS)


def test_read_columns_must_be_stored_feature_columns():
    assert _check_columns(["volume"]) == [c for c in DB_FEATURE_COLUMNS
                                          if c in {"volume", *REQUIRED_COLUMNS}]  # fmt: skip
    with pytest.raises(ValueError, match="not stored feature column"):
        _check_columns(["volume", "bogus"])
    with pytest.raises(ValueError, match="derived from 'venue'"):
        _check_columns([VENUE_COLUMNS[0]])
    with pytest.raises(TypeError, match="single string"):
        _check_columns("volume")


def test_prune_columns_must_be_a_boolean():
    cfg = small_config()
    with pytest.raises(ConfigError, match="prune_columns"):
        validate_config({**cfg, "data": {"prune_columns": "yes"}}, "test")
    validate_config({**cfg, "data": {"prune_columns": True}}, "test")


def test_the_champion_paths_refuse_a_pruned_config():
    check_not_pruned(small_config(), "retrain")  # default: fine
    with pytest.raises(ValueError, match="research runs"):
        check_not_pruned(pruned(small_config()), "retrain")


def test_with_prune_columns_leaves_the_config_untouched():
    cfg = small_config()
    out = with_prune_columns(cfg, True)
    assert out["data"]["prune_columns"] is True and "data" not in cfg


# --- the searchable lookback ---


def staggered_df(n_tokens: int = 6, n_rows: int = 80, step_s: int = 5, gap_s: int = 120):
    """``make_features_df`` with token ``k`` starting ``k * gap_s`` seconds after the first."""
    df = make_features_df(n_tokens=n_tokens, n_rows=n_rows, step_s=step_s)
    shift = df["mint"].str[1:].astype(int) * gap_s
    df["ts"] = df["ts"] + pd.to_timedelta(shift, unit="s")
    return df


def test_lookback_days_is_searchable_and_validated():
    cfg = apply_params(small_config(), {"lookback_days": 1.5})
    assert cfg["data"]["lookback_days"] == 1.5
    search.check_space({"lookback_days": {"choices": [1.0, 3.5]}}, small_config())
    with pytest.raises(ValueError, match="must be > 0"):
        search.check_space({"lookback_days": {"choices": [0]}}, small_config())
    with pytest.raises(ValueError, match="promotion gate"):
        apply_params(small_config(), {"lookback_days": 1e-6})  # shorter than the gate needs


def test_the_read_covers_the_longest_lookback_of_the_space():
    assert read_lookback({}, 2.0) == 2.0
    assert read_lookback({"lookback_days": {"choices": [2.0, 3.5]}}, 2.0) == 3.5
    assert read_lookback({"lookback_days": {"low": 0.5, "high": 1.0}}, 2.0) == 2.0


def test_lookback_tokens_replay_the_bounded_read_at_a_cutoff():
    df = staggered_df()
    first = df.groupby("mint")["ts"].min()
    epoch = {m: int(t.timestamp()) for m, t in first.items()}
    cutoff = epoch["M4"] + 30
    tokens = LookbackTokens(df)

    lookback = (cutoff - epoch["M2"]) / 86_400  # M2 starts exactly at the window's start
    assert tokens(cutoff, lookback, None) == {"M2", "M3", "M4"}  # M5 starts after the cutoff
    rows = {m: int(((df["mint"] == m) & (df["ts"] <= pd.Timestamp(cutoff, unit="s", tz="UTC")))
                   .sum()) for m in ("M2", "M3", "M4")}  # fmt: skip
    order = sorted(rows, key=lambda m: hashlib.md5(m.encode()).hexdigest())
    budget = rows[order[0]] + rows[order[1]]
    assert tokens(cutoff, lookback, budget) == set(order[:2])
    assert tokens(cutoff, lookback, budget - 1) == set(order[:1])


def test_without_lookback_in_the_space_the_search_trains_as_before(monkeypatch):
    calls = []
    monkeypatch.setattr(search, "LookbackTokens", lambda df: calls.append(df))
    run_search(make_features_df(), small_config(), space={"n_layers": {"choices": [1]}},
               n_trials=1, n_windows=1, seeds=[0])  # fmt: skip
    assert calls == []


def test_a_lookback_trial_trains_on_the_recent_tokens_of_each_slice_and_scores_the_same_windows(
    monkeypatch,
):
    df = staggered_df(n_tokens=8, gap_s=60)
    seen: list[tuple[set, tuple[int, ...]]] = []
    real = ablation.train_and_score

    def spy(train_ds, val_ds, eval_ds, cfg, **kwargs):
        mints = set(train_ds.samples.mint[train_ds.indices]) | set(
            val_ds.samples.mint[val_ds.indices]
        )
        seen.append((mints, tuple(eval_ds.samples.end_ts[eval_ds.indices])))
        return real(train_ds, val_ds, eval_ds, cfg, **kwargs)

    monkeypatch.setattr(search, "train_and_score", spy)
    span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86_400
    base = {**small_config(), "data": {"lookback_days": span}}
    short = 400 / 86_400  # the 4 x 60 s promotion floor, a bit over
    run_search(df, base, space={"lookback_days": {"choices": [short]}}, n_trials=2,
               n_windows=1, seeds=[0], eval_fraction=0.3)  # fmt: skip

    (base_mints, base_scored), (trial_mints, trial_scored) = seen
    assert trial_mints < base_mints  # the oldest tokens started before the short lookback
    assert trial_scored == base_scored
