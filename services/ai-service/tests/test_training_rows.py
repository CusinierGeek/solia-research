"""Compact training rows (#202 step 2, solia_ai.rows): the shared narrow matrix, the token-block
build, the streamed normalizer fit and their use by the search / throughput. In memory, CPU.
The database read (solia_ai.db.read_training_rows) is covered in tests/test_training_read.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from test_train import make_features_df, small_config

import solia_ai.search as search
from solia_ai import throughput
from solia_ai.dataset import (
    SequenceDataset,
    build_samples,
    config_samples,
    rows_samples,
    temporal_split,
)
from solia_ai.db import with_prune_columns
from solia_ai.features import (
    ALWAYS_WRITTEN,
    VENUE_COLUMNS,
    build_feature_matrix,
    group_columns,
)
from solia_ai.model import FeatureNormalizer, SoliaTransformer
from solia_ai.rows import TrainingRows
from solia_ai.search import run_search
from solia_ai.train import fit, training_feature_rows, training_row_mask

GROUPS = ["price", "liquidity", "flow", "holders", "context", "momentum", "concentration"]
SAMPLE_ARRAYS = ("pump", "rug", "expected_return", "mask", "end_ts", "mint",
                 "eligibility_inputs", "anchor_liquidity", "anchor_venue")  # fmt: skip


def pruned(cfg: dict, groups=GROUPS) -> dict:
    return with_prune_columns({**cfg, "features": {"groups": list(groups)}}, True)


def gated(cfg: dict) -> dict:
    """``cfg`` with an eligibility rule and a venue-aware liquidity floor that both drop windows."""
    return {
        **cfg,
        "eligibility": {"min_unique_traders": 1, "min_volume_sol": 0.0, "min_token_age_sec": 1.0},
        "labeling": {**cfg["labeling"], "liquidity_floor": {"default": 48.0, "pumpfun": 0.0}},
    }


def shuffled_df(seed: int = 3, **kwargs) -> pd.DataFrame:
    """``make_features_df`` with a venue column, a row without a mint, rows in random order."""
    df = make_features_df(**kwargs).drop(columns=list(VENUE_COLUMNS))
    rng = np.random.default_rng(seed)
    df["venue"] = rng.choice(np.array(["pumpfun", "pumpswap", None], dtype=object), len(df))
    df.loc[5, "mint"] = None
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def rows_of(df: pd.DataFrame, groups=GROUPS) -> TrainingRows:
    return TrainingRows.from_frame(df, group_columns(groups))


def assert_same_windows(a, b) -> None:
    """Same windows, labels and served (FEATURE_COLUMNS-wide) inputs, whatever the row layout."""
    assert len(a) == len(b) and a.columns == b.columns
    for name in SAMPLE_ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        if x.dtype == object:  # missing venues: NaN on both sides, and NaN != NaN
            assert pd.Series(x).equals(pd.Series(y)), name
        else:
            np.testing.assert_array_equal(x, y, err_msg=name)
    picks = np.arange(len(a))
    x_a, pad_a = a.padded_batch(picks)
    x_b, pad_b = b.padded_batch(picks)
    np.testing.assert_array_equal(pad_a, pad_b)
    np.testing.assert_array_equal(x_a, x_b)


# --- TrainingRows ---


def test_rows_from_a_frame_are_sorted_by_token_then_time_with_the_narrow_matrix():
    df = shuffled_df()
    columns = group_columns(GROUPS)
    rows = rows_of(df)

    known = df[df["mint"].notna()]
    expected = known.sort_values(["mint", "ts"], kind="stable")
    assert len(rows) == len(known) and rows.n_tokens == known["mint"].nunique()
    assert list(rows.frame["mint"].astype(object)) == list(expected["mint"])
    assert (
        rows.frame["ts"].to_numpy() == pd.to_datetime(expected["ts"], utc=True).to_numpy()
    ).all()
    np.testing.assert_array_equal(rows.frame["price"], expected["price"].to_numpy(np.float64))
    assert list(rows.frame["venue"]) == list(expected["venue"])
    np.testing.assert_array_equal(rows.features, build_feature_matrix(expected, columns=columns))
    assert rows.features.dtype == np.float32 and rows.columns == tuple(columns)
    assert rows.nbytes >= rows.features.nbytes


def test_rows_refuse_a_matrix_that_doesnt_match_the_frame():
    rows = rows_of(make_features_df(n_tokens=2, n_rows=10))
    with pytest.raises(ValueError, match="features must be"):
        TrainingRows(rows.frame, rows.features[:-1], rows.columns)
    with pytest.raises(ValueError, match="float32"):
        TrainingRows(rows.frame, rows.features.astype(np.float64), rows.columns)
    with pytest.raises(ValueError, match="ordered subset"):
        TrainingRows.from_frame(make_features_df(n_tokens=1, n_rows=4), ["volume", "price"])


def test_token_blocks_hold_whole_tokens_within_the_bound():
    rows = rows_of(make_features_df(n_tokens=5, n_rows=10))  # 5 tokens x 10 rows

    assert rows.token_blocks(25) == [(0, 20), (20, 40), (40, 50)]
    assert rows.token_blocks(10) == [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    assert rows.token_blocks(3) == rows.token_blocks(10)  # a token is never split
    assert rows.token_blocks(10**9) == [(0, 50)]
    with pytest.raises(ValueError, match="max_rows"):
        rows.token_blocks(0)


def test_rows_refuse_a_group_whose_rows_predate_it():
    df = make_features_df(n_tokens=2, n_rows=10)
    column = ALWAYS_WRITTEN["momentum"][0]
    df.loc[3, column] = np.nan
    rows = rows_of(df)

    assert rows.nulls[column] == 1
    with pytest.raises(ValueError, match="predate the 'momentum' group"):
        rows_samples(rows, pruned(small_config()))
    rows_samples(rows_of(df, ["price"]), pruned(small_config(), ["price"]))  # not enabled: fine


# --- the shared-matrix build ---


@pytest.mark.parametrize("block_rows", [1, 100, 10**9])
@pytest.mark.parametrize("stride", [0, 10])
@pytest.mark.parametrize("with_gates", [False, True])
def test_rows_samples_equal_the_frame_build(block_rows, stride, with_gates):
    df = shuffled_df()
    cfg = pruned({**small_config(), "data": {"window_stride_seconds": stride}})
    cfg = gated(cfg) if with_gates else cfg

    reference = config_samples(df, cfg)
    samples = rows_samples(rows_of(df), cfg, block_rows=block_rows)

    assert len(reference) > 0 and samples.shared_rows and not reference.shared_rows
    assert_same_windows(samples, reference)


def test_every_build_shares_the_one_matrix():
    rows = rows_of(shuffled_df())
    cfg = pruned(small_config())
    a = config_samples(rows, cfg)  # dispatches to rows_samples
    b = config_samples(rows, {**cfg, "data": {**cfg["data"], "window_stride_seconds": 20}})

    assert a.features is rows.features and b.features is rows.features
    assert len(b) < len(a)


def test_rows_samples_need_exactly_the_enabled_groups_columns():
    rows = rows_of(make_features_df(n_tokens=2, n_rows=20), ["price"])
    with pytest.raises(ValueError, match="enabled groups need"):
        rows_samples(rows, pruned(small_config()))


def test_rows_without_any_window_give_empty_shared_samples():
    rows = rows_of(make_features_df(n_tokens=2, n_rows=1))  # 1 row per token < min_seq_len
    samples = rows_samples(rows, pruned(small_config()))

    assert len(samples) == 0 and samples.shared_rows and samples.features is rows.features


def test_a_shared_build_refuses_rows_out_of_token_time_order_or_a_bad_matrix():
    df = make_features_df(n_tokens=2, n_rows=10)
    columns = group_columns(GROUPS)
    matrix = build_feature_matrix(df, columns=columns)
    kwargs = dict(seq_len=4, min_seq_len=2, horizons=("30s",), pump_thresholds={"30s": 0.1},
                  rug_thresholds={"30s": 0.3}, feature_columns=columns)  # fmt: skip

    build_samples(df, **kwargs, features_matrix=matrix)  # sorted: fine
    with pytest.raises(ValueError, match="contiguous and in time order"):
        build_samples(df.iloc[::-1].reset_index(drop=True), **kwargs, features_matrix=matrix)
    with pytest.raises(ValueError, match="features_matrix must be"):
        build_samples(df, **kwargs, features_matrix=matrix[:, :3])
    with pytest.raises(ValueError, match="features_matrix must be"):
        build_samples(df, **kwargs, features_matrix=matrix, row_offset=1)
    with pytest.raises(ValueError, match="features_matrix must be"):
        build_samples(df, **{**kwargs, "feature_columns": None}, features_matrix=matrix)


# --- the streamed normalizer fit ---


def test_the_training_row_mask_covers_exactly_the_training_windows():
    samples = config_samples(make_features_df(), pruned(small_config()))
    indices = np.arange(0, len(samples), 3)
    mask = training_row_mask(SequenceDataset(samples, indices))

    expected = np.zeros(len(samples.features), bool)
    for i in indices:
        expected[samples.start_row[i] : samples.end_row[i] + 1] = True
    np.testing.assert_array_equal(mask, expected)
    assert not training_row_mask(SequenceDataset(samples, np.array([], np.int64))).any()


@pytest.mark.parametrize("chunk_rows", [1, 7, 1 << 20])
def test_the_streamed_fit_matches_the_one_pass_fit(chunk_rows):
    samples = config_samples(make_features_df(), pruned(small_config()))
    dataset = SequenceDataset(samples, np.arange(0, len(samples), 2))
    one_pass = SoliaTransformer.from_config(pruned(small_config()))
    one_pass.fit_normalizer(training_feature_rows(dataset), columns=samples.columns)
    streamed = SoliaTransformer.from_config(pruned(small_config()))
    streamed.fit_normalizer_rows(
        samples.features, training_row_mask(dataset), columns=samples.columns,
        chunk_rows=chunk_rows,
    )  # fmt: skip

    for name in ("mean", "std"):
        torch.testing.assert_close(
            getattr(streamed.normalizer, name), getattr(one_pass.normalizer, name),
            rtol=1e-5, atol=1e-6,
        )  # fmt: skip
    assert bool(streamed.normalizer.fitted)


def test_the_streamed_fit_rejects_bad_input():
    model = SoliaTransformer.from_config(pruned(small_config()))
    columns = group_columns(GROUPS)
    rows = np.ones((4, len(columns)), np.float32)
    keep = np.ones(4, bool)

    with pytest.raises(ValueError, match="no row"):
        model.fit_normalizer_rows(rows, np.zeros(4, bool), columns=columns)
    with pytest.raises(ValueError, match="keep must be"):
        model.fit_normalizer_rows(rows, keep[:2], columns=columns)
    with pytest.raises(ValueError, match="expected"):
        model.fit_normalizer_rows(rows[:, :2], keep, columns=columns)
    with pytest.raises(ValueError, match="not among"):
        model.fit_normalizer_rows(rows[:, :1], keep, columns=["nope"])
    with pytest.raises(ValueError, match="chunk_rows"):
        model.fit_normalizer_rows(rows, keep, columns=columns, chunk_rows=0)
    bad = rows.copy()
    bad[1, columns.index("price")] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        model.fit_normalizer_rows(bad, keep, columns=columns)


def test_set_statistics_follows_the_fit_rules_and_validates():
    norm = FeatureNormalizer(4)
    norm.set_statistics(np.array([1.0, 2.0]), np.array([0.5, 0.0]), index=[1, 3])

    torch.testing.assert_close(norm.mean, torch.tensor([0.0, 1.0, 0.0, 2.0]))
    torch.testing.assert_close(norm.std, torch.tensor([1.0, 0.5, 1.0, 1.0]))  # std 0 -> 1
    assert bool(norm.fitted)
    with pytest.raises(ValueError, match=r"expected \[2\]"):
        norm.set_statistics(np.zeros(3), np.ones(3), index=[0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        norm.set_statistics(np.array([np.nan] * 4), np.ones(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_streamed_fit_works_on_a_cuda_model():
    samples = config_samples(make_features_df(), pruned(small_config()))
    keep = np.ones(len(samples.features), bool)
    on_cpu = SoliaTransformer.from_config(pruned(small_config()))
    on_cpu.fit_normalizer_rows(samples.features, keep, columns=samples.columns)
    on_cuda = SoliaTransformer.from_config(pruned(small_config())).to("cuda")
    on_cuda.fit_normalizer_rows(samples.features, keep, columns=samples.columns)

    torch.testing.assert_close(on_cuda.normalizer.mean.cpu(), on_cpu.normalizer.mean)


# --- training, search and throughput on the rows ---


def test_training_on_shared_rows_streams_the_normalizer_fit(monkeypatch):
    df = make_features_df()
    cfg = pruned(small_config())
    samples = config_samples(rows_of(df), cfg)
    train_idx, val_idx = temporal_split(samples.end_ts, val_fraction=0.25, embargo_seconds=60)
    calls = []
    real = SoliaTransformer.fit_normalizer_rows
    monkeypatch.setattr(
        SoliaTransformer, "fit_normalizer_rows",
        lambda self, *a, **k: calls.append(1) or real(self, *a, **k),
    )  # fmt: skip

    result = fit(SequenceDataset(samples, train_idx), SequenceDataset(samples, val_idx), cfg,
                 device="cpu")  # fmt: skip

    reference = config_samples(df, cfg)  # the frame build's normalizer, one pass
    model = SoliaTransformer.from_config(cfg)
    model.fit_normalizer(
        training_feature_rows(SequenceDataset(reference, train_idx)), columns=reference.columns
    )
    assert calls == [1]
    torch.testing.assert_close(result.model.normalizer.mean, model.normalizer.mean,
                               rtol=1e-5, atol=1e-6)  # fmt: skip


def test_a_lookback_search_on_rows_trains_and_scores_like_on_the_frame(monkeypatch):
    df = make_features_df(n_tokens=8)
    shift = df["mint"].str[1:].astype(int) * 60  # token k starts k minutes after the first
    df["ts"] = df["ts"] + pd.to_timedelta(shift, unit="s")
    span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86_400
    base = pruned({**small_config(), "data": {"lookback_days": span, "max_rows": 10**9}})
    space = {
        "lookback_days": {"choices": [400 / 86_400]},
        "window_stride_seconds": {"choices": [20]},
    }

    def splits(data):
        seen = []

        def spy(train_ds, val_ds, eval_ds, cfg, **kwargs):
            seen.append((tuple(sorted(set(train_ds.samples.mint[train_ds.indices]))),
                         tuple(train_ds.samples.end_ts[train_ds.indices]),
                         tuple(eval_ds.samples.end_ts[eval_ds.indices])))  # fmt: skip
            return {"pump/30s/pr_auc": 0.5}

        monkeypatch.setattr(search, "train_and_score", lambda *a, **k: (0, spy(*a, **k)))
        run_search(data, base, space=space, strategy="grid", n_trials=3, n_windows=1, seeds=[0],
                   eval_fraction=0.3)  # fmt: skip
        return seen

    on_frame, on_rows = splits(df), splits(rows_of(df))
    assert len(on_rows) == 2 and on_rows == on_frame  # baseline + the one grid point


def test_throughput_runs_on_rows_and_reports_their_size():
    report = throughput.run(
        rows_of(make_features_df()), pruned(small_config()), workers=[0], batches=2, device="cpu"
    )

    assert report["samples"] > 0 and report["tokens"] == 6 and report["rows_mb"] > 0
    assert report["runs"]["batched, 0 workers"]["batches"] == 2
