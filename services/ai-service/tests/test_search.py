"""Hyperparameter & labeling-threshold search (FE-T7, #116)."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime

import numpy as np
import pytest
from test_train import make_features_df, small_config

import solia_ai.ablation as ablation
import solia_ai.config
import solia_ai.db
import solia_ai.registry
import solia_ai.search as search
from solia_ai.config import load_config
from solia_ai.dataset import config_samples
from solia_ai.rows import TrainingRows
from solia_ai.search import (
    DEFAULT_SPACE,
    SearchReport,
    apply_params,
    candidate_params,
    check_budget,
    check_space,
    proposed_diff,
    run_objective,
    run_search,
)

SMALL_SPACE = {
    "seq_len": {"choices": [4, 6]},
    "d_model": {"choices": [8, 16]},
    "dropout": {"low": 0.0, "high": 0.2},
    "learning_rate": {"low": 0.001, "high": 0.02, "log": True},
    "pump_threshold_scale": {"choices": [0.5, 1.0]},
}


@pytest.fixture
def no_registry(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a search trial must never register a model")

    monkeypatch.setattr(solia_ai.registry, "register_model", forbidden)


# --- search space ---


def test_the_default_space_is_valid_for_the_shipped_config():
    check_space(DEFAULT_SPACE, load_config())


@pytest.mark.parametrize(
    ("space", "message"),
    [
        ({}, "at least one parameter"),
        ({"n_heads": {"choices": [2]}}, "unknown search parameter.*n_heads"),
        ({"dropout": 0.1}, "must be a mapping"),
        ({"dropout": {"choices": []}}, "non-empty list"),
        ({"dropout": {"choices": [0.1, 0.1]}}, "duplicates"),
        ({"dropout": {"low": 0.1}}, "either 'choices' or 'low'/'high'"),
        ({"dropout": {"choices": [0.1], "low": 0.0, "high": 0.2}}, "either"),
        ({"dropout": {"low": 0.3, "high": 0.1}}, "low .* <= high"),
        ({"dropout": {"choices": [1.0]}}, r"\[0, 1\)"),
        ({"learning_rate": {"low": 0.0, "high": 0.1, "log": True}}, "> 0"),
        ({"learning_rate": {"low": 0.01, "high": 0.1, "log": "yes"}}, "'log' must be a boolean"),
        ({"d_model": {"choices": [15]}}, "multiple of n_heads"),
        ({"d_model": {"low": 8, "high": 32}}, "explicit 'choices'"),
        ({"seq_len": {"choices": [1]}}, "min_seq_len"),
        ({"seq_len": {"choices": [4.5]}}, "integer"),
        ({"n_layers": {"choices": [0]}}, ">= 1"),
        ({"rug_threshold_scale": {"choices": [3.0]}}, "below 1"),
        ({"pump_threshold_scale": {"choices": [True]}}, "number"),
    ],
)
def test_invalid_search_spaces_are_rejected(space, message):
    with pytest.raises(ValueError, match=message):
        check_space(space, small_config())


def test_grid_search_needs_choices_only():
    with pytest.raises(ValueError, match="grid search needs 'choices'"):
        check_space({"dropout": {"low": 0.0, "high": 0.2}}, small_config(), strategy="grid")
    with pytest.raises(ValueError, match="strategy"):
        check_space(SMALL_SPACE, small_config(), strategy="bayes")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_trials": 0}, "n_trials"),
        ({"n_trials": 2.5}, "n_trials"),
        ({"max_seconds": 0}, "max_seconds"),
        ({"objective": "ece"}, "objective"),
    ],
)
def test_invalid_budgets_are_rejected(kwargs, message):
    settings = {"n_trials": 3, "max_seconds": None, "objective": "roc_auc", **kwargs}
    with pytest.raises(ValueError, match=message):
        check_budget(**settings)


def test_random_candidates_stay_in_the_space_and_are_deterministic():
    space = check_space(SMALL_SPACE, small_config())

    first = list(itertools.islice(candidate_params(space, strategy="random", seed=7), 10))
    second = list(itertools.islice(candidate_params(space, strategy="random", seed=7), 10))
    other = list(itertools.islice(candidate_params(space, strategy="random", seed=8), 10))

    assert first == second and first != other
    assert len({json.dumps(p, sort_keys=True) for p in first}) == len(first)  # no duplicate
    for p in first:
        assert p["seq_len"] in (4, 6) and p["d_model"] in (8, 16)
        assert 0.0 <= p["dropout"] <= 0.2 and 0.001 <= p["learning_rate"] <= 0.02


def test_grid_candidates_cover_the_grid_once_then_stop():
    space = check_space(
        {"seq_len": {"choices": [4, 6]}, "n_layers": {"choices": [1, 2, 3]}},
        small_config(),
        strategy="grid",
    )

    points = list(candidate_params(space, strategy="grid", seed=0))

    assert len(points) == 6
    assert {(p["seq_len"], p["n_layers"]) for p in points} == {
        (s, n) for s in (4, 6) for n in (1, 2, 3)
    }


def test_random_candidates_stop_once_a_tiny_space_is_exhausted():
    space = check_space({"seq_len": {"choices": [4, 6]}}, small_config())

    assert len(list(candidate_params(space, strategy="random", seed=0))) == 2


# --- applying a trial ---


def test_apply_params_overrides_the_model_and_keeps_the_base_untouched():
    base = small_config()

    cfg = apply_params(
        base, {"seq_len": 4, "d_model": 8, "n_layers": 2, "dropout": 0.1, "learning_rate": 0.005}
    )

    assert cfg["model"] | {} == {
        **base["model"],
        "seq_len": 4,
        "d_model": 8,
        "ff_dim": 16,  # base ratio ff_dim / d_model = 2
        "n_layers": 2,
        "dropout": 0.1,
    }
    assert cfg["training"]["learning_rate"] == 0.005
    assert base == small_config()


def test_apply_params_keeps_the_base_min_seq_len_even_when_implicit():
    base = small_config()
    del base["model"]["min_seq_len"]  # implicit: the base seq_len (6)

    assert apply_params(base, {"seq_len": 8})["model"]["min_seq_len"] == 6


def test_threshold_scales_multiply_every_horizon_of_their_head_only():
    cfg = apply_params(small_config(), {"pump_threshold_scale": 1.5})

    assert cfg["labeling"]["pump_threshold"] == {"30s": 0.15, "1m": 0.3}
    assert cfg["labeling"]["rug_threshold"] == small_config()["labeling"]["rug_threshold"]


def test_a_threshold_change_relabels_the_training_target():
    df, base = make_features_df(), small_config()
    looser = config_samples(df, apply_params(base, {"pump_threshold_scale": 0.5}))
    reference = config_samples(df, base)

    labeled = (reference.mask == 1) & (looser.mask[: len(reference)] == 1)
    # A lower bar can only turn non-events into events, never the reverse, and here it does.
    assert np.array_equal(looser.end_ts, reference.end_ts)
    assert np.all(looser.pump[labeled] >= reference.pump[labeled])
    assert looser.pump[labeled].sum() > reference.pump[labeled].sum()


def test_proposed_diff_shows_only_the_changed_keys():
    base = small_config()

    diff = proposed_diff(base, apply_params(base, {"n_layers": 3}))

    changed = [
        line for line in diff.splitlines() if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]
    assert changed == ["-  n_layers: 1", "+  n_layers: 3"]
    assert proposed_diff(base, base) == ""


def test_run_objective_averages_heads_and_horizons_skipping_undefined():
    metrics = {"pump/30s/roc_auc": 0.6, "pump/1m/roc_auc": None, "rug/30s/roc_auc": 0.8}

    assert run_objective(metrics, "roc_auc", ["30s", "1m"]) == pytest.approx(0.7)
    assert run_objective({}, "roc_auc", ["30s"]) is None


# --- orchestration ---


def test_search_respects_the_trial_budget_and_never_saves(no_registry, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # any checkpoint written would land here
    base = small_config()

    report = run_search(
        make_features_df(), base, space=SMALL_SPACE, n_trials=3, n_windows=2, seeds=[0, 1]
    )

    assert isinstance(report, SearchReport)
    assert [t.name for t in report.trials] == ["baseline", "trial-01", "trial-02"]
    assert report.stopped == "trials"
    assert all(len(t.runs) == 2 * 2 for t in report.trials)  # windows x seeds
    assert all(t.objective["n"] == 4 for t in report.trials)
    assert sorted(report.ranking) == sorted(t.name for t in report.trials)
    assert report.best == report.ranking[0]
    assert report.trials[0].params == {}
    if report.best == "baseline":
        assert report.proposed_diff == "" and report.comparison == {}
    else:
        assert report.proposed_diff.startswith("--- config.yaml")
        assert report.comparison["objective"]["n"] == 4
    payload = json.loads(report.to_json())
    assert payload["settings"]["eval_labels"]["pump_threshold"] == {"30s": 0.1, "1m": 0.2}
    assert "baseline" in report.render()
    assert base == small_config()  # never applied
    assert not list(tmp_path.iterdir())  # no checkpoint saved


def test_search_stops_when_the_time_budget_is_spent(no_registry, monkeypatch):
    ticks = itertools.count(0.0, 10.0)  # every clock read advances 10s

    report = run_search(
        make_features_df(),
        small_config(),
        space=SMALL_SPACE,
        n_trials=50,
        max_seconds=25.0,
        n_windows=1,
        seeds=[0],
        clock=lambda: next(ticks),
    )

    assert report.stopped == "time"
    assert 1 <= len(report.trials) < 50


def test_search_stops_when_the_space_is_exhausted(no_registry):
    report = run_search(
        make_features_df(),
        small_config(),
        space={"n_layers": {"choices": [1, 2]}},
        strategy="grid",
        n_trials=10,
        n_windows=1,
        seeds=[0],
    )

    assert report.stopped == "space"
    assert len(report.trials) == 3  # baseline + the two grid points


def test_a_trial_is_deterministic_for_a_fixed_seed(no_registry):
    df, cfg = make_features_df(), small_config()
    kwargs = {"space": SMALL_SPACE, "n_trials": 2, "n_windows": 1, "seeds": [3], "search_seed": 5}

    first = run_search(df, cfg, **kwargs)
    second = run_search(df, cfg, **kwargs)

    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert [r.metrics for t in first.trials for r in t.runs] == [
        r.metrics for t in second.trials for r in t.runs
    ]


def test_trials_train_on_their_labels_but_are_scored_on_the_base_labels(no_registry, monkeypatch):
    df, base = make_features_df(), small_config()
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    real = ablation.train_and_score

    def spy(train_ds, val_ds, eval_ds, cfg, **kwargs):
        seen.append((train_ds.samples.pump, eval_ds.samples.pump))
        return real(train_ds, val_ds, eval_ds, cfg, **kwargs)

    monkeypatch.setattr(search, "train_and_score", spy)
    run_search(
        df,
        base,
        space={"pump_threshold_scale": {"choices": [0.5]}},
        n_trials=2,
        n_windows=1,
        seeds=[0],
    )

    reference = config_samples(df, base).pump
    relabeled = config_samples(df, apply_params(base, {"pump_threshold_scale": 0.5})).pump
    (base_train, base_eval), (trial_train, trial_eval) = seen
    assert np.array_equal(base_train, reference) and np.array_equal(base_eval, reference)
    assert np.array_equal(trial_train, relabeled) and not np.array_equal(trial_train, reference)
    assert np.array_equal(trial_eval, reference)


def test_every_trial_scores_the_same_windows(no_registry, monkeypatch):
    scored: list[tuple[int, ...]] = []
    real = ablation.train_and_score

    def spy(train_ds, val_ds, eval_ds, cfg, **kwargs):
        scored.append(tuple(eval_ds.samples.end_ts[eval_ds.indices]))
        return real(train_ds, val_ds, eval_ds, cfg, **kwargs)

    monkeypatch.setattr(search, "train_and_score", spy)
    run_search(
        make_features_df(),
        small_config(),
        space={"seq_len": {"choices": [3]}, "pump_threshold_scale": {"choices": [2.0]}},
        n_trials=2,
        n_windows=2,
        seeds=[0],
    )

    assert len(scored) == 4 and scored[:2] == scored[2:]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seeds": []}, "seed"),
        ({"n_trials": 0}, "n_trials"),
        ({"space": {"bogus": {"choices": [1]}}}, "unknown search parameter"),
        ({"objective": "ece"}, "objective"),
        ({"n_windows": 50}, "empty"),
    ],
)
def test_run_search_rejects_bad_setups(kwargs, message, monkeypatch):
    with pytest.raises(ValueError, match=message):
        run_search(make_features_df(), small_config(), **{"space": SMALL_SPACE, **kwargs})


# --- CLI ---


FAKE_WINDOW_END = datetime(2026, 10, 30, 12, tzinfo=UTC)


def fake_bounded_read(_engine, settings, end=None, columns=None):
    df = make_features_df()

    class Bounds:
        @staticmethod
        def describe() -> str:
            return f"read {len(df)} rows (max_rows {settings.max_rows})"

        @staticmethod
        def as_dict() -> dict:
            return {"window_end": (end or FAKE_WINDOW_END).isoformat(), "rows_read": len(df)}

    return df, Bounds()


@pytest.fixture
def offline_cli(monkeypatch):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda *a, **k: object())
    monkeypatch.setattr(solia_ai.db, "read_training_features", fake_bounded_read)


def test_main_writes_the_report_and_the_proposed_diff(offline_cli, no_registry, tmp_path, capsys):
    space = tmp_path / "space.yaml"
    space.write_text("n_layers:\n  choices: [2]\n", encoding="utf-8")
    out, diff = tmp_path / "search.json", tmp_path / "proposed.diff"

    search.main(
        [
            "--space",
            str(space),
            "--trials",
            "2",
            "--windows",
            "1",
            "--seeds",
            "0",
            "--out",
            str(out),
            "--diff-out",
            str(diff),
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [t["name"] for t in payload["trials"]] == ["baseline", "trial-01"]
    assert diff.read_text(encoding="utf-8") == payload["proposed_diff"]
    assert "NOT applied" in capsys.readouterr().out or payload["best"] == "baseline"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--trials", "0"], "n_trials"),
        (["--max-minutes", "0"], "max_seconds"),
        (["--windows", "0"], "n_windows"),
        (["--space", "does-not-exist.yaml"], "cannot read search space"),
        (["--max-rows", "0"], "max_rows"),
    ],
)
def test_main_rejects_bad_settings_before_reading_data(monkeypatch, argv, message):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())

    def no_db(*_a, **_k):
        raise AssertionError("must fail before touching the database")

    monkeypatch.setattr(solia_ai.db, "make_engine", no_db)
    monkeypatch.setattr(solia_ai.db, "read_training_features", no_db)

    with pytest.raises(SystemExit, match=message):
        search.main(argv)


def test_main_rejects_an_invalid_space_file(monkeypatch, tmp_path):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    space = tmp_path / "space.yaml"
    space.write_text("d_model:\n  choices: [15]\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="multiple of n_heads"):
        search.main(["--space", str(space)])


# --- regularization & stride (#160) ---


def test_weight_decay_and_stride_are_searchable():
    cfg = apply_params(small_config(), {"weight_decay": 0.05, "window_stride_seconds": 30})

    assert cfg["training"]["weight_decay"] == 0.05
    assert cfg["data"]["window_stride_seconds"] == 30
    assert "window_stride_seconds" not in small_config().get("data", {})  # base untouched


@pytest.mark.parametrize(
    ("space", "message"),
    [
        ({"weight_decay": {"choices": [-0.01]}}, ">= 0"),
        ({"window_stride_seconds": {"choices": [-30]}}, ">= 0"),
        ({"window_stride_seconds": {"choices": ["30s"]}}, "number"),
    ],
)
def test_invalid_regularization_values_are_rejected(space, message):
    with pytest.raises(ValueError, match=message):
        check_space(space, small_config())


def test_a_stride_trial_trains_on_thinned_windows_but_is_scored_on_the_base_ones(
    no_registry, monkeypatch
):
    seen: list[tuple[int, int, tuple[int, ...]]] = []
    real = ablation.train_and_score

    def spy(train_ds, val_ds, eval_ds, cfg, **kwargs):
        scored = tuple(eval_ds.samples.end_ts[eval_ds.indices])
        seen.append((len(train_ds.samples), len(eval_ds.samples), scored))
        return real(train_ds, val_ds, eval_ds, cfg, **kwargs)

    monkeypatch.setattr(search, "train_and_score", spy)
    run_search(
        make_features_df(),
        small_config(),
        space={"window_stride_seconds": {"choices": [30]}},
        n_trials=2,
        n_windows=1,
        seeds=[0],
    )

    (base_n, base_eval_n, base_scored), (trial_n, trial_eval_n, trial_scored) = seen
    assert trial_n < base_n  # fewer training windows
    assert trial_eval_n == base_eval_n and trial_scored == base_scored  # the same held-out


def test_the_report_tracks_the_fit_and_the_rug_drift_per_slice(no_registry):
    report = run_search(
        make_features_df(),
        small_config(),
        space={"dropout": {"choices": [0.3]}},
        n_trials=2,
        n_windows=2,
        seeds=[0, 1],
    )

    for trial in report.trials:
        assert trial.fit["best_epoch"]["mean"] >= 1
        assert trial.fit["fit/gap"]["n"] == 4  # windows x seeds
        assert [row["window"] for row in trial.drift] == [0, 1]
        for row in trial.drift:
            assert set(row) >= {"eval_start", "eval_end", "rug/1m/roc_auc", "rug/1m/pr_auc"}
    text = report.render()
    assert "best ep" in text and "val-train gap" in text
    assert "Rug drift — baseline" in text
    payload = json.loads(report.to_json())
    assert payload["trials"][0]["drift"][0]["window"] == 0


def test_a_row_budget_keeps_whole_tokens_in_md5_order_like_the_bounded_read():
    import hashlib

    import pandas as pd

    rows = {"A": 3, "B": 5, "C": 2, "D": 4}
    mints = pd.Series([m for m, n in rows.items() for _ in range(n)] + [None])
    order = sorted(rows, key=lambda m: hashlib.md5(m.encode()).hexdigest())

    for budget in (1, 4, 8, 13, 14, 100):
        running, expected = 0, set()
        for m in order:
            running += rows[m]
            if running <= budget:
                expected.add(m)
        assert search.budget_tokens(mints, budget) == expected


def test_the_read_covers_the_largest_budget_of_the_space():
    assert search.read_rows({}, 4_000_000) == 4_000_000
    assert search.read_rows({"max_rows": {"choices": [2, 9_000_000]}}, 4_000_000) == 9_000_000
    assert search.read_rows({"max_rows": {"low": 1, "high": 3, "log": False}}, 10) == 10


def test_a_row_budget_trial_trains_on_fewer_tokens_but_scores_the_same_windows(
    no_registry, monkeypatch
):
    df = make_features_df()
    seen: list[tuple[set, tuple[int, ...]]] = []
    real = ablation.train_and_score

    def spy(train_ds, val_ds, eval_ds, cfg, **kwargs):
        mints = set(train_ds.samples.mint[train_ds.indices])
        seen.append((mints, tuple(eval_ds.samples.end_ts[eval_ds.indices])))
        return real(train_ds, val_ds, eval_ds, cfg, **kwargs)

    monkeypatch.setattr(search, "train_and_score", spy)
    budget = int(df["mint"].value_counts().max() * 2)
    run_search(df, small_config(), space={"max_rows": {"choices": [budget]}}, n_trials=2,
               n_windows=1, seeds=[0])  # fmt: skip

    (base_mints, base_scored), (trial_mints, trial_scored) = seen
    assert trial_mints < base_mints and trial_mints <= search.budget_tokens(df["mint"], budget)
    assert trial_scored == base_scored


def test_main_reads_the_largest_row_budget(offline_cli, no_registry, tmp_path, monkeypatch):
    read: list[int] = []

    def recording_read(engine, settings, end=None, columns=None):
        read.append(settings.max_rows)
        return fake_bounded_read(engine, settings, end)

    monkeypatch.setattr(solia_ai.db, "read_training_features", recording_read)
    space = tmp_path / "space.yaml"
    space.write_text("max_rows:\n  choices: [10000000]\n", encoding="utf-8")

    search.main(["--space", str(space), "--trials", "1", "--windows", "1", "--seeds", "0",
                 "--max-rows", "500", "--device", "cpu"])  # fmt: skip

    assert read == [10_000_000]


def test_main_reads_the_longest_lookback_and_prunes_on_request(
    offline_cli, no_registry, tmp_path, monkeypatch
):
    read: list[tuple[float, list[str] | None]] = []

    def recording_read(engine, settings, end=None, columns=None):
        read.append((settings.lookback_days, columns))
        return fake_bounded_read(engine, settings, end)

    def recording_rows(engine, settings, *, read_columns, matrix_columns, end=None):
        # --prune-columns: the compact read (solia_ai.rows), never a frame of every row
        read.append((settings.lookback_days, read_columns))
        df, bounds = fake_bounded_read(engine, settings, end)
        return TrainingRows.from_frame(df, matrix_columns), bounds

    monkeypatch.setattr(solia_ai.db, "read_training_features", recording_read)
    monkeypatch.setattr(solia_ai.db, "read_training_rows", recording_rows)
    space = tmp_path / "space.yaml"
    space.write_text("lookback_days:\n  choices: [1.0, 3.5]\n", encoding="utf-8")

    search.main(["--space", str(space), "--trials", "1", "--windows", "1", "--seeds", "0",
                 "--lookback-days", "0.5", "--device", "cpu"])  # fmt: skip
    search.main(["--space", str(space), "--trials", "1", "--windows", "1", "--seeds", "0",
                 "--prune-columns", "--device", "cpu"])  # fmt: skip

    (lookback, columns), (_, pruned) = read
    assert lookback == 3.5 and columns is None
    assert pruned == solia_ai.db.training_columns(
        solia_ai.db.with_prune_columns(small_config(), True)
    )


@pytest.mark.parametrize(
    "name",
    [
        "overfitting-data.yaml",
        "overfitting-regularization.yaml",
        "training-breadth.yaml",
        "recency-lookback.yaml",
        "ensemble.yaml",
    ],
)
def test_the_shipped_sweep_spaces_are_valid(name):
    from pathlib import Path

    by_path = search.load_space(Path(search.__file__).parent / "sweeps" / name)
    assert search.load_space(Path(name).stem) == by_path  # by name, as the CLI takes it
    check_space(by_path, load_config(), strategy="grid")


# --- resume (#181) ---

RESUME = {"space": SMALL_SPACE, "n_trials": 3, "n_windows": 1, "seeds": [0]}
DATA = {"window_end": "2026-10-30T12:00:00+00:00", "rows_read": 123}


class Interrupted(BaseException):
    """Stands for the reboot: not an Exception, so the search can't record it as a failed trial."""


def counting_trainer(monkeypatch, *, die_on: int | None = None) -> list[int]:
    """Counts train_and_score calls; the ``die_on``-th call (1-based) interrupts the process."""
    calls: list[int] = []
    real = ablation.train_and_score

    def trainer(*args, **kwargs):
        calls.append(1)
        if die_on is not None and len(calls) == die_on:
            raise Interrupted
        return real(*args, **kwargs)

    monkeypatch.setattr(search, "train_and_score", trainer)
    return calls


def trial_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rewrite(path, records) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_the_checkpoint_name_follows_the_report():
    assert search.checkpoint_path("/app/models/reports/x.json").name == "x.trials.jsonl"


def test_a_rerun_skips_finished_trials_and_reruns_the_interrupted_one(
    no_registry, tmp_path, monkeypatch
):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "s.trials.jsonl"
    full = run_search(df, cfg, **RESUME)

    counting_trainer(monkeypatch, die_on=3)  # baseline, trial-01 finish; trial-02 dies mid-way
    with pytest.raises(Interrupted):
        run_search(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    assert [r.get("name") for r in trial_lines(ckpt)] == [None, "baseline", "trial-01"]

    calls = counting_trainer(monkeypatch)
    resumed = run_search(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)

    assert len(calls) == 1  # only trial-02 trained again (1 window x 1 seed)
    assert resumed.resumed == ["baseline", "trial-01"]
    assert [t.name for t in resumed.trials] == [t.name for t in full.trials]
    assert [t.params for t in resumed.trials] == [t.params for t in full.trials]
    assert [r.metrics for t in resumed.trials for r in t.runs] == [
        r.metrics for t in full.trials for r in t.runs
    ]  # as if never interrupted
    assert resumed.windows == full.windows and resumed.ranking == full.ranking
    assert "Resumed from the checkpoint" in resumed.render()

    calls = counting_trainer(monkeypatch)
    again = run_search(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    assert calls == [] and again.resumed == ["baseline", "trial-01", "trial-02"]


def test_a_torn_last_line_reruns_that_trial(no_registry, tmp_path, monkeypatch):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "s.trials.jsonl"
    run_search(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    lines = ckpt.read_text(encoding="utf-8").splitlines()
    ckpt.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:40], encoding="utf-8")

    calls = counting_trainer(monkeypatch)
    report = run_search(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)

    assert len(calls) == 1 and report.resumed == ["baseline", "trial-01"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seeds": [0, 1]}, "settings.seeds"),
        ({"search_seed": 7}, "settings.search_seed"),
        ({"space": {"n_layers": {"choices": [1, 2]}}}, "settings.space"),
        ({"data": {**DATA, "rows_read": 124}}, "data.rows_read"),
        ({"config": True}, "config.model"),
    ],
)
def test_a_checkpoint_of_another_search_is_refused(
    change, message, no_registry, tmp_path, monkeypatch
):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "s.trials.jsonl"
    run_search(df, cfg, **{**RESUME, "n_trials": 1}, checkpoint=ckpt, data=DATA)
    before = ckpt.read_text(encoding="utf-8")
    kwargs = {**RESUME, "n_trials": 1, "data": DATA, **change}
    if kwargs.pop("config", False):
        cfg = {**cfg, "model": {**cfg["model"], "dropout": 0.3}}
    calls = counting_trainer(monkeypatch)

    with pytest.raises(ValueError, match=f"another search.*{message}"):
        run_search(df, cfg, **kwargs, checkpoint=ckpt)

    assert calls == [] and ckpt.read_text(encoding="utf-8") == before  # untouched


def test_a_trial_recorded_with_other_params_is_refused(no_registry, tmp_path):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "s.trials.jsonl"
    run_search(df, cfg, **{**RESUME, "n_trials": 2}, checkpoint=ckpt, data=DATA)
    lines = trial_lines(ckpt)
    lines[-1]["params"] = {"n_layers": 9}
    rewrite(ckpt, lines)

    with pytest.raises(ValueError, match="trial-01 ran with"):
        run_search(df, cfg, **{**RESUME, "n_trials": 2}, checkpoint=ckpt, data=DATA)


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path):
    path = tmp_path / "s.trials.jsonl"
    path.write_text('{"something": "else"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a search trials file"):
        search.read_checkpoint_header(path)
    assert search.read_checkpoint_header(tmp_path / "missing.jsonl") is None


def test_a_failed_trial_is_reported_and_the_search_goes_on(no_registry, tmp_path, monkeypatch):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "s.trials.jsonl"
    real = ablation.train_and_score
    calls: list[int] = []

    def oom_on_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("CUDA out of memory")
        return real(*args, **kwargs)

    monkeypatch.setattr(search, "train_and_score", oom_on_second)
    report = run_search(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)

    assert [t.name for t in report.trials] == ["baseline", "trial-02"]
    assert [(f.name, f.error) for f in report.failed] == [
        ("trial-01", "RuntimeError: CUDA out of memory")
    ]
    assert "trial-01" not in report.ranking
    assert "FAILED trial-01" in report.render()
    assert json.loads(report.to_json())["failed"][0]["params"] == report.failed[0].params

    calls = counting_trainer(monkeypatch)  # a rerun doesn't retry the recorded failure
    again = run_search(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    assert calls == [] and [f.name for f in again.failed] == ["trial-01"]


def test_a_failing_baseline_still_stops_the_search(no_registry, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(search, "train_and_score", broken)
    with pytest.raises(RuntimeError, match="boom"):
        run_search(make_features_df(), small_config(), **RESUME)


def test_resumed_time_counts_against_the_time_budget(no_registry, tmp_path):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "s.trials.jsonl"
    kwargs = {**RESUME, "n_trials": 50, "max_seconds": 1000.0, "data": DATA, "checkpoint": ckpt}
    run_search(df, cfg, **{**kwargs, "n_trials": 2})
    lines = trial_lines(ckpt)
    lines[0]["fingerprint"]["settings"]["n_trials"] = 50
    for record in lines[1:]:
        record["seconds"] = 600.0  # 1 200 s already spent before the "reboot"
    rewrite(ckpt, lines)

    report = run_search(df, cfg, **kwargs)

    assert report.stopped == "time" and len(report.trials) == 2  # no new trial started
    assert report.elapsed_seconds == 1200.0


def test_main_resumes_on_the_window_it_started_on(offline_cli, no_registry, tmp_path, monkeypatch):
    ends: list = []

    def recording_read(engine, settings, end=None, columns=None):
        ends.append(end)
        return fake_bounded_read(engine, settings, end)

    monkeypatch.setattr(solia_ai.db, "read_training_features", recording_read)
    space = tmp_path / "space.yaml"
    space.write_text("n_layers:\n  choices: [2]\n", encoding="utf-8")
    out = tmp_path / "search.json"
    argv = ["--space", str(space), "--trials", "2", "--windows", "1", "--seeds", "0",
            "--device", "cpu", "--out", str(out)]  # fmt: skip

    search.main(argv)
    ckpt = search.checkpoint_path(out)
    assert len(trial_lines(ckpt)) == 3  # header, baseline, trial-01
    calls = counting_trainer(monkeypatch)
    search.main(argv)
    assert calls == []  # everything resumed
    assert ends == [None, FAKE_WINDOW_END]  # the rerun read the first run's window
    assert json.loads(out.read_text(encoding="utf-8"))["resumed"] == ["baseline", "trial-01"]

    search.main([*argv, "--restart"])
    assert ends[-1] is None and len(calls) == 2  # discarded: today's window, trained again


def test_main_refuses_a_foreign_checkpoint_before_reading(offline_cli, tmp_path, monkeypatch):
    out = tmp_path / "search.json"
    search.checkpoint_path(out).write_text("not json\n", encoding="utf-8")
    monkeypatch.setattr(solia_ai.db, "read_training_features", None)  # must not be called
    with pytest.raises(SystemExit, match="not a search trials file"):
        search.main(["--trials", "1", "--out", str(out)])


def test_main_passes_the_stride_override_to_the_search(
    offline_cli, no_registry, tmp_path, monkeypatch
):
    seen: list[float] = []
    real_run = search.run_search

    def spy(df, config, **kwargs):
        seen.append(config["data"]["window_stride_seconds"])
        return real_run(df, config, **kwargs)

    monkeypatch.setattr(search, "run_search", spy)
    space = tmp_path / "space.yaml"
    space.write_text("recency_half_life_hours:\n  choices: [24]\n", encoding="utf-8")

    search.main(["--space", str(space), "--strategy", "grid", "--trials", "2", "--windows", "1",
                 "--seeds", "0", "--window-stride-seconds", "20", "--device", "cpu"])  # fmt: skip

    assert seen == [20.0]


@pytest.mark.parametrize("stride", ["-1", "nan", "inf"])
def test_main_rejects_a_bad_stride_before_reading_data(monkeypatch, stride):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())

    def no_db(*_a, **_k):
        raise AssertionError("must fail before touching the database")

    monkeypatch.setattr(solia_ai.db, "make_engine", no_db)
    with pytest.raises(SystemExit, match="window_stride_seconds"):
        search.main(["--window-stride-seconds", stride])


def test_the_combined_recency_grid_is_the_six_points_of_two_periods_and_three_half_lives():
    space = search.load_space("recency-lookback")

    assert space == {
        "lookback_days": {"choices": [3.5, 5.0]},
        "recency_half_life_hours": {"choices": [12, 24, 48]},
    }
