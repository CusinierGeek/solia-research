"""Hyperparameter & labeling-threshold search (FE-T7, #116)."""

from __future__ import annotations

import itertools
import json

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


@pytest.fixture
def offline_cli(monkeypatch):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda *a, **k: object())
    monkeypatch.setattr(solia_ai.db, "read_features", lambda *a, **k: make_features_df())


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
    ],
)
def test_main_rejects_bad_settings_before_reading_data(monkeypatch, argv, message):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())

    def no_db(*_a, **_k):
        raise AssertionError("must fail before touching the database")

    monkeypatch.setattr(solia_ai.db, "make_engine", no_db)
    monkeypatch.setattr(solia_ai.db, "read_features", no_db)

    with pytest.raises(SystemExit, match=message):
        search.main(argv)


def test_main_rejects_an_invalid_space_file(monkeypatch, tmp_path):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    space = tmp_path / "space.yaml"
    space.write_text("d_model:\n  choices: [15]\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="multiple of n_heads"):
        search.main(["--space", str(space)])
