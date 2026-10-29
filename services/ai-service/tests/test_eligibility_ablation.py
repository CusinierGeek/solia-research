"""The eligibility ablation: rule parsing, paired runs on each rule's population, and the CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_train import make_features_df, small_config

import solia_ai.config
import solia_ai.db
import solia_ai.eligibility_ablation as elig_ablation
import solia_ai.registry
from solia_ai.eligibility import EligibilityRule
from solia_ai.eligibility_ablation import (
    BASELINE,
    TRADES_KEY,
    parse_rule,
    rule_name,
    run_eligibility_ablation,
)

RULE = EligibilityRule(min_unique_traders=5)


def active_and_dead_tokens():
    """Even tokens always have 10 traders, odd ones never more than 1."""
    df = make_features_df()
    token = df["mint"].str[1:].astype(int)
    df["unique_traders"] = np.where(token % 2 == 0, 10, 1)
    df["volume"] = np.where(token % 2 == 0, 3.0, 0.01)
    df["token_age_sec"] = 60
    return df


@pytest.fixture
def no_registry(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an ablation run must never register a model")

    monkeypatch.setattr(solia_ai.registry, "register_model", forbidden)


# --------------------------------------------------------------------------- rules


def test_parse_rule_reads_any_subset_of_thresholds():
    assert parse_rule("traders=4") == EligibilityRule(min_unique_traders=4)
    assert parse_rule("traders=4, volume=1.5,age=30") == EligibilityRule(4, 1.5, 30)
    assert rule_name(parse_rule("volume=1,traders=2")) == "traders>=2,vol>=1"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("traders", "expected traders=N"),
        ("holders=3", "expected traders=N"),
        ("traders=x", "not a number"),
        ("traders=1.5", "not a number"),
        ("volume=-1", ">= 0"),
        ("traders=0", "filters nothing"),
        ("", "filters nothing"),
    ],
)
def test_parse_rule_rejects_bad_specs(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_rule(spec)


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ([], "at least one"),
        ([EligibilityRule()], "baseline"),
        ([RULE, RULE], "duplicate"),
        ([EligibilityRule(min_unique_traders=50)], "keeps no"),
    ],
)
def test_bad_rule_sets_are_rejected_before_training(rules, message, monkeypatch):
    def no_training(*_a, **_k):
        raise AssertionError("must fail before training")

    monkeypatch.setattr(elig_ablation, "fit", no_training)
    with pytest.raises(ValueError, match=message):
        run_eligibility_ablation(
            active_and_dead_tokens(), small_config(), rules=rules, n_windows=1, seeds=[0]
        )


# --------------------------------------------------------------------------- end to end


def test_each_rule_is_paired_with_the_every_token_model_on_its_own_population(
    no_registry, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # any checkpoint written would land here
    df = active_and_dead_tokens()
    trained_on: list[int] = []
    real_fit = elig_ablation.fit

    def recording_fit(train_ds, *args, **kwargs):
        trained_on.append(len(train_ds))
        return real_fit(train_ds, *args, **kwargs)

    monkeypatch.setattr(elig_ablation, "fit", recording_fit)

    report = run_eligibility_ablation(df, small_config(), rules=[RULE], n_windows=2, seeds=[0, 1])

    assert len(report.runs) == 2 * 2  # windows x seeds, each with its paired baseline
    assert len(trained_on) == 2 * 2 * 2  # the baseline and the rule's model per run
    assert trained_on[1] < trained_on[0]  # the rule's model sees only the active tokens
    rows = {r["name"]: r for r in report.rules}
    assert set(rows) == {BASELINE, "traders>=5"}
    assert rows[BASELINE]["n_tokens"] == 6 and rows["traders>=5"]["n_tokens"] == 3
    assert rows["traders>=5"]["n_rows"] == len(df) // 2
    assert 0 < rows["traders>=5"]["sample_share"] < 1
    assert 0 < rows["traders>=5"]["n_eval"] < rows[BASELINE]["n_eval"]
    run = report.runs[0]
    assert set(run.metrics) == set(run.baseline)
    assert TRADES_KEY in run.metrics
    assert set(report.lifts["traders>=5"]) == set(elig_ablation.metric_keys(["30s", "1m"]))
    rendered = report.render()
    assert "traders>=5" in rendered and "pump/30s/pr_auc" in rendered
    assert json.loads(json.dumps(report.as_dict()))["settings"]["seeds"] == [0, 1]
    assert not list(tmp_path.iterdir())  # no checkpoint saved


def test_the_config_s_own_rule_does_not_shrink_the_baseline(no_registry):
    cfg = small_config()
    cfg["eligibility"] = {"min_unique_traders": 5}

    report = run_eligibility_ablation(
        active_and_dead_tokens(),
        cfg,
        rules=[EligibilityRule(min_volume_sol=1.0)],
        n_windows=1,
        seeds=[0],
    )

    assert {r["name"]: r["n_tokens"] for r in report.rules} == {BASELINE: 6, "vol>=1": 3}


# --------------------------------------------------------------------------- CLI


def test_main_writes_the_json_report(no_registry, monkeypatch, tmp_path, capsys):
    df = active_and_dead_tokens()
    bounds = SimpleNamespace(describe=lambda: "read everything")
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda *a, **k: object())
    monkeypatch.setattr(solia_ai.db, "read_training_features", lambda *a, **k: (df, bounds))
    out = tmp_path / "eligibility.json"

    elig_ablation.main(
        ["--rule", "traders=5", "--windows", "1", "--seeds", "0", "--epochs", "1"]
        + ["--device", "cpu", "--out", str(out)]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [r["name"] for r in payload["rules"]] == [BASELINE, "traders>=5"]
    assert "traders>=5" in capsys.readouterr().out


def test_main_rejects_a_bad_rule_before_reading_data(monkeypatch):
    def unreachable(*_a, **_k):
        raise AssertionError("must fail before reading data")

    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", unreachable)

    with pytest.raises(SystemExit, match="expected traders=N"):
        elig_ablation.main(["--rule", "holders=3"])
