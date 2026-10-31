"""Overnight sweep launcher helpers (#181): preflight, stage chaining, summary."""

from __future__ import annotations

import json
import os
import re

import pytest
import yaml
from test_train import small_config

import solia_ai.config
import solia_ai.overnight as overnight
import solia_ai.search
from solia_ai.overnight import (
    STAGE1,
    STAGE2,
    STAGE2_CONFIG,
    SUMMARY,
    Preflight,
    env_flag,
    preflight_problems,
    running_sweeps,
    stage1_choice,
    stage2_config,
)

GB = 1_000_000_000
READY = Preflight(
    retrain_running=False,
    scheduler_enabled="false",
    paper_bot_enabled="",
    cuda="NVIDIA GeForce RTX 4070",
    free_bytes=100 * GB,
    other_sweeps=[],
)


def problems(**changes) -> list[str]:
    return preflight_problems(Preflight(**{**READY.__dict__, **changes}), min_free_bytes=20 * GB)


# --- preflight ---


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("", False), ("false", False), (" OFF ", False), ("0", False),
     ("true", True), ("1", True), ("Yes", True)],
)  # fmt: skip
def test_env_flags_read_like_the_services(value, expected):
    assert env_flag("X", value) is expected


def test_an_unreadable_env_flag_is_rejected():
    with pytest.raises(ValueError, match="X must be true or false"):
        env_flag("X", "maybe")


def test_a_ready_stack_passes_the_preflight():
    assert problems() == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"retrain_running": True}, "holds the retrain lock"),
        ({"retrain_running": None}, "database is unreachable"),
        ({"scheduler_enabled": "true"}, "RETRAIN_SCHEDULER_ENABLED=true"),
        ({"scheduler_enabled": None}, None),  # unset = the service's default: off
        ({"paper_bot_enabled": "true"}, "PAPER_BOT_ENABLED=true"),
        ({"paper_bot_enabled": "sometimes"}, "PAPER_BOT_ENABLED must be true or false"),
        ({"cuda": None}, "CUDA is not visible"),
        ({"free_bytes": 5 * GB}, "only 5.0 GB free"),
        ({"other_sweeps": [42]}, "already running \\(pid 42\\)"),
    ],
)
def test_each_precondition_refuses_on_its_own(changes, message):
    found = problems(**changes)
    if message is None:
        assert found == []
    else:
        assert len(found) == 1 and re.search(message, found[0]), found


def test_every_problem_is_listed_not_just_the_first():
    assert len(problems(retrain_running=True, cuda=None, paper_bot_enabled="true")) == 3


def test_running_sweeps_finds_other_searches_in_proc(tmp_path):
    def proc(pid, *argv):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")

    proc(10, "python", "-u", "-m", "solia_ai.overnight", "run", "--windows", "2")
    proc(11, "python", "-m", "solia_ai.search", "--space", "overfitting-data")
    proc(12, "python", "-m", "solia_ai.overnight", "preflight")  # the check itself
    proc(13, "uvicorn", "solia_ai.api:app")
    proc(os.getpid(), "python", "-m", "solia_ai.search")  # this process: never counted
    (tmp_path / "self").mkdir()  # not a pid

    assert running_sweeps(tmp_path) == [10, 11]
    assert running_sweeps(tmp_path / "missing") == []  # no /proc (Windows): nothing found


def test_main_preflight_exits_1_and_lists_the_problems(monkeypatch, capsys):
    monkeypatch.setattr(
        overnight,
        "gather_preflight",
        lambda _dir, paper: Preflight(**{**READY.__dict__, "paper_bot_enabled": paper}),
    )

    assert overnight.main(["preflight", "--paper-bot-enabled", "true"]) == 1
    assert "REFUSED: PAPER_BOT_ENABLED=true" in capsys.readouterr().err
    assert overnight.main(["preflight", "--paper-bot-enabled", ""]) == 0
    assert "preflight OK" in capsys.readouterr().out


# --- stage 1 -> stage 2 ---


def report(best="baseline", verdict="better", failed=(), stage_params=None):
    params = stage_params or {"window_stride_seconds": 30, "max_rows": 8_000_000}
    trials = [
        {"name": "baseline", "params": {}, "score": 0.5},
        {"name": "trial-01", "params": params, "score": 0.6},
    ]
    return {
        "settings": {"objective": "pr_auc"},
        "trials": trials,
        "ranking": ["trial-01", "baseline"] if best != "baseline" else ["baseline", "trial-01"],
        "best": best,
        "stopped": "trials",
        "comparison": (
            {}
            if best == "baseline"
            else {"objective": {"mean": 0.1, "std": 0.01, "verdict": verdict}}
        ),
        "failed": list(failed),
        "resumed": ["baseline"],
        "elapsed_seconds": 7200.0,
    }


BASE = {"data": {"window_stride_seconds": 0, "max_rows": 4_000_000, "lookback_days": 2}}


def test_stage_2_takes_the_best_stage_1_trial_when_it_beats_the_baseline():
    assert stage1_choice(report(best="trial-01"), BASE) == {
        "window_stride_seconds": 30,
        "max_rows": 8_000_000,
    }


@pytest.mark.parametrize(
    "stage1", [report(best="baseline"), report(best="trial-01", verdict="not significant")]
)
def test_stage_2_keeps_the_base_data_unless_stage_1_wins(stage1):
    assert stage1_choice(stage1, BASE) == {"window_stride_seconds": 0, "max_rows": 4_000_000}


def test_stage_2_config_applies_the_choice_and_is_validated():
    base = small_config()
    cfg = stage2_config(base, {"window_stride_seconds": 30, "max_rows": 1000})
    assert cfg["data"]["window_stride_seconds"] == 30 and cfg["data"]["max_rows"] == 1000
    assert cfg["model"] == base["model"]
    with pytest.raises(solia_ai.config.ConfigError):
        stage2_config(base, {"window_stride_seconds": -1})


class FakeSearch:
    """Stands for solia_ai.search.main: records each call and writes the stage's report."""

    def __init__(self, reports, *, fail_stage=None, stage1=None, stage2=None):
        self.calls: list[tuple[list[str], str | None]] = []
        self.fail_stage = fail_stage
        self.reports = {STAGE1: stage1 or report(best="trial-01"), STAGE2: stage2 or report()}

    def __call__(self, argv):
        space = argv[argv.index("--space") + 1]
        self.calls.append((argv, os.environ.get(solia_ai.config.CONFIG_ENV)))
        if space == self.fail_stage:
            raise SystemExit("the features table is empty: nothing to train on")
        out = argv[argv.index("--out") + 1]
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.reports[space], fh)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: {**small_config(), **BASE})
    monkeypatch.delenv(solia_ai.config.CONFIG_ENV, raising=False)

    def install(fake):
        monkeypatch.setattr(solia_ai.search, "main", fake)
        return fake

    return install


def test_run_chains_stage_1_then_stage_2_on_its_choice(offline, tmp_path):
    fake = offline(FakeSearch(tmp_path))

    code = overnight.main(["run", "--reports-dir", str(tmp_path), "--device", "cuda"])

    assert code == 0
    (argv1, env1), (argv2, env2) = fake.calls
    assert argv1[:6] == ["--space", STAGE1, "--strategy", "grid", "--trials", "5"]
    assert argv1[argv1.index("--seeds") + 1 : argv1.index("--objective")] == ["0", "1"]
    assert argv1[argv1.index("--objective") + 1] == "pr_auc"
    assert argv1[argv1.index("--out") + 1] == str(tmp_path / f"{STAGE1}.json")
    assert argv2[:6] == ["--space", STAGE2, "--strategy", "random", "--trials", "10"]
    assert env1 is None and env2 == str(tmp_path / STAGE2_CONFIG)
    cfg = yaml.safe_load((tmp_path / STAGE2_CONFIG).read_text(encoding="utf-8"))
    assert cfg["data"]["window_stride_seconds"] == 30 and cfg["data"]["max_rows"] == 8_000_000
    assert solia_ai.config.CONFIG_ENV not in os.environ  # restored
    summary = (tmp_path / SUMMARY).read_text(encoding="utf-8")
    assert "stage 2 runs on: {'window_stride_seconds': 30, 'max_rows': 8000000}" in summary
    assert "best: trial-01" in summary and "no failure" in summary


def test_run_passes_restart_to_both_stages(offline, tmp_path):
    fake = offline(FakeSearch(tmp_path))
    overnight.main(["run", "--reports-dir", str(tmp_path), "--restart"])
    assert all("--restart" in argv for argv, _ in fake.calls)


def test_a_failed_stage_1_skips_stage_2_and_exits_1(offline, tmp_path):
    fake = offline(FakeSearch(tmp_path, fail_stage=STAGE1))

    assert overnight.main(["run", "--reports-dir", str(tmp_path)]) == 1

    assert len(fake.calls) == 1
    summary = (tmp_path / SUMMARY).read_text(encoding="utf-8")
    assert f"{STAGE1} stopped: the features table is empty" in summary
    assert f"{STAGE2} skipped" in summary


def test_explicit_stride_and_budget_run_stage_2_without_stage_1(offline, tmp_path):
    fake = offline(FakeSearch(tmp_path, fail_stage=STAGE1))

    code = overnight.main(
        ["run", "--reports-dir", str(tmp_path), "--stride", "60", "--max-rows", "6000000"]
    )

    assert code == 1  # stage 1's failure is still reported
    assert [argv[1] for argv, _ in fake.calls] == [STAGE1, STAGE2]
    cfg = yaml.safe_load((tmp_path / STAGE2_CONFIG).read_text(encoding="utf-8"))
    assert cfg["data"]["window_stride_seconds"] == 60 and cfg["data"]["max_rows"] == 6_000_000


def test_a_failed_trial_is_reported_and_fails_the_run(offline, tmp_path):
    oom = {"name": "trial-03", "params": {"d_model": 128}, "error": "RuntimeError: CUDA OOM"}
    offline(FakeSearch(tmp_path, stage2=report(failed=[oom])))

    assert overnight.main(["run", "--reports-dir", str(tmp_path)]) == 1

    summary = (tmp_path / SUMMARY).read_text(encoding="utf-8")
    assert "FAILED trial-03" in summary and "1 failed" in summary
    assert f"{STAGE2} trial-03: RuntimeError: CUDA OOM" in summary


def test_a_crashing_stage_is_reported_not_swallowed(offline, tmp_path):
    def crash(argv):
        raise RuntimeError("driver lost")

    offline(crash)
    assert overnight.main(["run", "--reports-dir", str(tmp_path)]) == 1
    assert "crashed: RuntimeError: driver lost" in (tmp_path / SUMMARY).read_text(encoding="utf-8")


def test_the_summary_command_rewrites_it_from_the_reports(tmp_path, capsys):
    (tmp_path / f"{STAGE1}.json").write_text(json.dumps(report(best="trial-01")), encoding="utf-8")

    assert overnight.main(["summary", "--reports-dir", str(tmp_path)]) == 0

    text = (tmp_path / SUMMARY).read_text(encoding="utf-8")
    assert f"{STAGE1}: 2 trial(s) finished, 0 failed, stopped by trials, 2.0 h" in text
    assert "baseline: score 0.5000" in text and "trial-01: score 0.6000" in text
    assert "lift vs baseline 0.1000 ± 0.0100 (better)" in text
    assert f"{STAGE2}: no report" in text
    assert text in capsys.readouterr().out


def test_the_retrain_lock_check_sees_a_running_retrain(pg_url):
    from sqlalchemy import create_engine

    from solia_ai.retrain import retrain_lock

    engine = create_engine(pg_url)
    try:
        assert overnight.retrain_lock_held(engine) is False
        with retrain_lock(engine):  # a retrain holds it from another connection
            assert overnight.retrain_lock_held(engine) is True
        assert overnight.retrain_lock_held(engine) is False
    finally:
        engine.dispose()
