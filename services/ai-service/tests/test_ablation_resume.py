"""#188: the ablation CLI on the bounded training read, combined variants, and resume after a
reboot (checkpointed runs)."""

from __future__ import annotations

import json

import pytest
from test_search import FAKE_WINDOW_END, fake_bounded_read
from test_train import make_features_df, small_config

import solia_ai.ablation as ablation
import solia_ai.config
import solia_ai.db
import solia_ai.registry
from solia_ai.ablation import Variant, combined_variant, plan_variants, run_ablation
from solia_ai.features import BASELINE_GROUPS


@pytest.fixture
def no_registry(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an ablation run must never register a model")

    monkeypatch.setattr(solia_ai.registry, "register_model", forbidden)


@pytest.fixture
def offline_cli(monkeypatch):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", lambda *a, **k: object())
    monkeypatch.setattr(solia_ai.db, "read_training_features", fake_bounded_read)


# --- combined variants ---


def test_a_combination_adds_several_candidate_groups_as_one_variant():
    variant = combined_variant(["momentum", "concentration"])

    assert variant == Variant(
        "+momentum+concentration",
        (*BASELINE_GROUPS, "momentum", "concentration"),
        "momentum+concentration",
        "add",
    )


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (["momentum"], "at least two groups"),
        (["momentum", "momentum"], "at least two groups"),
        (["momentum", "bogus"], "unknown feature group.*bogus"),
        (["momentum", "price"], "baseline group.*price"),
    ],
)
def test_a_bad_combination_is_rejected(groups, message):
    with pytest.raises(ValueError, match=message):
        combined_variant(groups)


def test_main_measures_a_combination_next_to_the_single_groups(offline_cli, no_registry, tmp_path):
    out = tmp_path / "ablation.json"
    ablation.main(["--groups", "momentum", "concentration", "--combine", "momentum",
                   "concentration", "--windows", "1", "--seeds", "0", "--device", "cpu",
                   "--out", str(out)])  # fmt: skip

    payload = json.loads(out.read_text(encoding="utf-8"))
    names = [v["name"] for v in payload["variants"]]
    assert names == ["baseline", "+momentum", "+concentration", "+momentum+concentration"]
    assert set(payload["lifts"]) == set(names[1:])


def test_main_rejects_a_bad_combination_before_reading_data(monkeypatch):
    monkeypatch.setattr(solia_ai.config, "load_config", lambda *a, **k: small_config())
    monkeypatch.setattr(solia_ai.db, "make_engine", None)  # must not be called

    with pytest.raises(SystemExit, match="at least two groups"):
        ablation.main(["--combine", "momentum"])


# --- bounded read ---


def test_main_reads_the_bounded_training_window_not_the_whole_table(
    offline_cli, no_registry, monkeypatch, capsys
):
    read: list = []

    def recording_read(engine, settings, end=None):
        read.append((settings.lookback_days, settings.max_rows, end))
        return fake_bounded_read(engine, settings, end)

    monkeypatch.setattr(solia_ai.db, "read_training_features", recording_read)
    ablation.main(["--groups", "holders", "--windows", "1", "--seeds", "0", "--device", "cpu",
                   "--lookback-days", "1.5", "--max-rows", "500"])  # fmt: skip

    assert read == [(1.5, 500, None)]
    assert "max_rows 500" in capsys.readouterr().err  # the read's bounds are logged


def test_main_defaults_to_the_config_data_bounds(offline_cli, no_registry, monkeypatch):
    read: list = []

    def recording_read(engine, settings, end=None):
        read.append(settings)
        return fake_bounded_read(engine, settings, end)

    monkeypatch.setattr(solia_ai.db, "read_training_features", recording_read)
    ablation.main(["--groups", "holders", "--windows", "1", "--seeds", "0", "--device", "cpu"])

    assert read == [solia_ai.db.DataSettings.from_config(small_config())]


def test_main_reports_an_empty_read_cleanly(offline_cli, monkeypatch):
    def empty(*_a, **_k):
        raise ValueError("no feature rows in the window")

    monkeypatch.setattr(solia_ai.db, "read_training_features", empty)
    with pytest.raises(SystemExit, match="no feature rows"):
        ablation.main(["--windows", "1", "--seeds", "0", "--device", "cpu"])


# --- resume ---

RESUME = {"variants": plan_variants(["holders"]), "n_windows": 2, "seeds": [0]}
DATA = {"window_end": "2026-10-30T12:00:00+00:00", "rows_read": 123}


class Interrupted(BaseException):
    """Stands for the reboot: not an Exception, so nothing can swallow it."""


def counting_trainer(monkeypatch, *, die_on: int | None = None) -> list[int]:
    """Counts train_and_score calls; the ``die_on``-th call (1-based) interrupts the process."""
    calls: list[int] = []
    real = ablation.train_and_score

    def trainer(*args, **kwargs):
        calls.append(1)
        if die_on is not None and len(calls) == die_on:
            raise Interrupted
        return real(*args, **kwargs)

    monkeypatch.setattr(ablation, "train_and_score", trainer)
    return calls


def run_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_checkpoint_name_follows_the_report():
    assert ablation.checkpoint_path("/app/models/reports/x.json").name == "x.runs.jsonl"


def test_an_interrupted_ablation_resumes_where_it_stopped(no_registry, tmp_path, monkeypatch):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "a.runs.jsonl"
    full = run_ablation(df, cfg, **RESUME)

    counting_trainer(monkeypatch, die_on=3)  # both baseline runs finish, the third dies
    with pytest.raises(Interrupted):
        run_ablation(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    lines = run_lines(ckpt)
    assert lines[0]["kind"] == "header" and [r["kind"] for r in lines[1:]] == ["run", "run"]

    calls = counting_trainer(monkeypatch)
    resumed = run_ablation(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)

    assert len(calls) == 2 and resumed.resumed == 2  # only the two -holders runs trained
    assert [(r.variant, r.window, r.seed) for r in resumed.runs] == [
        (r.variant, r.window, r.seed) for r in full.runs
    ]
    for got, want in zip(resumed.runs, full.runs, strict=True):
        assert got.metrics == pytest.approx(want.metrics, nan_ok=True)
    assert resumed.lifts.keys() == full.lifts.keys()
    assert "resumed from the checkpoint: 2 run(s)" in resumed.render()

    calls = counting_trainer(monkeypatch)
    again = run_ablation(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    assert calls == [] and again.resumed == 4


def test_a_torn_last_line_reruns_that_run(no_registry, tmp_path, monkeypatch):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "a.runs.jsonl"
    run_ablation(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    text = ckpt.read_text(encoding="utf-8")
    ckpt.write_text(text[: len(text) - 40], encoding="utf-8")  # the process died mid-write

    calls = counting_trainer(monkeypatch)
    report = run_ablation(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)

    assert len(calls) == 1 and report.resumed == 3


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seeds": [0, 1]}, "settings.seeds"),
        ({"n_windows": 1}, "settings.n_windows"),
        ({"variants": plan_variants(["flow"])}, "variants"),
        ({"data": {**DATA, "rows_read": 124}}, "data.rows_read"),
        ({"config": True}, "config.model"),
    ],
)
def test_a_checkpoint_of_another_ablation_is_refused(
    change, message, no_registry, tmp_path, monkeypatch
):
    df, cfg, ckpt = make_features_df(), small_config(), tmp_path / "a.runs.jsonl"
    run_ablation(df, cfg, **RESUME, checkpoint=ckpt, data=DATA)
    before = ckpt.read_text(encoding="utf-8")
    kwargs = {**RESUME, "data": DATA, **change}
    if kwargs.pop("config", False):
        cfg = {**cfg, "model": {**cfg["model"], "dropout": 0.3}}
    calls = counting_trainer(monkeypatch)

    with pytest.raises(ValueError, match=f"another ablation.*{message}"):
        run_ablation(df, cfg, **kwargs, checkpoint=ckpt)

    assert calls == [] and ckpt.read_text(encoding="utf-8") == before  # untouched


def test_without_a_checkpoint_nothing_is_written(no_registry, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_ablation(make_features_df(), small_config(), **RESUME)
    assert not list(tmp_path.iterdir())


def test_main_resumes_on_the_window_it_started_on(offline_cli, no_registry, tmp_path, monkeypatch):
    ends: list = []

    def recording_read(engine, settings, end=None):
        ends.append(end)
        return fake_bounded_read(engine, settings, end)

    monkeypatch.setattr(solia_ai.db, "read_training_features", recording_read)
    out = tmp_path / "ablation.json"
    argv = ["--groups", "holders", "--windows", "1", "--seeds", "0", "--device", "cpu",
            "--out", str(out)]  # fmt: skip

    ablation.main(argv)
    ckpt = ablation.checkpoint_path(out)
    assert len(run_lines(ckpt)) == 3  # header, baseline, -holders
    calls = counting_trainer(monkeypatch)
    ablation.main(argv)
    assert calls == []  # everything resumed
    assert ends == [None, FAKE_WINDOW_END]  # the rerun read the first run's window
    assert json.loads(out.read_text(encoding="utf-8"))["resumed"] == 2

    ablation.main([*argv, "--restart"])
    assert ends[-1] is None and len(calls) == 2  # discarded: today's window, trained again


def test_main_refuses_a_foreign_checkpoint_before_reading(offline_cli, tmp_path, monkeypatch):
    out = tmp_path / "ablation.json"
    ablation.checkpoint_path(out).write_text("not json\n", encoding="utf-8")
    monkeypatch.setattr(solia_ai.db, "read_training_features", None)  # must not be called
    with pytest.raises(SystemExit, match="not an ablation runs file"):
        ablation.main(["--windows", "1", "--out", str(out)])
