"""Checkpoint retention: what is kept, dry run, MODEL_DIR containment, orphans, rollback refusal.

The database tests run against the shared Testcontainers PostgreSQL (skipped locally without
Docker); the path rules are unit-tested without it.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from test_retrain import (
    T0,
    fake_checkpoint,
    gate_config,
    promote_chain,
    run,
    stub,
)
from test_train import BOUNDS

from solia_ai import retrain
from solia_ai.config import ConfigError, validate_retention
from solia_ai.registry import read_champion, register_model
from solia_ai.retention import (
    PruneError,
    RetentionSettings,
    apply_prune,
    describe,
    fallback_directory,
    locate,
    model_directory,
    plan_prune,
    possible_locations,
)
from solia_ai.retrain import RetrainBusyError, RollbackRefusedError, prune, retrain_lock, rollback


def config(**retention) -> dict:
    return {**gate_config(), "retention": retention} if retention else gate_config()


# --------------------------------------------------------------------------- settings


def test_settings_default_and_read_the_config():
    assert RetentionSettings.from_config(config()) == RetentionSettings(5, 2)
    assert RetentionSettings.from_config(config(keep_candidates=0)) == RetentionSettings(0, 2)


@pytest.mark.parametrize(
    "retention, message",
    [
        ({"keep_candidates": -1}, "must be >= 0"),
        ({"rollback_depth": 1.5}, "must be an integer"),
        ({"keep": 3}, "unknown key"),
    ],
)
def test_invalid_settings_are_refused(retention, message):
    with pytest.raises(ConfigError, match=message):
        RetentionSettings.from_config(config(**retention))


@pytest.mark.parametrize("config_value", [["not", "a", "mapping"], {"retention": [3]}])
def test_a_malformed_retention_config_is_refused(config_value):
    with pytest.raises(ConfigError, match="must be a mapping|expected a mapping"):
        validate_retention(config_value)


def test_settings_ignore_a_broken_unrelated_section():
    broken = {**config(keep_candidates=1), "model": "not a mapping", "labeling": None}
    assert RetentionSettings.from_config(broken) == RetentionSettings(1, 2)


def test_the_packaged_config_has_valid_settings():
    from solia_ai.config import load_config

    assert RetentionSettings.from_config(load_config()).keep_candidates >= 1


def test_model_directory_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert model_directory() == tmp_path.resolve()
    assert model_directory(tmp_path / "other") == (tmp_path / "other").resolve()


def test_the_fallback_follows_serving(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert fallback_directory() is None  # serving has no file-name fallback either
    assert model_directory() == (Path.cwd() / "models").resolve()
    assert fallback_directory(tmp_path) == tmp_path.resolve()  # --model-dir plays $MODEL_DIR
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "env"))
    assert fallback_directory() == (tmp_path / "env").resolve()


# --------------------------------------------------------------------------- path rules


def test_locate_finds_a_file_inside_the_directory(tmp_path):
    file = tmp_path / "a.pt"
    file.write_bytes(b"x")
    assert locate(str(file), tmp_path, tmp_path) == (file.resolve(), None)


def test_locate_falls_back_to_the_file_name_in_the_directory(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"x")
    found, refusal = locate("/elsewhere/on/the/host/a.pt", tmp_path, tmp_path)
    assert (found, refusal) == ((tmp_path / "a.pt").resolve(), None)
    found, _ = locate("C:\\Users\\me\\models\\a.pt", tmp_path, tmp_path)  # Windows path
    assert found == (tmp_path / "a.pt").resolve()


def test_locate_without_a_fallback_only_checks_the_registered_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))  # an explicit None ignores $MODEL_DIR
    (tmp_path / "a.pt").write_bytes(b"x")
    assert locate("/elsewhere/on/the/host/a.pt", tmp_path, None) == (None, None)


def test_locate_tolerates_a_missing_file(tmp_path):
    assert locate(str(tmp_path / "gone.pt"), tmp_path, tmp_path) == (None, None)


def test_locate_refuses_a_file_outside_the_directory(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    for path in (str(outside), str(directory / ".." / "outside.pt")):
        found, refusal = locate(path, directory.resolve(), directory.resolve())
        assert found is None and "outside MODEL_DIR" in refusal


def test_locate_never_resolves_to_the_directory_itself(tmp_path):
    assert locate(str(tmp_path / ".."), tmp_path, tmp_path) == (None, None)
    assert locate("..", tmp_path, tmp_path)[0] is None


def test_locate_refuses_a_symlink_leaving_the_directory(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    target = tmp_path / "secret.pt"
    target.write_bytes(b"x")
    link = directory / "link.pt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable here")
    found, refusal = locate(str(link), directory.resolve(), directory.resolve())
    assert found is None and "outside MODEL_DIR" in refusal


def test_possible_locations_cover_the_file_name_fallback(tmp_path):
    assert possible_locations("/host/models/a.pt", (tmp_path, None)) == {
        Path("/host/models/a.pt").resolve(),
        (tmp_path / "a.pt").resolve(),
    }


# --------------------------------------------------------------------------- database


@pytest.fixture
def engine(pg_url):
    eng = create_engine(pg_url)
    with eng.begin() as conn:
        conn.exec_driver_sql(
            "TRUNCATE training_events, models, logs, trades, tokens_history RESTART IDENTITY "
            "CASCADE"
        )
    yield eng
    eng.dispose()


def pruned_versions(engine) -> set[str]:
    with engine.connect() as conn:
        return set(
            conn.execute(text("SELECT version FROM models WHERE pruned_at IS NOT NULL")).scalars()
        )


def age(file, *, hours):
    old = time.time() - hours * 3600
    os.utime(file, (old, old))


def add_candidates(engine, directory, versions):
    """Rejected candidates registered after the chain, newest last."""
    for version in versions:
        path = fake_checkpoint(directory / f"{version}.pt", version)
        register_model(engine, version=version, path=str(path))


@pytest.fixture
def history(engine, tmp_path):
    """Champions c0..c4 promoted in turn (c4 serves), then candidates r0..r3 rejected."""
    promote_chain(engine, tmp_path, [f"c{i}" for i in range(5)])
    add_candidates(engine, tmp_path, [f"r{i}" for i in range(4)])
    return tmp_path


def test_keeps_the_champion_its_rollback_chain_and_the_newest_candidates(engine, history):
    settings = RetentionSettings(keep_candidates=3, rollback_depth=2)
    plan = plan_prune(engine, settings, history)
    kept = {m.version: why for m, why in plan.kept}
    assert kept == {
        "c4": "champion",
        "c3": "rollback target 1",
        "c2": "rollback target 2",
        "r3": "recent",
        "r2": "recent",
        "r1": "recent",
    }
    assert {p.model.version for p in plan.prune} == {"c0", "c1", "r0"}

    done = apply_prune(engine, plan)
    assert set(done.marked) == {"c0", "c1", "r0"} and not done.errors
    assert sorted(p.name for p in history.glob("*.pt")) == [
        "c2.pt", "c3.pt", "c4.pt", "r1.pt", "r2.pt", "r3.pt",
    ]  # fmt: skip
    assert pruned_versions(engine) == {"c0", "c1", "r0"}
    with engine.connect() as conn:  # the rows stay, with their history
        assert conn.execute(text("SELECT count(*) FROM models")).scalar() == 9
    assert read_champion(engine).version == "c4"
    # Already pruned rows are not planned again.
    assert plan_prune(engine, settings, history).prune == ()


def test_keep_candidates_zero_keeps_only_the_champion_and_the_chain(engine, history):
    plan = plan_prune(engine, RetentionSettings(0, 1), history)
    assert {m.version for m, _ in plan.kept} == {"c4", "c3"}


def test_a_dry_run_deletes_and_writes_nothing(engine, history):
    plan = plan_prune(engine, RetentionSettings(1, 1), history)
    done = apply_prune(engine, plan, dry_run=True, delete_orphans=True)
    assert done.dry_run and done.marked == () and done.deleted == ()
    assert len(list(history.glob("*.pt"))) == 9
    assert pruned_versions(engine) == set()
    report = describe(done)
    assert (
        report[-1] == "would delete 6 checkpoint(s), 0 orphan(s) found — dry run: nothing changed"
    )
    assert f"prune   c0: {(history / 'c0.pt').resolve()}" in report


def test_a_missing_file_is_reported_and_left_unmarked(engine, history):
    (history / "c0.pt").unlink()
    plan = plan_prune(engine, RetentionSettings(4, 2), history)
    assert [m.version for m in plan.missing] == ["c0"]
    assert "c0" not in {p.model.version for p in plan.prune}
    done = apply_prune(engine, plan)
    assert "c0" not in done.marked and not done.errors
    assert "c0" not in pruned_versions(engine)
    assert "missing c0: checkpoint not found, row left as is" in describe(done)


def test_the_wrong_directory_is_refused_and_nothing_is_marked(engine, history, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PruneError, match="no registered checkpoint found"):
        plan_prune(engine, RetentionSettings(0, 0), empty, fallback_dir=empty)
    with pytest.raises(PruneError):
        prune(engine, config(keep_candidates=0, rollback_depth=0), model_dir=empty)
    assert pruned_versions(engine) == set()
    assert len(list(history.glob("*.pt"))) == 9


def test_main_prune_reports_the_wrong_directory(monkeypatch, engine, history, tmp_path):
    monkeypatch.setattr("solia_ai.db.make_engine", lambda: engine)
    monkeypatch.setattr("solia_ai.config.load_config", lambda: config())
    with pytest.raises(SystemExit, match="wrong --model-dir"):
        retrain.main(["prune", "--model-dir", str(tmp_path / "nowhere")])
    assert pruned_versions(engine) == set()


def test_an_empty_registry_prunes_nothing_without_error(engine, tmp_path):
    plan = plan_prune(engine, RetentionSettings(0, 0), tmp_path)
    assert plan.prune == () and plan.kept == ()


def test_a_file_outside_the_model_dir_is_never_deleted(engine, tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    outside = fake_checkpoint(tmp_path / "outside.pt", "old")
    register_model(engine, version="old", path=str(outside))
    promote_chain(engine, directory, ["c0"])
    plan = plan_prune(engine, RetentionSettings(0, 0), directory)
    assert plan.prune == ()
    assert [(m.version, "outside MODEL_DIR" in why) for m, why in plan.refused] == [("old", True)]
    apply_prune(engine, plan)
    assert outside.is_file() and pruned_versions(engine) == set()


def test_a_traversal_path_is_refused(engine, tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    victim = fake_checkpoint(tmp_path / "victim.pt", "old")
    register_model(engine, version="old", path=str(directory / ".." / "victim.pt"))
    promote_chain(engine, directory, ["c0"])
    plan = plan_prune(engine, RetentionSettings(0, 0), directory)
    assert [m.version for m, _ in plan.refused] == ["old"]
    apply_prune(engine, plan)
    assert victim.is_file()


def test_the_champions_file_is_never_deleted_even_if_an_old_row_shares_it(engine, tmp_path):
    champion = promote_chain(engine, tmp_path, ["c0"])
    # An older row registered on another host with the same file name: same fallback file.
    register_model(engine, version="twin", path="/other/host/c0.pt")
    with engine.begin() as conn:  # make it the oldest
        conn.execute(text("UPDATE models SET created_at = :t WHERE version = 'twin'"),
                     {"t": T0 - timedelta(days=1)})  # fmt: skip
    plan = plan_prune(engine, RetentionSettings(0, 0), tmp_path)
    assert [m.version for m, _ in plan.refused] == ["twin"]
    apply_prune(engine, plan)
    assert (tmp_path / "c0.pt").is_file()
    assert read_champion(engine).id == champion.id


def test_the_database_refuses_a_pruned_champion(engine, tmp_path):
    promote_chain(engine, tmp_path, ["c0"])
    with pytest.raises(IntegrityError, match="models_champion_not_pruned_check"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE models SET pruned_at = now() WHERE is_champion"))


def test_orphans_are_reported_and_deleted_only_with_the_flag(engine, history):
    orphan = history / "crashed-run.pt"
    orphan.write_bytes(b"x")
    age(orphan, hours=2)
    (history / "notes.txt").write_text("not a checkpoint")
    plan = plan_prune(engine, RetentionSettings(10, 10), history)
    assert plan.orphans == (orphan.resolve(),)
    apply_prune(engine, plan)
    assert orphan.is_file()
    assert any("--delete-orphans" in line for line in describe(apply_prune(engine, plan)))
    done = apply_prune(engine, plan, delete_orphans=True)
    assert done.orphans_deleted == (orphan.resolve(),) and not orphan.exists()
    assert (history / "notes.txt").is_file()


def test_a_file_that_cant_be_deleted_is_reported_and_not_marked(engine, history, monkeypatch):
    plan = plan_prune(engine, RetentionSettings(4, 2), history)
    real_unlink = type(history).unlink

    def locked(self, *args, **kwargs):
        if self.name == "c0.pt":
            raise PermissionError("in use")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(type(history), "unlink", locked)
    done = apply_prune(engine, plan)
    assert "c0" not in done.marked and any("c0" in e for e in done.errors)
    assert "c0" not in pruned_versions(engine)
    report = describe(done)
    assert f"failed  c0: {(history / 'c0.pt').resolve()}" in report
    assert not any(line.startswith("pruned  c0") for line in report)


def test_rollback_to_a_pruned_model_is_refused(engine, tmp_path):
    promote_chain(engine, tmp_path, ["a", "b"])
    plan = plan_prune(engine, RetentionSettings(0, 0), tmp_path)
    assert [p.model.version for p in plan.prune] == ["a"]
    apply_prune(engine, plan)
    with pytest.raises(RollbackRefusedError, match="pruned by the retention policy"):
        rollback(engine, gate_config())
    assert read_champion(engine).version == "b"


def test_prune_takes_the_retrain_lock(engine, history):
    with retrain_lock(engine):
        with pytest.raises(RetrainBusyError):
            prune(engine, config(keep_candidates=0, rollback_depth=0), model_dir=history)
    done = prune(engine, config(keep_candidates=0, rollback_depth=0), model_dir=history)
    assert set(done.marked) == {"c0", "c1", "c2", "c3", "r0", "r1", "r2", "r3"}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM logs WHERE level = 'info'")).scalar() == 1


def test_a_retrain_prunes_once_it_is_recorded(engine, history, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(history))
    cfg = config(keep_candidates=2, rollback_depth=1)
    outcome = retrain.run_retrain(
        engine,
        cfg,
        trigger="manual",
        threshold=5,
        evaluate_fn=stub(0.4, 0.5),  # rejected: c4 stays champion
        read_data_fn=lambda engine, data: (pd.DataFrame({"x": [1, 2]}), BOUNDS),
        load_champion_fn=lambda champion, config: None,
    )
    assert outcome.status == "rejected"
    # Kept: c4 (champion), c3 (rollback), cand + r3 (newest).
    assert set(outcome.pruned) == {"c0", "c1", "c2", "r0", "r1", "r2"}
    assert sorted(p.name for p in history.glob("*.pt")) == ["c3.pt", "c4.pt", "r3.pt"]


def test_a_failed_prune_never_fails_the_run(engine, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    def broken(*args, **kwargs):
        raise OSError("volume unmounted")

    monkeypatch.setattr(retrain, "plan_prune", broken)
    outcome = run(engine, stub(0.6))
    assert outcome.status == "promoted" and outcome.pruned == ()
    with engine.connect() as conn:
        message = conn.execute(
            text("SELECT message FROM logs WHERE message LIKE 'retention after%'")
        ).scalar_one()
    assert "failed: OSError: volume unmounted" in message


def test_main_prune_dry_run(monkeypatch, capsys, engine, history):
    monkeypatch.setattr("solia_ai.db.make_engine", lambda: engine)
    seen = {}
    real = retrain.prune

    def spy(engine, **kwargs):
        seen.update(kwargs)
        return real(engine, config(keep_candidates=0, rollback_depth=0), **kwargs)

    monkeypatch.setattr(retrain, "prune", spy)
    retrain.main(["prune", "--dry-run", "--model-dir", str(history)])
    out = capsys.readouterr().out
    assert seen["dry_run"] and not seen["delete_orphans"]
    assert "would delete 8 checkpoint(s)" in out and "dry run: nothing changed" in out
    assert len(list(history.glob("*.pt"))) == 9


# --------------------------------------------------------------------------- review follow-ups


def test_pruned_rows_and_rollback_targets_take_no_candidate_slot(engine, history):
    apply_prune(engine, plan_prune(engine, RetentionSettings(1, 2), history))
    assert pruned_versions(engine) == {"c0", "c1", "r0", "r1", "r2"}
    # Raised back: the newest rows are pruned, so the slots go to the next checkpoints still there.
    plan = plan_prune(engine, RetentionSettings(5, 2), history)
    kept = {m.version: why for m, why in plan.kept}
    assert kept == {
        "c4": "champion",
        "c3": "rollback target 1",
        "c2": "rollback target 2",
        "r3": "recent",
    }
    assert plan.prune == ()


def test_a_pruned_rollback_target_is_shown_as_such(engine, tmp_path):
    promote_chain(engine, tmp_path, ["a", "b"])
    apply_prune(engine, plan_prune(engine, RetentionSettings(0, 0), tmp_path))
    plan = plan_prune(engine, RetentionSettings(0, 1), tmp_path)
    assert dict((m.version, why) for m, why in plan.kept)["a"] == (
        "rollback target 1, checkpoint already pruned"
    )


def test_a_row_changed_since_the_plan_is_skipped_and_its_file_kept(engine, history):
    plan = plan_prune(engine, RetentionSettings(4, 2), history)
    assert {p.model.version for p in plan.prune} == {"c0", "c1"}
    with engine.begin() as conn:  # promoted outside the retrain lock after the plan
        conn.execute(text("UPDATE models SET is_champion = false WHERE is_champion"))
        conn.execute(text("UPDATE models SET is_champion = true WHERE version = 'c0'"))
    done = apply_prune(engine, plan)
    assert done.marked == ("c1",)
    assert any("c0 changed since the plan" in e for e in done.errors)
    assert (history / "c0.pt").is_file() and not (history / "c1.pt").exists()
    assert read_champion(engine).version == "c0"
    assert pruned_versions(engine) == {"c1"}


def test_a_recent_orphan_is_never_deleted(engine, history):
    fresh = history / "being-written.pt"
    fresh.write_bytes(b"x")
    plan = plan_prune(engine, RetentionSettings(10, 10), history)
    assert plan.orphans == () and plan.recent_orphans == (fresh.resolve(),)
    done = apply_prune(engine, plan, delete_orphans=True)
    assert fresh.is_file() and done.orphans_deleted == ()
    assert any("written < 60 min ago, kept" in line for line in describe(done))


def test_an_orphan_rewritten_after_the_plan_is_kept(engine, history):
    orphan = history / "reused.pt"
    orphan.write_bytes(b"x")
    age(orphan, hours=2)
    plan = plan_prune(engine, RetentionSettings(10, 10), history)
    assert plan.orphans == (orphan.resolve(),)
    orphan.write_bytes(b"new run")  # fresh mtime again
    assert apply_prune(engine, plan, delete_orphans=True).orphans_deleted == ()
    assert orphan.is_file()


def test_lasting_warnings_are_logged_once(engine, history, tmp_path):
    outside = fake_checkpoint(tmp_path.parent / f"{tmp_path.name}-outside.pt", "far")
    register_model(engine, version="far", path=str(outside))
    with engine.begin() as conn:  # the oldest row, so not kept as recent
        conn.execute(text("UPDATE models SET created_at = :t WHERE version = 'far'"),
                     {"t": T0 - timedelta(days=1)})  # fmt: skip
    orphan = history / "crashed-run.pt"
    orphan.write_bytes(b"x")
    age(orphan, hours=2)
    cfg = config(keep_candidates=4, rollback_depth=10)  # keeps every row but "far"
    for _ in range(3):
        prune(engine, cfg, model_dir=history)
    with engine.connect() as conn:
        warnings = (
            conn.execute(text("SELECT message FROM logs WHERE level = 'warn' ORDER BY id"))
            .scalars()
            .all()
        )
    assert len(warnings) == 2
    assert any("far not pruned" in w for w in warnings)
    assert any("crashed-run.pt" in w for w in warnings)


def test_a_dry_run_needs_no_lock(engine, history):
    with retrain_lock(engine):
        done = prune(engine, config(keep_candidates=0, rollback_depth=0), model_dir=history,
                     dry_run=True)  # fmt: skip
    assert done.dry_run and len(done.plan.prune) == 8
    assert pruned_versions(engine) == set() and len(list(history.glob("*.pt"))) == 9
