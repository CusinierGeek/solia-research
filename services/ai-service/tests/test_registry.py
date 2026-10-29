"""The models registry against a real PostgreSQL (Testcontainers), plus train() writing a row."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import check_values
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from test_train import make_features_df, small_config

from solia_ai.registry import (
    NO_MODEL_TRAINED_DECISIONS,
    RESET_TRADE_COUNT_DECISIONS,
    TRAINING_DECISIONS,
    TRAINING_TRIGGERS,
    last_train_cutoff,
    read_champion,
    register_model,
)
from solia_ai.train import train

T0 = datetime(2026, 1, 10, tzinfo=UTC)


@pytest.fixture
def engine(pg_url):
    eng = create_engine(pg_url)
    with eng.begin() as conn:
        conn.exec_driver_sql("TRUNCATE models RESTART IDENTITY CASCADE")
    yield eng
    eng.dispose()


def _rows(engine):
    with engine.connect() as conn:
        columns = "id, version, path, is_champion, metrics, hyperparams, n_params"
        return conn.execute(text(f"SELECT {columns} FROM models")).mappings().all()


def test_register_model_persists_every_column(engine):
    record = register_model(
        engine,
        version="v1",
        path="/models/v1.pt",
        metrics={"val_total": 0.42},
        hyperparams={"model": {"d_model": 16}, "horizons": ["30s"]},
        n_params=12345,
        is_champion=True,
    )

    (row,) = _rows(engine)
    assert record.id == row["id"]
    assert row["version"] == "v1"
    assert row["path"] == "/models/v1.pt"
    assert row["is_champion"] is True
    assert row["metrics"] == {"val_total": 0.42}  # JSONB round-trips to a dict
    assert row["hyperparams"] == {"model": {"d_model": 16}, "horizons": ["30s"]}
    assert row["n_params"] == 12345
    assert record.created_at is not None


def test_null_metrics_and_hyperparams_are_allowed(engine):
    register_model(engine, version="bare", path="/models/bare.pt")

    (row,) = _rows(engine)
    assert row["metrics"] is None
    assert row["hyperparams"] is None
    assert row["n_params"] is None
    assert row["is_champion"] is False


def test_promoting_a_champion_demotes_the_previous_one(engine):
    register_model(engine, version="old", path="/old.pt", is_champion=True)
    register_model(engine, version="new", path="/new.pt", is_champion=True)

    champions = [r["version"] for r in _rows(engine) if r["is_champion"]]
    assert champions == ["new"]  # exactly one champion, the latest


def test_non_champion_rows_coexist(engine):
    register_model(engine, version="a", path="/a.pt", is_champion=False)
    register_model(engine, version="b", path="/b.pt", is_champion=False)

    assert len([r for r in _rows(engine) if not r["is_champion"]]) == 2


def test_read_champion_is_none_on_an_empty_registry(engine):
    assert read_champion(engine) is None


def test_read_champion_is_none_when_nothing_was_promoted(engine):
    register_model(engine, version="plain", path="/plain.pt", is_champion=False)

    assert read_champion(engine) is None


def test_read_champion_returns_the_promoted_row(engine):
    registered = register_model(
        engine, version="champ", path="/models/champ.pt", n_params=42, is_champion=True
    )
    register_model(engine, version="other", path="/other.pt", is_champion=False)

    record = read_champion(engine)

    assert record == registered  # every column mapped back from the row
    assert record.version == "champ"
    assert record.path == "/models/champ.pt"
    assert record.is_champion is True
    assert record.n_params == 42


def test_read_champion_follows_a_promotion(engine):
    register_model(engine, version="old", path="/old.pt", is_champion=True)
    register_model(engine, version="new", path="/new.pt", is_champion=True)

    assert read_champion(engine).version == "new"


def test_read_champion_maps_a_null_n_params(engine):
    register_model(engine, version="bare", path="/bare.pt", is_champion=True)

    assert read_champion(engine).n_params is None


def test_duplicate_version_is_rejected(engine):
    register_model(engine, version="dup", path="/dup.pt")

    with pytest.raises(ValueError, match="already exists"):
        register_model(engine, version="dup", path="/dup2.pt")


def test_non_finite_metrics_are_rejected_before_writing(engine):
    with pytest.raises(ValueError, match="finite"):
        register_model(engine, version="nan", path="/nan.pt", metrics={"val_return": float("nan")})

    assert _rows(engine) == []


def test_blank_version_is_rejected(engine):
    with pytest.raises(ValueError, match="non-empty"):
        register_model(engine, version="   ", path="/x.pt")


def test_train_writes_a_champion_registry_row(engine, tmp_path):
    result = train(
        make_features_df(),
        small_config(),
        model_dir=tmp_path,
        engine=engine,
        version="trained",
    )

    (row,) = _rows(engine)
    assert result.model_id == row["id"]
    assert row["version"] == "trained"
    assert row["is_champion"] is True
    assert row["n_params"] == result.n_params
    assert row["path"] == str(result.checkpoint_path)
    assert "val_total" in row["metrics"]
    assert row["hyperparams"]["horizons"] == ["30s", "1m"]


# --- IntegrityError mapping (no database needed) ---


class _FailingEngine:
    """An engine whose INSERT fails with a constraint violation named ``constraint``."""

    def __init__(self, constraint):
        orig = SimpleNamespace(diag=SimpleNamespace(constraint_name=constraint))
        self.error = IntegrityError("INSERT INTO models ...", {}, orig)

    @contextmanager
    def begin(self):
        error = self.error

        class Conn:
            def execute(self, statement, params=None):
                if str(statement).startswith("INSERT"):
                    raise error

            @contextmanager
            def begin_nested(self):
                yield

        yield Conn()


def test_a_concurrent_champion_promotion_is_reported_as_such():
    with pytest.raises(ValueError, match="promoted concurrently"):
        register_model(
            _FailingEngine("models_single_champion_idx"),
            version="v1",
            path="/v1.pt",
            is_champion=True,
        )


def test_a_duplicate_version_constraint_is_reported_as_such():
    with pytest.raises(ValueError, match="already exists"):
        register_model(_FailingEngine("models_version_key"), version="v1", path="/v1.pt")


def test_an_unknown_integrity_error_propagates_unchanged():
    with pytest.raises(IntegrityError):
        register_model(_FailingEngine("some_other_constraint"), version="v1", path="/v1.pt")


def test_training_triggers_mirror_the_sql_check(engine):
    assert set(TRAINING_TRIGGERS) == check_values(
        engine, "training_events", "training_events_trigger_check"
    )
    assert "manual" in TRAINING_TRIGGERS


def test_training_decisions_mirror_the_sql_check(engine):
    assert set(TRAINING_DECISIONS) == check_values(
        engine, "training_events", "training_events_decision_check"
    )


def test_only_a_run_that_trained_a_model_resets_the_trade_count():
    assert set(RESET_TRADE_COUNT_DECISIONS) == {"promoted", "rejected"}
    assert set(RESET_TRADE_COUNT_DECISIONS) | set(NO_MODEL_TRAINED_DECISIONS) == set(
        TRAINING_DECISIONS
    )


def _event(conn, started_at, decision, trigger="n_trades"):
    conn.execute(
        text(
            "INSERT INTO training_events (started_at, finished_at, trigger, decision) "
            "VALUES (:s, :f, :t, :d)"
        ),
        {
            "s": started_at,
            "f": None if decision is None else started_at,
            "d": decision,
            "t": trigger,
        },
    )


def test_last_train_cutoff_is_none_before_any_run(engine):
    assert last_train_cutoff(engine) is None


def test_last_train_cutoff_is_none_when_no_run_trained_a_model(engine):
    with engine.begin() as conn:
        _event(conn, T0, "failed")
        _event(conn, T0 + timedelta(hours=1), "rollback")
        _event(conn, T0 + timedelta(hours=2), None)
    assert last_train_cutoff(engine) is None


def test_last_train_cutoff_ignores_later_failed_in_flight_and_rollback_runs(engine):
    with engine.begin() as conn:
        _event(conn, T0 - timedelta(hours=3), "promoted", trigger="initial")
        _event(conn, T0, "rejected")
        _event(conn, T0 + timedelta(minutes=75), "failed", trigger="manual")
        _event(conn, T0 + timedelta(minutes=90), None)
        _event(conn, T0 + timedelta(minutes=100), "rollback", trigger="manual")
    assert last_train_cutoff(engine) == T0


def test_training_events_accept_every_trigger_and_decision(engine):
    with engine.begin() as conn:
        for trigger in TRAINING_TRIGGERS:
            for decision in TRAINING_DECISIONS:
                conn.execute(
                    text(
                        "INSERT INTO training_events (trigger, finished_at, decision) "
                        "VALUES (:t, now(), :d)"
                    ),
                    {"t": trigger, "d": decision},
                )
        count = conn.execute(text("SELECT count(*) FROM training_events")).scalar_one()
    assert count == len(TRAINING_TRIGGERS) * len(TRAINING_DECISIONS)


@pytest.mark.parametrize(
    ("column", "value"), [("trigger", "cron"), ("trigger", "Manual"), ("decision", "error")]
)
def test_training_events_reject_unknown_values(engine, column, value):
    with pytest.raises(IntegrityError, match="check"), engine.begin() as conn:
        conn.execute(text(f"INSERT INTO training_events ({column}) VALUES (:v)"), {"v": value})


@pytest.mark.parametrize(
    "columns",
    ["(trigger, decision) VALUES ('n_trades', 'promoted')", "(finished_at) VALUES (now())"],
)
def test_training_events_set_decision_and_finished_at_together(engine, columns):
    with pytest.raises(IntegrityError, match="training_events_finished_decision_check"):
        with engine.begin() as conn:
            conn.execute(text(f"INSERT INTO training_events {columns}"))
