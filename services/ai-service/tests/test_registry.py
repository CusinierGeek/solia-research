"""The models registry against a real PostgreSQL (Testcontainers), plus train() writing a row."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from test_train import make_features_df, small_config

from solia_ai.registry import read_champion, register_model
from solia_ai.train import train


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
