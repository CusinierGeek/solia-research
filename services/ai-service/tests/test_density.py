"""The features density check: its verdict and CLI (the query: test_density_integration)."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import OperationalError

from solia_ai import density
from solia_ai.config import ConfigError
from solia_ai.density import SlotDensity, main, over_limit

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
WINDOW = ["--from", "2026-11-01T12:00Z", "--to", "2026-11-01T13:00Z"]


def _use_slots(monkeypatch, slots):
    monkeypatch.setattr(density, "make_engine", lambda: None)
    monkeypatch.setattr(density, "read_density", lambda *a, **k: slots)


def test_density_is_rows_per_token_bucket():
    assert SlotDensity(T0, n_rows=200, n_buckets=100).density == 2.0


def test_over_limit_keeps_only_slots_above_the_limit():
    single = SlotDensity(T0, 100, 100)
    at_limit = SlotDensity(T0, 110, 100)
    doubled = SlotDensity(T0, 200, 100)
    assert over_limit([single, at_limit, doubled], 1.1) == [doubled]


def test_cli_exits_0_on_a_single_series(monkeypatch, capsys):
    _use_slots(monkeypatch, [SlotDensity(T0, 100, 100)])
    assert main(WINDOW) == 0
    assert "0 above" in capsys.readouterr().out


def test_cli_exits_1_and_lists_a_doubled_slot(monkeypatch, capsys):
    _use_slots(monkeypatch, [SlotDensity(T0, 100, 100), SlotDensity(T0, 200, 100)])
    assert main(WINDOW) == 1
    out = capsys.readouterr().out
    assert "2026-11-01 12:00Z  density 2.000" in out
    assert "1 above" in out


def test_cli_max_density_moves_the_limit(monkeypatch):
    _use_slots(monkeypatch, [SlotDensity(T0, 200, 100)])
    assert main([*WINDOW, "--max-density", "2.5"]) == 0


def test_cli_rejects_a_window_without_rows(monkeypatch, capsys):
    _use_slots(monkeypatch, [])
    assert main(WINDOW) == 2
    assert "no features rows" in capsys.readouterr().out


def test_cli_rejects_reversed_bounds(capsys):
    assert main(["--from", "2026-11-01T13:00Z", "--to", "2026-11-01T12:00Z"]) == 2
    assert "must be before" in capsys.readouterr().out


@pytest.mark.parametrize(
    "bad", [["--from", "not-a-time", "--to", "2026-11-01"], [*WINDOW, "--slot-minutes", "0"]]
)
def test_cli_rejects_invalid_arguments(bad):
    with pytest.raises(SystemExit) as exc:
        main(bad)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "error",
    [ConfigError("no database url"), OperationalError("SELECT", {}, Exception("refused"))],
)
def test_cli_reports_a_failed_read(monkeypatch, capsys, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(density, "make_engine", lambda: None)
    monkeypatch.setattr(density, "read_density", fail)
    assert main(WINDOW) == 2
    assert "density check failed" in capsys.readouterr().out
