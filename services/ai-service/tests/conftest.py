"""Shared test helpers: repo/migration discovery and an ephemeral PostgreSQL."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "services" / "db" / "migrations").is_dir():
            return parent
    raise FileNotFoundError("repo root with services/db/migrations not found")


@pytest.fixture(scope="session")
def migration_files() -> list[Path]:
    """Every migration, in the order the runner applies them (ordinal file-name order)."""
    files = sorted(
        (_repo_root() / "services" / "db" / "migrations").glob("*.sql"), key=lambda p: p.name
    )
    assert files, "no migrations found"
    return files


def strip_sql_comments(sql: str) -> str:
    """Drop ``--`` comments (the migrations have none inside string literals)."""
    lines = []
    for line in sql.splitlines():
        marker = line.find("--")
        lines.append(line if marker < 0 else line[:marker])
    return "\n".join(lines)


_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


def split_sql_statements(sql: str) -> list[str]:
    """Split on ``;`` except inside dollar-quoted bodies (``$$ ... $$``, ``$tag$ ... $tag$``).

    Empty statements are dropped. An unterminated dollar quote raises ``ValueError``.
    """
    statements, start, pos = [], 0, 0
    while True:
        quote = _DOLLAR_QUOTE.search(sql, pos)
        semicolon = sql.find(";", pos)
        if semicolon >= 0 and (quote is None or semicolon < quote.start()):
            statements.append(sql[start:semicolon])
            start = pos = semicolon + 1
        elif quote is not None:
            close = sql.find(quote.group(), quote.end())
            if close < 0:
                raise ValueError(f"unterminated dollar quote {quote.group()}")
            pos = close + len(quote.group())
        else:
            statements.append(sql[start:])
            break
    return [s.strip() for s in statements if s.strip()]


def _apply_migrations(engine: Engine, files: list[Path]) -> None:
    for sql_file in files:
        for statement in split_sql_statements(strip_sql_comments(sql_file.read_text("utf-8"))):
            with engine.begin() as conn:
                conn.exec_driver_sql(statement)


def _postgres_container_class():
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:
        from testcontainers.postgres import PostgresContainer
    return PostgresContainer


def start_container(make_container):
    """Start a container, or skip when the environment can't run one.

    Skipped locally when testcontainers or Docker is unavailable, but a hard failure in CI
    (``CI`` set), so the database tests can never silently stop running there.
    """
    in_ci = bool(os.environ.get("CI"))
    try:
        container = make_container()
    except ImportError:
        if in_ci:
            raise
        pytest.skip("testcontainers not installed")

    try:
        container.start()
    except Exception as exc:  # docker daemon unreachable, image pull failure, ...
        if in_ci:
            raise
        pytest.skip(f"Docker unavailable: {exc}")
    return container


@pytest.fixture(scope="session")
def pg_url(migration_files: list[Path]) -> Iterator[str]:
    """URL of a throwaway PostgreSQL 16 with every migration applied (see start_container)."""
    container = start_container(
        lambda: _postgres_container_class()("postgres:16-alpine", driver="psycopg")
    )
    try:
        url = container.get_connection_url()
        engine = create_engine(url)
        _apply_migrations(engine, migration_files)
        engine.dispose()
        yield url
    finally:
        container.stop()
