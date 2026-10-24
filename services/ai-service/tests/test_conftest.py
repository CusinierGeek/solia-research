"""The DB fixture's helpers: SQL splitting, and environment handling (skip locally, fail in CI)."""

import re
from collections import Counter

import pytest
from conftest import split_sql_statements, start_container, strip_sql_comments


class DockerDown:
    def start(self):
        raise ConnectionError("Error while fetching server API version")


def missing_testcontainers():
    raise ImportError("No module named 'testcontainers'")


def test_docker_unavailable_skips_locally(monkeypatch):
    monkeypatch.delenv("CI", raising=False)

    with pytest.raises(pytest.skip.Exception, match="Docker unavailable"):
        start_container(DockerDown)


def test_docker_unavailable_fails_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")

    with pytest.raises(ConnectionError):
        start_container(DockerDown)


def test_missing_testcontainers_skips_locally_and_fails_in_ci(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception, match="testcontainers not installed"):
        start_container(missing_testcontainers)

    monkeypatch.setenv("CI", "true")
    with pytest.raises(ImportError):
        start_container(missing_testcontainers)


def test_a_started_container_is_returned(monkeypatch):
    monkeypatch.delenv("CI", raising=False)

    class Up:
        started = False

        def start(self):
            self.started = True

    container = start_container(Up)

    assert container.started


def test_statements_split_on_semicolons_but_not_inside_dollar_quoted_bodies():
    sql = (
        "CREATE TABLE t (id INT);\n"
        "CREATE FUNCTION f() RETURNS trigger LANGUAGE plpgsql AS $$\n"
        "BEGIN PERFORM 1; RETURN NULL; END\n"
        "$$;\n"
        "CREATE TRIGGER g AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION f();\n\n"
    )

    assert split_sql_statements(sql) == [
        "CREATE TABLE t (id INT)",
        "CREATE FUNCTION f() RETURNS trigger LANGUAGE plpgsql AS $$\n"
        "BEGIN PERFORM 1; RETURN NULL; END\n$$",
        "CREATE TRIGGER g AFTER INSERT ON t FOR EACH ROW EXECUTE FUNCTION f()",
    ]


def test_tagged_dollar_quotes_keep_their_body_whole_even_around_bare_ones():
    sql = (
        "CREATE FUNCTION f() RETURNS text LANGUAGE plpgsql AS $body$\n"
        "BEGIN RETURN $$a;b$$; END\n"
        "$body$;\n"
        "SELECT $1::int; SELECT 2"
    )

    assert split_sql_statements(sql) == [
        "CREATE FUNCTION f() RETURNS text LANGUAGE plpgsql AS $body$\n"
        "BEGIN RETURN $$a;b$$; END\n$body$",
        "SELECT $1::int",
        "SELECT 2",
    ]


def test_an_unterminated_dollar_quote_is_rejected():
    with pytest.raises(ValueError, match=r"\$fn\$"):
        split_sql_statements("CREATE FUNCTION f() AS $fn$ BEGIN; END;")


def test_every_migration_splits_into_whole_statements(migration_files):
    for sql_file in migration_files:
        statements = split_sql_statements(strip_sql_comments(sql_file.read_text("utf-8")))
        assert statements, sql_file.name
        for statement in statements:
            tags = Counter(re.findall(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$", statement))
            assert all(n % 2 == 0 for n in tags.values()), (sql_file.name, statement)
