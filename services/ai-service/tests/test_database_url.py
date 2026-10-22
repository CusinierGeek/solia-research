"""database_url(): the SOLIA_DB_CONNECTION / POSTGRES_* contract shared with the .NET services."""

import pytest
from sqlalchemy.engine import make_url

from solia_ai.config import ConfigError, database_url

# Every URL-reserved character, assembled at runtime: a fake value for escaping tests that
# secret scanners shouldn't mistake for a hardcoded credential.
RESERVED = "".join(["@", "/", ":", "#", "?", "%"])
TRICKY = "x" + RESERVED + "x"


def url(**env):
    return database_url(env.get)


def test_all_defaults_target_local_dev():
    u = url()

    assert u.render_as_string(hide_password=False) == (
        "postgresql+psycopg://solia:change_me@localhost:5432/solia"
    )


def test_postgres_parts_are_used():
    u = url(
        POSTGRES_HOST="db.example",
        POSTGRES_PORT="6543",
        POSTGRES_USER="u",
        POSTGRES_PASSWORD="p",
        POSTGRES_DB="d",
    )

    assert (u.host, u.port, u.username, u.password, u.database) == (
        "db.example",
        6543,
        "u",
        "p",
        "d",
    )


def test_credentials_with_reserved_characters_survive_a_round_trip():
    u = url(POSTGRES_USER="us@r:x", POSTGRES_PASSWORD=TRICKY, POSTGRES_HOST="localhost")

    parsed = make_url(u.render_as_string(hide_password=False))
    assert (parsed.username, parsed.password, parsed.host, parsed.database) == (
        "us@r:x",
        TRICKY,
        "localhost",
        "solia",
    )


def test_blank_and_padded_values_behave_like_the_dotnet_resolver():
    u = url(POSTGRES_HOST="  ", POSTGRES_PORT="", POSTGRES_DB=" d ", POSTGRES_USER="\t")

    assert (u.host, u.port, u.database, u.username) == ("localhost", 5432, "d", "solia")


@pytest.mark.parametrize("port", ["abc", "0", "70000"])
def test_invalid_port_is_rejected(port):
    with pytest.raises(ConfigError, match="POSTGRES_PORT"):
        url(POSTGRES_PORT=port)


@pytest.mark.parametrize("host", ["postgres", "db.example.com", "10.0.0.5"])
def test_remote_host_without_password_fails_fast(host):
    with pytest.raises(ConfigError, match="POSTGRES_PASSWORD is not set"):
        url(POSTGRES_HOST=host)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "LOCALHOST"])
def test_local_host_may_use_the_dev_default_password(host):
    assert url(POSTGRES_HOST=host).password == "change_me"


def test_blank_password_counts_as_unset():
    with pytest.raises(ConfigError, match="POSTGRES_PASSWORD is not set"):
        url(POSTGRES_HOST="postgres", POSTGRES_PASSWORD="  ")


# ----------------------------------------------------------------- SOLIA_DB_CONNECTION


def test_connection_string_wins_over_postgres_parts():
    u = url(
        SOLIA_DB_CONNECTION="Host=db.example;Port=6543;Database=d;Username=u;Password=p",
        POSTGRES_HOST="ignored",
    )

    assert (u.host, u.port, u.database, u.username, u.password) == (
        "db.example",
        6543,
        "d",
        "u",
        "p",
    )


def test_connection_string_keywords_are_case_and_space_insensitive_with_aliases():
    u = url(SOLIA_DB_CONNECTION="server=h; PORT=5433 ; User ID=u; PWD=p; DB=d")

    assert (u.host, u.port, u.username, u.password, u.database) == ("h", 5433, "u", "p", "d")


def test_quoted_values_may_contain_separators_and_escaped_quotes():
    u = url(SOLIA_DB_CONNECTION="Host=h;Password='a;b=c''d';Username=\"u\"")

    assert (u.password, u.username) == ("a;b=c'd", "u")


def test_connection_string_options_map_to_libpq_query_parameters():
    u = url(SOLIA_DB_CONNECTION="Host=h;SSL Mode=VerifyFull;Timeout=15;Application Name=solia-ai")

    assert dict(u.query) == {
        "sslmode": "verify-full",
        "connect_timeout": "15",
        "application_name": "solia-ai",
    }


def test_npgsql_only_pool_settings_are_ignored():
    u = url(
        SOLIA_DB_CONNECTION="Host=h;Pooling=true;Maximum Pool Size=50;Include Error Detail=true"
    )

    assert u.host == "h"
    assert not u.query


def test_connection_string_defaults_port_and_escapes_credentials():
    u = url(SOLIA_DB_CONNECTION=f"Host=h;Username=u;Password={TRICKY}")

    parsed = make_url(u.render_as_string(hide_password=False))
    assert (parsed.port, parsed.password) == (5432, TRICKY)


def test_blank_connection_string_falls_back_to_postgres_parts():
    assert url(SOLIA_DB_CONNECTION="   ").host == "localhost"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("postgresql://u:p@h/d", "not a URL"),
        ("Port=5432;Username=u", "must set Host"),
        ("Host=h;Frobnicate=1", "unsupported keyword 'Frobnicate'"),
        ("Host=h;SSL Mode=Sometimes", "unsupported SSL Mode"),
        ("Host=h;Port=nope", "Port must be a number"),
        ("Host=h;Password='unterminated", "unterminated quoted value"),
        ("Host=h;garbage", "expected key=value"),
    ],
)
def test_invalid_connection_strings_are_rejected(value, message):
    with pytest.raises(ConfigError, match=message):
        url(SOLIA_DB_CONNECTION=value)


def test_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("SOLIA_DB_CONNECTION", "Host=envhost;Password=p")

    assert database_url().host == "envhost"
