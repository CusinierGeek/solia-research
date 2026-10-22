"""Configuration loading.

Model/training knobs come from ``config.yaml`` (shipped inside the package, overridable with
``SOLIA_AI_CONFIG``) and are validated on load. The database connection comes from the
environment, using the same contract as the .NET services (``Solia.Shared``'s
``DbConnectionSettings``): ``SOLIA_DB_CONNECTION`` wins, else the ``POSTGRES_*`` parts. Kept
intentionally import-light (no torch) so tooling and tests stay fast.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any, NoReturn

import yaml
from sqlalchemy.engine import URL

CONFIG_ENV = "SOLIA_AI_CONFIG"
CONNECTION_ENV = "SOLIA_DB_CONNECTION"

_DRIVER = "postgresql+psycopg"
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DEV_PASSWORD = "change_me"

Lookup = Callable[[str], str | None]


class ConfigError(ValueError):
    """Invalid or missing configuration (config file or database environment)."""


# --------------------------------------------------------------------------- config.yaml


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and validate the YAML config.

    Resolution: ``path`` if given, else ``$SOLIA_AI_CONFIG``, else the ``config.yaml`` packaged
    with ``solia_ai`` (present in source checkouts and installed wheels alike).
    """
    override = path if path is not None else _clean(os.environ.get(CONFIG_ENV))
    if override is not None:
        source = str(override)
        try:
            text = Path(override).read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file {source}: {exc}") from exc
    else:
        packaged = resources.files("solia_ai").joinpath("config.yaml")
        source = str(packaged)
        text = packaged.read_text(encoding="utf-8")

    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from exc

    _validate(config, source)
    return config


def _validate(config: Any, source: str) -> None:
    def fail(problem: str) -> NoReturn:
        raise ConfigError(f"{source}: {problem}")

    if not isinstance(config, dict):
        fail(f"expected a mapping at the top level, got {type(config).__name__}")

    missing = [s for s in ("model", "training", "loss", "horizons", "labeling") if s not in config]
    if missing:
        fail(f"missing section(s): {', '.join(missing)}")

    def section(name: str) -> dict[str, Any]:
        value = config[name]
        if not isinstance(value, dict):
            fail(f"'{name}' must be a mapping")
        return value

    def number(sec: str, key: str, *, positive: bool = False, integer: bool = False) -> float:
        value = section(sec).get(key)
        valid_type = isinstance(value, int) if integer else isinstance(value, int | float)
        if isinstance(value, bool) or not valid_type:
            fail(f"'{sec}.{key}' must be {'an integer' if integer else 'a number'}, got {value!r}")
        if positive and value <= 0:
            fail(f"'{sec}.{key}' must be > 0, got {value!r}")
        if not positive and value < 0:
            fail(f"'{sec}.{key}' must be >= 0, got {value!r}")
        return value

    for key in ("d_model", "n_heads", "n_layers", "seq_len"):
        number("model", key, positive=True, integer=True)
    if section("model")["d_model"] % section("model")["n_heads"]:
        fail("'model.d_model' must be divisible by 'model.n_heads'")
    if not 0 <= number("model", "dropout") < 1:
        fail("'model.dropout' must be in [0, 1)")

    for key in ("learning_rate", "batch_size", "epochs"):
        number("training", key, positive=True)

    if not 0 <= number("loss", "alpha") <= 1:
        fail("'loss.alpha' must be in [0, 1] (L = alpha*BCE + (1-alpha)*Focal)")
    for key in ("focal_gamma", "strategy_weight", "regression_weight"):
        number("loss", key)

    horizons = config["horizons"]
    if (
        not isinstance(horizons, list)
        or not horizons
        or not all(isinstance(h, str) for h in horizons)
    ):
        fail("'horizons' must be a non-empty list of strings")

    labeling = section("labeling")
    for kind in ("pump_threshold", "rug_threshold"):
        thresholds = labeling.get(kind)
        if not isinstance(thresholds, dict):
            fail(f"'labeling.{kind}' must be a mapping of horizon -> threshold")
        uncovered = [h for h in horizons if h not in thresholds]
        if uncovered:
            fail(f"'labeling.{kind}' has no threshold for horizon(s): {', '.join(uncovered)}")
        for horizon, value in thresholds.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                fail(f"'labeling.{kind}.{horizon}' must be a positive number, got {value!r}")


# --------------------------------------------------------------------------- database


def database_url(lookup: Lookup | None = None) -> URL:
    """Build the SQLAlchemy URL from the environment (or ``lookup``, for tests).

    ``SOLIA_DB_CONNECTION`` (an Npgsql connection string, shared with the .NET services) wins;
    otherwise the ``POSTGRES_*`` parts. Blank values count as unset, values are trimmed, and
    credentials are escaped by ``URL.create``. The local-dev default password is only accepted
    for a local host: a remote database without ``POSTGRES_PASSWORD`` fails fast instead of
    trying a known placeholder.
    """
    get = lookup or os.environ.get

    explicit = _clean(get(CONNECTION_ENV))
    if explicit is not None:
        return _from_npgsql(explicit)

    host = _clean(get("POSTGRES_HOST")) or "localhost"
    password = _clean(get("POSTGRES_PASSWORD"))
    if password is None:
        if host.lower() not in _LOCAL_HOSTS:
            raise ConfigError(
                f"POSTGRES_PASSWORD is not set for database host '{host}'; the local-dev default "
                "is only used for localhost. Set POSTGRES_PASSWORD or SOLIA_DB_CONNECTION."
            )
        password = _DEV_PASSWORD

    return URL.create(
        _DRIVER,
        username=_clean(get("POSTGRES_USER")) or "solia",
        password=password,
        host=host,
        port=_port(_clean(get("POSTGRES_PORT")), "POSTGRES_PORT"),
        database=_clean(get("POSTGRES_DB")) or "solia",
    )


# Npgsql keyword -> canonical name (keywords are case- and space-insensitive).
_KEYWORDS = {
    "host": "host",
    "server": "host",
    "port": "port",
    "database": "database",
    "db": "database",
    "username": "username",
    "userid": "username",
    "user": "username",
    "password": "password",
    "pwd": "password",
    "psw": "password",
    "sslmode": "sslmode",
    "timeout": "connect_timeout",
    "applicationname": "application_name",
}

# Npgsql-only tuning that has no libpq/SQLAlchemy URL equivalent (SQLAlchemy pools itself).
_IGNORED_KEYWORDS = frozenset(
    {
        "pooling",
        "minpoolsize",
        "maxpoolsize",
        "maximumpoolsize",
        "minimumpoolsize",
        "connectionidlelifetime",
        "connectionlifetime",
        "commandtimeout",
        "includeerrordetail",
        "trustservercertificate",
        "keepalive",
        "multiplexing",
    }
)

_SSL_MODES = {
    "disable": "disable",
    "allow": "allow",
    "prefer": "prefer",
    "require": "require",
    "verifyca": "verify-ca",
    "verifyfull": "verify-full",
}


def _from_npgsql(connection_string: str) -> URL:
    if "://" in connection_string:
        raise ConfigError(
            f"{CONNECTION_ENV} must be an Npgsql connection string (Host=...;Username=...;...), "
            "the format the .NET services read — not a URL."
        )

    parts: dict[str, str] = {}
    for key, value in _split_keywords(connection_string):
        normalized = key.replace(" ", "").lower()
        if normalized in _IGNORED_KEYWORDS:
            continue
        canonical = _KEYWORDS.get(normalized)
        if canonical is None:
            raise ConfigError(f"{CONNECTION_ENV}: unsupported keyword '{key}'")
        parts[canonical] = value

    host = parts.get("host")
    if not host:
        raise ConfigError(f"{CONNECTION_ENV} must set Host")

    query: dict[str, str] = {}
    if "sslmode" in parts:
        mode = _SSL_MODES.get(parts["sslmode"].replace("-", "").lower())
        if mode is None:
            raise ConfigError(f"{CONNECTION_ENV}: unsupported SSL Mode '{parts['sslmode']}'")
        query["sslmode"] = mode
    for key in ("connect_timeout", "application_name"):
        if key in parts:
            query[key] = parts[key]

    return URL.create(
        _DRIVER,
        username=parts.get("username"),
        password=parts.get("password"),
        host=host,
        port=_port(parts.get("port"), f"{CONNECTION_ENV} Port"),
        database=parts.get("database"),
        query=query,
    )


def _split_keywords(connection_string: str) -> list[tuple[str, str]]:
    """Split ``key=value;...`` honoring single/double-quoted values (quotes doubled to escape)."""
    pairs: list[tuple[str, str]] = []
    i, n = 0, len(connection_string)
    while i < n:
        end = connection_string.find("=", i)
        if end < 0:
            if connection_string[i:].strip(" ;"):
                raise ConfigError(
                    f"{CONNECTION_ENV}: expected key=value near '{connection_string[i:]}'"
                )
            break
        key = connection_string[i:end].strip(" ;")
        i = end + 1
        while i < n and connection_string[i] == " ":
            i += 1

        if i < n and connection_string[i] in "'\"":
            quote = connection_string[i]
            i += 1
            value_chars: list[str] = []
            while True:
                if i >= n:
                    raise ConfigError(f"{CONNECTION_ENV}: unterminated quoted value for '{key}'")
                if connection_string[i] == quote:
                    if i + 1 < n and connection_string[i + 1] == quote:
                        value_chars.append(quote)
                        i += 2
                        continue
                    i += 1
                    break
                value_chars.append(connection_string[i])
                i += 1
            value = "".join(value_chars)
            semicolon = connection_string.find(";", i)
            i = n if semicolon < 0 else semicolon + 1
        else:
            semicolon = connection_string.find(";", i)
            value = connection_string[i : n if semicolon < 0 else semicolon].strip()
            i = n if semicolon < 0 else semicolon + 1

        if not key:
            raise ConfigError(f"{CONNECTION_ENV}: empty keyword")
        pairs.append((key, value))
    return pairs


def _port(raw: str | None, name: str) -> int:
    if raw is None or raw == "":
        return 5432
    try:
        port = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got '{raw}'") from None
    if not 1 <= port <= 65535:
        raise ConfigError(f"{name} must be in 1..65535, got {port}")
    return port


def _clean(value: str | None) -> str | None:
    """Trimmed value, or None when unset or blank (matches the .NET resolver)."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
