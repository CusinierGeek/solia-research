"""Configuration loading.

Model/training knobs come from ``config.yaml`` (shipped inside the package, overridable with
``SOLIA_AI_CONFIG``) and are validated on load. The database connection comes from the
environment, using the same contract as the .NET services (``Solia.Shared``'s
``DbConnectionSettings``): ``SOLIA_DB_CONNECTION`` wins, else the ``POSTGRES_*`` parts. Kept
intentionally import-light (no torch) so tooling and tests stay fast.
"""

from __future__ import annotations

import math
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

# solia_ai.calibration's methods and default; defined here so validation stays torch-free.
CALIBRATION_METHODS = ("none", "temperature", "platt")
CALIBRATION_DEFAULT_METHOD = "platt"
# The largest BCE pos_weight (neg/pos ratio) ever applied: ``auto`` clips to it, a fixed
# ``loss.pos_weight`` above it is rejected rather than silently clipped.
MAX_POS_WEIGHT = 1000.0

# solia_ai.promotion's gate knobs and defaults (``config.yaml`` ``promotion``); defined here so
# validation stays torch-free. ``horizon`` defaults to the middle configured horizon.
PROMOTION_METRICS = ("roc_auc", "pr_auc")
PROMOTION_HEADS = ("pump", "rug")
PROMOTION_DEFAULTS: dict[str, Any] = {
    "metric": "pr_auc",
    "head": "pump",
    "horizon": None,
    "margin": 0.01,
    "windows": 3,
    "seeds": 3,
    "min_win_share": 2 / 3,
    "ece_tolerance": 0.01,
    "secondary_tolerance": 0.02,
    "pnl_veto": True,
    "bootstrap_samples": 2000,
    "confidence": 0.95,
    "seed": 0,
    # The entry-selection guard (solia_ai.entry_selection); omitted or null = disabled.
    "entry_selection": None,
}
# Its knobs, when the ``promotion.entry_selection`` block is given (an omitted knob keeps these).
ENTRY_SELECTION_DEFAULTS: dict[str, Any] = {
    "top_fraction": 0.01,
    "min_top_lift": 1.0,
    "min_tokens": 20,
    "max_token_share": 0.5,
    # The horizons it can reject on (e.g. those the bot may trade); null = every configured horizon.
    # The others are still measured and reported ("report-only"), never blocking.
    "horizons": None,
}

# The most recent share of labeled windows the walk-forward evaluation holds out
# (solia_ai.ablation, hence the promotion gate). Defined here, torch-free, so the candidate's
# held-out cut defaults to the very same share: they can't drift apart.
DEFAULT_EVAL_FRACTION = 0.3

# solia_ai.candidate's knobs and defaults (``config.yaml`` ``candidate``); defined here so
# validation stays torch-free.
CANDIDATE_ON_INCOMPATIBLE = ("scratch", "error")
CANDIDATE_DEFAULTS: dict[str, Any] = {
    "warm_start": True,
    "warm_start_epochs": 10,
    "warm_start_lr_factor": 0.1,
    "on_incompatible": "scratch",
    "heldout_fraction": DEFAULT_EVAL_FRACTION,
    "val_fraction": 0.2,
    "min_train_samples": 1000,
}

# solia_ai.retention's knobs and defaults (``config.yaml`` ``retention``); defined here so
# validation stays torch-free.
RETENTION_DEFAULTS: dict[str, Any] = {
    "keep_candidates": 5,
    "rollback_depth": 2,
}

# The bounded training read's knobs and defaults (``config.yaml`` ``data``, solia_ai.db
# ``read_training_features``): every retrain / candidate / train run reads only the most recent
# ``lookback_days`` of ``features`` (relative to the newest row) and at most ``max_rows`` rows of
# whole tokens. ``window_stride_seconds`` keeps at most one training window per that many seconds
# of a token's life (0 = one per snapshot; solia_ai.dataset). Defined here so validation stays
# torch-free.
DATA_DEFAULTS: dict[str, Any] = {
    "lookback_days": 2.0,
    "max_rows": 4_000_000,
    "window_stride_seconds": 0,
    # Read only the enabled feature groups' columns (+ those labels / eligibility need) and hold
    # the windows' features as a matrix of those columns only (solia_ai.db.training_columns). Set by
    # the research CLIs' --prune-columns (search, throughput); the candidate / retrain refuse it.
    "prune_columns": False,
}

# The token eligibility rule's thresholds and defaults (``config.yaml`` ``eligibility``,
# solia_ai.eligibility): 0 disables a criterion, so all-zero keeps every token. Defined here so
# validation stays torch-free.
ELIGIBILITY_DEFAULTS: dict[str, Any] = {
    "min_unique_traders": 0,
    "min_volume_sol": 0.0,
    "min_token_age_sec": 0.0,
}

_HORIZON_UNITS = {"s": 1, "m": 60, "h": 3600}

Lookup = Callable[[str], str | None]


class ConfigError(ValueError):
    """Invalid or missing configuration (config file or database environment)."""


# --------------------------------------------------------------------------- horizons


def horizon_seconds(label: str) -> int:
    """Parse a horizon like ``'30s'``, ``'5m'``, ``'1h'`` into seconds."""
    text = label.strip().lower()
    if len(text) < 2 or text[-1] not in _HORIZON_UNITS or not text[:-1].isdigit():
        raise ValueError(f"invalid horizon {label!r} (expected e.g. '30s', '5m', '1h')")
    seconds = int(text[:-1]) * _HORIZON_UNITS[text[-1]]
    if seconds <= 0:
        raise ValueError(f"invalid horizon {label!r} (must be positive)")
    return seconds


def min_lookback_seconds(config: Any) -> int:
    """The shortest lookback the promotion gate can work with: one longest-horizon embargo per
    held-out window plus one for the training pool before them (``(windows + 1) x embargo``)."""
    windows = config.get("promotion", {}).get("windows", PROMOTION_DEFAULTS["windows"])
    return (int(windows) + 1) * max(horizon_seconds(h) for h in config["horizons"])


def data_bounds_error(config: Any, lookback_days: Any, max_rows: Any) -> str | None:
    """Why ``lookback_days`` / ``max_rows`` can't bound a training read under ``config`` (``None``
    when they can). Shared by the config validation and the CLI overrides."""
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int | float):
        return f"'data.lookback_days' must be a number, got {lookback_days!r}"
    if not lookback_days > 0:
        return f"'data.lookback_days' must be > 0, got {lookback_days!r}"
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        return f"'data.max_rows' must be an integer, got {max_rows!r}"
    if max_rows < 1:
        return f"'data.max_rows' must be >= 1, got {max_rows!r}"
    floor = min_lookback_seconds(config)
    if lookback_days * 86400 < floor:
        return (
            f"'data.lookback_days' ({lookback_days} d) is shorter than the promotion gate needs: "
            f"(promotion.windows + 1) x the longest horizon's embargo = {floor / 3600:g} h"
        )
    return None


def window_stride_error(stride: Any) -> str | None:
    """Why ``stride`` can't be a ``data.window_stride_seconds`` (a finite number >= 0), or None."""
    if isinstance(stride, bool) or not isinstance(stride, int | float):
        return f"'data.window_stride_seconds' must be a number, got {stride!r}"
    if not (math.isfinite(stride) and stride >= 0):
        return f"'data.window_stride_seconds' must be a finite number >= 0, got {stride!r}"
    return None


# --------------------------------------------------------------------------- model shape


def eligibility_error(values: Any) -> str | None:
    """Why an ``eligibility`` section is invalid (unknown key, non-numeric or negative threshold,
    a fractional trader count), or ``None`` when it is valid."""
    if not isinstance(values, dict):
        return f"'eligibility' must be a mapping, got {values!r}"
    unknown = [str(k) for k in values if k not in ELIGIBILITY_DEFAULTS]
    if unknown:
        return f"'eligibility' has unknown key(s): {', '.join(unknown)}"
    for key, value in values.items():
        integer = key == "min_unique_traders"
        valid = isinstance(value, int) if integer else isinstance(value, int | float)
        if isinstance(value, bool) or not valid:
            kind = "an integer" if integer else "a number"
            return f"'eligibility.{key}' must be {kind}, got {value!r}"
        if not 0 <= value < float("inf"):  # also rejects NaN
            return (
                f"'eligibility.{key}' must be a finite number >= 0 (0 disables it), got {value!r}"
            )
    return None


def liquidity_floor_error(values: Any) -> str | None:
    """Why a ``labeling.liquidity_floor`` mapping (``default`` and venue -> SOL floor) is invalid
    (non-string key, a venue listed twice once normalized, non-numeric, negative or non-finite
    floor), or ``None`` when it is valid."""
    name = "labeling.liquidity_floor"
    if not isinstance(values, dict):
        return f"'{name}' must be a mapping of venue -> SOL floor, got {values!r}"
    seen: set[str] = set()
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            return f"'{name}' keys must be venue names (or 'default'), got {key!r}"
        venue = key.strip().lower()
        if venue in seen:
            return f"'{name}' lists venue {venue!r} twice"
        seen.add(venue)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"'{name}.{key}' must be a number (SOL), got {value!r}"
        if not 0 <= value < float("inf"):  # also rejects NaN
            return f"'{name}.{key}' must be a finite number >= 0 (0 disables it), got {value!r}"
    return None


def model_shape_error(*, d_model: int, n_heads: int, seq_len: int, min_seq_len: int) -> str | None:
    """Why a model shape is invalid, or None. Shared by config validation and the model itself
    (so the rules live in one place, without importing torch here)."""
    if d_model % n_heads:
        return f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    if not 1 <= min_seq_len <= seq_len:
        return f"min_seq_len ({min_seq_len}) must be in [1, seq_len ({seq_len})]"
    return None


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


def validate_config(config: Any, source: str = "config") -> None:
    """Raise :class:`ConfigError` unless ``config`` is a valid config mapping (the rules
    :func:`load_config` applies to a file), e.g. for a config built in memory."""
    _validate(config, source)


def validate_retention(config: Any, source: str = "config") -> None:
    """Raise :class:`ConfigError` unless the ``retention`` section (optional) is valid. Checks only
    that section, so retention keeps working when an unrelated section is broken."""
    if not isinstance(config, dict):
        raise ConfigError(
            f"{source}: expected a mapping at the top level, got {type(config).__name__}"
        )
    if "retention" not in config:
        return
    retention = config["retention"]
    if not isinstance(retention, dict):
        raise ConfigError(f"{source}: 'retention' must be a mapping")
    unknown = [str(k) for k in retention if k not in RETENTION_DEFAULTS]
    if unknown:
        raise ConfigError(f"{source}: 'retention' has unknown key(s): {', '.join(unknown)}")
    for key in RETENTION_DEFAULTS:
        value = retention.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{source}: 'retention.{key}' must be an integer, got {value!r}")
        if value < 0:
            raise ConfigError(f"{source}: 'retention.{key}' must be >= 0, got {value!r}")


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

    model = section("model")
    for key in ("d_model", "n_heads", "n_layers", "seq_len"):
        number("model", key, positive=True, integer=True)
    for key in ("ff_dim", "min_seq_len"):
        if key in model:
            number("model", key, positive=True, integer=True)
    if problem := model_shape_error(
        d_model=model["d_model"],
        n_heads=model["n_heads"],
        seq_len=model["seq_len"],
        min_seq_len=model.get("min_seq_len", model["seq_len"]),
    ):
        fail(f"'model': {problem}")
    if not 0 <= number("model", "dropout") < 1:
        fail("'model.dropout' must be in [0, 1)")

    for key in ("learning_rate", "batch_size", "epochs"):
        number("training", key, positive=True)
    balanced = section("training").get("balanced_sampling", True)
    if not isinstance(balanced, bool):
        fail(f"'training.balanced_sampling' must be a boolean, got {balanced!r}")
    if section("training").get("num_workers", "auto") != "auto":
        number("training", "num_workers", integer=True)
    pin_memory = section("training").get("pin_memory", "auto")
    if pin_memory != "auto" and not isinstance(pin_memory, bool):
        fail(f"'training.pin_memory' must be a boolean or 'auto', got {pin_memory!r}")
    if "prefetch_factor" in section("training"):
        number("training", "prefetch_factor", positive=True, integer=True)
    if "recency_half_life_hours" in section("training"):
        half_life = number("training", "recency_half_life_hours", positive=True)
        if not math.isfinite(half_life):
            fail(f"'training.recency_half_life_hours' must be finite, got {half_life!r}")

    if not 0 <= number("loss", "alpha") <= 1:
        fail("'loss.alpha' must be in [0, 1] (L = alpha*BCE + (1-alpha)*Focal)")
    for key in ("focal_gamma", "strategy_weight", "regression_weight"):
        number("loss", key)

    def positive_weight(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            fail(f"'{name}' must be a positive number, got {value!r}")
        if value > MAX_POS_WEIGHT:
            fail(f"'{name}' must be at most {MAX_POS_WEIGHT:g}, got {value!r}")

    if "pos_weight" in section("loss"):
        spec = section("loss")["pos_weight"]
        if isinstance(spec, dict):
            unknown = [str(k) for k in spec if k not in ("pump", "rug")]
            if unknown:
                fail(f"'loss.pos_weight' has unknown head(s): {', '.join(unknown)} (pump, rug)")
            for head, value in spec.items():
                positive_weight(f"loss.pos_weight.{head}", value)
        elif spec not in ("auto", "none"):
            positive_weight("loss.pos_weight", spec)
        if balanced and spec != "none":
            fail(
                "'loss.pos_weight' needs 'training.balanced_sampling: false': balanced batches "
                "already correct the class imbalance, weighting on top would correct it twice"
            )

    if "calibration" in config:
        calibration = section("calibration")
        method = calibration.get("method", CALIBRATION_DEFAULT_METHOD)
        if method not in CALIBRATION_METHODS:
            fail(
                f"'calibration.method' must be one of {', '.join(CALIBRATION_METHODS)}, "
                f"got {method!r}"
            )
        if "fraction" in calibration and not 0 < number("calibration", "fraction") < 1:
            fail("'calibration.fraction' must be in (0, 1)")
        for key in ("min_samples", "min_class_samples"):
            if key in calibration:
                number("calibration", key, positive=True, integer=True)

    horizons = config["horizons"]
    if (
        not isinstance(horizons, list)
        or not horizons
        or not all(isinstance(h, str) for h in horizons)
    ):
        fail("'horizons' must be a non-empty list of strings")
    for horizon in horizons:
        try:
            horizon_seconds(horizon)
        except ValueError as exc:
            fail(f"'horizons': {exc}")

    horizon_weights = section("loss").get("horizon_weights")
    if horizon_weights is not None:
        if not isinstance(horizon_weights, dict):
            fail("'loss.horizon_weights' must be a mapping of horizon -> weight")
        unknown = [str(h) for h in horizon_weights if h not in horizons]
        if unknown:
            fail(f"'loss.horizon_weights' has unknown horizon(s): {', '.join(unknown)}")
        for horizon, value in horizon_weights.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
                fail(
                    f"'loss.horizon_weights.{horizon}' must be a non-negative number, got {value!r}"
                )

    if "features" in config:
        groups = section("features").get("groups")
        if (
            not isinstance(groups, list)
            or not groups
            or not all(isinstance(g, str) for g in groups)
        ):
            fail("'features.groups' must be a non-empty list of feature group names")
        from .features import FEATURE_GROUPS  # lazy: keeps this module import-light

        unknown = [g for g in groups if g not in FEATURE_GROUPS]
        if unknown:
            fail(
                f"'features.groups' has unknown group(s): {', '.join(unknown)} "
                f"(known: {', '.join(FEATURE_GROUPS)})"
            )

    if "promotion" in config:
        promotion = section("promotion")
        unknown = [str(k) for k in promotion if k not in PROMOTION_DEFAULTS]
        if unknown:
            fail(f"'promotion' has unknown key(s): {', '.join(unknown)}")
        choices = {"metric": PROMOTION_METRICS, "head": PROMOTION_HEADS, "horizon": horizons}
        for key, allowed in choices.items():
            # An explicit null horizon is the default: the middle configured horizon.
            if key == "horizon" and key in promotion and promotion[key] is None:
                continue
            if key in promotion and promotion[key] not in allowed:
                fail(
                    f"'promotion.{key}' must be one of {', '.join(map(str, allowed))}, "
                    f"got {promotion[key]!r}"
                )
        for key in ("margin", "ece_tolerance", "secondary_tolerance"):
            if key in promotion:
                number("promotion", key)
        for key in ("windows", "seeds", "bootstrap_samples"):
            if key in promotion:
                number("promotion", key, positive=True, integer=True)
        if "seed" in promotion:
            number("promotion", "seed", integer=True)
        if "min_win_share" in promotion and not 0 < number("promotion", "min_win_share") <= 1:
            fail("'promotion.min_win_share' must be in (0, 1]")
        if "confidence" in promotion and not 0 < number("promotion", "confidence") < 1:
            fail("'promotion.confidence' must be in (0, 1)")
        if not isinstance(promotion.get("pnl_veto", True), bool):
            fail(f"'promotion.pnl_veto' must be a boolean, got {promotion['pnl_veto']!r}")
        entry = promotion.get("entry_selection")
        if entry is not None:
            if not isinstance(entry, dict):
                fail("'promotion.entry_selection' must be a mapping (or null to disable it)")
            unknown = [str(k) for k in entry if k not in ENTRY_SELECTION_DEFAULTS]
            if unknown:
                fail(f"'promotion.entry_selection' has unknown key(s): {', '.join(unknown)}")
            ranges = {
                "top_fraction": (lambda v: 0 < v <= 0.5, "in (0, 0.5]"),
                "min_top_lift": (lambda v: v >= 0, ">= 0"),
                "max_token_share": (lambda v: 0 < v <= 1, "in (0, 1]"),
            }
            for key, (ok, bounds) in ranges.items():
                value = entry.get(key, ENTRY_SELECTION_DEFAULTS[key])
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(value)
                    or not ok(value)
                ):
                    fail(
                        f"'promotion.entry_selection.{key}' must be a number {bounds}, "
                        f"got {value!r}"
                    )
            guarded = entry.get("horizons")
            if guarded is not None:
                if (
                    not isinstance(guarded, list)
                    or not guarded
                    or not all(isinstance(h, str) for h in guarded)
                ):
                    fail(
                        "'promotion.entry_selection.horizons' must be a non-empty list of "
                        f"horizons (or null for all), got {guarded!r}"
                    )
                unknown_h = [h for h in guarded if h not in config.get("horizons", [])]
                if unknown_h:
                    fail(
                        "'promotion.entry_selection.horizons' has horizon(s) not in 'horizons': "
                        f"{', '.join(unknown_h)}"
                    )
            tokens = entry.get("min_tokens", ENTRY_SELECTION_DEFAULTS["min_tokens"])
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
                fail(
                    "'promotion.entry_selection.min_tokens' must be an integer >= 1, "
                    f"got {tokens!r}"
                )

    if "candidate" in config:
        candidate = section("candidate")
        unknown = [str(k) for k in candidate if k not in CANDIDATE_DEFAULTS]
        if unknown:
            fail(f"'candidate' has unknown key(s): {', '.join(unknown)}")
        if not isinstance(candidate.get("warm_start", True), bool):
            fail(f"'candidate.warm_start' must be a boolean, got {candidate['warm_start']!r}")
        policy = candidate.get("on_incompatible", CANDIDATE_DEFAULTS["on_incompatible"])
        if policy not in CANDIDATE_ON_INCOMPATIBLE:
            fail(
                f"'candidate.on_incompatible' must be one of "
                f"{', '.join(CANDIDATE_ON_INCOMPATIBLE)}, got {policy!r}"
            )
        for key in ("warm_start_epochs", "min_train_samples"):
            if key in candidate:
                number("candidate", key, positive=True, integer=True)
        for key in ("heldout_fraction", "val_fraction"):
            if key in candidate and not 0 < number("candidate", key) < 1:
                fail(f"'candidate.{key}' must be in (0, 1)")
        if "warm_start_lr_factor" in candidate and not (
            0 < number("candidate", "warm_start_lr_factor") <= 1
        ):
            fail("'candidate.warm_start_lr_factor' must be in (0, 1]")

    validate_retention(config, source)

    data = section("data") if "data" in config else {}
    unknown = [str(k) for k in data if k not in DATA_DEFAULTS]
    if unknown:
        fail(f"'data' has unknown key(s): {', '.join(unknown)}")
    bounds = {**DATA_DEFAULTS, **data}
    problem = data_bounds_error(config, bounds["lookback_days"], bounds["max_rows"])
    if problem or (problem := window_stride_error(bounds["window_stride_seconds"])):
        fail(problem)
    if not isinstance(bounds["prune_columns"], bool):
        fail(f"'data.prune_columns' must be true or false, got {bounds['prune_columns']!r}")

    if "eligibility" in config and (problem := eligibility_error(config["eligibility"])):
        fail(problem)

    labeling = section("labeling")
    if "max_gap_fraction" in labeling and not 0 < number("labeling", "max_gap_fraction") <= 1:
        fail("'labeling.max_gap_fraction' must be in (0, 1]")
    if "min_gap_seconds" in labeling:
        number("labeling", "min_gap_seconds")
    if "max_return" in labeling:
        number("labeling", "max_return", positive=True)
    if "liquidity_floor" in labeling and (
        problem := liquidity_floor_error(labeling["liquidity_floor"])
    ):
        fail(problem)
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
