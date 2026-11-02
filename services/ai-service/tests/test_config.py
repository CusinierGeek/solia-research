"""load_config(): file resolution and validation."""

from pathlib import Path

import pytest
import yaml

from solia_ai.config import CONFIG_ENV, DATA_DEFAULTS, ConfigError, load_config

VALID = load_config()  # the packaged config is the reference valid document


def write(tmp_path: Path, content) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        content if isinstance(content, str) else yaml.safe_dump(content), encoding="utf-8"
    )
    return path


def variant(**sections):
    """A deep-ish copy of the valid config with some sections replaced or removed (None)."""
    cfg = {
        k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
        for k, v in VALID.items()
    }
    for name, value in sections.items():
        if value is None:
            cfg.pop(name)
        else:
            cfg[name] = value
    return cfg


def test_explicit_path_wins(tmp_path):
    path = write(tmp_path, variant(model={**VALID["model"], "seq_len": 70}))

    assert load_config(path)["model"]["seq_len"] == 70


def test_env_override_is_used_when_no_path_given(tmp_path, monkeypatch):
    monkeypatch.setenv(
        CONFIG_ENV, str(write(tmp_path, variant(model={**VALID["model"], "seq_len": 9})))
    )

    assert load_config()["model"]["seq_len"] == 9


def test_blank_env_override_falls_back_to_the_packaged_config(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, "   ")

    assert load_config() == VALID


def test_missing_file_is_a_clear_error(tmp_path):
    missing = tmp_path / "nope.yaml"

    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(missing)


def test_invalid_yaml_names_the_file(tmp_path):
    path = write(tmp_path, "model: [unclosed")

    with pytest.raises(ConfigError, match=r"config\.yaml: invalid YAML"):
        load_config(path)


@pytest.mark.parametrize("content", ["", "just a string", "- a\n- list\n"])
def test_non_mapping_document_is_rejected(tmp_path, content):
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(write(tmp_path, content))


def test_missing_sections_are_named(tmp_path):
    with pytest.raises(ConfigError, match="missing section\\(s\\): loss, labeling"):
        load_config(write(tmp_path, variant(loss=None, labeling=None)))


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"model": "nope"}, "'model' must be a mapping"),
        ({"model": {**VALID["model"], "d_model": 0}}, "'model.d_model' must be > 0"),
        ({"model": {**VALID["model"], "n_heads": 2.5}}, "'model.n_heads' must be an integer"),
        ({"model": {**VALID["model"], "d_model": 130}}, r"d_model \(130\) must be divisible"),
        ({"model": {**VALID["model"], "min_seq_len": 65}}, r"min_seq_len \(65\) must be in"),
        ({"model": {**VALID["model"], "ff_dim": 0}}, "'model.ff_dim' must be > 0"),
        ({"model": {**VALID["model"], "dropout": 1.0}}, r"'model.dropout' must be in \[0, 1\)"),
        (
            {"training": {**VALID["training"], "learning_rate": -1}},
            "'training.learning_rate' must be > 0",
        ),
        ({"loss": {**VALID["loss"], "alpha": 1.5}}, r"'loss.alpha' must be in \[0, 1\]"),
        ({"loss": {**VALID["loss"], "alpha": True}}, "'loss.alpha' must be a number"),
        ({"loss": {**VALID["loss"], "focal_gamma": -2}}, "'loss.focal_gamma' must be >= 0"),
        (
            {"training": {**VALID["training"], "balanced_sampling": "yes"}},
            "'training.balanced_sampling' must be a boolean",
        ),
        (
            {"loss": {**VALID["loss"], "horizon_weights": {"1hr": 3.0}}},
            r"'loss.horizon_weights' has unknown horizon\(s\): 1hr",
        ),
        (
            {"loss": {**VALID["loss"], "horizon_weights": {"1h": -1}}},
            "'loss.horizon_weights.1h' must be a non-negative number",
        ),
        ({"horizons": []}, "'horizons' must be a non-empty list"),
        ({"horizons": ["30s", 5]}, "'horizons' must be a non-empty list of strings"),
        ({"features": []}, "'features' must be a mapping"),
        ({"features": {"groups": []}}, "'features.groups' must be a non-empty list"),
        ({"features": {"groups": "price"}}, "'features.groups' must be a non-empty list"),
        (
            {"features": {"groups": ["price", "nope"]}},
            r"'features.groups' has unknown group\(s\): nope",
        ),
    ],
)
def test_invalid_values_are_rejected_with_the_key_named(tmp_path, sections, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, variant(**sections)))


def test_the_packaged_config_ships_the_data_settings_172_chose():
    # #172: 8 M rows at a 60 s stride; omitting `data` keeps the legacy behaviour (DATA_DEFAULTS).
    assert VALID["data"]["max_rows"] == 8_000_000
    assert VALID["data"]["window_stride_seconds"] == 60
    assert DATA_DEFAULTS["max_rows"] == 4_000_000 and DATA_DEFAULTS["window_stride_seconds"] == 0


def test_the_packaged_config_keeps_the_regularization_172_did_not_beat():
    assert (VALID["model"]["dropout"], VALID["model"]["d_model"]) == (0.1, 128)
    assert (VALID["training"]["weight_decay"], VALID["training"]["learning_rate"]) == (0.01, 3e-4)


def test_the_packaged_config_enables_the_groups_the_ablation_retained():
    # #188: momentum + concentration cleared the pump/5m margin; the other candidates did not.
    from solia_ai.features import BASELINE_GROUPS, group_columns

    groups = VALID["features"]["groups"]
    assert groups == [*BASELINE_GROUPS, "momentum", "concentration"]
    assert not {"buy_pressure", "burn", "venue"} & set(groups)
    assert len(group_columns(groups)) > len(group_columns(BASELINE_GROUPS))


def test_omitting_features_falls_back_to_the_baseline_groups(tmp_path):
    from solia_ai.features import BASELINE_GROUPS

    cfg = load_config(write(tmp_path, variant(features=None)))
    assert cfg.get("features", {}).get("groups", BASELINE_GROUPS) == BASELINE_GROUPS


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"max_gap_fraction": 0}, r"'labeling.max_gap_fraction' must be in \(0, 1\]"),
        ({"max_gap_fraction": 1.5}, r"'labeling.max_gap_fraction' must be in \(0, 1\]"),
        ({"min_gap_seconds": -5}, "'labeling.min_gap_seconds' must be >= 0"),
        ({"max_return": 0}, "'labeling.max_return' must be > 0"),
        ({"max_return": "big"}, "'labeling.max_return' must be a number"),
    ],
)
def test_invalid_gap_settings_are_rejected(tmp_path, extra, message):
    labeling = {**VALID["labeling"], **extra}

    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, variant(labeling=labeling)))


def test_every_horizon_needs_a_threshold(tmp_path):
    labeling = {
        "pump_threshold": {
            k: v for k, v in VALID["labeling"]["pump_threshold"].items() if k != "1h"
        },
        "rug_threshold": VALID["labeling"]["rug_threshold"],
    }

    with pytest.raises(
        ConfigError, match="'labeling.pump_threshold' has no threshold for horizon\\(s\\): 1h"
    ):
        load_config(write(tmp_path, variant(labeling=labeling)))


def test_thresholds_must_be_positive(tmp_path):
    labeling = {
        "pump_threshold": VALID["labeling"]["pump_threshold"],
        "rug_threshold": {**VALID["labeling"]["rug_threshold"], "5m": 0},
    }

    with pytest.raises(ConfigError, match="'labeling.rug_threshold.5m' must be a positive number"):
        load_config(write(tmp_path, variant(labeling=labeling)))


def test_old_split_loss_weights_are_not_accepted(tmp_path):
    loss = {k: v for k, v in VALID["loss"].items() if k != "alpha"} | {
        "bce_weight": 0.7,
        "focal_weight": 0.5,
    }

    with pytest.raises(ConfigError, match="'loss.alpha' must be a number"):
        load_config(write(tmp_path, variant(loss=loss)))


NATURAL = {**VALID["training"], "balanced_sampling": False}


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"calibration": "platt"}, "'calibration' must be a mapping"),
        ({"calibration": {"method": "isotonic"}}, "'calibration.method' must be one of"),
        ({"calibration": {"fraction": 0}}, r"'calibration.fraction' must be in \(0, 1\)"),
        ({"calibration": {"fraction": 1.0}}, r"'calibration.fraction' must be in \(0, 1\)"),
        ({"calibration": {"min_samples": 0}}, "'calibration.min_samples' must be > 0"),
        ({"calibration": {"min_samples": 2.5}}, "'calibration.min_samples' must be an integer"),
        (
            {"loss": {**VALID["loss"], "pos_weight": "auto"}},  # balanced_sampling: true
            "'loss.pos_weight' needs 'training.balanced_sampling: false'",
        ),
        (
            {"training": NATURAL, "loss": {**VALID["loss"], "pos_weight": -1}},
            "'loss.pos_weight' must be a positive number",
        ),
        (
            {"training": NATURAL, "loss": {**VALID["loss"], "pos_weight": "heavy"}},
            "'loss.pos_weight' must be a positive number",
        ),
        (
            {"training": NATURAL, "loss": {**VALID["loss"], "pos_weight": {"return": 2.0}}},
            r"'loss.pos_weight' has unknown head\(s\): return",
        ),
        (
            {"training": NATURAL, "loss": {**VALID["loss"], "pos_weight": {"pump": 0}}},
            "'loss.pos_weight.pump' must be a positive number",
        ),
        (
            {"calibration": {"min_class_samples": 0}},
            "'calibration.min_class_samples' must be > 0",
        ),
        (
            {"training": NATURAL, "loss": {**VALID["loss"], "pos_weight": 5000}},
            "'loss.pos_weight' must be at most 1000",
        ),
        (
            {"training": NATURAL, "loss": {**VALID["loss"], "pos_weight": {"rug": 2000.0}}},
            "'loss.pos_weight.rug' must be at most 1000",
        ),
    ],
)
def test_invalid_calibration_and_class_weighting_are_rejected(tmp_path, sections, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, variant(**sections)))


@pytest.mark.parametrize(
    "pos_weight", ["auto", "none", 3.0, 1000, {"pump": 5.0, "rug": 2}, {"rug": 1.5}]
)
def test_class_weighting_is_accepted_with_natural_sampling(tmp_path, pos_weight):
    cfg = load_config(
        write(tmp_path, variant(training=NATURAL, loss={**VALID["loss"], "pos_weight": pos_weight}))
    )

    assert cfg["loss"]["pos_weight"] == pos_weight


def test_calibration_section_is_optional(tmp_path):
    assert "calibration" not in load_config(write(tmp_path, variant(calibration=None)))
    assert VALID["calibration"]["method"] == "platt"


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ("scratch", "'candidate' must be a mapping"),
        ({"warmstart": True}, r"'candidate' has unknown key\(s\): warmstart"),
        ({"warm_start": "yes"}, "'candidate.warm_start' must be a boolean"),
        ({"on_incompatible": "ignore"}, "'candidate.on_incompatible' must be one of"),
        ({"warm_start_epochs": 0}, "'candidate.warm_start_epochs' must be > 0"),
        ({"min_train_samples": 2.5}, "'candidate.min_train_samples' must be an integer"),
        ({"heldout_fraction": 1.0}, r"'candidate.heldout_fraction' must be in \(0, 1\)"),
        ({"val_fraction": 0}, r"'candidate.val_fraction' must be in \(0, 1\)"),
    ],
)
def test_invalid_candidate_settings_are_rejected(tmp_path, candidate, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, variant(candidate=candidate)))


def test_packaged_candidate_settings_are_valid():
    assert load_config()["candidate"]["on_incompatible"] in ("scratch", "error")


# --------------------------------------------------------------------------- data bounds


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("all", "'data' must be a mapping"),
        ({"lookback": 2}, r"'data' has unknown key\(s\): lookback"),
        ({"lookback_days": 0}, "'data.lookback_days' must be > 0"),
        ({"lookback_days": -1.5}, "'data.lookback_days' must be > 0"),
        ({"lookback_days": "2d"}, "'data.lookback_days' must be a number"),
        ({"lookback_days": True}, "'data.lookback_days' must be a number"),
        ({"max_rows": 0}, "'data.max_rows' must be >= 1"),
        ({"max_rows": 1.5}, "'data.max_rows' must be an integer"),
        (
            {"window_stride_seconds": -10},
            "'data.window_stride_seconds' must be a finite number >= 0",
        ),
        ({"window_stride_seconds": float("nan")}, "'data.window_stride_seconds' must be a finite"),
        ({"window_stride_seconds": "30s"}, "'data.window_stride_seconds' must be a number"),
        ({"window_stride_seconds": True}, "'data.window_stride_seconds' must be a number"),
        # 3 gate windows + 1, x the 1 h embargo = 4 h > 0.1 d (2.4 h)
        ({"lookback_days": 0.1}, r"shorter than the promotion gate needs: .* = 4 h"),
    ],
)
def test_invalid_data_bounds_are_rejected(tmp_path, data, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, variant(data=data)))


def test_the_lookback_floor_follows_the_gate_windows_and_the_longest_horizon(tmp_path):
    short = variant(data={"lookback_days": 0.1}, horizons=["30s", "5m"])
    short["labeling"] = {**short["labeling"]}
    assert load_config(write(tmp_path, short))["data"]["lookback_days"] == 0.1  # 4 x 5 m fits
    short["promotion"] = {**short["promotion"], "windows": 50}  # 51 x 5 m = 4.25 h does not
    with pytest.raises(ConfigError, match="4.25 h"):
        load_config(write(tmp_path, short))


def test_an_unparseable_horizon_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="'horizons': invalid horizon '5 min'"):
        load_config(write(tmp_path, variant(horizons=["5 min"])))


def test_data_bounds_default_when_omitted(tmp_path):
    from solia_ai.db import DataSettings

    cfg = load_config(write(tmp_path, variant(data=None)))
    assert DataSettings.from_config(cfg) == DataSettings(
        DATA_DEFAULTS["lookback_days"], DATA_DEFAULTS["max_rows"]
    )


def test_overrides_win_over_the_config_and_are_validated():
    from solia_ai.db import DataSettings

    cfg = variant(data={"lookback_days": 3, "max_rows": 10})
    assert DataSettings.from_config(cfg) == DataSettings(3.0, 10)
    assert DataSettings.from_config(cfg, lookback_days=1.5) == DataSettings(1.5, 10)
    assert DataSettings.from_config(cfg, max_rows=99) == DataSettings(3.0, 99)
    with pytest.raises(ConfigError, match="'data.max_rows' must be >= 1"):
        DataSettings.from_config(cfg, max_rows=0)
    with pytest.raises(ConfigError, match="'data.lookback_days' must be > 0"):
        DataSettings.from_config(cfg, lookback_days=-1)
    with pytest.raises(ConfigError, match="shorter than the promotion gate needs"):
        DataSettings.from_config(cfg, lookback_days=0.01)


@pytest.mark.parametrize("stride", [0, 30, 45.5])
def test_a_window_stride_is_accepted(tmp_path, stride):
    cfg = load_config(write(tmp_path, variant(data={"window_stride_seconds": stride})))
    assert cfg["data"]["window_stride_seconds"] == stride


def test_the_shipped_config_sets_a_window_stride():
    assert load_config()["data"]["window_stride_seconds"] >= 0
