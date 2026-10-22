"""load_config(): file resolution and validation."""

from pathlib import Path

import pytest
import yaml

from solia_ai.config import CONFIG_ENV, ConfigError, load_config

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
    path = write(tmp_path, variant(model={**VALID["model"], "seq_len": 7}))

    assert load_config(path)["model"]["seq_len"] == 7


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
        ({"model": {**VALID["model"], "d_model": 130}}, "divisible by 'model.n_heads'"),
        ({"model": {**VALID["model"], "dropout": 1.0}}, r"'model.dropout' must be in \[0, 1\)"),
        (
            {"training": {**VALID["training"], "learning_rate": -1}},
            "'training.learning_rate' must be > 0",
        ),
        ({"loss": {**VALID["loss"], "alpha": 1.5}}, r"'loss.alpha' must be in \[0, 1\]"),
        ({"loss": {**VALID["loss"], "alpha": True}}, "'loss.alpha' must be a number"),
        ({"loss": {**VALID["loss"], "focal_gamma": -2}}, "'loss.focal_gamma' must be >= 0"),
        ({"horizons": []}, "'horizons' must be a non-empty list"),
        ({"horizons": ["30s", 5]}, "'horizons' must be a non-empty list of strings"),
    ],
)
def test_invalid_values_are_rejected_with_the_key_named(tmp_path, sections, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, variant(**sections)))


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
