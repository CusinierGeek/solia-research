import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solia_ai.config import ConfigError, load_config, validate_config
from solia_ai.eligibility import EligibilityRule

FIXTURE = Path(__file__).resolve().parents[2] / "shared" / "testdata" / "eligibility-cases.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))


def rows_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["unique_traders", "volume", "token_age_sec"])


# --------------------------------------------------------------------------- parity fixture


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["name"])
def test_the_shared_fixture_cases(case):
    """The .NET bot's EligibilityParityTests run the very same cases."""
    rule = EligibilityRule.from_mapping(case["rule"])

    eligible = rule.eligible_frame(rows_frame(case["rows"]))

    assert eligible.tolist() == case["eligible"]


@pytest.mark.parametrize("rule", CASES["invalid_rules"], ids=str)
def test_the_shared_fixture_invalid_rules_are_rejected(rule):
    with pytest.raises(ValueError, match="eligibility"):
        EligibilityRule.from_mapping(rule)


# --------------------------------------------------------------------------- the rule


def test_a_row_depends_only_on_itself_and_earlier_rows():
    rule = EligibilityRule(min_unique_traders=3)
    traders = np.array([1, 2, 5, 1, 1], np.float64)
    vol = np.zeros(5)
    age = np.arange(5, dtype=np.float64)

    full = rule.eligible(traders, vol, age)

    for i in range(len(traders)):  # truncating the future never changes the past
        assert rule.eligible(traders[: i + 1], vol[: i + 1], age[: i + 1]).tolist() == (
            full[: i + 1].tolist()
        )


def test_disabled_rule_needs_no_input_columns():
    rows = pd.DataFrame({"price": [1.0, 2.0]})

    assert EligibilityRule().eligible_frame(rows).tolist() == [True, True]


def test_an_enabled_rule_refuses_a_frame_without_its_columns():
    with pytest.raises(ValueError, match="unique_traders"):
        EligibilityRule(min_volume_sol=1.0).eligible_frame(pd.DataFrame({"volume": [1.0]}))


def test_numeric_strings_from_the_database_are_read_as_numbers():
    rows = pd.DataFrame(
        {"unique_traders": [None, 4], "volume": ["0.1", "2.5"], "token_age_sec": [1, 2]},
        dtype=object,
    )

    rule = EligibilityRule(min_unique_traders=4, min_volume_sol=2.5)

    assert rule.eligible_frame(rows).tolist() == [False, True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_unique_traders": -1},
        {"min_volume_sol": -1.0},
        {"min_token_age_sec": -0.5},
        {"min_volume_sol": math.nan},
        {"min_token_age_sec": math.inf},
    ],
)
def test_constructor_rejects_bad_thresholds(kwargs):
    with pytest.raises(ValueError, match=">= 0"):
        EligibilityRule(**kwargs)


def test_round_trips_through_its_dict_and_reads_a_model_without_a_rule_as_disabled():
    rule = EligibilityRule(min_unique_traders=3, min_volume_sol=0.5, min_token_age_sec=10)

    assert EligibilityRule.of({"eligibility": rule.as_dict()}) == rule
    assert EligibilityRule.of({"model": {}}) == EligibilityRule()
    assert EligibilityRule.of(None) == EligibilityRule()
    assert not EligibilityRule.of(None).enabled
    assert rule.enabled
    assert "traders>=3" in rule.describe()
    assert "disabled" in EligibilityRule().describe()


# --------------------------------------------------------------------------- config


def test_the_packaged_config_has_a_valid_rule():
    cfg = load_config()

    assert EligibilityRule.from_config(cfg) == EligibilityRule.from_mapping(cfg["eligibility"])


def test_an_omitted_section_is_the_disabled_rule():
    cfg = load_config()
    cfg.pop("eligibility", None)

    validate_config(cfg)
    assert EligibilityRule.from_config(cfg) == EligibilityRule()


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"min_unique_traders": -2}, "eligibility.min_unique_traders' must be a finite number"),
        ({"min_volume_sol": -0.1}, "eligibility.min_volume_sol' must be a finite number"),
        ({"min_token_age_sec": "old"}, "eligibility.min_token_age_sec' must be a number"),
        ({"min_unique_traders": 1.5}, "eligibility.min_unique_traders' must be an integer"),
        ({"min_traders": 3}, "unknown key"),
        ([3], "'eligibility' must be a mapping"),
    ],
)
def test_invalid_config_fails_fast_with_a_clear_message(section, message):
    cfg = load_config()
    cfg["eligibility"] = section

    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)
