"""The shared JSON-lines checkpoint used by long unattended runs (search, ablation)."""

from __future__ import annotations

import json
import math

import pytest

from solia_ai import runlog

FINGERPRINT = {"settings": {"seeds": [0, 1], "n": 2}, "data": {"rows_read": 10}}
KINDS = {"what": "a demo file", "other": "another demo"}


def test_canonical_reads_back_like_json():
    assert runlog.canonical({"a": (1, 2), "b": math.nan}) == {"a": [1, 2], "b": None}


def test_differences_name_the_changed_keys():
    a = {"settings": {"seeds": [0], "n": 2}, "config": 1}
    b = {"settings": {"seeds": [0, 1], "n": 2}, "config": 2}
    assert runlog.differences(a, b) == ["config", "settings.seeds"]


def test_a_missing_or_empty_file_has_no_header(tmp_path):
    assert runlog.read_header(tmp_path / "missing.jsonl", "a demo file") is None
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    assert runlog.read_header(tmp_path / "empty.jsonl", "a demo file") is None


@pytest.mark.parametrize("first", ["not json", '{"kind": "run"}', "[1, 2]"])
def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path, first):
    path = tmp_path / "x.jsonl"
    path.write_text(first + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a demo file"):
        runlog.read_header(path, "a demo file")


def test_open_log_starts_a_file_then_returns_its_records(tmp_path):
    path = tmp_path / "sub" / "x.jsonl"

    assert runlog.open_log(path, FINGERPRINT, **KINDS) == []
    runlog.append(path, {"kind": "run", "i": 1})
    runlog.append(path, {"kind": "run", "i": 2})

    assert runlog.open_log(path, FINGERPRINT, **KINDS) == [
        {"kind": "run", "i": 1},
        {"kind": "run", "i": 2},
    ]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["kind"] == "header"


def test_open_log_refuses_another_fingerprint_and_names_the_difference(tmp_path):
    path = tmp_path / "x.jsonl"
    runlog.open_log(path, FINGERPRINT, **KINDS)
    other = {**FINGERPRINT, "data": {"rows_read": 11}}

    with pytest.raises(ValueError, match=r"another demo \(differs in: data.rows_read\)"):
        runlog.open_log(path, other, **KINDS)


def test_a_torn_last_line_is_dropped_but_a_corrupt_middle_line_is_an_error(tmp_path):
    path = tmp_path / "x.jsonl"
    runlog.open_log(path, FINGERPRINT, **KINDS)
    runlog.append(path, {"kind": "run", "i": 1})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "run", "i"')  # died mid-write

    assert runlog.open_log(path, FINGERPRINT, **KINDS) == [{"kind": "run", "i": 1}]

    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"kind": "run", "i": 2}) + "\n")
    with pytest.raises(ValueError, match="line 3 is corrupt"):
        runlog.open_log(path, FINGERPRINT, **KINDS)
