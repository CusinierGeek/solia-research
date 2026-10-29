"""The throughput benchmark on a small in-memory frame (no database, CPU)."""

from __future__ import annotations

import sys

import pytest
from test_train import make_features_df, small_config

from solia_ai import throughput


def test_run_measures_the_build_and_every_loader_setting():
    report = throughput.run(
        make_features_df(), small_config(), workers=[0], batches=3, device="cpu", per_item=True
    )

    assert report["samples"] > 0 and report["build_seconds"] > 0
    assert list(report["runs"]) == ["per-item, 0 workers", "batched, 0 workers"]
    for run in report["runs"].values():
        assert run["batches"] == 3 and run["samples_per_s"] > 0
    if sys.platform.startswith("linux"):
        assert report["runs"]["batched, 0 workers"]["memory"]["processes"] == 1
    else:
        assert report["runs"]["batched, 0 workers"]["memory"] is None  # no /proc


def test_process_memory_is_none_without_proc():
    assert throughput.process_memory(pid="no-such-process") is None


@pytest.mark.parametrize("argv", [["--workers", "-1"], ["--batches", "0"]])
def test_main_rejects_invalid_arguments(argv):
    with pytest.raises(SystemExit, match="--workers"):
        throughput.main(argv)
