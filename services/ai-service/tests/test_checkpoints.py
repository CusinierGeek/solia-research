"""Shared checkpoint location rules (training, serving, retention)."""

from __future__ import annotations

from pathlib import Path

from solia_ai.checkpoints import (
    checkpoint_file_name,
    fallback_path,
    find_checkpoint,
    model_directory,
    resolve_checkpoint_path,
    serving_fallback_dir,
)


def test_model_directory_prefers_the_argument_then_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert model_directory() == Path("models")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert model_directory() == tmp_path
    assert model_directory(tmp_path / "x") == tmp_path / "x"


def test_serving_falls_back_only_when_model_dir_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert serving_fallback_dir() is None
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert serving_fallback_dir() == tmp_path


def test_file_names_split_on_both_separators():
    assert checkpoint_file_name(r"C:\models\a.pt") == "a.pt"
    assert checkpoint_file_name("/app/models/a.pt") == "a.pt"
    assert fallback_path("/x/..", Path("/d")) is None
    assert fallback_path("/x/a.pt", None) is None


def test_find_checkpoint(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"x")
    assert find_checkpoint(str(tmp_path / "a.pt"), None) == tmp_path / "a.pt"
    assert find_checkpoint("/host/a.pt", tmp_path) == tmp_path / "a.pt"
    assert find_checkpoint("/host/a.pt", None) is None
    assert find_checkpoint("/host/b.pt", tmp_path) is None


def test_resolve_checkpoint_path_uses_model_dir(tmp_path, monkeypatch):
    (tmp_path / "a.pt").write_bytes(b"x")
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert resolve_checkpoint_path("/host/a.pt") == Path("/host/a.pt")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert resolve_checkpoint_path("/host/a.pt") == tmp_path / "a.pt"
