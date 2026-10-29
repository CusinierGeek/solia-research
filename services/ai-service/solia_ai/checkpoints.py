"""Where checkpoint files live, shared by training, serving and retention. Kept torch-free.

- :func:`model_directory`: the directory training writes to and retention prunes (``model_dir``,
  else ``$MODEL_DIR``, else ``./models``);
- :func:`find_checkpoint`: the file a registered path is served from (the registered path, else
  the same file name in a fallback directory); :func:`resolve_checkpoint_path` is serving's use of
  it, with ``$MODEL_DIR`` as the fallback.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_MODEL_DIR = "models"
CHECKPOINT_GLOB = "*.pt"


def model_directory(model_dir: str | os.PathLike[str] | None = None) -> Path:
    """``model_dir``, else ``$MODEL_DIR``, else ``./models`` (not resolved)."""
    return Path(model_dir or os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR)


def serving_fallback_dir() -> Path | None:
    """The directory serving falls back to by file name: ``$MODEL_DIR``, or none when unset."""
    model_dir = os.environ.get("MODEL_DIR")
    return Path(model_dir) if model_dir else None


def checkpoint_file_name(path: str | os.PathLike[str]) -> str:
    # Split on both separators: a Windows-registered path read on Linux has no "/" in it.
    return re.split(r"[\\/]", str(path))[-1]


def fallback_path(path: str | os.PathLike[str], directory: Path | None) -> Path | None:
    """The same file name in ``directory`` (``None`` without a directory or a usable name)."""
    name = checkpoint_file_name(path)
    if directory is None or name in ("", ".", ".."):
        return None
    return directory / name


def find_checkpoint(path: str | os.PathLike[str], fallback_dir: Path | None) -> Path | None:
    """The registered checkpoint ``path`` if it exists here, else the same file name in
    ``fallback_dir`` (no fallback when ``None``); ``None`` when missing everywhere.

    The registry stores the absolute path where training wrote the checkpoint, which is only
    valid on that machine: a champion trained on the host (``./models``) and served from the
    container (``/app/models`` volume) is found by name in the serving process's model directory.
    """
    registered = Path(path)
    if registered.is_file():
        return registered
    candidate = fallback_path(path, fallback_dir)
    if candidate is not None and candidate.is_file():
        return candidate
    return None


def resolve_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    """Serving's lookup: :func:`find_checkpoint` with ``$MODEL_DIR`` as the fallback. Missing
    everywhere, the registered path is returned so ``load_checkpoint`` raises on it."""
    return find_checkpoint(path, serving_fallback_dir()) or Path(path)
