"""Append-only JSON-lines checkpoints for long unattended runs (search trials, ablation runs).

The first line is a header holding the run's **fingerprint** (settings, config, data window); every
later line is one finished unit of work, appended and fsynced as it ends. A rerun with the same
fingerprint skips what the file already holds, so a reboot loses at most the unit in progress; a
different fingerprint is refused rather than mixing two runs. A torn last line (the process died
mid-write) is ignored: that unit simply runs again.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evaluate import json_safe


def canonical(value: Any) -> Any:
    """``value`` as it reads back from JSON (tuples as lists, non-finite floats as None)."""
    return json.loads(json.dumps(json_safe(value), sort_keys=True, default=str))


def differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """The ``section.key`` entries where two fingerprints differ."""
    out = []
    for section in sorted(a.keys() | b.keys()):
        x, y = a.get(section), b.get(section)
        if isinstance(x, dict) and isinstance(y, dict):
            out += [f"{section}.{k}" for k in sorted(x.keys() | y.keys()) if x.get(k) != y.get(k)]
        elif x != y:
            out.append(section)
    return out


def read_header(path: str | Path, what: str) -> dict[str, Any] | None:
    """The fingerprint ``path`` was started with; None when there is no file yet.

    Raises ``ValueError`` when the file is not a checkpoint (``what`` names the expected kind in
    the message, e.g. "a search trials file")."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(encoding="utf-8") as fh:
        first = fh.readline()
    try:
        header = json.loads(first)
    except json.JSONDecodeError:
        header = None
    if not isinstance(header, dict) or header.get("kind") != "header":
        raise ValueError(f"{path} is not {what}; delete it or choose another --out")
    return header["fingerprint"]


def open_log(
    path: str | Path, fingerprint: Mapping[str, Any], *, what: str, other: str
) -> list[dict[str, Any]]:
    """Start ``path`` with ``fingerprint`` or check it holds the same one; returns its records.

    ``other`` names the kind of run in the refusal ("another search"). Raises ``ValueError`` on a
    different fingerprint or a corrupt line other than the last one."""
    path = Path(path)
    fingerprint = canonical(fingerprint)
    stored = read_header(path, what)
    if stored is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        append(path, {"kind": "header", "fingerprint": fingerprint}, mode="w")
        return []
    if stored != fingerprint:
        raise ValueError(
            f"{path} holds the work of {other} (differs in: "
            f"{', '.join(differences(fingerprint, stored))}); rerun with the same settings, "
            "or delete it (--restart) to start over"
        )
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn last write: that unit runs again
            raise ValueError(f"{path}: line {i + 2} is corrupt; delete it to start over") from None
    return records


def append(path: str | Path, record: Mapping[str, Any], mode: str = "a") -> None:
    """Write one record as a JSON line and sync it to disk before returning."""
    with Path(path).open(mode, encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
