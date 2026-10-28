"""requirements.lock (the Docker image's dependency set) stays in sync with pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def lock_entries() -> list[Requirement]:
    lines = (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    return [Requirement(line) for line in lines if line.strip() and not line.startswith("#")]


LOCK = {canonicalize_name(r.name): r for r in lock_entries()}


def pinned_version(req: Requirement) -> Version:
    (spec,) = req.specifier
    return Version(spec.version)


def test_every_lock_entry_is_an_exact_pin():
    for req in lock_entries():
        specs = list(req.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"{req} is not an exact pin"
        assert not req.extras and req.marker is None, f"{req}: extras/markers are resolved away"


@pytest.mark.parametrize("dependency", PROJECT["dependencies"])
def test_every_runtime_dependency_is_locked_at_its_pinned_version(dependency):
    req = Requirement(dependency)
    name = canonicalize_name(req.name)

    assert name in LOCK, f"{req.name} missing from requirements.lock: regenerate it"
    locked = pinned_version(LOCK[name])
    assert req.specifier.contains(
        locked, prereleases=True
    ), f"{req.name}: pyproject says {req.specifier}, lock has {locked}: regenerate it"


def test_torch_is_locked_to_the_cpu_build():
    assert pinned_version(LOCK["torch"]).local == "cpu"


@pytest.mark.parametrize("dependency", PROJECT["optional-dependencies"]["dev"])
def test_no_dev_dependency_is_locked(dependency):
    assert canonicalize_name(Requirement(dependency).name) not in LOCK
