"""The Docker image locks stay in sync with pyproject.toml: requirements.lock (CPU, the default
image) and requirements-cu130.lock (the GPU image), which must be the same set but for torch's build
and the CUDA runtime wheels it pulls in."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
LOCK_FILES = ["requirements.lock", "requirements-cu130.lock"]

# What CUDA torch adds on top of the CPU set (its runtime wheels and triton, its GPU compiler).
CUDA_ONLY_PREFIXES = ("nvidia-", "cuda-")
CUDA_ONLY_NAMES = {"triton"}


def lock_entries(file: str) -> list[Requirement]:
    lines = (ROOT / file).read_text(encoding="utf-8").splitlines()
    return [Requirement(line) for line in lines if line.strip() and not line.startswith("#")]


def lock(file: str) -> dict[str, Requirement]:
    return {canonicalize_name(r.name): r for r in lock_entries(file)}


def pinned_version(req: Requirement) -> Version:
    (spec,) = req.specifier
    return Version(spec.version)


@pytest.mark.parametrize("file", LOCK_FILES)
def test_every_lock_entry_is_an_exact_pin(file):
    for req in lock_entries(file):
        specs = list(req.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"{file}: {req} is not an exact pin"
        assert (
            not req.extras and req.marker is None
        ), f"{file}: {req}: extras/markers are resolved away"


@pytest.mark.parametrize("file", LOCK_FILES)
@pytest.mark.parametrize("dependency", PROJECT["dependencies"])
def test_every_runtime_dependency_is_locked_at_its_pinned_version(file, dependency):
    req = Requirement(dependency)
    name = canonicalize_name(req.name)
    locked_set = lock(file)

    assert name in locked_set, f"{req.name} missing from {file}: regenerate it"
    locked = pinned_version(locked_set[name])
    assert req.specifier.contains(
        locked, prereleases=True
    ), f"{req.name}: pyproject says {req.specifier}, {file} has {locked}: regenerate it"


@pytest.mark.parametrize(
    ("file", "build"), [("requirements.lock", "cpu"), ("requirements-cu130.lock", "cu130")]
)
def test_torch_is_locked_to_the_variant_build(file, build):
    assert pinned_version(lock(file)["torch"]).local == build


def test_gpu_lock_matches_the_cpu_lock_except_torch_and_its_cuda_wheels():
    cpu, gpu = lock("requirements.lock"), lock("requirements-cu130.lock")

    for name, req in cpu.items():
        assert (
            name in gpu
        ), f"{name} is in requirements.lock but not requirements-cu130.lock: regenerate it"
        if name != "torch":
            assert pinned_version(gpu[name]) == pinned_version(req), (
                f"{name}: requirements.lock has {pinned_version(req)}, requirements-cu130.lock has "
                f"{pinned_version(gpu[name])}: regenerate the GPU lock constrained on the CPU one"
            )
    assert pinned_version(gpu["torch"]).public == pinned_version(cpu["torch"]).public

    extra = set(gpu) - set(cpu)
    assert extra, "the GPU lock adds no CUDA runtime: was it compiled against the CPU index?"
    for name in extra:
        assert (
            name.startswith(CUDA_ONLY_PREFIXES) or name in CUDA_ONLY_NAMES
        ), f"unexpected GPU-only package {name}"


@pytest.mark.parametrize("file", LOCK_FILES)
@pytest.mark.parametrize("dependency", PROJECT["optional-dependencies"]["dev"])
def test_no_dev_dependency_is_locked(file, dependency):
    assert canonicalize_name(Requirement(dependency).name) not in lock(file)


@pytest.mark.parametrize("file", LOCK_FILES)
@pytest.mark.parametrize("dependency", PROJECT["optional-dependencies"]["benchmark"])
def test_the_benchmark_extra_stays_out_of_the_serving_image(file, dependency):
    assert canonicalize_name(Requirement(dependency).name) not in lock(file)
