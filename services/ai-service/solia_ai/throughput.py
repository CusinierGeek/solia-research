"""Training throughput benchmark: how fast the dataset builds and batches reach the model.

``python -m solia_ai.throughput`` reads ``features`` like a retrain (the bounded read, same flags),
builds the samples (timed), then trains the configured model for ``--batches`` steps per loader
setting (``--workers 0 3 ...``; ``--per-item`` adds the pre-#152 per-sample loader) and reports,
for each: samples/s, the GPU utilisation (``nvidia-smi``, when present) and the memory of the
training process plus its loader workers — RSS, and PSS (shared pages split between the processes
sharing them: workers that share the rows add little PSS; workers holding their own copy would
add a full copy each). Memory is read from ``/proc`` (Linux, i.e. the container; ``None``
elsewhere). Writes nothing: no checkpoint, no registry row. Numbers go to docs/ML-MODEL.md
"Training throughput".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import (
    LoaderSettings,
    SequenceDataset,
    balanced_sampler,
    config_samples,
    make_loaders,
)
from .loss import SoliaLoss
from .model import SoliaTransformer


def process_memory(pid: int | str = "self") -> dict[str, int] | None:
    """``{"rss_kb", "pss_kb"}`` of a process from ``/proc/<pid>/smaps_rollup`` (``None``
    without it)."""
    path = Path(f"/proc/{pid}/smaps_rollup")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    fields = {}
    for line in lines:
        name, _, rest = line.partition(":")
        if name in ("Rss", "Pss"):
            fields[f"{name.lower()}_kb"] = int(rest.split()[0])
    return fields or None


def peak_rss_mb() -> float | None:
    """The process's peak RSS so far (``ru_maxrss``; ``None`` where unavailable)."""
    try:
        import resource  # POSIX only
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class GpuSampler:
    """Mean GPU utilisation (``nvidia-smi``) while running; ``mean`` is ``None`` without it."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuSampler:
        if shutil.which("nvidia-smi"):
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        query = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(query, capture_output=True, text=True, timeout=5).stdout
                self.values.append(float(out.split()[0]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                return

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.values)) if self.values else None


def _per_item_loader(train: SequenceDataset, batch_size: int, seed: int) -> DataLoader:
    """The pre-#152 train loader: one ``__getitem__`` per sample, default collation."""
    return DataLoader(train, batch_size=batch_size, sampler=balanced_sampler(train, seed=seed))


def _memory(loader: DataLoader) -> dict[str, float] | None:
    """RSS / PSS (MB) of this process plus the loader's live workers."""
    iterator = getattr(loader, "_iterator", None)
    workers = getattr(iterator, "_workers", None) or []
    parts = [process_memory()] + [process_memory(w.pid) for w in workers]
    if any(p is None for p in parts):
        return None
    return {
        "rss_mb": sum(p["rss_kb"] for p in parts) / 1024,
        "pss_mb": sum(p["pss_kb"] for p in parts) / 1024,
        "processes": len(parts),
    }


def measure_loader(
    model: SoliaTransformer,
    loss_fn: SoliaLoss,
    loader: DataLoader,
    *,
    batches: int,
    device: torch.device,
    warmup: int = 5,
) -> dict[str, Any]:
    """Train ``batches`` steps from ``loader`` (after ``warmup`` untimed ones); samples/s, the
    GPU utilisation and the memory while the workers run."""
    optimizer = AdamW(model.parameters(), lr=1e-4)
    model.train()
    pin = bool(getattr(loader, "pin_memory", False))
    n = seen = 0
    started = 0.0
    with GpuSampler() as gpu:
        for x, padding_mask, targets in loader:
            if n == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started, seen = time.perf_counter(), 0
            optimizer.zero_grad()
            outputs = model(
                x.to(device, non_blocking=pin), padding_mask.to(device, non_blocking=pin)
            )
            loss = loss_fn(outputs, {k: v.to(device, non_blocking=pin) for k, v in targets.items()})
            loss["total"].backward()
            optimizer.step()
            seen += len(x)
            n += 1
            if n == warmup + batches:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        memory = _memory(loader)
    return {
        "batches": n - warmup,
        "seconds": seconds,
        "samples_per_s": seen / seconds if seconds > 0 else None,
        "gpu_util_pct": gpu.mean,
        "memory": memory,
    }


def run(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    workers: Sequence[int],
    batches: int,
    device: str | torch.device,
    per_item: bool = False,
) -> dict[str, Any]:
    """Build the samples and measure each loader setting (module doc)."""
    device = torch.device(device)
    started = time.perf_counter()
    samples = config_samples(df, config)
    report: dict[str, Any] = {
        "rows": len(df),
        "tokens": int(df["mint"].nunique()),
        "samples": len(samples),
        "build_seconds": time.perf_counter() - started,
        "rss_after_build_mb": (process_memory() or {}).get("rss_kb", 0) / 1024 or None,
        "runs": {},
    }
    train = SequenceDataset(samples)
    training = config["training"]
    batch_size, seed = int(training["batch_size"]), int(training["seed"])
    model = SoliaTransformer.from_config(config).to(device)
    model.fit_normalizer(samples.features)
    loss_fn = SoliaLoss.from_config(config).to(device)
    pin = device.type == "cuda"

    settings: list[tuple[str, DataLoader]] = []
    if per_item:
        settings.append(("per-item, 0 workers", _per_item_loader(train, batch_size, seed)))
    for n in workers:
        loader_settings = LoaderSettings(num_workers=n, pin_memory=pin)
        loader, _ = make_loaders(
            train, train, batch_size=batch_size, seed=seed, loader=loader_settings
        )
        settings.append((f"batched, {n} workers", loader))
    for name, loader in settings:
        report["runs"][name] = measure_loader(
            model, loss_fn, loader, batches=batches, device=device
        )
        del loader
    report["peak_rss_mb"] = peak_rss_mb()
    return report


def main(argv: list[str] | None = None) -> None:
    from .db import DataSettings, add_data_bounds_arguments, make_engine, read_training_features

    parser = argparse.ArgumentParser(description="Measure the training throughput (no writes).")
    add_data_bounds_arguments(parser)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--batches", type=int, default=200, help="timed steps per setting")
    parser.add_argument("--per-item", action="store_true", help="also the pre-#152 loader")
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)
    if any(n < 0 for n in args.workers) or args.batches < 1:
        raise SystemExit("--workers must be >= 0 and --batches >= 1")

    config = load_config()
    data = DataSettings.from_config(
        config, lookback_days=args.lookback_days, max_rows=args.max_rows
    )
    started = time.monotonic()
    df, bounds = read_training_features(make_engine(), data)
    read_seconds = time.monotonic() - started
    print(bounds.describe(), flush=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = run(
        df, config, workers=args.workers, batches=args.batches, device=device,
        per_item=args.per_item,
    )  # fmt: skip
    print(json.dumps({"read_seconds": read_seconds, "device": device, "cpus": os.cpu_count(),
                      **report}, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
