"""Unattended, resumable overnight sweep (#181): the #160 overfitting search, stage 1 then stage 2.

Run inside the ai-service container (``infra/scripts/overnight-sweep.sh`` launches it detached):

* ``preflight`` — refuses (exit 1, every problem listed) unless the night can run undisturbed: no
  retrain holds the retrain lock, the retrain scheduler and the paper bot are off (both share the
  GPU and the database), CUDA is visible, the reports volume has room, no other sweep is running.
* ``run`` — stage 1 (``overfitting-data``: stride x row budget) then stage 2
  (``overfitting-regularization``) on stage 1's chosen stride / row budget, written as a config
  file next to the reports and handed to stage 2 via ``SOLIA_AI_CONFIG``. Both stages checkpoint
  every finished trial (:mod:`solia_ai.search`), so rerunning the same command after a reboot
  resumes where it stopped and loses at most the trial in progress. Ends with
  ``overnight-summary.txt``; exits 1 if a stage or a trial failed.
* ``summary`` — (re)writes the summary from the reports already there.

Stage 1's choice: its best trial's ``window_stride_seconds`` / ``max_rows`` only when that trial
beats the baseline (paired objective verdict ``better``); otherwise the base config's values.
``--stride`` / ``--max-rows`` override it (e.g. to run stage 2 alone).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REPORTS_DIR = "/app/models/reports"
STAGE1, STAGE2 = "overfitting-data", "overfitting-regularization"
STAGE2_CONFIG = f"{STAGE2}.config.yaml"
SUMMARY = "overnight-summary.txt"
DEFAULT_MIN_FREE_GB = 20.0
STAGE_DATA_PARAMS = ("window_stride_seconds", "max_rows")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


# --------------------------------------------------------------------------- preflight


def env_flag(name: str, value: str | None) -> bool:
    """A boolean env value as the services read it (unset / empty = off); ValueError otherwise."""
    raw = (value or "").strip().lower()
    if raw not in _TRUE | _FALSE:
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return raw in _TRUE


@dataclass(frozen=True)
class Preflight:
    """What the night depends on, gathered by :func:`gather_preflight` (injectable in tests)."""

    retrain_running: bool | None  # None: the database could not be asked
    scheduler_enabled: str | None  # RETRAIN_SCHEDULER_ENABLED of this (ai-service) container
    paper_bot_enabled: str | None  # PAPER_BOT_ENABLED of the backend (empty: not running / unset)
    cuda: str | None  # the GPU's name, None when CUDA isn't available
    free_bytes: int
    other_sweeps: list[int]  # pids of other running searches / overnight runs


def preflight_problems(p: Preflight, *, min_free_bytes: int) -> list[str]:
    """Every reason not to start; empty when the night can run."""
    problems = []
    if p.retrain_running is None:
        problems.append("cannot check the retrain lock: the database is unreachable")
    elif p.retrain_running:
        problems.append("a retrain (or rollback) holds the retrain lock: wait for it to finish")
    for name, value in (
        ("RETRAIN_SCHEDULER_ENABLED", p.scheduler_enabled),
        ("PAPER_BOT_ENABLED", p.paper_bot_enabled),
    ):
        try:
            if env_flag(name, value):
                problems.append(
                    f"{name}=true in the running stack: set it to false (.env) and recreate the "
                    "container, it shares the GPU / database with the sweep"
                )
        except ValueError as exc:
            problems.append(str(exc))
    if p.cuda is None:
        problems.append(
            "CUDA is not visible: run the GPU image (docker-compose.gpu.yml), never the CPU one"
        )
    if p.free_bytes < min_free_bytes:
        problems.append(
            f"only {p.free_bytes / 1e9:.1f} GB free on the reports volume "
            f"(need {min_free_bytes / 1e9:.0f} GB): free some space first"
        )
    if p.other_sweeps:
        problems.append(
            f"a sweep is already running (pid {', '.join(map(str, p.other_sweeps))}): let it "
            "finish, or stop it before starting another"
        )
    return problems


_SWEEP_MARKERS = ("solia_ai.search", "solia_ai.overnight run", "solia_ai.overnight\0run")


def running_sweeps(proc: Path = Path("/proc")) -> list[int]:
    """Other pids whose command line runs a search or an overnight run (Linux ``/proc``)."""
    pids = []
    for entry in proc.glob("[0-9]*/cmdline") if proc.is_dir() else ():
        pid = int(entry.parent.name)
        try:
            cmdline = entry.read_bytes().decode("utf-8", "replace")
        except OSError:
            continue  # gone meanwhile
        if pid != os.getpid() and any(m in cmdline for m in _SWEEP_MARKERS):
            pids.append(pid)
    return sorted(pids)


def retrain_lock_held(engine) -> bool:
    """Whether any session holds the retrain/rollback advisory lock right now."""
    from sqlalchemy import text

    from .retrain import LOCK_KEY

    with engine.connect() as conn:
        return bool(
            conn.execute(
                text(
                    "SELECT count(*) > 0 FROM pg_locks WHERE locktype = 'advisory' AND granted "
                    "AND classid::bigint = :hi AND objid::bigint = :lo AND objsubid = 1"
                ),
                {"hi": LOCK_KEY >> 32, "lo": LOCK_KEY & 0xFFFFFFFF},
            ).scalar()
        )


def gather_preflight(reports_dir: Path, paper_bot_enabled: str | None) -> Preflight:
    import torch

    from .db import make_engine
    from .retrain_service import ENABLED_ENV

    try:
        engine = make_engine()
        try:
            running = retrain_lock_held(engine)
        finally:
            engine.dispose()
    except Exception as exc:  # reported as a problem, not a crash
        print(f"retrain lock check failed: {exc}", file=sys.stderr)
        running = None
    existing = reports_dir if reports_dir.exists() else Path(reports_dir.anchor or ".")
    return Preflight(
        retrain_running=running,
        scheduler_enabled=os.environ.get(ENABLED_ENV),
        paper_bot_enabled=paper_bot_enabled,
        cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        free_bytes=shutil.disk_usage(existing).free,
        other_sweeps=running_sweeps(),
    )


# --------------------------------------------------------------------------- stages


def stage1_choice(report: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """The stride / row budget stage 2 runs on: the best stage-1 trial's when it beats the baseline
    (objective verdict ``better``), else the base config's."""
    data = base.get("data", {})
    chosen = {k: data[k] for k in STAGE_DATA_PARAMS if k in data}
    if report.get("best", "baseline") == "baseline":
        return chosen
    if report.get("comparison", {}).get("objective", {}).get("verdict") != "better":
        return chosen
    best = next(t for t in report["trials"] if t["name"] == report["best"])
    return {**chosen, **{k: v for k, v in best["params"].items() if k in STAGE_DATA_PARAMS}}


def stage2_config(base: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    """The base config with stage 1's ``data`` choice applied (validated)."""
    from .config import validate_config

    cfg = {**base, "data": {**base.get("data", {}), **data}}
    validate_config(cfg, "stage 2 config")
    return cfg


def search_argv(
    space: str,
    reports_dir: Path,
    *,
    strategy: str,
    trials: int,
    windows: int,
    seeds: Sequence[int],
    objective: str,
    device: str | None,
) -> list[str]:
    argv = [
        "--space", space, "--strategy", strategy, "--trials", str(trials),
        "--windows", str(windows), "--seeds", *map(str, seeds), "--objective", objective,
        "--out", str(reports_dir / f"{space}.json"),
        "--diff-out", str(reports_dir / f"{space}.diff"),
    ]  # fmt: skip
    return [*argv, "--device", device] if device else argv


def run_stage(
    name: str, argv: list[str], search_main: Callable[[list[str]], None], log: Callable[[str], None]
) -> str | None:
    """Runs one search; the failure message, or None when it finished."""
    log(f"=== {name}: python -m solia_ai.search {' '.join(argv)}")
    started = time.monotonic()
    try:
        search_main(argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            return f"{name} stopped: {exc.code}"
    except Exception as exc:  # reported in the summary, never swallowed
        return f"{name} crashed: {type(exc).__name__}: {exc}"
    finally:
        log(f"=== {name}: {time.monotonic() - started:.0f} s")
    return None


# --------------------------------------------------------------------------- summary


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def summarize_report(stage: str, report: Mapping[str, Any]) -> list[str]:
    """A few human-readable lines for one stage's JSON report."""
    by_name = {t["name"]: t for t in report["trials"]}
    baseline, best = by_name.get("baseline"), by_name.get(report["best"])
    objective = report["settings"]["objective"]
    lines = [
        f"{stage}: {len(report['trials'])} trial(s) finished, {len(report.get('failed', []))} "
        f"failed, stopped by {report['stopped']}, {report.get('elapsed_seconds', 0) / 3600:.1f} h "
        f"of training ({len(report.get('resumed', []))} resumed from the checkpoint)",
        f"  objective: mean {objective}, ranked by mean - std",
        f"  baseline: score {_fmt(baseline and baseline['score'])}",
    ]
    for name in report["ranking"][:3]:
        t = by_name[name]
        params = ", ".join(f"{k}={v}" for k, v in t["params"].items()) or "(base config)"
        lines.append(f"  {name}: score {_fmt(t['score'])} — {params}")
    if best is None or report["best"] == "baseline":
        lines.append("  best: the baseline — no change proposed")
    else:
        verdict = report["comparison"]["objective"]
        lines.append(
            f"  best: {report['best']}, lift vs baseline {_fmt(verdict['mean'])} ± "
            f"{_fmt(verdict['std'])} ({verdict['verdict']})"
        )
    for f in report.get("failed", []):
        lines.append(f"  FAILED {f['name']} ({f['params']}): {f['error']}")
    return lines


def render_summary(
    reports_dir: Path,
    *,
    failures: Sequence[str] = (),
    choice: Mapping[str, Any] | None = None,
    elapsed_seconds: float | None = None,
) -> str:
    lines = ["Overnight sweep (#160 / #172)", ""]
    for stage in (STAGE1, STAGE2):
        path = reports_dir / f"{stage}.json"
        if not path.is_file():
            lines += [f"{stage}: no report ({path} missing)", ""]
            continue
        lines += [*summarize_report(stage, json.loads(path.read_text(encoding="utf-8"))), ""]
        if stage == STAGE1 and choice is not None:
            lines += [f"stage 2 runs on: {dict(choice)}", ""]
    if elapsed_seconds is not None:
        lines.append(f"wall time of this run: {elapsed_seconds / 3600:.1f} h")
    if failures:
        lines += ["FAILURES:", *(f"  {f}" for f in failures)]
    else:
        lines.append("no failure")
    lines.append(f"reports: {reports_dir}/{{{STAGE1},{STAGE2}}}.json / .diff")
    return "\n".join(lines) + "\n"


def write_summary(reports_dir: Path, text: str) -> Path:
    path = reports_dir / SUMMARY
    path.write_text(text, encoding="utf-8")
    return path


def failed_trials(reports_dir: Path) -> list[str]:
    out = []
    for stage in (STAGE1, STAGE2):
        path = reports_dir / f"{stage}.json"
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            out += [f"{stage} {f['name']}: {f['error']}" for f in report.get("failed", [])]
    return out


# --------------------------------------------------------------------------- run


def run(
    args: argparse.Namespace,
    *,
    search_main: Callable[[list[str]], None] | None = None,
    load_config: Callable[[], dict[str, Any]] | None = None,
    log: Callable[[str], None] = lambda msg: print(msg, flush=True),
) -> int:
    from . import config as config_module
    from . import search

    search_main = search_main or search.main
    load_config = load_config or config_module.load_config
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    common = {
        "windows": args.windows,
        "seeds": args.seeds,
        "objective": args.objective,
        "device": args.device,
    }
    restart = ["--restart"] if args.restart else []
    failures: list[str] = []
    base = load_config()

    failure = run_stage(
        STAGE1,
        search_argv(STAGE1, reports, strategy="grid", trials=args.stage1_trials, **common)
        + restart,
        search_main,
        log,
    )
    if failure:
        failures.append(failure)
    choice = None
    stage1 = reports / f"{STAGE1}.json"
    if stage1.is_file() and not failure:
        choice = stage1_choice(json.loads(stage1.read_text(encoding="utf-8")), base)
    if args.stride is not None or args.max_rows is not None:
        choice = {**(choice or {}), **_overrides(args)}

    if choice is None:
        failures.append(f"{STAGE2} skipped: no stage 1 report to take the stride / budget from")
    else:
        cfg_path = reports / STAGE2_CONFIG
        try:
            cfg = stage2_config(base, choice)
        except Exception as exc:
            failures.append(f"{STAGE2} skipped: invalid stage 2 config {choice}: {exc}")
        else:
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            log(f"stage 2 config {cfg_path}: data {cfg['data']}")
            previous = os.environ.get(config_module.CONFIG_ENV)
            os.environ[config_module.CONFIG_ENV] = str(cfg_path)
            try:
                failure = run_stage(
                    STAGE2,
                    search_argv(STAGE2, reports, strategy="random", trials=args.stage2_trials,
                                **common) + restart,
                    search_main,
                    log,
                )  # fmt: skip
            finally:
                if previous is None:
                    os.environ.pop(config_module.CONFIG_ENV, None)
                else:
                    os.environ[config_module.CONFIG_ENV] = previous
            if failure:
                failures.append(failure)

    failures += failed_trials(reports)
    text = render_summary(
        reports, failures=failures, choice=choice, elapsed_seconds=time.monotonic() - started
    )
    log(text)
    log(f"summary written to {write_summary(reports, text)}")
    return 1 if failures else 0


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if args.stride is not None:
        out["window_stride_seconds"] = args.stride
    if args.max_rows is not None:
        out["max_rows"] = args.max_rows
    return out


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="refuse unless the night can run undisturbed")
    pre.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    pre.add_argument(
        "--paper-bot-enabled",
        default="",
        help="the backend's PAPER_BOT_ENABLED (empty: backend not running or unset = off)",
    )
    pre.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)

    go = sub.add_parser("run", help="stage 1 then stage 2, resumable; writes the summary")
    go.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    go.add_argument("--stage1-trials", type=int, default=5, help="baseline included")
    go.add_argument("--stage2-trials", type=int, default=10, help="baseline included")
    go.add_argument("--windows", type=int, default=2)
    go.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    go.add_argument("--objective", choices=("roc_auc", "pr_auc"), default="pr_auc")
    go.add_argument("--stride", type=float, help="stage 2 stride, instead of stage 1's choice")
    go.add_argument("--max-rows", type=int, help="stage 2 row budget, instead of stage 1's choice")
    go.add_argument("--device", help="torch device (default: cuda if available)")
    go.add_argument("--restart", action="store_true", help="discard the checkpoints, start over")

    summ = sub.add_parser("summary", help="(re)write the summary from the reports")
    summ.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)

    args = parser.parse_args(argv)
    reports = Path(args.reports_dir)
    if args.command == "preflight":
        problems = preflight_problems(
            gather_preflight(reports, args.paper_bot_enabled),
            min_free_bytes=int(args.min_free_gb * 1e9),
        )
        for problem in problems:
            print(f"REFUSED: {problem}", file=sys.stderr)
        if not problems:
            print("preflight OK")
        return 1 if problems else 0
    if args.command == "run":
        return run(args)
    text = render_summary(reports, failures=failed_trials(reports))
    print(text)
    print(f"summary written to {write_summary(reports, text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
