"""Checkpoint retention: delete old checkpoint files from ``MODEL_DIR`` without breaking serving or
rollback (docs/ML-MODEL.md "Checkpoint retention").

Every retrain writes a checkpoint; :func:`plan_prune` decides which ones go and
:func:`apply_prune` deletes them. Kept, whatever their age:

- the **champion** (also enforced by the database: a pruned model can't be champion, migration
  0011);
- its **rollback chain**: the ``rollback_depth`` previous champions a rollback walks back to
  (:func:`solia_ai.registry.previous_champion`, the same walk :func:`solia_ai.retrain.rollback`
  does);
- the ``keep_candidates`` newest other models whose checkpoint is still there (not pruned, not
  already kept as champion or rollback target).

Every other model's file is deleted but its ``models`` row is kept (version, metrics, history)
with ``pruned_at`` set, so a rollback to it is refused cleanly
(:func:`solia_ai.promotion.check_rollback`). Safety rules:

- files are found exactly as serving finds them (:func:`solia_ai.checkpoints.find_checkpoint`),
  with ``--model-dir``, else ``$MODEL_DIR``, else nothing as the file-name fallback;
- a model whose file is not found is reported as missing and left unmarked (a rollback refuses it
  as missing anyway); when no registered checkpoint at all is found in the directory, the prune is
  refused (:class:`PruneError`): it is almost certainly pointed at the wrong directory;
- only files resolving inside ``MODEL_DIR`` are ever deleted (symlinks and ``..`` resolved first);
  a row pointing elsewhere is refused and left as is;
- a file that is (or may be, through the file-name fallback) the checkpoint of a kept model is
  never deleted;
- the rows are locked (``FOR UPDATE``) and re-checked before their files are deleted, so a
  promotion running outside the retrain lock either wins (the model is then skipped) or waits and
  is refused by the database;
- orphan ``*.pt`` files (no ``models`` row) are only reported, deleted with ``delete_orphans`` and
  only once older than ``orphan_min_age`` seconds: a training run outside the retrain lock
  (``train``, ``candidate``, ``search``) writes its checkpoint before inserting its row.

Runs under the retrain lock (:func:`solia_ai.retrain.prune`), except a dry run, which only reads.
Kept torch-free. DB writes are raw SQL.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from . import checkpoints
from .checkpoints import CHECKPOINT_GLOB, fallback_path, find_checkpoint
from .config import RETENTION_DEFAULTS, validate_retention
from .registry import ModelRecord, previous_champion, record_columns, row_to_record

logger = logging.getLogger(__name__)

# An orphan younger than this may be a checkpoint whose models row is about to be inserted.
DEFAULT_ORPHAN_MIN_AGE_SECONDS = 3600.0
# plan_prune's default file-name fallback: the prune directory itself.
SAME_DIRECTORY = object()


class PruneError(RuntimeError):
    """The prune is refused before touching anything (e.g. the wrong checkpoint directory)."""


@dataclass(frozen=True)
class RetentionSettings:
    """The ``config.yaml`` ``retention`` knobs (see the file for each one's meaning)."""

    keep_candidates: int
    rollback_depth: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RetentionSettings:
        """Validated settings (:class:`solia_ai.config.ConfigError` on bad values). Only the
        ``retention`` section is checked: a broken unrelated section doesn't stop retention."""
        validate_retention(config)
        return cls(**{**RETENTION_DEFAULTS, **config.get("retention", {})})


def model_directory(model_dir: str | os.PathLike[str] | None = None) -> Path:
    """The checkpoint directory, resolved: where :mod:`solia_ai.train` / :mod:`solia_ai.candidate`
    write (:func:`solia_ai.checkpoints.model_directory`)."""
    return checkpoints.model_directory(model_dir).resolve()


def fallback_directory(model_dir: str | os.PathLike[str] | None = None) -> Path | None:
    """The file-name fallback used to find a registered checkpoint: ``model_dir`` (the prune
    directory plays ``$MODEL_DIR``'s role), else ``$MODEL_DIR``, else none, as in serving."""
    if model_dir:
        return Path(model_dir).resolve()
    serving = checkpoints.serving_fallback_dir()
    return serving.resolve() if serving is not None else None


def _inside(path: Path, directory: Path) -> bool:
    return path != directory and path.is_relative_to(directory)


def possible_locations(path: str, directories: tuple[Path | None, ...]) -> set[Path]:
    """Every file the registered ``path`` may be served from: the path itself and the same file
    name in each of ``directories`` (resolved), whether they exist or not."""
    locations = {Path(path).resolve()}
    for directory in directories:
        candidate = fallback_path(path, directory)
        if candidate is not None:
            locations.add(candidate.resolve())
    return locations


def locate(path: str, directory: Path, fallback_dir: Path | None) -> tuple[Path | None, str | None]:
    """The file a registered ``path`` is served from (:func:`find_checkpoint` with
    ``fallback_dir``): ``(file, None)``; ``(None, None)`` when it is missing; ``(None, reason)``
    when it resolves outside ``directory`` (never deleted)."""
    found = find_checkpoint(path, fallback_dir)
    if found is None:
        return None, None
    resolved = found.resolve()
    if not _inside(resolved, directory):
        return None, f"{resolved} is outside MODEL_DIR {directory}"
    return resolved, None


@dataclass(frozen=True)
class Pruned:
    """A model whose checkpoint ``file`` goes."""

    model: ModelRecord
    file: Path


@dataclass(frozen=True)
class PrunePlan:
    directory: Path
    kept: tuple[tuple[ModelRecord, str], ...]  # (model, why it is kept)
    prune: tuple[Pruned, ...]
    refused: tuple[tuple[ModelRecord, str], ...]  # (model, why its file is not deleted)
    orphans: tuple[Path, ...]  # checkpoint files without a models row, old enough to delete
    missing: tuple[ModelRecord, ...] = ()  # not kept, file not found: left unmarked
    recent_orphans: tuple[Path, ...] = ()  # orphans too recent to delete (may be mid-write)
    orphan_min_age: float = DEFAULT_ORPHAN_MIN_AGE_SECONDS


@dataclass(frozen=True)
class PruneOutcome:
    plan: PrunePlan
    dry_run: bool
    deleted: tuple[Path, ...] = ()  # model checkpoints deleted
    marked: tuple[str, ...] = ()  # versions whose row was actually marked pruned
    orphans_deleted: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()


def _models(engine: Engine) -> list[ModelRecord]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(f"SELECT {record_columns()} FROM models ORDER BY created_at DESC, id DESC")
        ).all()
    return [row_to_record(row) for row in rows]


def plan_prune(
    engine: Engine,
    settings: RetentionSettings,
    directory: Path,
    *,
    fallback_dir: Path | None | object = SAME_DIRECTORY,
    orphan_min_age: float = DEFAULT_ORPHAN_MIN_AGE_SECONDS,
    now: Callable[[], float] = time.time,
) -> PrunePlan:
    """What :func:`apply_prune` would do now (reads only). ``fallback_dir`` is the file-name
    fallback (:func:`fallback_directory`; ``None``: none), the prune directory when not given.

    Raises:
        PruneError: registered checkpoints exist but none is found in ``directory``.
    """
    directory = directory.resolve()
    if fallback_dir is SAME_DIRECTORY:
        fallback: Path | None = directory
    else:
        fallback = Path(fallback_dir).resolve() if fallback_dir is not None else None
    models = _models(engine)
    keep: dict[int, str] = {}
    champion = next((m for m in models if m.is_champion), None)
    if champion is not None:
        keep[champion.id] = "champion"
        current = champion
        for depth in range(1, settings.rollback_depth + 1):
            previous = previous_champion(engine, current)
            if previous is None or previous.id in keep:
                break
            keep[previous.id] = f"rollback target {depth}"
            current = previous
    recent = [m for m in models if m.id not in keep and m.pruned_at is None]
    for model in recent[: settings.keep_candidates]:
        keep[model.id] = "recent"

    protected = {
        location
        for m in models
        if m.id in keep
        for location in possible_locations(m.path, (directory, fallback))
    }
    referenced = {
        location for m in models for location in possible_locations(m.path, (directory, fallback))
    }

    kept, prune, refused, missing = [], [], [], []
    found_any = False
    for model in models:
        if model.pruned_at is not None:
            if model.id in keep:
                kept.append((model, f"{keep[model.id]}, checkpoint already pruned"))
            continue
        file, refusal = locate(model.path, directory, fallback)
        found_any = found_any or file is not None
        if model.id in keep:
            kept.append((model, keep[model.id]))
        elif refusal is not None:
            refused.append((model, refusal))
        elif file is None:
            missing.append(model)
        elif file in protected:
            refused.append((model, f"{file} is also the checkpoint of a kept model"))
        else:
            prune.append(Pruned(model, file))
    if not found_any and any(m.pruned_at is None for m in models):
        raise PruneError(
            f"no registered checkpoint found in {directory}: wrong --model-dir or MODEL_DIR? "
            "Nothing was pruned"
        )

    cutoff = now() - orphan_min_age
    orphans, recent_orphans = [], []
    for file in sorted(directory.glob(CHECKPOINT_GLOB)):
        if not file.is_file():
            continue
        resolved = file.resolve()
        if not _inside(resolved, directory) or resolved in referenced:
            continue
        (orphans if resolved.stat().st_mtime <= cutoff else recent_orphans).append(resolved)
    return PrunePlan(
        directory,
        tuple(kept),
        tuple(prune),
        tuple(refused),
        tuple(orphans),
        missing=tuple(missing),
        recent_orphans=tuple(recent_orphans),
        orphan_min_age=orphan_min_age,
    )


def apply_prune(
    engine: Engine,
    plan: PrunePlan,
    *,
    dry_run: bool = False,
    delete_orphans: bool = False,
    ensure_held: Callable[[], None] = lambda: None,
    now: Callable[[], float] = time.time,
) -> PruneOutcome:
    """Carry out ``plan``: in one transaction, lock the planned rows, skip any that changed since
    the plan (became champion, already pruned), delete their files, then mark them ``pruned_at``.
    A dry run touches nothing. ``ensure_held`` is the retrain lock's check, run before any change.

    A file that can't be deleted is reported in ``errors`` and its row left unmarked. Deleting
    before committing the mark keeps a failure safe either way: a deleted but unmarked checkpoint
    is refused by a rollback as missing.
    """
    if dry_run:
        return PruneOutcome(plan, dry_run=True)
    ensure_held()
    deleted: list[Path] = []
    marked: list[str] = []
    errors: list[str] = []
    if plan.prune:
        with engine.begin() as conn:
            # Locked until commit: a concurrent promotion of one of these rows waits, then fails
            # on models_champion_not_pruned_check instead of serving a deleted checkpoint.
            unchanged = set(
                conn.execute(
                    text(
                        "SELECT id FROM models "
                        "WHERE id IN :ids AND NOT is_champion AND pruned_at IS NULL FOR UPDATE"
                    ).bindparams(bindparam("ids", expanding=True)),
                    {"ids": [item.model.id for item in plan.prune]},
                ).scalars()
            )
            to_mark = []
            for item in plan.prune:
                version = item.model.version
                if item.model.id not in unchanged:
                    errors.append(f"{version} changed since the plan (champion or pruned): skipped")
                    continue
                try:
                    item.file.unlink()
                    deleted.append(item.file)
                except FileNotFoundError:
                    pass  # gone meanwhile: nothing left to delete
                except OSError as exc:
                    errors.append(f"could not delete {item.file} ({version}): {exc}")
                    continue
                to_mark.append(item.model.id)
            if to_mark:
                ensure_held()
                marked = list(
                    conn.execute(
                        text(
                            "UPDATE models SET pruned_at = now() "
                            "WHERE id IN :ids AND NOT is_champion AND pruned_at IS NULL "
                            "RETURNING version"
                        ).bindparams(bindparam("ids", expanding=True)),
                        {"ids": to_mark},
                    ).scalars()
                )
    orphans_deleted = []
    if delete_orphans:
        cutoff = now() - plan.orphan_min_age
        for orphan in plan.orphans:
            try:
                if orphan.stat().st_mtime > cutoff:
                    continue  # rewritten since the plan: may be in use again
                orphan.unlink()
                orphans_deleted.append(orphan)
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"could not delete orphan {orphan}: {exc}")
    return PruneOutcome(
        plan,
        dry_run=False,
        deleted=tuple(deleted),
        marked=tuple(marked),
        orphans_deleted=tuple(orphans_deleted),
        errors=tuple(errors),
    )


def describe(outcome: PruneOutcome) -> list[str]:
    """Human-readable report lines (the CLI's output)."""
    plan = outcome.plan
    marked = set(outcome.marked)
    lines = [f"checkpoint directory {plan.directory}"]
    lines += [f"keep    {m.version} ({why})" for m, why in plan.kept]
    for item in plan.prune:
        if outcome.dry_run:
            state = "prune "
        else:
            state = "pruned" if item.model.version in marked else "failed"
        lines.append(f"{state}  {item.model.version}: {item.file}")
    lines += [f"missing {m.version}: checkpoint not found, row left as is" for m in plan.missing]
    lines += [f"refused {m.version}: {why}" for m, why in plan.refused]
    deleted_orphans = set(outcome.orphans_deleted)
    for orphan in plan.orphans:
        state = "deleted" if orphan in deleted_orphans else "kept (pass --delete-orphans)"
        lines.append(f"orphan  {orphan}: no models row, {state}")
    minutes = round(plan.orphan_min_age / 60)
    for orphan in plan.recent_orphans:
        lines.append(f"orphan  {orphan}: no models row, written < {minutes} min ago, kept")
    lines += [f"error   {e}" for e in outcome.errors]
    orphan_count = len(plan.orphans) + len(plan.recent_orphans)
    if outcome.dry_run:
        summary = f"would delete {len(plan.prune)} checkpoint(s), {orphan_count} orphan(s) found"
        summary += " — dry run: nothing changed"
    else:
        summary = f"deleted {len(outcome.deleted)} checkpoint(s), {orphan_count} orphan(s) found"
    lines.append(summary)
    return lines
