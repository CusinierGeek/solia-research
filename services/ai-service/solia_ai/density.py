"""Feature-row density check: were two writers' series interleaved over a period?

``features`` is keyed by ``(mint, ts)``, but a replay and live ingestion tick at offset times, so
that key never collides: a replay over hours live already wrote adds a second series for each token
instead of being skipped. A coverage check can't see it (every hour has rows), and it changes what a
model trains on (a fixed-length sequence then spans half the time). The check counts each token's
rows per snapshot-sized bucket per time slot: ~1 for a single series, ~2 for two.

    python -m solia_ai.density --from 2026-11-01T00:00Z --to 2026-11-02T00:00Z

Exit code: 0 = no slot above ``--max-density``; 1 = at least one; 2 = invalid arguments, no rows in
the window or an unreachable database. Read-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from solia_ai.config import ConfigError
from solia_ai.db import make_engine

SNAPSHOT_SECONDS = 10  # the pipeline's per-token snapshot cadence
DEFAULT_SLOT_MINUTES = 5
# A slot holding the seam between two writers measured up to 1.04 after the 2026-11-04 cleanup;
# a doubled one about 2. A partial overlap lands in between, proportionally to the share it covers.
DEFAULT_MAX_DENSITY = 1.1
# One query per hour: a period's bucket count stays out of a single aggregate's memory.
QUERY_STEP = timedelta(hours=1)

_DENSITY_SQL = text(
    "SELECT slot, sum(n) AS n_rows, count(*) AS n_buckets FROM ("
    "  SELECT floor(extract(epoch FROM ts) / :slot_seconds) * :slot_seconds AS slot, mint,"
    "         floor(extract(epoch FROM ts) / :bucket_seconds) AS bucket, count(*) AS n"
    "  FROM features WHERE ts >= :start AND ts < :end GROUP BY 1, 2, 3"
    ") per_bucket GROUP BY slot ORDER BY slot"
)


@dataclass(frozen=True)
class SlotDensity:
    start: datetime
    n_rows: int
    n_buckets: int

    @property
    def density(self) -> float:
        return self.n_rows / self.n_buckets


def read_density(
    engine: Engine,
    start: datetime,
    end: datetime,
    *,
    slot_minutes: int = DEFAULT_SLOT_MINUTES,
    bucket_seconds: int = SNAPSHOT_SECONDS,
) -> list[SlotDensity]:
    """Each slot's rows and distinct (token, bucket) pairs in ``[start, end)``, in time order; a
    slot without rows is absent."""
    totals: dict[float, tuple[int, int]] = {}
    with engine.connect() as conn:
        step_start = start
        while step_start < end:
            step_end = min(step_start + QUERY_STEP, end)
            params = {
                "slot_seconds": slot_minutes * 60,
                "bucket_seconds": bucket_seconds,
                "start": step_start,
                "end": step_end,
            }
            for slot, n_rows, n_buckets in conn.execute(_DENSITY_SQL, params):
                seen_rows, seen_buckets = totals.get(float(slot), (0, 0))
                totals[float(slot)] = (seen_rows + int(n_rows), seen_buckets + int(n_buckets))
            step_start = step_end
    return [
        SlotDensity(datetime.fromtimestamp(slot, UTC), n_rows, n_buckets)
        for slot, (n_rows, n_buckets) in sorted(totals.items())
    ]


def over_limit(slots: Sequence[SlotDensity], max_density: float) -> list[SlotDensity]:
    """The slots whose density exceeds ``max_density``."""
    return [slot for slot in slots if slot.density > max_density]


def _timestamp(value: str) -> datetime:
    """An ISO-8601 time; without an offset it is UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flag time slots whose features rows hold more than one series per token."
    )
    parser.add_argument("--from", dest="start", type=_timestamp, required=True)
    parser.add_argument("--to", dest="end", type=_timestamp, required=True, help="exclusive")
    parser.add_argument("--slot-minutes", type=_positive_int, default=DEFAULT_SLOT_MINUTES)
    parser.add_argument("--max-density", type=float, default=DEFAULT_MAX_DENSITY)
    args = parser.parse_args(argv)
    if args.start >= args.end:
        print("--from must be before --to")
        return 2

    try:
        slots = read_density(make_engine(), args.start, args.end, slot_minutes=args.slot_minutes)
    except (ConfigError, SQLAlchemyError) as exc:
        print(f"density check failed: {exc}")
        return 2
    if not slots:
        print(f"no features rows in {args.start.isoformat()} .. {args.end.isoformat()}")
        return 2

    flagged = over_limit(slots, args.max_density)
    for slot in flagged:
        print(f"{slot.start:%Y-%m-%d %H:%M}Z  density {slot.density:.3f}  rows {slot.n_rows}")
    highest = max(slot.density for slot in slots)
    print(
        f"{len(slots)} slot(s) with rows, {len(flagged)} above {args.max_density}; "
        f"highest density {highest:.3f}"
    )
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
