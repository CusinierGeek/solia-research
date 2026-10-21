# Sprint 1 — Data pipeline

> **This file is the sprint brief.** The actual tasks are tracked as GitHub Issues (T1–T7) under **Milestone #1** and on the **Solia** Project board — each issue is the source of truth for its task and becomes one PR. The task checklist below is kept as the sprint overview.

**Milestone:** Data pipeline · **Depends on:** — · **Service:** `services/data-pipeline` (.NET worker), `services/db`

## Goal
Stand up PostgreSQL with the full schema, and a **.NET worker** that ingests pump.io events (live WebSocket **and** historical replay), computes features across the full token lifecycle (pre/post-bonding), and persists them.

## Scope
- PostgreSQL schema + migrations for all tables in [`../DATA-MODEL.md`](../DATA-MODEL.md).
- `IDataSource` abstraction + `PumpApiStreamSource` (live) + `PumpApiReplaySource` (historical `.jsonl.zst`) + `FileReplaySource` (tests).
- Normalization (live + replay through the same path), rolling windows, feature computation ([`../DATA-PIPELINE.md`](../DATA-PIPELINE.md)).
- Idempotent persistence to `tokens_live`, `tokens_history`, `features`.
- Post-bonding tracking (venue + migration markers).
- Health endpoint, structured logging to `logs`.

## Out of scope
- ML (Sprint 2). The autonomous `SolanaRpcSource` fallback (ported from SolGemFinder) — stubbed/interface only; implemented later if needed.
- Full Docker wiring (Sprint 6) — a local Postgres container is enough here.

## Task breakdown
- [ ] .NET worker project skeleton (`Solia.DataPipeline`) + solution wiring; analyzers, `dotnet format`, xUnit test project.
- [ ] Choose migration tool (EF Core migrations or raw SQL via a migrator) and set up `services/db/migrations`.
- [ ] Write DDL for all 8 tables + indexes (incl. `venue` / post-bonding fields).
- [ ] DB access layer (EF Core).
- [ ] `RawEvent` canonical model + `IDataSource` interface.
- [ ] `PumpApiStreamSource`: connect `wss://stream.pumpapi.io/`, subscribe, reconnect w/ backoff, resume.
- [ ] `PumpApiReplaySource`: download/stream `.jsonl.zst` hourly archives, decompress (zstd), emit `RawEvent` (same format as live).
- [ ] `FileReplaySource`: read recorded fixtures for tests.
- [ ] Normalizer: provider payload → `RawEvent` (price from pool reserves, bonding progress, is_buy, venue, migration).
- [ ] Rolling-window manager (bounded memory, eviction of inactive tokens).
- [ ] Feature computation (all features in the pipeline doc); holders via unique-trader approximation (free) with optional paid enrichment flag off by default.
- [ ] Persistence (idempotent upserts, batching).
- [ ] Health endpoint + structured logging.
- [ ] Record a small real event sample (live capture) for `FileReplaySource` fixtures.
- [ ] Script/notes to mirror a chosen replay **window** (start ~2–4 recent weeks) to local storage.

## Deliverables
- `services/db/migrations/*`
- `services/data-pipeline/` runnable .NET worker
- Unit tests (feature math) + integration test (`FileReplaySource` → ephemeral Postgres via Testcontainers)
- Service README with run instructions

## Definition of Done
- [ ] Postgres + migrations apply cleanly.
- [ ] Running against pumpapi (live or replay) produces rows in `features`, including post-bonding events.
- [ ] Reconnect verified (kill/restore WS).
- [ ] Unit + integration tests green in CI.
- [ ] Feature definitions documented and matched to the spec.

## Decisions to confirm at kickoff
- Exact pumpapi subscription/message schema (verify current docs; capture a live sample).
- EF Core migrations vs a standalone SQL migrator.
- The initial replay **window** to mirror for Sprint 2 training.

## Kickoff prompt
> We're implementing **Sprint 1 — Data pipeline** of the Solia project. Read `docs/ARCHITECTURE.md`, `docs/DATA-MODEL.md`, `docs/DATA-SOURCES.md`, and `docs/DATA-PIPELINE.md`. Create the PostgreSQL migrations for all tables (including venue/post-bonding fields), then build the **.NET worker** `Solia.DataPipeline`: an `IDataSource` abstraction with `PumpApiStreamSource` (live WS `wss://stream.pumpapi.io/`), `PumpApiReplaySource` (historical `.jsonl.zst` via zstd), and `FileReplaySource` (fixtures); event normalization (price from pool reserves, bonding progress, venue, migration) shared by live and replay; rolling windows; feature computation (holders approximated from unique traders, paid enrichment off by default); idempotent persistence via EF Core. Add analyzers, `dotnet format`, and xUnit unit tests (feature math) + an integration test using `FileReplaySource` against an ephemeral Postgres (Testcontainers). Keep the autonomous `SolanaRpcSource` (SolGemFinder RPC+IDL) as an interface stub only. Work on a branch `sprint/01-data-pipeline` and open a PR following `CONTRIBUTING.md`. Confirm the exact pumpapi message schema and the replay window with me first.
