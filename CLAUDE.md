# CLAUDE.md — Solia operating manual

Read this first. It encodes how we work on Solia so it doesn't have to be re-explained each session.

## What Solia is

Predictive **paper-trading** research system for Pump.fun tokens: an ML model (Transformer) estimates pump/rug probability + a recommended strategy; a bot paper-trades on it; a continuous-training loop retrains; a Blazor dashboard shows it live.

> ⚠️ **Paper trading only.** No real orders, no wallet signing, no on-chain execution — ever. Never add a real-execution path.

## Workflow model (important)

```
Sprint  = GitHub Milestone  = a set of Tasks
Task    = GitHub Issue      = exactly ONE pull request
```

- **One task per conversation.** Pick an issue, do it, open its PR, stop.
- **Tasks are specified in their GitHub Issue** (the issue body is the source of truth), using the template below. The `docs/sprints/SPRINT-0N-*.md` files are only the sprint *brief* (goal + context); the per-task detail lives in issues.
- **No code is pre-written in a task.** The issue states the *what/why* (business need, acceptance criteria, technical details) — the *how* is decided when the task is implemented.

### Task issue template (`.github/ISSUE_TEMPLATE/task.md`)
Every task issue has: **Business need** · **Acceptance criteria** · **Technical details** · **Dependencies** · **Out of scope**.

### Task lifecycle
1. Pick the lowest-numbered unblocked issue in the current milestone.
2. Branch off `main`: `sprint/<n>-<task-slug>` or `feat/…`, `fix/…`, `chore/…`, `docs/…`.
3. Implement **+ tests** (see *Tests are mandatory* below); meet the issue's acceptance criteria.
4. Open a PR into `main` using the PR template; `Closes #<issue>`; set the milestone.
5. CI green + (convention) one review → **squash & merge**; delete the branch.

### Working lean (token budget)
Default to the cheapest path that still ships quality:
- **One task = one fresh conversation** (or `/clear` between tasks). Don't chain many tasks in one long chat — the whole history is re-processed every turn.
- **Skip plan mode for well-scoped tasks** (issue is clear). Use it only for ambiguous/architectural work, or when asked; otherwise go straight to implement → PR.
- **Local = fast unit tests; let CI run the heavy/integration tests** (e.g. Testcontainers). Don't re-run the full suite locally every task.
- **Trim tool output** (`-q`, `| tail`/`grep`) and keep summaries terse.
- Don't re-read files already in context; don't spawn sub-agents (cold start = expensive).

### Hard conventions
- **Conventional Commits**: `type(scope): summary` (`feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `ci`).
- **Never** add AI/Claude attribution to commits or PRs (no `Co-authored-by`, no "Generated with…", no 🤖). Authorship is 100% the user's.
- **No direct pushes to `main`** — always branch + PR. (Branch protection isn't server-enforced on this free private repo; follow it by discipline.)
- **Never commit secrets**; never log decrypted secrets.
- **Tests are mandatory.** A task is not finished — and its PR is not opened — until it ships
  tests for the logic it adds or changes, and `dotnet test` / `pytest` pass locally.
  - Cover the behaviour the acceptance criteria describe, plus the error paths (invalid
    config, bad input, unreachable dependency).
  - Never defer a task's tests to a later task. A test project that doesn't exist yet is
    created in the same PR.
  - Only purely non-code changes (docs, templates) are exempt; say so in the PR's
    *How to test* section.

## GitHub structure

- **Project (v2) "Solia"** — board/table for all tasks.
- **Milestones #1–#8** — the 8 sprints (see `docs/ROADMAP.md`).
- **Labels** — `sprint-1`…`sprint-8`, `type:task|feat|bug|docs`.
- Owner/repo: `CusinierGeek/solia` (private).

## Architecture (summary)

```
pumpapi.io WS ─▶ data-pipeline (.NET) ─▶ PostgreSQL ─▶ ai-service (FastAPI+PyTorch)
                                             │                 │
                                   backend (ASP.NET Core) ◀────┘
                                   ├─ paper-trading bot
                                   └─ dashboard (Blazor Server)
```

- **Full detail:** `docs/ARCHITECTURE.md` · data model `docs/DATA-MODEL.md` · model `docs/ML-MODEL.md` · API `docs/API-CONTRACTS.md` · secrets `docs/SECRETS.md`.
- **Data sources & cost strategy:** `docs/DATA-SOURCES.md`. Primary = **pumpapi.io** (free stream + free historical replay, pre/post-bonding). Fallbacks behind a swappable `DataSource`: PumpPortal (trade stream metered), Solana RPC+IDL (from SolGemFinder).
- **Golden rule — "derive, don't fetch":** compute price/MC/volume/bonding from free stream events; only pay for rare immutable enrichment (cached). Never fetch per-event.

## Services & stack

| Path | Stack | Role |
|------|-------|------|
| `services/data-pipeline` | **.NET 10 worker** | pumpapi ingestion (live + replay), features → DB |
| `services/ai-service` | **Python 3.11** (PyTorch, FastAPI) | model: train/infer/retrain |
| `services/backend` | **.NET 10** (ASP.NET Core) | REST API, AI client, bot host, AES secrets |
| `services/dashboard` | **.NET 10** (Blazor Server) | live UI |
| `services/db` | **PostgreSQL 16** | schema + migrations (raw SQL = source of truth) |
| `infra/` | Docker / Compose | containers, volumes, network |

Installed locally: .NET 10, Python 3.11, PostgreSQL 16 (Docker), git, gh (scopes incl. `project`+`workflow`), Docker daemon up.

## Common commands

```bash
docker compose up -d postgres          # local Postgres
# .NET:   dotnet build ; dotnet test ; dotnet format --verify-no-changes
# Python: ruff check . ; black --check . ; pytest
```

## SolGemFinder (old project) — reuse policy

The sibling `SolGemFinder` repository (.NET Aspire, untested, mid-refactor): **harvest, don't adopt.** The only valuable piece is its hand-written Pump.fun Anchor IDL decoder (`Shared/Services/Solana/PumpFunIdl/` + `PumpFunIDL.json`) — reuse it **only** for the autonomous `SolanaRpcSource` fallback. **Do not** reuse its `TradingWorker` (real execution).
