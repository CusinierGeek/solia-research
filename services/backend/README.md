# backend (ASP.NET Core · .NET 10)

REST API for the dashboard and external callers; AI-service client; hosts the paper-trading bot; AES secrets.

- **Implemented in:** [Sprint 4](../../docs/sprints/SPRINT-04-backend.md) (bot logic from [Sprint 3](../../docs/sprints/SPRINT-03-paper-bot.md)).
- **Design:** [API-CONTRACTS.md](../../docs/API-CONTRACTS.md) · [SECRETS.md](../../docs/SECRETS.md)

## Projects

| Project | Role |
|---------|------|
| `Solia.Api` | ASP.NET Core host — composition, logging, error-shape, OpenAPI, `/health`. Endpoints land in T4/T5. |
| `Solia.Core` | Domain types, options and contracts — no ASP.NET dependency. |
| `Solia.Secrets` | AES secrets provider + CLI (T7). Scaffolded empty in T1. |
| `Solia.Bot` | Paper-trading bot (Sprint 3). |
| `*.Tests` | `Solia.Api.Tests` (integration, `WebApplicationFactory`), `Solia.Core.Tests` + `Solia.Bot.Tests` (unit, plus Testcontainers tests tagged `Category=Integration` — skip locally with `--filter "Category!=Integration"`). |

## Host conventions (T1)

- **Base path** `/api/v1` (`ApiRoutes.BasePath`); liveness at `/health` (real readiness checks in T8).
- **Error shape** — every failure returns `{ "error": { "code", "message" } }` via `ErrorShapeMiddleware`; throw an `ApiException` (e.g. `NotFoundException`) for a well-known failure.
- **Logging** — console plus a best-effort mirror of notable entries into the `logs` table (`service = backend`), configured under the `Postgres` logger-provider alias (defaults to `Warning`).
- **Configuration** — Options pattern with `ValidateOnStart`; flat `.env` keys (e.g. `BACKEND_URL`) map onto option sections, mirroring the data-pipeline/bot convention.
- **OpenAPI/Swagger UI** — served in Development only (`/openapi/v1.json` + Swagger UI).

## Hosted paper-trading bot (T6)

- Off by default; `Bot:Enabled` / `PAPER_BOT_ENABLED=true` runs `PaperTradingBot` inside the API host (`GatedPaperTradingBot`).
- **Feed** (`DbBotEventSource`): polls new `features` rows every `Bot:PollInterval` (by id; an insert that commits after a higher id is waited for up to `Bot:MaxCommitDelay`) → price ticks of active `tokens_live` tokens. Each token with a new row is queued (one entry per token, newest row, oldest-waiting first — none is dropped) and a separate loop scores it on its feature window *ending at that row* through the ai-service for `Bot:Horizon` (default `1m`), so a slow ai-service never holds ticks back. The prediction is written to `predictions` and fed to the bot. Events go through a `Bot:ReorderWindow` (default 2 s) buffer so they reach the engine in time order. Rows older than `Bot:MaxMarkAgeSeconds` are skipped. When the ai-service or the DB fails, or scoring fails unexpectedly, the token or poll is skipped; the bot keeps running.
- **Persistence**: `trades` via `PostgresTradeStore` (also the closed-trade counter), bot milestones in `logs` with `service = bot`.

Paper trading only — no real execution, ever.
