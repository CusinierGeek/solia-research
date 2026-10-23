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

Paper trading only — no real execution, ever.
