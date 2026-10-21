# Sprint 4 — Backend (ASP.NET Core, .NET 10)

**Milestone:** Backend API · **Depends on:** Sprints 1–3 · **Service:** `services/backend`

## Goal
The ASP.NET Core REST API that serves the dashboard, talks to the AI service, hosts the paper-trading bot, and manages AES secrets.

## Scope
- All endpoints in [`../API-CONTRACTS.md`](../API-CONTRACTS.md) (`/api/v1/...`).
- EF Core (or Dapper) data access over the Sprint-1 schema.
- Typed `HttpClient` for `ai-service`.
- Paper-trading bot as a `BackgroundService` (from Sprint 3).
- `ISecretsProvider` with AES-256-GCM per [`../SECRETS.md`](../SECRETS.md) + `solia-secrets` CLI util.
- Structured logging to `logs`; health endpoints; OpenAPI/Swagger.

## Out of scope
- UI (Sprint 5). Container wiring (Sprint 6). Retrain internals (Sprint 7 — expose the proxy endpoints only).

## Task breakdown
- [ ] Solution/project layout (`Solia.Api`, `Solia.Core`, `Solia.Bot`, `Solia.Secrets`, tests).
- [ ] DB context + entities mapped to the schema.
- [ ] Endpoints: `/predict`, `/tokens/live`, `/tokens/history`, `/trades/history`, `/stats/winrate`, `/stats/performance`, `/logs/bot`, `/model/status`, `/model/retrain`.
- [ ] AI service client (typed HttpClient, Polly retries).
- [ ] Host the paper-trading bot (`BackgroundService`).
- [ ] AES secrets provider + encrypt/rotate CLI + log redaction.
- [ ] Error-shape middleware, validation, pagination.
- [ ] Swagger + health checks.
- [ ] Unit tests (services) + integration tests (WebApplicationFactory + Testcontainers Postgres).

## Deliverables
- Running API (`dotnet run`) with Swagger.
- `solia-secrets` CLI.
- Test suite (unit + integration).

## Definition of Done
- [ ] All endpoints return the documented shapes.
- [ ] `/predict` and `/model/*` correctly proxy ai-service.
- [ ] Bot runs as a hosted service and writes `trades`.
- [ ] Secrets decrypt from `secrets.enc` using `SOLIA_MASTER_KEY`; nothing secret is logged.
- [ ] Tests green in CI.

## Kickoff prompt
> We're implementing **Sprint 4 — Backend** of Solia (ASP.NET Core, .NET 10). Read `docs/ARCHITECTURE.md`, `docs/API-CONTRACTS.md`, `docs/DATA-MODEL.md`, `docs/SECRETS.md`, and the Sprint 3 bot. Create the solution (`Solia.Api/Core/Bot/Secrets` + tests), EF Core entities over the existing schema, all `/api/v1` endpoints, a typed HttpClient to the ai-service with retries, the paper-trading bot as a BackgroundService, and an `ISecretsProvider` using AES-256-GCM plus a `solia-secrets` encrypt/rotate CLI with log redaction. Add Swagger, health checks, error middleware, and unit + integration tests (WebApplicationFactory + Testcontainers Postgres). Branch `sprint/04-backend`, PR per `CONTRIBUTING.md`.
