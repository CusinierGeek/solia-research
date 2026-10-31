# backend (ASP.NET Core · .NET 10)

REST API for the dashboard and external callers; AI-service client; hosts the paper-trading bot; AES secrets.

- **Implemented in:** [Sprint 4](../../docs/sprints/SPRINT-04-backend.md) (bot logic from [Sprint 3](../../docs/sprints/SPRINT-03-paper-bot.md)).
- **Design:** [API-CONTRACTS.md](../../docs/API-CONTRACTS.md) · [SECRETS.md](../../docs/SECRETS.md)

## Projects

| Project | Role |
|---------|------|
| `Solia.Api` | ASP.NET Core host — composition, logging, error-shape, OpenAPI, health probes, `/api/v1` endpoints, hosted bot. |
| `Solia.Core` | Domain types, options and contracts — no ASP.NET dependency. |
| `Solia.Secrets` | AES-256-GCM crypto core, `ISecretsProvider`, `SecretRedactor` (T7). No dependencies beyond the BCL. |
| `Solia.Secrets.Cli` | `solia-secrets` CLI (`encrypt` / `rotate` / `get` / `keygen`) over the same crypto core. |
| `Solia.Bot` | Paper-trading bot (Sprint 3). |
| `*.Tests` | `Solia.Api.Tests` (integration, `WebApplicationFactory`), `Solia.Secrets.Tests` (crypto + CLI), `Solia.Core.Tests` + `Solia.Bot.Tests` (unit, plus Testcontainers tests tagged `Category=Integration` — skip locally with `--filter "Category!=Integration"`). |

## Host conventions (T1)

- **Base path** `/api/v1` (`ApiRoutes.BasePath`); health probes under `/health` (see *Health & end-to-end suite* below).
- **Error shape** — every failure returns `{ "error": { "code", "message" } }` via `ErrorShapeMiddleware`; throw an `ApiException` (e.g. `NotFoundException`) for a well-known failure.
- **Logging** — console plus a best-effort mirror of notable entries into the `logs` table (`service = backend`), configured under the `Postgres` logger-provider alias (defaults to `Warning`).
- **Configuration** — Options pattern with `ValidateOnStart`; flat `.env` keys (e.g. `BACKEND_URL`) map onto option sections, mirroring the data-pipeline/bot convention.
- **OpenAPI/Swagger UI** — served in Development only (`/openapi/v1.json` + Swagger UI). Every operation has an id, a tag, a summary, its success schema and the error envelope (`ApiErrorResponse`) for each failure status it can return (`ProducesError`); `ApiSurfaceTests` enumerates the routing table so a new endpoint can't go undocumented.

## Hosted paper-trading bot (T6)

- Off by default; `Bot:Enabled` / `PAPER_BOT_ENABLED=true` runs `PaperTradingBot` inside the API host (`GatedPaperTradingBot`).
- **Feed** (`DbBotEventSource`): polls new `features` rows every `Bot:PollInterval` (by id; an insert that commits after a higher id is waited for up to `Bot:MaxCommitDelay`) → price ticks of active `tokens_live` tokens. Each token with a new row is queued (one entry per token, newest row, oldest-waiting first — none is dropped) and a separate loop scores it on its feature window *ending at that row* through the ai-service for `Bot:Horizon` (default `15m`), so a slow ai-service never holds ticks back. The prediction is written to `predictions` and fed to the bot. Events go through a `Bot:ReorderWindow` (default 2 s) buffer so they reach the engine in time order. Rows older than `Bot:MaxMarkAgeSeconds` are skipped. When the ai-service or the DB fails, or scoring fails unexpectedly, the token or poll is skipped; the bot keeps running.
- **Fills** (#165, `Bot:RealisticFills`, default on): each tick carries its `features` row's venue and liquidity; entry needs that pool above its venue's `Bot:LiquidityFloorSol` (AMM 1 SOL, bonding curves 0), size ≤ `Bot:MaxPoolFraction` (2 %) of its SOL reserve (curves: + `Bot:VirtualSol`), both sides pay x·y=k impact and `Bot:FeePct` (per venue). Per-venue tables bind as `Bot__FeePct__pumpswap=0.003`. Details: [docs/ML-MODEL.md "Realistic fills"](../../docs/ML-MODEL.md).
- **Execution realism** (#175): orders fill after `Bot:EntryLatencyMs` / `Bot:ExitLatencyMs` (3 s / 2 s, event time) at the first mark after it, pay `Bot:NetworkFeeSol` per transaction and `Bot:AccountRentSol` per entry, and fail past `Bot:MaxSlippagePct` (15 %; exits retried up to `Bot:MaxExitRetries`). Each at 0 disables it. Details: [docs/ML-MODEL.md "Execution realism"](../../docs/ML-MODEL.md).
- **Persistence**: `trades` via `PostgresTradeStore` (also the closed-trade counter), bot milestones in `logs` with `service = bot`.
- **Sessions** (#169, `BotSessions` / `PostgresBotSessionStore`): every trade records its bot session — one configuration + champion. On start the bot continues the current session when its fingerprint (normalized options + `Bot:Horizon` + champion version) is unchanged, else opens a new one (named by the optional `Bot:SessionLabel`); it re-checks every 30 s before an entry, so a promotion while it runs rolls over too. `/stats/*` and `/trades/history` read the current session by default (`?session=all|<id>`); `GET /bot/sessions` lists them with open / orphaned trades. Table: [docs/DATA-MODEL.md](../../docs/DATA-MODEL.md) `bot_sessions`.

## Secrets (T7)

- At startup the host decrypts `secrets/secrets.enc` (`SOLIA_SECRETS_FILE` to override) with `SOLIA_MASTER_KEY` into memory (`ISecretsProvider`). No file at the default path → plaintext config only; an explicit `SOLIA_SECRETS_FILE` that doesn't exist, a missing/wrong key or a tampered file → startup fails.
- **DB**: a `db_connection` secret replaces the configured connection string; otherwise `db_password` replaces its password. The secrets-aware `NpgsqlDataSource` is shared by EF, the log sink and the bot.
- **Redaction**: the `ILoggerFactory` is wrapped (`AddSecretRedaction`) so every logger scrubs known secret values and the master key (message, structured values, exception text, scopes) before any provider sees them. The bot's `logs` rows bypass `ILogger`, so its sink (`RedactingBotLogSink`) scrubs message and context itself. The password inside a `db_connection` secret is redacted on its own.
- CLI: `dotnet run --project Solia.Secrets.Cli -- <command>` (see [SECRETS.md](../../docs/SECRETS.md)).

## Health & end-to-end suite (T8)

- `GET /health/live` — liveness: runs no check (never depends on the DB or the ai-service).
- `GET /health/ready` — readiness: `database` (a `SELECT 1` through `SoliaDbContext`, on the secrets-aware data source) and `ai_service` (`ISoliaAiClient` → ai-service `/health`), each bounded to 5 s.
- `GET /health` — every check. Unhealthy → `503`, otherwise `200`; both carry the health JSON `{ "status", "total_duration", "entries": { "<check>": { "status", "description", "duration", "tags" } } }`. Descriptions are fixed strings — failure details are logged (redacted), never returned.
- `Solia.Api.Tests/EndToEnd` proves the Sprint-4 DoD on the production host: Testcontainers Postgres (seeded), the bot enabled, the ai-service stubbed, the DB password only in an encrypted `secrets.enc`, and every log entry captured to check nothing secret is logged (`Category=Integration`, runs in CI).

Paper trading only — no real execution, ever.
