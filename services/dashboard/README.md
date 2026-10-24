# dashboard (Blazor Server · .NET 10)

Live UI: predictions, prediction/trade history, win-rate, total P&L, performance curve, bot logs.
**Paper trading only** — the app bar always shows a *PAPER TRADING* badge; nothing here touches real funds.

- **Implemented in:** [Sprint 5](../../docs/sprints/SPRINT-05-dashboard.md)
- **Design:** [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) · [API-CONTRACTS.md](../../docs/API-CONTRACTS.md)

## Projects

| Path | Role |
|------|------|
| `Solia.Dashboard` | Blazor Server host (interactive server render mode), [MudBlazor](https://mudblazor.com) components |
| `Solia.Dashboard.Tests` | xUnit + [bUnit](https://bunit.dev) component tests, plus host smoke tests |

## Run locally against a local backend

```bash
# 1. backend on http://localhost:8080 (see services/backend/README.md)
# 2. dashboard on http://localhost:8081 (launch profile "http")
dotnet run --project services/dashboard/Solia.Dashboard
```

Open <http://localhost:8081>. Liveness probe: `GET /health/live` (checks nothing — the dashboard stays
up while the backend is down; views show their error state with a retry instead).

## Configuration

| Key | Flat env alias | Notes |
|-----|----------------|-------|
| `Backend:BaseUrl` | `BACKEND_URL` | Absolute http(s) URL of the backend. **Required** — a missing or invalid value fails startup. Development defaults to `http://localhost:8080` (`appsettings.Development.json`); compose uses `http://backend:8080`. The flat key wins. |

The listen address comes from the launch profile (`http://localhost:8081`); outside `dotnet run`, set
`ASPNETCORE_URLS` (e.g. `http://+:8081`).

## Layout & building blocks

- `Components/Layout` — `MainLayout` (app bar, drawer, dark/light toggle, paper-trading badge), `NavMenu`
  rendered from `Navigation/DashboardNav.Items` (the single list of views; a test keeps it in sync with the
  pages' `@page` routes), `PageHeader`.
- `Components/State` — `LoadingState` (skeleton / spinner), `EmptyState`, `ErrorState` (message + retry)
  and `AsyncContent<TItem>`, the *load → loading / empty / error / content* pattern views adopt:

```razor
<AsyncContent TItem="IReadOnlyList<TradeDto>" Load="Api.GetTradesAsync" IsEmpty="t => t.Count == 0"
              EmptyMessage="No trades yet.">
    @foreach (var trade in context) { ... }
</AsyncContent>
```

## Test

```bash
dotnet test services/dashboard/Solia.Dashboard.Tests
dotnet format --verify-no-changes
```
