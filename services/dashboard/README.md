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
| `Backend:AttemptTimeout` | — | Bound on one HTTP attempt to the backend (default `00:00:05`). Between 10 ms and 24 h. |
| `Backend:TotalTimeout` | — | Bound on a whole call, its one retry included (default `00:00:12`). Between 10 ms and 24 h, and ≥ 2 × `AttemptTimeout` + 250 ms (two attempts plus the jittered retry delay). |

The listen address comes from the launch profile (`http://localhost:8081`); outside `dotnet run`, set
`ASPNETCORE_URLS` (e.g. `http://+:8081`).

## Layout & building blocks

- `Components/Layout` — `MainLayout` (app bar, drawer, dark/light toggle, paper-trading badge), `NavMenu`
  rendered from `Navigation/DashboardNav.Items` (the single list of views; a test keeps it in sync with the
  pages' `@page` routes), `PageHeader`.
- `Api` — `ISoliaApiClient`, the typed client of the backend's read endpoints (`{BaseUrl}/api/v1/…`;
  no write/control endpoints). Dashboard-owned DTOs pinned to `docs/API-CONTRACTS.md` by fixture tests,
  typed query objects (null/blank omitted), one retry on transient GET failures. Every failure is a
  `BackendException` whose `Message` is safe to display: `BackendErrorException` (envelope `Code` + message +
  `StatusCode`) or `BackendUnavailableException` (network failure, timeout, gateway 502/503/504).
- `Live` — `ILiveFeed`, the shared subscription to the backend's SignalR hub (`{BaseUrl}/hubs/live`):
  typed events (`LiveFeedEvents.PredictionCreated`, `TradeOpened`, `TradeClosed`, `LogWritten`, `StatsChanged`)
  carrying the same DTOs as REST. **One connection per app, not per circuit** — the hub broadcasts the same
  events to everyone, so per-tab connections would only multiply sockets against the backend. Started with the
  host without blocking it; a failed start and a dropped connection are retried with backoff (0 s, 2 s, 5 s,
  10 s, then every 30 s, forever); a connection the server keeps closing backs off the same way (the backoff
  only resets after 30 s connected). `State` is *connected / reconnecting / disconnected*, shown in the app bar by
  `LiveFeedIndicator` (LIVE / RECONNECTING / OFFLINE). Delivery is at most once, so every (re)connect raises
  `SubscribeResynced` — re-fetch over REST then. Handlers run off the UI thread and are started in order but not
  awaited (a slow circuit never stalls the others); a state handler that renders later should read `State` then.
  Subscribe in `OnInitialized`, marshal with `InvokeAsync`, and dispose the subscription with the component.
  Host tests swap the `ILiveConnection` service for a fake so no test opens a socket:

```razor
@implements IDisposable
@inject ILiveFeed LiveFeed
@code {
    private readonly List<IDisposable> _subscriptions = [];

    protected override void OnInitialized()
    {
        _subscriptions.Add(LiveFeed.Subscribe(LiveFeedEvents.TradeOpened, t => InvokeAsync(() => { /* apply t */ StateHasChanged(); })));
        _subscriptions.Add(LiveFeed.SubscribeResynced(() => InvokeAsync(ReloadAsync)));
    }

    public void Dispose() => _subscriptions.ForEach(s => s.Dispose());
}
```
- `Components/State` — `LoadingState` (skeleton / spinner), `EmptyState`, `ErrorState` (message + retry)
  and `AsyncContent<TItem>`, the *load → loading / empty / error / content* pattern views adopt:

```razor
<AsyncContent TItem="IReadOnlyList<LiveTokenDto>" Load="ct => Api.GetLiveTokensAsync(cancellationToken: ct)" IsEmpty="t => t.Count == 0"
              EmptyMessage="No tracked tokens yet.">
    @foreach (var token in context) { ... }
</AsyncContent>
```

## Test

```bash
dotnet test services/dashboard/Solia.Dashboard.Tests
dotnet format --verify-no-changes
```
