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

## Run locally (Postgres + backend + dashboard)

From the repo root (container wiring for the dashboard itself comes in Sprint 6):

```bash
# 1. Postgres 16 + schema (see services/db/README.md)
docker compose up -d postgres
POSTGRES_HOST=localhost dotnet run --project services/db/Solia.DbMigrator

# 2. backend on http://localhost:8080 (see services/backend/README.md; POSTGRES_* / AI_SERVICE_URL from .env,
#    PAPER_BOT_ENABLED=true to have the bot trade — without the ai-service the views simply stay empty)
POSTGRES_HOST=localhost dotnet run --project services/backend/Solia.Api

# 3. dashboard on http://localhost:8081 (launch profile "http"; Development points it at http://localhost:8080)
dotnet run --project services/dashboard/Solia.Dashboard
```

Open <http://localhost:8081>. Liveness probe: `GET /health/live` (checks nothing — the dashboard stays
up while the backend is down; views show their error state with a retry instead).

## Loading, empty, error and offline states

Every view follows the same rules (audited in Sprint 5 T12):

| Situation | What the view shows |
|-----------|---------------------|
| First load in flight | `LoadingState` — a skeleton for tables, a spinner in stat cards. Nothing is fetched while prerendering. |
| Loaded, nothing to show | `EmptyState` with a view-specific message (*No tracked tokens yet.*, *No trades match these filters.*…). |
| Backend unreachable / failing on the first load | `ErrorState` — a safe message (never the exception text) and a **Retry**. Each stat card fails on its own. |
| A refresh of the same data fails (live update, resync, reconcile tick, retry) | The last content **stays**, under a *stale* warning with a **Retry** (`.solia-stale`); it clears on the next success. |
| A load of different data fails (new filter, page, window) | `ErrorState`: the old content would not answer the new question. |
| A request the backend rejects (400, e.g. a page past the depth cap) | An inline warning without retry, with a way back to the first page. |
| Hub disconnected | The app-bar chip turns RECONNECTING / OFFLINE; views keep their data and REST keeps working (stat cards and live predictions still reconcile every 60 s). Live pushes pause. |
| Hub reconnected | Chip back to LIVE; every live view re-fetches its snapshot (resync), so events missed while offline show up. |

## Configuration

| Key | Flat env alias | Notes |
|-----|----------------|-------|
| `Backend:BaseUrl` | `BACKEND_URL` | Absolute http(s) URL of the backend. **Required** — a missing or invalid value fails startup. Development defaults to `http://localhost:8080` (`appsettings.Development.json`); compose uses `http://backend:8080`. The flat key wins. |
| `Backend:AttemptTimeout` | — | Bound on one HTTP attempt to the backend (default `00:00:05`). Between 10 ms and 24 h. |
| `Backend:TotalTimeout` | — | Bound on a whole call, its one retry included (default `00:00:12`). Between 10 ms and 24 h, and ≥ 2 × `AttemptTimeout` + 250 ms (two attempts plus the jittered retry delay). |

| `ASPNETCORE_URLS` | — | Listen address. The launch profile sets `http://localhost:8081`; outside `dotnet run` (containers) set it, e.g. `http://+:8081`. |
| `ASPNETCORE_ENVIRONMENT` | — | `Development` enables the detailed error page and the local `Backend:BaseUrl` default; anything else needs `BACKEND_URL`. |
| `Logging:LogLevel:*` | `Logging__LogLevel__Default`… | Standard ASP.NET Core logging (console). Failed loads are logged as warnings with the exception; the UI never shows it. |

There is no auth (single-user, internal network) and nothing secret in the dashboard's configuration.

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
- `Components/Charts` + `Stats` — the performance charts. **Library: MudBlazor charts** (chosen for Sprint 5
  T10 over Blazor-ApexCharts, Radzen and Chart.js): it is already the UI library (no new package, one theme),
  MIT, and renders SVG on the server with no JS interop of its own. The trade-off: no dual axis, no mixed
  line/bar series, and value ticks spaced only by whole units. The library lives behind two thin wrappers,
  `EquityChart` (equity line + starting-balance reference, then a per-trade P&L panel on the same time axis)
  and `WinRateDonut`. They draw only what the pure mappers hand them: `EquityCurve.From(PerformanceResponse)`
  (points ordered by `ts`, exact decimals for display, the starting balance derived as latest equity −
  `total_pnl`, time-label spacing, tooltips) and `WinRateSlices.From(WinRateResponse)`. `ChartAxis` scales a
  series by a power of ten, so a fractional tick step (a P&L of a few hundredths) becomes a whole one, and
  labels the ticks back. In MudBlazor 9 the tooltip template gets the hovered *series* index rather than the
  point's, so the wrapper writes the point's time round-trip (`o`) and `EquityCurve.Tooltip` finds the trade
  from it. Swapping the library means rewriting the two wrappers only. Page tests stub them
  (`ComponentFactories.AddStub<EquityChart>()`) and assert the mapped parameters, not library markup; under
  bUnit the library doesn't plot series anyway (it waits for a browser layout pass).

## Test

```bash
dotnet test services/dashboard/Solia.Dashboard.Tests                                    # everything (CI)
dotnet test services/dashboard/Solia.Dashboard.Tests --filter "Category!=Integration"   # fast local loop
dotnet format --verify-no-changes
```

- **Component tests (bUnit)** — every page (`Pages/*Tests`) against `FakeSoliaApiClient` + `FakeLiveFeed`, plus the
  building blocks (`AsyncContent`, state components, charts, layout, live-feed indicator) and the pure mappers.
- **Host tests** — the real host via `WebApplicationFactory` with a fake hub connection: startup validation, health,
  prerendered shell, 404 page.
- **End to end** (`EndToEnd/`, in memory — no socket, no Docker — so part of the fast local loop) — the Sprint-5 DoD: the real
  host (DI, typed REST client with its resilience pipeline, SignalR live feed) bound to `StubBackend`, an in-memory
  contract-faithful backend (seeded snake_case JSON per docs/API-CONTRACTS.md, the documented error envelope, a
  SignalR hub at `/hubs/live`). Each view renders the seeded data; pushed predictions, trades and logs appear
  without a reload; REST down shows error states that recover on retry; a hub outage flips the indicator, keeps the
  views, and the reconnect's resync re-fetches what was missed. Pages render under bUnit on the host's own services,
  because the views only fetch once interactive (an HTTP GET sees their loading skeleton).
