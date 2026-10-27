# Sprint 5 — Dashboard (Blazor Server, .NET 10)

**Milestone:** Dashboard · **Depends on:** Sprint 4 · **Service:** `services/dashboard`

## Goal
A live Blazor Server dashboard over the backend API with real-time updates.

## Scope
Views:
- Live predictions (pump prob, rug prob, recommended strategy, confidence).
- Prediction history.
- Trade history.
- Win-rate.
- Total gain/loss (P&L).
- Performance curve (equity chart).
- Bot logs.

Real-time via SignalR. UI components: **MudBlazor** (decided in T3). Charts: MudBlazor's built-in charts
first; if they fall short (e.g. a zoomable/time-scaled equity curve), a small Blazor wrapper over
[TradingView lightweight-charts](https://github.com/tradingview/lightweight-charts) via JS interop (T10).

## Out of scope
- Auth (reserved; add if the repo ever goes multi-user). Container wiring (Sprint 6).

## Task breakdown
- [x] Blazor Server project + layout/nav.
- [x] Typed API client to the backend.
- [x] Live predictions view (auto-refresh / SignalR push).
- [x] Prediction history (paged table + filters).
- [x] Trade history (paged table + filters).
- [x] Stats cards: win-rate, total P&L.
- [x] Performance/equity curve chart.
- [x] Bot logs view (level/service filter, live tail).
- [x] Loading/empty/error states.
- [x] Component tests (bUnit) for key views.

## Deliverables
- Running dashboard (`dotnet run`) bound to the backend.
- bUnit tests for core components.

## Definition of Done
- [x] All listed views render real data from the backend.
- [x] Live updates work (new predictions/trades appear without reload).
- [x] Charts render the equity curve and win-rate.
- [x] Tests green in CI.

Verified by `services/dashboard/Solia.Dashboard.Tests/EndToEnd` (T12): the real dashboard host against a
contract-faithful stub backend (seeded REST + SignalR hub) — every view renders the seeded data, pushed
predictions / trades / logs appear without a reload, the equity curve and win-rate reach the charts, and REST or hub
outages degrade and recover. Runs in memory with the bUnit suite (locally and in CI).

## Kickoff prompt
> We're implementing **Sprint 5 — Dashboard** of Solia (Blazor Server, .NET 10). Read `docs/ARCHITECTURE.md` and `docs/API-CONTRACTS.md`. Build a Blazor Server dashboard with: live predictions, prediction history, trade history, win-rate and total P&L stat cards, an equity/performance curve chart, and a live bot-logs view — all over a typed backend API client, with SignalR for real-time updates. Add loading/empty/error states and bUnit component tests. Branch `sprint/05-dashboard`, PR per `CONTRIBUTING.md`. Ask me which charting library to use before wiring the chart.
