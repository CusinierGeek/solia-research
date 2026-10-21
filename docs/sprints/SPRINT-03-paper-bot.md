# Sprint 3 — Paper-trading bot

**Milestone:** Paper-trading bot · **Depends on:** Sprints 1, 2 · **Service:** bot logic (lands in backend in Sprint 4, prototyped here)

## Goal
A bot that consumes predictions, simulates buy/sell with market orders only, manages open positions (TP/SL/timeout), and records every trade for training.

## Scope
- Decision logic: act on `strategy` + `confidence` + sizing.
- Position manager: open/close, apply dynamic TP/SL, timeout, exit reasons.
- Simulated fills against observed market price (no slippage model v1, or a simple one).
- Persist `trades` (entry/exit/price/duration/pnl/strategy/confidence/features_snapshot).
- Feed closed trades toward the retrain counter.

## Out of scope
- Any real order execution or wallet/keys — **paper only, forever**.
- The retrain trigger itself (Sprint 7) — here we just increment/record.

## Task breakdown
- [ ] Define bot config (starting balance, max concurrent positions, confidence threshold, timeouts).
- [ ] Prediction intake (poll inference / read `predictions`).
- [ ] Entry rule (buy when strategy=buy & confidence ≥ threshold & capacity available).
- [ ] Position sizing from model output + balance.
- [ ] Price feed for marking positions (from `features`/live).
- [ ] Exit rules: TP hit, SL hit, sell signal, timeout → `exit_reason`.
- [ ] Record `trades` rows on open and close.
- [ ] P&L accounting + balance update.
- [ ] Logging to `logs` (service=`bot`).
- [ ] Tests on a deterministic replay of prices.

## Deliverables
- Bot module (language TBD: prototype can be Python; final host is .NET backend in Sprint 4 — decide at kickoff).
- Deterministic backtest/replay harness.
- Unit tests for entry/exit/accounting.

## Definition of Done
- [ ] Given a replay of predictions+prices, the bot opens/closes positions and writes correct `trades`.
- [ ] P&L and win-rate computed correctly on a known fixture.
- [ ] No real execution path exists anywhere.
- [ ] Tests green in CI.

## Decision to confirm at kickoff
- **Bot language**: prototype in Python now, or write it directly as the .NET `BackgroundService` (Sprint 4 home)? Recommendation: write directly in .NET to avoid rewrite, using a replay harness for tests.

## Kickoff prompt
> We're implementing **Sprint 3 — Paper-trading bot** of Solia. Read `docs/ARCHITECTURE.md`, `docs/DATA-MODEL.md`, `docs/ML-MODEL.md`, `docs/API-CONTRACTS.md`. Build the paper-trading bot: prediction intake, entry rule (strategy+confidence+capacity), model-driven sizing, position marking from the price feed, exit rules (dynamic TP/SL, sell signal, timeout) with exit reasons, and `trades` persistence with a features snapshot. **Paper only — no real execution anywhere.** Provide a deterministic replay harness and unit tests for entry/exit/P&L. First ask me whether to write the bot directly as the .NET BackgroundService (recommended) or prototype in Python. Branch `sprint/03-paper-bot`, PR per `CONTRIBUTING.md`.
