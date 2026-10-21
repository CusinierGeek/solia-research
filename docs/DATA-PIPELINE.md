# Data pipeline

## Purpose

Turn raw Pump.fun activity into clean, time-aligned **feature snapshots** stored in PostgreSQL, ready for training and inference.

## Source: PumpPortal WebSocket

- Endpoint: `PUMPPORTAL_WS_URL` (default `wss://pumpportal.fun/api/data`).
- Subscriptions (confirmed against docs during Sprint 1):
  - new token creations,
  - trades (buys/sells) for tracked tokens,
  - optionally migration/graduation events.
- The client must handle reconnect with backoff, heartbeat/ping, and resubscription on reconnect.

### Swappable `DataSource` abstraction

```
class DataSource(Protocol):
    async def stream(self) -> AsyncIterator[RawEvent]: ...

# adapters:
#   PumpPortalWebSocketSource   (default, Sprint 1)
#   SolanaRpcSource             (future)
#   ReplaySource                (feed recorded events for tests/backtests)
```

Downstream code depends only on `RawEvent` → nothing else knows about PumpPortal.

## Stages

```
RawEvent ─▶ normalize ─▶ per-token rolling window ─▶ feature compute ─▶ persist
                                                                         │
                     tokens_live / tokens_history ◀───────────────────────┘
                                                      features ◀──────────┘
```

1. **Normalize**: map source payload to a canonical event (`mint`, `ts`, `type`, `price`, `amount`, `is_buy`, ...).
2. **Window**: maintain in-memory rolling windows per active token (deque with time/length bounds).
3. **Feature compute**: at a fixed cadence (e.g. every few seconds) or per event, compute the feature vector.
4. **Persist**: upsert `tokens_live`, append `tokens_history` (first seen), insert `features`.

## Feature set

Base: `price, volume, market_cap, holders, liquidity, buy_events, sell_events, token_age`.

Derived:
- `buy_sell_ratio`
- `holders_growth_rate`, `volume_growth_rate` (Δ over window)
- `tx_creation_rate` (tx/sec)
- `volatility` (rolling std of returns)
- `ema`, `rsi`
- `price_derivative` (first time-derivative)
- `trending_score`, `social_mentions` (if available)
- `onchain_signals` (Pump.fun-specific; JSONB)
- `behavioral_signals` (pump/rug pattern flags; JSONB)

Feature definitions must stay **in sync** with `ai-service/features.py` — keep a single documented spec and mirror it (parity test in Sprint 2).

## Reliability

- Reconnect with exponential backoff + jitter.
- Idempotent writes (dedupe by `(mint, ts, type)`).
- Bounded memory: cap tracked tokens; evict stale/inactive (`is_active=false`).
- Backpressure: if DB writes lag, batch and/or drop low-value snapshots (log it).

## Config

- `PUMPPORTAL_WS_URL`, DB connection, window length, compute cadence, max tracked tokens — all via env / a pipeline config file (Sprint 1).

## Testing

- Unit: feature math on synthetic windows.
- Integration: `ReplaySource` feeds a recorded event file → assert rows in an ephemeral Postgres.
