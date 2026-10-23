# Data pipeline

## Purpose

Turn raw Pump.fun activity (live **and** historical) into clean, time-aligned **feature snapshots** stored in PostgreSQL, ready for training and inference — covering the **full token lifecycle, pre- and post-bonding**.

## Implementation: .NET worker

The pipeline is a **.NET worker service** (`services/data-pipeline`), not Python. Rationale:
- the on-chain decoding know-how already exists in C# (SolGemFinder — the fallback `SolanaRpcSource`);
- unified stack with the .NET backend (shared models, EF Core, tooling);
- Python is reserved exclusively for the ML service.

## Source: pumpapi.io (primary)

Provider selection and cost model are documented in [`DATA-SOURCES.md`](DATA-SOURCES.md). Summary:
- **Live:** free WebSocket `wss://stream.pumpapi.io/` — token creations, buys, sells, migrations, pool events, across Pump.fun + AMMs (PumpSwap/Raydium/Meteora/Orca). Pre- and post-bonding in one stream.
- **Historical:** free hourly replay archives (`.jsonl.zst`) since 2026-04-18, **same event format** as live.
- **Cost:** streaming + replay are free; only action/enrichment REST calls cost 10 000 lamports; Solia derives from the stream and avoids these (see *derive, don't fetch*).

### Swappable `DataSource` abstraction

```csharp
public interface IDataSource
{
    IAsyncEnumerable<RawEvent> StreamAsync(CancellationToken ct);
}

// adapters (see DATA-SOURCES.md):
//   PumpApiStreamSource   — live WS                       [primary]
//   PumpApiReplaySource   — historical .jsonl.zst         [training backfill]
//   PumpPortalSource      — alternative live              [optional]
//   SolanaRpcSource       — logsSubscribe + Pump.fun IDL  [fallback, from SolGemFinder]
//   FileReplaySource      — recorded fixtures             [tests]
```

Downstream code depends only on `RawEvent` → nothing else knows about the provider.

## Stages

```
RawEvent ─▶ normalize ─▶ per-token rolling window ─▶ feature compute ─▶ persist
                                                                         │
                     tokens_live / tokens_history ◀───────────────────────┘
                                                      features ◀──────────┘
```

1. **Normalize**: map provider payload to a canonical event (`mint`, `ts`, `type`, `venue`, `price`, `amount`, `is_buy`, pool reserves, `bonding_progress`, `is_migrated`, ...). Live and replay normalize through the **same** path.
2. **Window**: maintain in-memory rolling windows per active token.
3. **Feature compute**: at a fixed cadence (or per event), compute the feature vector.
4. **Persist**: upsert `tokens_live`, append `tokens_history` (first seen), insert `features`.

## Feature set

Base: `price, volume, market_cap, holders, liquidity, buy_events, sell_events, token_age`.

Derived: `buy_sell_ratio, holders_growth_rate, volume_growth_rate, tx_creation_rate, volatility, ema, rsi, price_derivative, trending_score, social_mentions`, plus `onchain_signals` and `behavioral_signals` (JSONB).

**Derived directly from stream events (no paid fetch):** price (pool reserves ratio), market cap, volume, buy/sell ratio, bonding progress.

**Holders** is the one genuinely hard metric from the stream alone:
- *Free (default):* approximate by counting **unique traders** seen in the stream per token.
- *Paid (optional):* periodic `Get Token Info` (10 000 lamports), cached. Off by default.

Feature definitions must stay **in sync** with `ai-service/features.py` — single documented spec, mirrored, with a parity test (Sprint 2).

### Granularity: aggregated snapshots, not raw trades

The pipeline persists **one aggregated `FeatureSnapshot` per token per cadence** — it does **not** store individual trade events. `TokenState.Apply` streams every `RawEvent` into an in-memory rolling window (carrying `Ts`, `Kind`, `QuoteAmount` in SOL, `QuoteMint`, and the `Trader`/txSigner), and `Snapshot` reduces that window to the feature row. The `features.raw` JSONB column exists but is currently **unused**.

Consequence: **per-trade granularity is not recoverable from the database** — a feature that needs it (net SOL flow, holder concentration from `txSigner`, large-buy distribution) cannot be backfilled from stored rows.

This is **not** a problem, because the real source of truth is **re-playable**:

- The raw material already flows through `RawEvent` (SOL amounts, reserves, txSigner), so new per-trade features are computed in `TokenState`/`Snapshot` with **no new ingestion path** — just a wider feature set. Adding them is low-impact; see the enrichment epic.
- To apply a new feature to historical data, **re-run the free pumpapi replay** (`PumpApiReplaySource`) through the same path — it regenerates the whole `features` table enriched. See *Historical backfill* below.

Persisting raw per-trade rows (a dedicated `trade_events` table) is deliberately **avoided**: it adds write volume and retention burden for no gain, since re-replay already covers backfill.

## Post-bonding coverage

Events carry a `venue` and migration markers, so a token is tracked **continuously through graduation** (bonding curve → AMM). This is required for correct `pump`/`rug` labels over the full lifecycle (see [`ML-MODEL.md`](ML-MODEL.md)).

## Reliability

- Reconnect with exponential backoff + jitter; resume cleanly.
- Idempotent writes (dedupe by `(mint, ts, type, signature)`).
- Bounded memory: cap tracked tokens; evict stale/inactive (`is_active=false`).
- Backpressure: batch DB writes; if lagging, drop low-value snapshots (logged).

## Historical backfill (training)

- Download a **window** of replay archives (start ~2–4 recent weeks — the full archive is ~1.7 TB, do not pull it all), decompress, feed through `PumpApiReplaySource`.
- **Mirror archives to our own storage** to own the dataset long-term.
- Same normalization/feature path as live → consistent training/serving features.

## Config

- `PUMPAPI_WS_URL`, `PUMPAPI_REPLAY_BASE_URL`, DB connection, window length, compute cadence, max tracked tokens, `ENRICH_HOLDERS` (bool) — via env / worker config.

## Testing

- Unit: feature math on synthetic windows.
- Integration: `FileReplaySource` feeds a recorded event file → assert rows in an ephemeral Postgres (Testcontainers).
