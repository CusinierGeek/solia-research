# Data sources & cost strategy

Solia ingests Pump.fun activity through a **swappable `DataSource` abstraction**, so the provider can change without touching downstream code. This document records the providers evaluated, the cost model, and the decision.

## Guiding principle: *derive, don't fetch*

Streaming (push) is free or cheap; per-token REST *fetches* are where cost accrues.

- Compute **price, market cap, volume, buy/sell ratio, bonding progress** directly from decoded **stream events** (they carry the pool reserves). No per-token call.
- Build our own token state in PostgreSQL by accumulating events → **we become our own data provider**.
- Pay for a REST fetch (e.g. `Get Token Info`) only for **rare, immutable** enrichment (metadata), **on-demand, rate-limited, and cached permanently**.
- **Never** fetch per event / in a hot loop — that is the only way costs explode.

## Provider comparison (verified 2026-10-21)

| Provider | Trade stream | New-token / migration | Historical backfill | Post-bonding (AMM) | Notes |
|----------|--------------|-----------------------|---------------------|--------------------|-------|
| **pumpapi.io** ✅ primary | **Free** (`wss://stream.pumpapi.io/`) | Free | **Free** replay, hourly since **2026-04-18** | **Yes** (Pump.fun AMM, Raydium, Meteora, Orca) | Fixed 10 000 lamports (0.00001 SOL) per *action/enrichment* call (`Get Token Info`, etc.); 0.25% per *real* trade (N/A — Solia is paper-only) |
| PumpPortal | **Metered** 0.01 SOL / 10 000 trades | Free | None | Yes (auto-follows migration) | Trade stream is **not** free |
| Solana RPC `logsSubscribe` + IDL | Free (self-decoded) | Free | None (would need `getSignaturesForAddress` replay — slow/limited) | **No** unless you also subscribe to each AMM program and decode it | The SolGemFinder approach — full autonomy, more work |
| BitQuery / Solana Tracker | Paid tiers | — | Paid / limited free tier | Yes | Backup only |

### Why pumpapi.io wins
1. **Trade stream is free** (PumpPortal charges for it).
2. **Free historical replay** solves the initial-training dataset with zero wait and zero backfill cost.
3. **Covers post-bonding** natively → no need to write PumpSwap/Raydium decoders.
4. **Live stream and replay share the same event format** → one parser feeds both `StreamSource` (live) and `ReplaySource` (historical).

### Risks & hedges
- pumpapi.io is a third-party service → reliability/longevity risk.
- **Hedge 1:** the SolGemFinder **RPC + Pump.fun IDL decoder** stays as a `SolanaRpcSource` fallback behind the abstraction → full independence if pumpapi disappears or starts charging. (PumpSwap/Raydium decoding only added then, if needed.)
- **Hedge 2:** **mirror the replay archives to our own storage** now, so we permanently own the training set regardless of the provider's terms.

## Historical data strategy (initial training set)

- Historical replay files: `https://replay.pumpapi.io/YYYY/MM/DD/HH.jsonl.zst` — ~400 MB compressed / ~2 GB in memory per hour, same schema as live.
- **Volume reality:** the full archive since 2026-04-18 is ~1.7 TB compressed. **Do not download everything.** Pick a **window** (start with ~2–4 recent weeks) and expand if the model needs more.
- Flow: download window → decompress → feed through `ReplaySource` → same normalization/feature pipeline as live → PostgreSQL. The first model trains on this; continuous learning then runs off the free live stream + paper-trade outcomes.

## `DataSource` adapters

```
IDataSource (canonical RawEvent stream)
├── PumpApiStreamSource   — live WS (wss://stream.pumpapi.io/)         [primary]
├── PumpApiReplaySource   — historical .jsonl.zst files                [training backfill]
├── PumpPortalSource      — alternative live (trade stream metered)    [optional]
├── SolanaRpcSource       — logsSubscribe + Pump.fun IDL (SolGemFinder)[fallback, autonomous]
└── FileReplaySource      — recorded fixtures                          [tests]
```

Downstream (normalization, features, persistence) depends only on `RawEvent` — nothing knows which provider produced it.
