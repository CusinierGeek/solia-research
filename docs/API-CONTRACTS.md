# API contracts

Two API surfaces:
- **backend** (ASP.NET Core) — consumed by the dashboard and external callers.
- **ai-service** (FastAPI) — consumed by the backend only (internal).

All payloads are JSON. Timestamps are ISO-8601 UTC. These are **draft** contracts; finalized per sprint.

---

## backend (public REST)

Base: `/api/v1`

### `POST /predict`
Request a prediction for a token (proxied to ai-service).
```jsonc
// request
{ "mint": "So1...", "horizons": ["30s","1m","5m","15m","1h"] }
// response
{
  "mint": "So1...",
  "model_version": "v3",
  "predictions": [
    { "horizon": "1m", "pump_prob": 0.72, "rug_prob": 0.08,
      "expected_return": 0.35, "confidence": 81.5 }
  ]
}
```

The backend loads the token's latest `features` rows — at most `AiService:MaxFeatureWindowSize` (default 256,
a payload cap, not a copy of the model's `seq_len`) — and sends them oldest first; the ai-service owns the
window bounds (it keeps its champion's latest `seq_len` rows). `horizons` is optional (all when omitted) and,
when given, a non-empty list, trimmed and deduplicated; which horizon names exist is the ai-service's call (an
unknown one is a `400`). Errors: blank or missing `mint` / malformed body → `400 invalid_request`; unknown
mint, or a mint with no feature snapshot yet → `404 not_found`; fewer rows than the champion's `min_seq_len`
(a token too young to score) → `422 insufficient_history`; no champion / ai-service unreachable →
`503 unavailable`; ai-service failure → `502 internal_error` (upstream details are logged, never returned).

> The model is a **signal model**: predictions only. Trade **decisions** (strategy, position
> sizing, take-profit, stop-loss) are the paper-trading bot's **policy layer** (Sprints 3–4),
> derived from these predictions — not returned here. See [`ML-MODEL.md`](ML-MODEL.md#policy-layer-not-the-model).

### Read endpoints — shared conventions
- Response properties are **snake_case**; money/prices are exact decimals, probabilities/ratios doubles.
- `from`/`to` are ISO-8601 timestamps, inclusive; either may be omitted, `from > to` is a `400`.
- Enum-like params (`outcome`, `status`, `horizon`, `strategy`, `level`, `service`, `sort`, `window`) are case-insensitive; an
  unknown value is a `400 invalid_request` naming the parameter and its allowed values. An empty or blank
  value (`?status=`) is the same as omitting it.
- **Paginated** endpoints take `page` (1-based, default 1) and `pageSize` (1–200, default 50) and return:
```jsonc
{ "items": [ ... ], "page": 1, "page_size": 50, "total": 1234 }   // total = rows matching the filters
```
  A page past the end returns empty `items` with the same `total`. A page may reach at most 10 000 rows deep
  (`page × pageSize ≤ 10000`, else `400`) — narrow `from`/`to` to go further back. `total` and `items` come
  from one snapshot.

### `GET /tokens/live`
Currently tracked tokens (`is_active`) with their latest snapshot, as a plain array.
`?limit=` (1–500, default 50) `&sort=` `updated_at` (default) | `market_cap` | `liquidity` | `holders` |
`trending_score` | `created_at` — always descending, missing values last.
```jsonc
[ { "mint": "So1...", "symbol": "ABC", "name": "...", "created_at": "...", "last_price": 0.0000001234,
    "last_market_cap": 1000.5, "last_liquidity": 42.0, "holders": 311, "trending_score": 0.7,
    "venue": "pumpfun", "is_migrated": false, "updated_at": "..." } ]
```

### `GET /tokens/history`
Paginated token registry, newest `first_seen_at` first. `?from=&to=` bound `first_seen_at`;
`?outcome=` `pump` | `rug` | `neutral`; `?page=&pageSize=`.
Items: `{ mint, symbol, name, created_at, first_seen_at, last_seen_at, final_outcome }`.

### `GET /predictions/live`
The latest prediction of every currently tracked token (`tokens_live.is_active`), **one row per
(mint, horizon)** — flat, same item shape as the history, so a token predicted at three horizons appears three
times. A horizon the token was never predicted at has no row. `?limit=` (1–1000, default 100)
`&horizon=` `30s` | `1m` | `5m` | `15m` | `1h` (only that horizon) `&sort=` `ts` (default) | `pump_prob` |
`rug_prob` | `confidence` — always descending, missing values last.
```jsonc
[ { "id": 42, "mint": "So1...", "symbol": "ABC", "ts": "...", "horizon": "1m", "pump_prob": 0.81,
    "rug_prob": 0.07, "strategy": "buy", "confidence": 74.0, "position_size": 0.5,
    "take_profit": 0.0000001500, "stop_loss": 0.0000001100, "model_version": "v3" } ]
```
`symbol` comes from the token registry; `model_version` from `models`, `null` when the row has no model.
Probabilities are `0–1`, `confidence` `0–100` (doubles); `position_size` / `take_profit` / `stop_loss` are
exact decimals. There is no `expected_return` (the table doesn't store it; see `POST /predict`).

### `GET /predictions/history`
Paginated predictions, newest `ts` first. `?mint=` (exact, case-sensitive) `&horizon=` `&strategy=` `buy` |
`sell` | `hold` `&from=&to=` (bound `ts`) `&page=&pageSize=`.
Items: the `/predictions/live` item, `{ id, mint, symbol, ts, horizon, pump_prob, rug_prob, strategy,
confidence, position_size, take_profit, stop_loss, model_version }`.

### `GET /trades/history`
Paginated paper trades, most recently opened first. `?status=` `open` | `closed`; `?from=&to=` bound
`opened_at`; `?page=&pageSize=`.
Items: `{ id, mint, prediction_id, opened_at, closed_at, entry_price, exit_price, size, duration_sec, pnl,
pnl_pct, strategy, confidence, exit_reason, status }`.

### `GET /stats/winrate`
`wins / closed trades` — the backtest's (`ReplayResult`) definition: win = `pnl > 0`, loss = `pnl < 0`,
break-even (`0` or no P&L) is neither but still counts in the denominator; `0` when nothing closed. Closed
trades = `status = 'closed'` with a `closed_at` (the same set as `/stats/performance`). `?window=` `all`
(default) | `24h` | `7d` | `30d`: closed strictly after now − window, by `closed_at`.
```jsonc
{ "winrate": 0.58, "wins": 290, "losses": 190, "break_even": 20, "window": "all" }
```

### `GET /stats/performance`
Equity curve points for the dashboard chart: one point per closed trade in `closed_at` order, limited to the
latest `?limit=` (1–10000, default 1000), `equity` = the bot's configured starting balance
(`Bot:StartingBalance` / `PAPER_STARTING_BALANCE`) + realized P&L so far (earlier trades included), `pnl` =
that trade's P&L; `total_pnl` covers every closed trade.
```jsonc
{ "points": [ { "ts": "...", "equity": 11.5, "pnl": 1.5 } ], "total_pnl": 3.2 }
```

### `GET /logs/bot`
Recent bot/system logs, newest first, as a plain array. `?level=` is a **minimum** severity
(`debug` < `info` < `warn` < `error`); `?service=` `data-pipeline` | `ai-service` | `backend` | `bot`;
`?limit=` (1–1000, default 100).
```jsonc
[ { "id": 1, "ts": "...", "service": "bot", "level": "warn", "message": "...", "context": { ... } } ]
```

### `GET /model/status`
```jsonc
{ "champion_version": "v3", "created_at": "...", "metrics": {...},
  "trades_since_last_train": 142, "retrain_threshold": 500 }
```
Champion fields come from the ai-service (`metrics` is `{}` when it reports none). When the ai-service fails
(unreachable, no champion yet, error) the call still answers `200`, with `champion_version` / `created_at`
`null` and `metrics` `{}`, so the retrain progress stays visible. `trades_since_last_train` and
`retrain_threshold` are one consistent pair: the ai-service's own when it reports **both**; otherwise the
backend's count — closed trades (same set as `/stats/*`) closed strictly after the `started_at` of the latest
*finished* `training_events` run (any decision; an unfinished or crashed run does not reset it), or all of
them before any run finished — against `AiService:RetrainThreshold` (`RETRAIN_EVERY_N_TRADES`).

### `POST /model/retrain`
Manually trigger a retrain (proxied to ai-service, never retried). Returns `202 Accepted` +
`{ "training_event_id": 17 }`.

---

## ai-service (internal REST, FastAPI)

Base: `/`

### `POST /predict`
```jsonc
// request
{ "mint": "So1...", "features": { /* latest vector */ },
  "horizons": ["1m","5m"] }
// `features` is either the latest vector (one object = a one-row window) or the token's recent
// window, oldest row first: a non-empty list of objects keyed by feature name, or of arrays in
// FEATURE_COLUMNS order. `horizons` is optional (default: all); when present it must be non-empty.
// response: same prediction shape as backend /predict "predictions"
// a window keeps its latest `seq_len` rows; one shorter than the champion's training `min_seq_len`
// is rejected.
// errors: 400 invalid_features / invalid_request, 422 invalid_request (malformed body),
//         422 window_too_short, 503 no_champion / champion_unavailable, 500 internal_error
```

### `GET /model/status`
Champion version, `created_at`, metrics, `param_count`, and its window bounds `seq_len` / `min_seq_len`.
Trades-since-last-train and the retrain threshold are added by the retrain orchestrator (Sprint 7).

### `POST /model/retrain`
Kick off the retrain orchestrator. Body: `{ "trigger": "manual" }`. Returns `202` + `training_event_id`.

### `GET /health`, `GET /health/ready`, `GET /health/live`
Probes (outside `/api/v1`). `/health/live` checks nothing (process up); `/health/ready` checks the database and
the ai-service (its `/health`); `/health` runs every check. Unhealthy → `503`, otherwise `200`, both with:
```jsonc
{ "status": "Unhealthy", "total_duration": "00:00:00.0123",
  "entries": {
    "database":   { "status": "Healthy",   "description": "database reachable",     "duration": "00:00:00.004", "tags": ["ready"] },
    "ai_service": { "status": "Unhealthy", "description": "ai-service unreachable", "duration": "00:00:00.008", "tags": ["ready"] } } }
```
Descriptions are fixed strings (`database reachable|unreachable`, `ai-service reachable|unreachable`,
`ai-service reports it is not healthy`); anything else — e.g. a check that throws or times out — is reported as
`check failed`, never with an exception message. `/health/live` is always `200`. The ai-service probe is a single
attempt (no retry) that the circuit breaker neither counts nor blocks; each check is bounded to 5 s. A failing
check is logged once when it starts failing and once when it recovers, not on every probe.

---

## Error shape (both services)
```jsonc
{ "error": { "code": "not_found", "message": "unknown mint" } }
```

Backend codes by status (stable — clients may switch on them): 400/422 `invalid_request`, 401 `unauthorized`,
403 `forbidden`, 404 `not_found`, 405 `method_not_allowed`, 406 `not_acceptable`, 409 `conflict`,
413 `payload_too_large`, 415 `unsupported_media_type`, 429 `rate_limited`, 503 `unavailable`, other 5xx
`internal_error`. Headers a status relies on (`Allow`, `WWW-Authenticate`, `Retry-After`) are preserved.

## Versioning
- Public API is versioned by path (`/api/v1`).
- Prediction responses always include `model_version` for traceability.
