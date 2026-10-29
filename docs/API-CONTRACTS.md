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
`training_events` run whose decision is `promoted` or `rejected` (an unfinished run, a `failed` one or a
`rollback` trained no model on those trades and does not reset it), or all of them before any such run — against `AiService:RetrainThreshold` (`RETRAIN_EVERY_N_TRADES`).

### `POST /model/retrain`
Manually trigger a retrain (proxied to ai-service, never retried). Returns `202 Accepted` +
`{ "training_event_id": 17 }`; `409 conflict` while a retrain is already running.

### Live hub (SignalR) — `/hubs/live`
Server → client push so the dashboard updates without a reload. Mapped at **`/hubs/live`** on the backend host,
outside `/api/v1` (not versioned with the REST surface). Standard SignalR negotiate/transports, JSON protocol,
payloads **snake_case** like REST. The hub has no client → server methods: connect and handle the events.
Consumer: the dashboard, through one app-wide connection (not one per browser tab) that re-fetches over REST after
every (re)connect.

| Event | When | Payload |
|-------|------|---------|
| `prediction_created` | a `predictions` row is inserted | one `/predictions/history` item |
| `trade_opened` | a `trades` row is inserted open | one `/trades/history` item |
| `trade_closed` | a trade is closed (or inserted already closed) | one `/trades/history` item |
| `log_written` | a non-`debug` `logs` row is inserted, by **any** service | one `/logs/bot` item |
| `stats_changed` | right after every `trade_closed` (sent even if that broadcast failed) | `{ "trade_id": 7, "closed_at": "..." }` — a signal: re-fetch `/stats/winrate` and `/stats/performance` |

```jsonc
// prediction_created
{ "id": 812, "mint": "...", "symbol": "ABC", "ts": "...", "horizon": "1m", "pump_prob": 0.82, "rug_prob": 0.07,
  "strategy": null, "confidence": 81.0, "position_size": null, "take_profit": null, "stop_loss": null,
  "model_version": "v3" }
```

- **Source:** the database, not the writer. Triggers (migration `0004_live_notify.sql`) `NOTIFY solia_live,
  '<event>:<id>'` on commit; the backend `LISTEN`s, re-reads the row by id and pushes it. Rows from the in-process
  bot, the data-pipeline and the ai-service are all covered. A payload is the row **as of that re-read** (a trade
  that closed within milliseconds of opening may already show `closed` in its `trade_opened`).
- **Delivery: at-most-once, best-effort.** Nothing is replayed: events committed while a client is disconnected,
  while the backend's listen connection is down (re-opened with backoff, 1 s → 30 s), or dropped under load are lost.
  Load shedding: two bounded queues, oldest dropped — prediction/trade events (always served first) and log rows
  (re-read in batches of 100), so a log flood never pushes a trade event out; a send stalled > 5 s is cancelled and
  abandoned, and while 16 abandoned sends are still running new events are dropped instead of piling up. Clients **re-fetch over REST after every
  (re)connect** and treat events as increments on top of that snapshot, de-duplicating by `id`.
- **Never slows the writers:** `NOTIFY` doesn't wait for listeners, the push runs on its own queue — a slow or
  disconnected client can't hold up the bot — and the listen connection is unpooled, so it never takes a slot of the
  pool the writers use. Nothing logged while pushing is mirrored into `logs` (it would announce itself).
- **Security:** none beyond the internal network — single-user, no auth, no CORS policy (same-origin / server-side
  clients only). No scale-out backplane: one backend instance.

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
// FEATURE_COLUMNS order. An object row carries `venue` as its raw string or null (the service one-hot
// encodes it; null = venue_other) — not beside venue_* columns; an array row carries the encoded venue_*
// columns. Object rows must all name the same features (else 400 invalid_features). `horizons` is optional (default: all); when present it must be non-empty.
// response: same prediction shape as backend /predict "predictions"
// a window keeps its latest `seq_len` rows; one shorter than the champion's training `min_seq_len`
// is rejected.
// errors: 400 invalid_features / invalid_request, 422 invalid_request (malformed body),
//         422 window_too_short, 503 no_champion / champion_incompatible / champion_unavailable, 500 internal_error
```

### `GET /model/status`
```jsonc
{ "champion_version": "v3", "created_at": "...", "param_count": 123456, "metrics": {...},
  "seq_len": 64, "min_seq_len": 8,
  "eligibility": { "min_unique_traders": 3, "min_volume_sol": 0.0, "min_token_age_sec": 0.0 },
  "trades_since_last_train": 142, "retrain_threshold": 500,
  "latest_run": { "id": 17, "trigger": "n_trades", "started_at": "...", "finished_at": null, "decision": null,
                  "progress": { "candidate": 1, "candidates": 3, "seed": 42, "epoch": 7, "epochs": 50,
                                "train_loss": 0.41, "val_loss": 0.52, "seconds": 64.2, "samples_per_s": 21000.0,
                                "eta_seconds": 2780.0, "since_improved": 1, "patience": 8 } } }
```
`eligibility` is the token eligibility rule the champion was trained under (all `0` = every token; a model
recorded before the rule reads as all `0`) — the bot only scores the tokens it accepts (docs/ML-MODEL.md
"Token eligibility"). The champion fields as before; `503 no_champion` / `champion_incompatible` / `champion_unavailable` while none
can be served (the backend then reports its own progress). `trades_since_last_train` / `retrain_threshold` are the
orchestrator's (same rule as the backend's count, against `RETRAIN_EVERY_N_TRADES`) and always come as a pair:
both `null` when the database read fails. `latest_run` is the newest `training_events` row — `finished_at` and
`decision` `null` while it runs — or `null` before any run (or when the read fails). While the answering
ai-service process trains that run, `latest_run.progress` is its live progress — the candidate being trained
(`candidate` of `candidates`, its `seed`) and its last finished epoch (`epoch` is `0` until the first one ends;
`eta_seconds` = the rest of the candidate's epoch budget at its pace so far, early stopping may end it sooner);
absent otherwise (another process's run, a finished run).

### `POST /model/retrain`
```jsonc
// request
{ "trigger": "manual" }
// 202
{ "training_event_id": 17 }
```
Starts a `manual` run of the retrain orchestrator in the background (training never runs on a request thread:
`/predict` and `/health` keep answering) and answers once its `training_events` row is recorded. Errors:
`409 retrain_in_progress` (a retrain — manual, scheduled or another process's — is running),
`422 invalid_request` (missing body, or a `trigger` other than `manual`), `503 retrain_unavailable` (the run could
not be started, e.g. database down, or was not recorded within 20 s — below the backend's 30 s
`AiService:RetrainTimeout`, so the backend always hears the answer). A run whose process stopped mid-training is
recorded as `failed` at the next startup (or by the next run), so `latest_run` never stays in flight forever.

**Background trigger.** While `RETRAIN_SCHEDULER_ENABLED` (default off; on in compose), the ai-service checks the
closed-trade counter every `RETRAIN_CHECK_INTERVAL_SECONDS` (default 300) and starts an `n_trades` run once it
reaches `RETRAIN_EVERY_N_TRADES`. A check never overlaps a running retrain; a failed check (database blip) or a
failed run is logged and retried on the next tick.

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
