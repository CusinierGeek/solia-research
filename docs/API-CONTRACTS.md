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

> The model is a **signal model**: predictions only. Trade **decisions** (strategy, position
> sizing, take-profit, stop-loss) are the paper-trading bot's **policy layer** (Sprints 3–4),
> derived from these predictions — not returned here. See [`ML-MODEL.md`](ML-MODEL.md#policy-layer-not-the-model).

### `GET /tokens/live`
Currently tracked tokens with latest snapshot. Supports `?limit=&sort=`.

### `GET /tokens/history`
Paginated token registry. `?from=&to=&outcome=&page=&pageSize=`.

### `GET /trades/history`
Paper trades. `?status=&from=&to=&page=&pageSize=`.

### `GET /stats/winrate`
```jsonc
{ "winrate": 0.58, "wins": 290, "losses": 210, "window": "all" }
```

### `GET /stats/performance`
Equity curve points for the dashboard chart.
```jsonc
{ "points": [ { "ts": "...", "equity": 10.0, "pnl": 0.0 } ], "total_pnl": 3.2 }
```

### `GET /logs/bot`
Recent bot/system logs. `?level=&service=&limit=`.

### `GET /model/status`
```jsonc
{ "champion_version": "v3", "created_at": "...", "metrics": {...},
  "trades_since_last_train": 142, "retrain_threshold": 500 }
```

### `POST /model/retrain`
Manually trigger a retrain (proxied to ai-service). Returns `202 Accepted` + a `training_event_id`.

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
// errors: 400 invalid_features / invalid_request, 422 invalid_request (malformed body),
//         503 no_champion / champion_unavailable, 500 internal_error
```

### `GET /model/status`
Champion version, metrics, param count, trades-since-last-train.

### `POST /model/retrain`
Kick off the retrain orchestrator. Body: `{ "trigger": "manual" }`. Returns `202` + `training_event_id`.

### `GET /health`
Liveness/readiness.

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
