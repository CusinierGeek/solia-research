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
      "strategy": "buy", "position_size": 0.15,
      "take_profit": 1.4, "stop_loss": 0.85, "confidence": 81.5 }
  ]
}
```

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
// response: same prediction shape as backend /predict "predictions"
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

## Versioning
- Public API is versioned by path (`/api/v1`).
- Prediction responses always include `model_version` for traceability.
