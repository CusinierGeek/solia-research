# ML model specification

## Problem framing

A **signal model**: a multi-task, multi-horizon time-series **Transformer**. Given a *sequence* of
feature snapshots for a token, it predicts — **for each horizon** (30s, 1m, 5m, 15m, 1h) — only
targets that have an objective label:

1. **pump probability** `p_pump ∈ [0,1]`
2. **rug probability** `p_rug ∈ [0,1]`
3. **expected forward return** `r̂` (regression)

plus a derived **confidence ∈ [0,100]** (from calibration, not a learned head).

> **Prediction ≠ decision.** The model does *not* output buy/sell/hold, position size, or TP/SL.
> Those are **decisions** with no objective training label; they live in a separate, tunable
> **policy layer** (the paper-trading bot, Sprints 3–4) that consumes the model's predictions
> (see [Policy layer](#policy-layer-not-the-model)). This plays to the Transformer's strength — a
> rich temporal representation feeding well-labeled predictions — and keeps risk tunable without
> retraining.

## Why a Transformer (and how we exploit it)

- **Sequence attention is the whole point.** The input is a *sequence* of the token's recent
  feature snapshots, so attention can capture the *shape* of a run-up (accelerating buys, holder
  growth, volatility regime, early dumps). Feeding a single row would waste the architecture — a
  plain MLP would do.
- **Shared encoder + many prediction heads is cheap and helps.** One context vector feeds all
  heads; multi-task, multi-horizon supervision regularizes and strengthens the representation. We
  add heads only for targets with real labels (pump, rug, return) — never for un-labelable
  decisions.

## Architecture

- **Type**: Transformer encoder for time series (no decoder).
- **Input**: `[batch, seq_len, n_features]`. `seq_len` = rolling window of snapshots (e.g. 64);
  `n_features` = the `FEATURE_COLUMNS` spec (`services/ai-service/solia_ai/features.py`, parity-checked against [`DATA-MODEL.md`](DATA-MODEL.md)).
- **Blocks**:
  - Input projection (`Linear(n_features → d_model)`) + positional encoding.
  - `N` × TransformerEncoderLayer (multi-head self-attention + FFN, pre-norm).
  - Masked mean / attention pooling over the sequence → context vector.
- **Heads** (per horizon): small MLPs on the shared context vector
  - `pump_head`, `rug_head` → sigmoid
  - `return_head` → linear (expected forward return)
- Default hyperparams (tunable in `services/ai-service/solia_ai/config.yaml`): `d_model=128`, `n_heads=8`, `n_layers=4`, `dropout=0.1`, `seq_len=64`.

## Losses

Total loss = weighted sum (per-horizon and per-task weights configurable):

- **pump / rug**: BCE **+ Focal Loss** (class imbalance — pumps/rugs are rare).
  `L = α · BCE + (1-α) · Focal(γ)`
- **expected return**: Huber (smooth L1).

Class imbalance is corrected **once**: class-balanced batches (`training.balanced_sampling`, the
default), *or* natural batches with a BCE `loss.pos_weight` (`auto` = neg/pos per horizon on the
training split, a fixed number, or per head). The two together are rejected by config validation.

## Calibration

The policy acts on probability thresholds, so the served probabilities must be calibrated. After
training, `p = sigmoid(logit / T + b)` is fit **per head × horizon** (`calibration.method`:
`platt` = T and b, `temperature` = T only, `none`) on the earliest `calibration.fraction` of the
validation windows, by minimizing log-loss. Platt is the default because balanced batches bias the
raw probabilities toward the ~50% event rate — a shift a temperature alone can't undo. The fit is
stored in the checkpoint (`calibration`, `calibration_end_ts`) and applied identically by
`/predict` and the ONNX graph; a checkpoint without it loads as the identity. Both maps are
strictly increasing; the fit is bounded (`T` ∈ [0.1, 10], `|b|` ≤ 10) so the sigmoid doesn't
saturate into ties, and `evaluate` scores in float64, so ROC-AUC / PR-AUC are unchanged. A
head × horizon keeps the identity when it has fewer than `calibration.min_samples` labels or fewer
than `calibration.min_class_samples` of its minority class (a rare event can't support a fit). The
calibration window is a strict prefix of the validation split (too short a split keeps the
identity, with a warning); `evaluate` reports ECE before/after and only scores windows after it,
and the ablation reports the calibrated − raw effect over all runs.

## Optimizer & training

- **Optimizer**: AdamW (`lr`, `weight_decay` in config).
- **Scheduler**: cosine with warmup.
- **Regularization**: dropout, gradient clipping, early stopping on the validation metric.
- **Mixed precision** optional.
- **Split**: temporal split (train on past, validate on more recent) to avoid leakage. **Never** shuffle across the time boundary within a token.

## Labeling

All labels come straight from the realized future price/liquidity trajectory — **objective, no
policy required**, so the model trains from day one, before any paper trades exist:

- `pump` at horizon `h` = price rose ≥ threshold within `h`.
- `rug` at horizon `h` = liquidity/price collapsed ≥ threshold within `h`.
- `expected_return` at horizon `h` = realized forward return over `h`.

Thresholds live in `config.yaml` (documented with the dataset builder, Sprint 2).

Labels require the token's **full lifecycle** — including **post-bonding** price/liquidity on the
AMM. The pipeline captures pre- and post-bonding events (see [`DATA-SOURCES.md`](DATA-SOURCES.md)),
so a pump/rug after migration is labeled correctly; a pre-bonding-only dataset would bias labels.

## Confidence

Confidence (0–100) is **derived** from the calibrated probabilities (their margin from 0.5, after
the per-head × horizon calibration of the logits), not a learned head; it must be monotonic with historical hit-rate. See
"Calibration" above.

## Outputs contract

The inference response (see [`API-CONTRACTS.md`](API-CONTRACTS.md)) returns, per requested horizon:
`pump_prob`, `rug_prob`, `expected_return`, `confidence` — plus the `model_version`. Nothing about
trade decisions (those are the policy layer's job).

## Policy layer (not the model)

Buy/sell/hold, position **sizing**, dynamic **take-profit / stop-loss**, and the entry threshold
are **rules**, not model outputs — implemented in the paper-trading bot (Sprints 3–4) and
configurable without retraining. They consume the model's predictions, for example:

- enter when `pump_prob ≥ P` **and** `rug_prob ≤ R` (and liquidity/age gates pass);
- size = fractional Kelly on `expected_return` (capped at TP) vs `rug_prob` (loss capped at SL), scaled by confidence, capped;
- TP/SL from `expected_return` and recent volatility.

Keeping decisions out of the model lets risk appetite change instantly and keeps the model purely
predictive. A *learned* policy (e.g. RL) is possible later only if it clearly beats the rules.

## Continuous training loop

1. Count new **closed** trades since last training (`training_events`).
2. When `>= RETRAIN_EVERY_N_TRADES`: build an updated dataset (historical + live).
3. Train a **candidate** from the champion checkpoint (warm start) or from scratch (configurable).
4. Evaluate candidate vs champion on a held-out recent window.
5. **Promote** only if it beats the champion by a margin on the primary metric; otherwise **reject**
   / **rollback**.
6. Record everything in `training_events` and `models`; log to `logs`.

Checkpoints, versioning, comparison, and rollback are first-class — see Sprint 7.

### Promotion gate: guard against luck-driven false champions

The promotion gate is a **controlled offline comparison**: candidate and champion are scored on the
**same held-out recent window** under the same policy (apples-to-apples). It is **not** the live
paper-trading account — a single live path has no counterfactual for the champion. Live P&L drives
*when* to retrain (`RETRAIN_EVERY_N_TRADES`) and drift detection, not promotion.

- **Primary metric is a *predictive* one** (ROC-AUC / PR-AUC / calibration), computed over thousands
  of labeled windows → statistically stable. **Policy-simulated P&L is a secondary sanity check**,
  not the gate: Pump.fun P&L is heavy-tailed (a few 10× tokens dominate), so promoting on a short
  window's backtest P&L is the classic way to crown a **false champion** (skill vs luck).
- **Promote by a margin**, never on a razor-thin edge, and use **rolling** held-out windows (don't
  re-select against the same window forever — multiple-comparisons overfitting).
- When building Sprint 7, add **statistical rigor** to the P&L comparison: significance test /
  confidence interval (bootstrap, Deflated Sharpe) and require the lift to hold across **multiple
  windows / seeds**, not a single run. `seed` makes *training* reproducible, not eval-sample luck.

### Ablation: every feature group must earn its place

New features are accepted only on **measured lift** (`solia_ai/ablation.py`), with the same rules
as the promotion gate: predictive metrics primary, reference-policy P&L secondary, several windows
and seeds.

- **Feature groups** are declared once in `features.py` (`FEATURE_GROUPS`: group → columns of
  `FEATURE_COLUMNS`; every column in exactly one group). `BASELINE_GROUPS` = the original 19 columns. An
  enrichment task adds its columns *and* a new group, outside the baseline.
- **Candidate groups stay disabled by default.** The model only sees the groups listed in the config's
  `features.groups` (default: `BASELINE_GROUPS`); the other columns are zeroed by a mask stored in the
  model (`feature_mask`, same in training, inference and the ONNX graph, and saved with the checkpoint).
  A group is enabled there only once its ablation lift clears the margin.
- **Candidates so far:** `buy_pressure` (FE-T2, #111) — `net_sol_flow`, `buy_volume_ratio`,
  `avg_buy_sol`, `max_buy_sol`, `large_buy_count`; `momentum` (FE-T3, #112) — `return_short`,
  `return_medium`, `return_long`, `price_acceleration`, `ema_slope`, `volume_acceleration`,
  `volume_burst_ratio`, `price_history_sec` (windows: `Pipeline:Momentum{Short,Medium,Long}Seconds`);
  `concentration` (FE-T4, #113) — `unique_traders`, `top_trader_share`, `trader_hhi`,
  `new_trader_ratio`: a holder-concentration proxy from trade signers over the window;
  `burn` (FE-T5, #114) — `burned_liquidity`, % of the AMM pool's LP burned (rug signal);
  `venue` (FE-T5, #114) — the `venue` column one-hot encoded (`VENUE_COLUMNS`).
- **Venue one-hot is encoded at load time, not stored.** `features.py` derives `venue_<name>` from the
  raw `venue` column (training reads it from the DB, `/predict` rows carry it as a string), over a
  fixed category list (`VENUES`: `pumpfun`, `pumpswap`, `raydium-launchpad`, `raydium`,
  `meteora-launchpad`, `meteora`, `bonk`) plus `venue_other` for anything else, NULL included. No
  schema churn when the list changes; appending a venue appends a column (an ordered-superset spec,
  so the champion keeps serving). `DB_FEATURE_COLUMNS` is the stored part the parity tests check.
- **Rows predating a group are refused, not zero-filled.** A NULL in a column the pipeline always writes
  since the group's migration (`ALWAYS_WRITTEN`, e.g. `net_sol_flow`/`large_buy_count` for
  `buy_pressure`, `volume_burst_ratio`/`price_history_sec` for `momentum`, `unique_traders` for `concentration`; none for `burn`, whose
  migration 0008 backfills earlier rows from `onchain_signals`, nor `venue`) means "never computed": training, evaluation and ablation raise if an enabled (or
  ablated) group has such rows — re-replay them first. Other NULLs are imputed per `NULL_FILL`
  (`buy_volume_ratio` → 0.5, "no pressure either way"; `volume_burst_ratio` → 1, "steady"; `new_trader_ratio` → −1, "no recent trader", as 0 is "all returning"; `burned_liquidity` → −1, "unknown", as 0% burned is a real rug signal; else 0 —
  e.g. a return a young token has no history for, which `price_history_sec` tells apart from a flat
  price).
- **Adding columns keeps the champion serving.** A checkpoint trained before columns were added (its
  `feature_columns` an ordered subset of `FEATURE_COLUMNS`) is rebuilt on its own columns, and the
  `Predictor` feeds it just those out of each current-spec window, until a retrain on the new spec is
  promoted. A spec with a removed or reordered column is refused at load (`IncompatibleCheckpointError`):
  `/predict` answers 503 `champion_incompatible` (the backend treats it like `no_champion`: not retried).
- **Variants**: the baseline, plus one per group under test — a non-baseline group is **added**
  (`+group`), a baseline group is **dropped** (`-group`). A disabled group's columns are zeroed in
  the input; architecture, labels and hyperparameters are identical.
- **Walk-forward windows**: the most recent `--eval-fraction` of labeled windows is cut into
  `--windows` consecutive slices. Each slice is scored by a model trained only on samples ending
  before it minus the label embargo; early stopping uses a time split of that training pool, never
  the scored slice. Every variant is scored on the same slices.
- **Seeds**: every (window, variant) is trained under each `--seeds` value.
- **Lift** = metric *with group* − *without group*, paired run for run, positive = better (ECE
  negated). Reported as mean ± std per metric (ROC-AUC / PR-AUC / ECE per head and horizon, then
  policy `total_pnl` / `avg_pnl`). Verdict: **better** if `mean − std > margin`, **worse** if
  `mean + std < −margin`, else **not significant**; **n/a** when the metric is undefined in every
  run (e.g. a single-class horizon).
- Ablation never saves a checkpoint, registers a model or touches `is_champion`.

```bash
# from services/ai-service, database from SOLIA_DB_CONNECTION / POSTGRES_* (a replayed history)
python -m solia_ai.ablation                                   # baseline vs dropping each group
python -m solia_ai.ablation --groups flow holders --windows 4 --seeds 0 1 2 3 \
    --margin 0.01 --pnl-margin 0 --out reports/ablation.json  # JSON report + readable summary
```

## Files (ai-service, built in Sprints 2 & 7)

| File | Role |
|------|------|
| `model.py` | Transformer + prediction heads (pump / rug / return) |
| `dataset.py` | Dataset/DataLoader, sequence windowing, labeling |
| `features.py` | Shared feature spec / parity with pipeline |
| `train.py` | Initial training entrypoint |
| `evaluate.py` | Held-out evaluation (predictive metrics + reference-policy P&L) |
| `ablation.py` | Per-feature-group lift across windows & seeds |
| `inference.py` | Load champion, run predictions (+ ONNX export) |
| `retrain.py` | Continuous-training orchestrator (Sprint 7) |
| `api.py` | FastAPI app (`/predict`, `/model/status`, `/model/retrain`) |
| `config.yaml` | All hyperparameters & thresholds |

## Evaluation metrics

- pump / rug: ROC-AUC, PR-AUC, precision/recall at the operating threshold, calibration (ECE).
- expected return: information coefficient (IC) / correlation, MAE.
- economic: simulated win-rate & P&L on the validation window **via the policy layer** — the metric that ultimately matters.
