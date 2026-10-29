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
- `expected_return` at horizon `h` = realized forward return over `h`, capped at `labeling.max_return`
  (default 10 = +1000 %; floor -1). A price read off a pool drained to dust can be 10^8x the anchor
  (seen on the 2026-10-28 replay), and a single such label dominated the Huber loss (`val_return`
  ≈ 3.6·10^7 on the first 4 M-row training).

Thresholds live in `config.yaml` (documented with the dataset builder, Sprint 2).

Labels require the token's **full lifecycle** — including **post-bonding** price/liquidity on the
AMM. The pipeline captures pre- and post-bonding events (see [`DATA-SOURCES.md`](DATA-SOURCES.md)),
so a pump/rug after migration is labeled correctly; a pre-bonding-only dataset would bias labels.


## Token eligibility

Most Pump.fun tokens are created and never really traded (2026-10-28 00:00–03:00 UTC replay, 22 569
tokens: 52.9 % never had more than 1 unique trader, 72 % never more than 3, 5.8 % passed 50). One rule,
`config.yaml` `eligibility` (`solia_ai/eligibility.py`), decides which rows the model learns on and the bot
may trade. It is **causal** — a row is judged on itself and the token's earlier rows only, never on what
happened next (dropping the tokens that *turned out* dead would train on an optimistic population and
miscalibrate the model on the tokens it sees live) — and **identical** in training and trading:

| Threshold (0 = off) | A row passes when… |
|---|---|
| `min_unique_traders` | the most distinct traders any snapshot of the token counted so far ≥ it |
| `min_volume_sol` | the highest window SOL `volume` any snapshot reported so far ≥ it |
| `min_token_age_sec` | the oldest `token_age_sec` known so far ≥ it |

The pipeline's `unique_traders` / `volume` are per rolling window (`Pipeline:WindowSeconds`, 300 s), so the
"so far" value is their **running maximum** — a causal lower bound of the traders seen, and never a sum of
overlapping windows (which would count each trade many times). A NULL input carries no information: it
doesn't raise a running maximum, so a token with no known value yet fails that criterion. Every criterion is
a running maximum, hence monotone: once eligible, a token stays eligible. All thresholds 0 (or the section
omitted) = every token, today's behaviour; a negative, non-finite or non-numeric threshold (or a fractional
trader count, an unknown key) fails the config load.

- **Dataset** (`build_samples`): only windows whose **anchor** (last) row is eligible become training /
  validation / held-out samples. Labels are unchanged, and the rows before the token became eligible may
  still be in the window's history. Each sample records its anchor's inputs (`Samples.eligibility_inputs`),
  so another rule is a mask, not a rebuild.
- **Recorded with the model**: `hyperparams["eligibility"]`, in the checkpoint and the `models` row;
  `GET /model/status` reports the served champion's. A model recorded without one predates the rule and was
  trained on every token (read as all 0).
- **Bot** (`Solia.Bot.Policy.EligibilityRule`, `Solia.Api.Bot.ChampionEligibility`): the bot never configures
  its own rule — it reads its **champion's** from `models.hyperparams` (re-read every 30 s, and as soon as
  `/predict` answers with another model version), so it can't trade a population its model never saw. Before
  `/predict`, the feed computes the token's running maxima over its rows up to the scored one (one SQL
  aggregate); an ineligible token is not sent to the ai-service, hence never entered. Its price ticks still
  flow, so an already-open position is managed as before. A champion recorded without a rule scores every
  token (warning); one recording an invalid rule scores **none** (fail closed, error) until it changes.
- **Rule change**: a retrain whose champion was trained under another rule than the config's logs a note
  (`training_events` / `logs`): both models are scored on the config's eligible held-out windows, and the bot
  keeps trading the champion's population until a candidate trained under the new rule is promoted.
- **Parity**: `services/shared/testdata/eligibility-cases.json` (threshold edges, NULLs, combined criteria,
  invalid rules) runs in both `tests/test_eligibility.py` and `EligibilityRuleTests`.
- **Read size**: the rule filters samples, not the training read — `read_training_features` still loads every
  token of the window (bounding the read is the `data` section's job). The ablation reports how many
  `features` rows the eligible tokens hold, i.e. what a filtered read would load.

### Choosing the thresholds: eligibility ablation

```bash
python -m solia_ai.eligibility_ablation --windows 3 --seeds 0 1 --out reports/eligibility.json
# --rule traders=4,volume=1.0,age=30 (repeatable); --lookback-days / --max-rows like a retrain
```

For each rule, every walk-forward (window, seed) run pairs the rule's model (trained on the eligible windows
only) with the every-token model, **both scored on the rule's eligible held-out windows** — the population the
bot would trade. The slices are cut once, by time, so every rule is scored on the same time ranges. Lift =
rule − every-token (positive = better, ECE negated), judged like the feature ablation. Predictive metrics are
primary; the samples / tokens / rows kept and the reference-policy trades & P&L are reported beside them.

**Default: disabled (all thresholds 0)** until measured. The first run (2026-10-29) was stopped: the
database held a stale replay, to be redone with the current pipeline. Run the ablation on the fresh replay
(after the training-throughput task, #152) and record the result and the chosen defaults here.

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
5. **Promote** only if it passes the promotion gate (primary metric over a margin, no guard-rail
   regression); otherwise **reject** (the candidate stays registered). **Rollback** to the previous
   champion is a separate, on-demand operation.
6. Record everything in `training_events` and `models`; log to `logs`.

Checkpoints, versioning, comparison, and rollback are first-class — shipped in Sprint 7.

**Candidate training** (`solia_ai/candidate.py`, config `candidate`): the candidate is trained on every
labeled window ending at or before the held-out start minus the label embargo (held-out = the most
recent `heldout_fraction` — by default the walk-forward eval share the gate scores, defined once —
or an explicit start); the newest tail whose forward horizons have not elapsed carries no label and is
excluded. Closed paper trades only trigger the retrain, they are never labels. `warm_start: true`
loads the champion's weights (normalizer refit on the new training split) with a fresh
optimizer/scheduler, a `warm_start_epochs` budget and the peak learning rate scaled by
`warm_start_lr_factor`; a champion whose feature columns, enabled groups, horizons or model dims
differ, whose recorded data (`data_end_ts`) reaches past the held-out start minus the embargo (its
weights would leak held-out windows), or whose checkpoint can't be loaded falls back to scratch
(logged, recorded in the result and checkpoint) or raises with `on_incompatible: error`. Calibration
is fit on the candidate's own validation split; the checkpoint's `data_end_ts` (end of that split)
keeps evaluation off every window the candidate saw. The recorded `hyperparams` are the configured
ones, with a warm start's effective budget/LR under `hyperparams.warm_start`. The checkpoint is saved
but never registered; `n_samples` (train + validation) feeds `training_events.n_samples`.

**Orchestrator** (`solia_ai/retrain.py`, `python -m solia_ai.retrain [--trigger n_trades|scheduled|manual] [--force]`
or `python -m solia_ai.retrain rollback`): due when the closed trades since the last run that trained a model
(`promoted` / `rejected`, the backend's `/model/status` rule) reach `RETRAIN_EVERY_N_TRADES` (default 500); a
`manual` or `--force` run skips the check. One run at a time (PostgreSQL advisory lock): a concurrent run or
rollback is refused, not queued. Each run writes a `training_events` row at start (trigger, champion) and
completes it (`n_samples`, the gate's comparison in `metrics`, decision, notes); anything that raises ends it as
`failed` with the error in `notes`. The gate's walk-forward windows are cut once (eval share =
`candidate.heldout_fraction`); one candidate is trained per gate seed, pinned to the first window's start, and
every candidate and the champion (a fixed model: its metrics per window are shared by the seeds) are scored on
the same windows. The first seed's candidate always gets a `models` row (val metrics + held-out means under
`heldout`); a promotion flips `is_champion` in the same transaction that completes the event. A champion whose
checkpoint can't be loaded or that serves other horizons fails the run (no paired comparison). **Rollback**
restores the champion the current one replaced (the `promoted` event that crowned it, so a second rollback
walks further back), after `check_rollback`, and records a `rollback` event; refused when there is none, its
checkpoint is missing or incompatible. Steps are logged to stdout and `logs` (service `ai-service`). The API
re-reads the registry champion every 30 s (`api.DEFAULT_REFRESH_SECONDS`), so a promotion or rollback is
served without a restart.

**In the API process** (`solia_ai/retrain_service.py`): `POST /model/retrain` starts a `manual` run on a worker
thread and answers `202` with its `training_event_id` once recorded (`409` while a retrain runs here or holds the
lock elsewhere). With `RETRAIN_SCHEDULER_ENABLED` (on in compose) a background trigger reads the counter every
`RETRAIN_CHECK_INTERVAL_SECONDS` (default 300) and starts an `n_trades` run at the threshold; a tick never overlaps
a running retrain, and a failed check or run is logged and retried on the next tick. Training never runs on a
request thread: `/predict` and `/health` keep serving. `/model/status` adds `trades_since_last_train`,
`retrain_threshold` and `latest_run` (see `docs/API-CONTRACTS.md`).

**End-to-end check** (`tests/test_continuous_training_e2e.py`, the Sprint 7 Definition of Done, run in CI): the
API app with its lifespan scheduler over an ephemeral PostgreSQL (Testcontainers), configured from the environment
(`SOLIA_AI_CONFIG` tiny config, `MODEL_DIR`, `RETRAIN_EVERY_N_TRADES`), with features read back from the
`features` table and a real tiny champion. `N − 1` closed trades start nothing; the `N`-th starts an `n_trades`
run that trains and scores a real candidate. Only the gate's metrics are controlled (candidate = champion ±
lift), so a better candidate is `promoted` (`is_champion` flipped, `/model/status` serves the new version with
`trades_since_last_train` back to 0) and a worse one `rejected` (champion unchanged, candidate row kept); a
rollback restores the prior champion (`decision = rollback`); `POST /model/retrain` with the backend's body
answers `202` + id, recorded as `manual`. The backend's relay of `/api/v1/model/*` is covered by its own
end-to-end tests (`services/backend/Solia.Api.Tests/EndToEnd`). About 25 s.

### Bounded training read

A retrain must not read the whole `features` table: a 48 h replay holds ~44 M rows (~34 GB) and live adds
~0.95 M rows per stream hour. Every training run — `python -m solia_ai.retrain`, `POST /model/retrain`, the
background trigger, `python -m solia_ai.candidate` and `python -m solia_ai.train` — reads through
`solia_ai.db.read_training_features`, bounded by config `data` (CLI `--lookback-days` / `--max-rows` override it):

1. **Lookback window** (`data.lookback_days`, default 2): rows in `[newest − N days, newest]`, `newest` being the
   newest `features` row (not the clock), so a database holding only replayed history works too.
2. **Whole tokens only**: a token with any row before the window is dropped, never cut.
3. **Row cap** (`data.max_rows`, default 4 000 000): when the whole tokens hold more rows, tokens are kept in
   `md5(mint)` order while the running total fits. The subset is deterministic across runs, independent of time
   (it spans the whole window, unlike a "first/last N rows" cut) and mostly stable as data grows (a prefix of
   the same order). With an eligibility filter (#149) the cap applies after it.

Only the model's columns are read (no JSONB), NUMERIC cast to `float8` in SQL, streamed by 200 000 rows with
`mint` / `venue` held as categories until the frame is built (a `str` per row fragments the heap: RSS was ~4×
the frame without it). Invalid bounds — ≤ 0, or a window shorter than `(promotion.windows + 1) ×` the longest
horizon (the gate's held-out windows plus the label embargo; 4 h with the defaults) — fail before the run is
locked or recorded (and at API startup); too few rows still fail on `candidate.min_train_samples`; an empty
table or a window with no whole token fitting the cap is an error. The bounds used (window start/end, rows and
tokens in the window, tokens dropped as partial or over the cap, rows read) are logged and recorded in
`training_events.metrics.data_bounds` + `notes` and in the candidate's `models.metrics.data_bounds`
(`train.py` records them on its row too). The champion and every candidate are scored on this one frame, so the
paired comparison stays fair.

Measured on a 48 h replay (2026-10-29, ai-service image, 2-day window holding ~34–38 M rows / ~140 k tokens;
peak RSS of the process, `ru_maxrss`, read + `config_samples` windows; read times inflated by a replay writing
to the same PostgreSQL):

| `max_rows` | rows read | tokens kept | frame | peak RSS (read) | peak RSS (+ windows) | read time |
|-----------:|----------:|------------:|------:|----------------:|---------------------:|----------:|
| 1 000 000  | 999 043   | 4 215       | 0.45 GB | 1.54 GB | 1.55 GB | 159 s |
| 2 000 000  | 1 999 402 | 8 607       | 0.89 GB | 2.19 GB | 2.42 GB | 306 s |
| 4 000 000  | 3 999 144 | 16 455      | 1.78 GB | 3.31 GB | 4.06 GB | 804 s |

≈ 0.9 GB fixed (imports, the chunk being decoded) + ≈ 0.6 GB per million rows read (+ ~0.2 GB/M for the
windows). The default leaves the rest of the ~16 GB Docker budget to training. Before the bound, every retrain
(the background trigger included) read the full table with its JSONB columns.

### Checkpoint retention

Every retrain writes a checkpoint to `MODEL_DIR`; `solia_ai/retention.py` keeps the volume bounded without breaking
serving or rollback. It runs after each run that recorded a decision (a failed prune is only logged: the run
already committed) and on demand, `python -m solia_ai.retrain prune [--dry-run] [--delete-orphans]`, under the
retrain lock (`--dry-run` only reads: no lock, changes nothing). Kept: the **champion**, the
`retention.rollback_depth` (default 2) **previous champions** a rollback walks back to, and the
`retention.keep_candidates` (default 5) **newest** other models whose checkpoint is still there. Every other
checkpoint file is deleted, but its `models` row stays (version, metrics, history) with `pruned_at` set (migration
0011; the champion can never be pruned, CHECK), so `check_rollback` refuses a rollback to it cleanly. Files are
found as serving finds them (`solia_ai/checkpoints.py`: registered path, else the file name in `--model-dir` /
`$MODEL_DIR`). A model whose file is not found is reported as missing and left unmarked; if no registered
checkpoint at all is found in the directory, the prune is refused (wrong directory). The planned rows are locked
and re-checked before deletion (a model promoted meanwhile is skipped); a file that can't be deleted is reported
and left unmarked. Only files resolving inside `MODEL_DIR` are deleted (symlinks / `..` resolved first): a row
pointing elsewhere is refused, and a file that is (or may be, through the file-name fallback) a kept model's
checkpoint is never deleted. Orphan `*.pt` files (no `models` row, e.g. a crashed run's) are reported and only
deleted with `--delete-orphans`, and only once older than an hour (a training run outside the retrain lock writes
its checkpoint before its row). Refused models and orphans are warned about once in `logs`, not after every run.

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
- **Implementation:** `solia_ai/promotion.py` (pure, config `promotion` in `config.yaml`). Paired
  `(window, seed)` runs, metrics named as in `ablation.run_metrics`. Promote only if the primary
  metric (default `pump/5m/pr_auc`) lifts by > `margin` on average **and** on ≥ `min_win_share` of
  runs, with no ECE / other ROC-AUC·PR-AUC regression beyond tolerance. P&L lift gets a bootstrap
  CI: it can veto (`pnl_veto`, CI entirely < 0), never promote. No champion → first valid candidate
  is promoted (`initial`); missing/NaN metrics (candidate or champion), or runs that are not
  exactly `windows` distinct windows × `seeds` distinct seeds → reject. Only predictive metrics
  (not P&L, not the uncalibrated `raw/` twins) can reject a candidate for being undefined.
  `check_rollback` validates restoring the previous champion (exists, checkpoint present and
  recording its version / feature spec / horizons, feature spec an ordered subset of the code's,
  same horizons).

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

### Hyperparameter & labeling-threshold search

Once the feature set is settled, `solia_ai/search.py` tunes what is left — scored by the ablation
harness (same walk-forward windows, same seeds, paired run for run), so a config only wins on lift
that survives eval-sample and initialization luck.

- **Space**: `seq_len`, `d_model`, `n_layers`, `dropout`, `learning_rate`, and the labels as one
  scale per head over every horizon (`pump_threshold_scale`, `rug_threshold_scale` — the term
  structure across horizons is kept). Each is `{choices: [...]}` or `{low, high[, log]}` (default
  space: `search.DEFAULT_SPACE`, override with `--space file.yaml`); omitted parameters keep the base
  value. `ff_dim` follows `d_model` at the base ratio; `min_seq_len` stays the base one. A space is
  validated before any data is read (`d_model` a multiple of `n_heads`, `seq_len >= min_seq_len`,
  every scaled rug threshold < 1, …).
- **Strategy & budget**: seeded random search (or `--strategy grid` over `choices`). The base config
  always runs first as `baseline`; then at most `--trials` trials in all, and no trial *starts* after
  `--max-minutes`.
- **Fixed metric definitions under a label change.** A threshold change alters the target, so each
  trial *trains and calibrates* on its own labels but is **scored against the base config's labels**
  on the same windows (the report's `settings.eval_labels`): every trial answers "how well does it
  rank the reference events". The objective is therefore a ranking metric — mean ROC-AUC (or
  `--objective pr_auc`) over both heads and every horizon; ECE is reported but never the objective,
  since a model trained on other labels is calibrated for another event.
- **Ranking**: by `mean − std` of the per-run objective (a noisy config doesn't win on luck). The best
  trial is compared to the baseline with the ablation verdict (objective and every metric).
- **Output**: a JSON report (`--out`) and a **proposed** `config.yaml` diff (`--diff-out`), never
  applied. Trials never save a checkpoint, register a model or touch `is_champion`.

```bash
python -m solia_ai.search --trials 30 --max-minutes 240 --windows 3 --seeds 0 1 2     --out reports/search.json --diff-out reports/search.diff
```

## Files (ai-service, built in Sprints 2 & 7)

| File | Role |
|------|------|
| `model.py` | Transformer + prediction heads (pump / rug / return) |
| `dataset.py` | Dataset/DataLoader, sequence windowing, labeling |
| `features.py` | Shared feature spec / parity with pipeline |
| `train.py` | Initial training entrypoint |
| `candidate.py` | Candidate training for a retrain (warm start from champion or scratch, refreshed time split, never registered) |
| `evaluate.py` | Held-out evaluation (predictive metrics + reference-policy P&L) |
| `ablation.py` | Per-feature-group lift across windows & seeds |
| `eligibility.py` | Token eligibility rule (causal, shared with the bot) |
| `eligibility_ablation.py` | Lift of eligibility rules (rule's model vs every-token model on the rule's population) |
| `search.py` | Hyperparameter & labeling-threshold search (proposes a `config.yaml` diff) |
| `inference.py` | Load champion, run predictions (+ ONNX export) |
| `retrain.py` | Continuous-training orchestrator: trade-count trigger, one run at a time, `training_events` audit, promote/reject, rollback, prune |
| `retrain_service.py` | Retrains inside the API: manual start (`POST /model/retrain`), background trade-count trigger, status progress |
| `retention.py` | Checkpoint retention policy: keep champion + rollback chain + newest candidates, mark pruned rows |
| `api.py` | FastAPI app (`/predict`, `/model/status`, `/model/retrain`) |
| `config.yaml` | All hyperparameters & thresholds |

## Evaluation metrics

- pump / rug: ROC-AUC, PR-AUC, precision/recall at the operating threshold, calibration (ECE).
- expected return: information coefficient (IC) / correlation, MAE.
- economic: simulated win-rate & P&L on the validation window **via the policy layer** — the metric that ultimately matters.
