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
- **Mixed precision** optional (not enabled: not measured to help yet, see "Training throughput").
- **Batch feeding** (`training.num_workers` / `pin_memory` / `prefetch_factor`): batches are gathered
  whole (one vectorized gather per batch, no per-sample Python objects), in the training process by
  default (`0`, measured fastest) or by persistent loader workers; `num_workers: auto` = half the CPUs
  (max 4) where workers fork and share the rows copy-on-write (Linux / the container), 0 elsewhere.
  The weights don't depend on the worker count (sampling stays in the training process; the loaders
  have their own seeded RNG).
- **Progress**: every epoch logs epoch / max, train & val loss, duration, samples/s, the candidate's
  ETA and the early-stopping counter (stdout; a retrain also writes it to `logs`, and
  `/model/status` shows the run's current candidate and epoch).
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

### Liquidity floor (#166)

A price read off a near-empty AMM pool is an artifact: one micro-trade moves it by multiples. Such prices
created **fake pump labels** and windows the model can "predict" by recognizing an empty pool that jumps. Live
paper trading picked exactly those (41 of the first 58 trades on 2026-10-30 entered pools with < 1 SOL, #165).
`labeling.liquidity_floor` (`solia_ai/liquidity.py`, SOL in the SOL-quoted pool = `features.liquidity`) is
**venue-aware**. A row's floor is its venue's floor, else `default` (NULL venue included). A row is *tradeable*
when its floor is 0 or its liquidity is ≥ the floor. An unknown liquidity can't be proven above a positive
floor.

- **Windows**: a window whose anchor row is below its floor is not a training / validation / held-out sample.
  It is filtered with the eligibility rule, before the stride.
- **pump / expected_return**: a price in the horizon read off a row below the floor doesn't count. `pump` is the
  highest *counted* price. `expected_return` is the last counted price at or before the horizon end. The anchor
  counts, so it is the fallback when the whole horizon read dust (return 0).
- **rug: unchanged.** A liquidity collapse (into dust or not) and a price crash are still rugs.
- **Recorded with every model**: `hyperparams["liquidity_floor"]`, in the checkpoint and the `models` row,
  reported by `GET /model/status`. A model recorded without one predates the floor (read as disabled). A retrain
  whose champion was trained under another floor logs a note: both models are scored on the config's windows and
  labels. **The bot's liquidity gate (#165) must not trade below its champion's floor**, so that the bot trades
  the population the model learned on: its `Bot:LiquidityFloorSol` defaults mirror `labeling.liquidity_floor`
  (see "Realistic fills" below) — keep them in step.

**Per-venue rule, from data** (6 h of `features`, 2026-10-30, share of consecutive same-venue snapshots whose
price moved ×2 or more):

| venue | < 0.1 SOL | 0.1–1 | 1–10 | ≥ 10 |
|---|---:|---:|---:|---:|
| pumpswap | 1.18 % | 0.42 % | 0.03 % | 0.02 % |
| raydium | 0.34 % | 0.09 % | 0.02 % | 0.00 % |
| meteora | 0.64 % | 0.09 % | 0.07 % | 0.09 % |
| pumpfun (curve) | 0.73 % | 2.21 % | 0.34 % | 0.05 % |

On the AMMs the artifacts stop at ~1 SOL, hence `default: 1.0`. Bonding curves (`pumpfun`, `raydium-launchpad`,
`meteora-launchpad`, `bonk`) price off **virtual** reserves (pumpfun: ~30 SOL virtual + the real SOL), so a
micro-trade can't multiply the price. Their jumps under 1 SOL are real early-life moves and dumps. They get a
floor of 0. 65 % of pumpfun rows hold < 1 SOL of real SOL, and an AMM-style floor would have dropped most of the
curve population by mistake.

**Diagnostic by liquidity bucket** (`python -m solia_ai.liquidity_report`, read-only; buckets of the anchor:
`amm` / `curve` × `<1` / `1-10` / `>=10` SOL):

```bash
python -m solia_ai.liquidity_report --champion --labels none --out /app/models/reports/liquidity-none.json
# several models on the same windows (after every model's data), labeled under the config's floor:
python -m solia_ai.liquidity_report --champion --checkpoint /app/models/x.pt --labels config --out ...
#   --no-sweep (skip the per-bucket entry-rule sweep)  --bootstrap-samples 200  --device cuda (default if available)
```

Champion `solia-20261030T012910179875Z`, 2 406 074 held-out windows, labels without the floor, entry rule 15m
`pump ≥ 0.25, rug ≤ 0.2` (report: [`reports/liquidity-none-solia-20261030T012910179875Z.txt`](reports/liquidity-none-solia-20261030T012910179875Z.txt)):

| bucket | % windows | % pumps | pump PR-AUC 15m | base 15m | % rule trades | precision | median return | % rule P&L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 100 | 100 | 0.258 | 0.064 | 100 (4 112) | 0.28 | −1.5 % | 100 |
| **amm ≥ 10** | 25.6 | 60.8 | **0.342** | 0.089 | 38.9 | **0.44** | **+23.4 %** | 80.2 |
| amm 1–10 | 5.9 | 2.1 | 0.141 | 0.029 | 3.5 | 0.15 | −12.7 % | −3.5 |
| amm < 1 | 13.9 | 7.9 | 0.167 | 0.065 | **43.9** | 0.19 | −12.1 % | 24.1 |
| curve < 1 | 37.6 | 15.2 | 0.207 | 0.052 | 1.3 | 0.33 | −3.3 % | 1.7 |
| curve 1–10 | 10.0 | 5.2 | 0.067 | 0.026 | 3.8 | 0.00 | −1.0 % | −0.3 |
| curve ≥ 10 | 7.0 | 8.8 | 0.148 | 0.047 | 8.5 | 0.18 | −8.4 % | −2.1 |

- **The model's quality does not come from dust pools.** It is best on realistic AMM pools (≥ 10 SOL: PR-AUC
  0.34 against 0.26 overall, entry precision 44 %, 80 % of the rule's P&L).
- **The entry rule wastes 44 % of its entries in AMM pools under 1 SOL.** They are half as precise, with a
  negative median, and their "gains" are the artifact prices. The dust bucket's pump base rate (6.5 %) is more
  than twice the 1–10 SOL one (2.9 %): those are the fake pumps the floor removes.
- Curve tokens under 1 SOL keep a usable signal (PR-AUC 0.21), which confirms the curve exemption.

**Retrain with the floor** (same bounded read, 4 M rows, scratch, `solia_ai.candidate`; 1 427 105 training
windows, best epoch 1: the #160 overfitting pattern) vs the champion, both scored on the same held-out windows
**labeled under the floor**:

754 375 shared held-out windows (after both models' data; AMM < 1 SOL anchors are dropped by the floor; reports:
[`champion`](reports/liquidity-compare-solia-20261030T012910179875Z.txt),
[`floor candidate`](reports/liquidity-compare-solia-floor-166.txt)):

| pump PR-AUC | 30s | 1m | 5m | 15m | 1h |
|---|---:|---:|---:|---:|---:|
| all — champion | 0.213 | 0.217 | 0.284 | 0.215 | 0.108 |
| all — **floor candidate** | **0.254** | **0.257** | **0.307** | **0.239** | **0.144** |
| amm ≥ 10 — champion | 0.214 | 0.227 | 0.321 | 0.255 | 0.129 |
| amm ≥ 10 — **floor candidate** | **0.254** | **0.262** | **0.336** | **0.271** | **0.180** |

Rug PR-AUC also rises on every horizon (all, 15m: 0.829 → 0.842; amm ≥ 10, 15m: 0.738 → 0.748).

| 15m entry rule `pump ≥ 0.25, rug ≤ 0.2` | trades | precision | win | mean / median return |
|---|---:|---:|---:|---:|
| champion, all | 752 | 0.326 | 0.57 | +0.245 / +0.094 |
| floor candidate, all | 984 | 0.236 | 0.49 | +0.042 / −0.008 |
| champion, amm ≥ 10 | 536 | 0.410 | 0.65 | +0.319 / +0.208 |
| floor candidate, amm ≥ 10 | 940 | 0.238 | 0.50 | +0.036 / 0.000 |

- **Predictive metrics (primary) favour the floor candidate** on the realistic bucket and overall. The
  candidate's rule trades sit almost entirely in realistic pools (95.5 % in AMM ≥ 10, none in curve dust), where
  the champion's rule still spread 29 % of its trades over thinner pools.
- **The fixed entry thresholds don't transfer.** The candidate's probability scale differs (it fires 1.3× more
  often at 0.25), so its rule P&L at the champion's thresholds is not comparable. The entry sweep must be re-run
  for any new champion (see "Choosing the entry rule").
- **Caveats.** One seed, one held-out slice. The candidate is also trained on fresher data than the champion
  (2026-10-30 01:29), so this comparison mixes the floor with recency. The promotion gate (windows × seeds,
  same config for both) is what decides a promotion. The floor ships enabled, so the next retrain trains its
  candidate under it, and the gate scores the champion on the floored labels.

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

### Realistic fills (#165)

The first hours of live paper trading booked +13 SOL on 10, almost all fake: 41 of 58 trades entered AMM pools
holding < 1 SOL and the biggest "wins" were 1–3.7 SOL positions filled at prices read off 0.00–0.03 SOL pools.
The bot now fills against the pool (`Solia.Bot` `Trading/Fills.cs`; amounts in **SOL**; `Bot:RealisticFills`,
default on — off is the idealized simulator, kept for before/after comparisons only):

| rule | config (default) | |
|---|---|---|
| **Liquidity gate** | `LiquidityFloorSol` (`default: 1`, `pumpfun` / `raydium-launchpad` / `meteora-launchpad` / `bonk`: 0) | no entry when the pool's real SOL (`features.liquidity`) is unknown or below its venue's floor (`default` for unlisted / unknown venues). Same rule and values as the training floor ("Liquidity floor" above): AMM prices under ~1 SOL are artifacts; bonding curves price off virtual reserves, so their real SOL may be ~0 while the price is valid. |
| **Pool-sized positions** | `MaxPoolFraction` (0.02) | size ≤ 2 % of the pool's SOL reserve `R`, on top of the balance caps; below `MinPositionSize` → skipped. `R` = real SOL + the venue's `VirtualSol` (curves: 30, Pump.fun's initial virtual SOL; AMMs: 0). |
| **Slippage** | — | constant product: a buy of `S` (after the fee, `n = S(1−f)`) gets `n/P · R/(R+n)` tokens; a sale worth `v` at the mark gets `v · R/(R+v)` SOL. Entry price recorded = all-in price per token. |
| **Fees** | `FeePct` per venue, per side (`default` 0.25 %, `pumpfun` 1.25 %, `pumpswap` 0.30 %, launchpads 1 %) | from the venues' published schedules (approximate); charged on entry and exit, so in `pnl`. |
| **Exit bound** | — | proceeds `< R` at exit: a pool drained to 0 SOL (or unknown) pays nothing, whatever its last price. |

Approximations, chosen pessimistic: each side is priced against the pool as observed, which never contains the
bot's own paper trade, so a round trip pays the impact twice where on-chain it would partly unwind (bounded by
the 2 % cap, ~2 % per side). TP/SL/dead-feed-stop triggers still read the observed mark; the fill applies on top, so a
take-profit can realize less than its level. `Equity` marks open positions at the mark (no exit impact). The
Kelly sizing doesn't price the costs in. Out of scope: latency, MEV / priority fees, order splitting.

**Before/after on the live period** (2026-10-30 14:06–20:05, 104 closed trades, recomputed from the stored
trades and their pools at entry/exit: [`reports/paper-fills-165.sql`](reports/paper-fills-165.sql) →
[`.txt`](reports/paper-fills-165.txt)):

| bucket (liquidity at entry) | trades before → after | P&L before → after (SOL) | win rate before → after |
|---|---:|---:|---:|
| AMM < 1 SOL | 67 → 0 | −2.51 → 0 | 0.16 → — |
| AMM 1–10 | 5 → 5 | −1.71 → −0.19 | 0 → 0 |
| AMM ≥ 10 | 31 → 31 | +4.41 → −4.65 | 0.39 → 0.32 |
| curve 1–10 | 1 → 1 | −0.00 → −0.04 | 0 → 0 |
| **all** | **104 → 37** | **+0.19 → −4.89** | **0.22 → 0.27** |

Two thirds of the trades are gated out. In AMM ≥ 10 SOL, three trades exited into a pool pulled to 0 SOL in the
same snapshot (+0.74 → −5.77 SOL); the 28 others go from +3.68 to +1.11 SOL (impact, fees, pool-capped sizes). The
sample is far too small to judge the model; the point is that paper P&L can no longer be faked by dust pools.

### Stale marks on exit (#176)

A mark is the mint's market while it is at most `Bot:MaxMarkAgeSeconds` (60) old; an entry needs one. A **signal or
timeout exit** on an older mark fills:

| feed state at the exit (event time) | fill at | why |
|---|---|---|
| mark fresh | the mark | — |
| mark stale, **feed live** — some mint marked within `Bot:MaxFeedSilenceSeconds` (60) | the last mark | the mint went quiet; a bonding curve only moves on trades, so its price *is* its last trade |
| mark stale, **feed silent** — no mint marked within it | `min(last mark, stop-loss)` | our feed died: assume the unobserved market hit the stop, never book a stale break-even |

Realistic fills apply on top in every case. The previous rule used the stop for every stale mark and booked
phantom stops on quiet mints: on the live DB (2026-10-30) 54 of 74 timeouts filled at the stop, 49 of them with a
last price unchanged vs entry (≈ −28 SOL). The new option changes the bot-session fingerprint, so P&L before and
after the fix sits in separate sessions (#171).

### Choosing the entry rule

The entry thresholds are chosen **from data**, not by hand: calibrated probabilities of a rare event
are low (pumps are 0.6 % of 1m windows, 1.6 % of 15m ones), so the pre-calibration rule
`pump ≥ 0.6, rug ≤ 0.2` fired once in ~880 k held-out windows and the reference P&L was 0.
`solia_ai/entry_sweep.py` scores a checkpoint once over its held-out windows (the retrain's bounded
read, windows after the checkpoint's data) and reports, per horizon and per rule — absolute
`pump_prob ≥ P` × `rug_prob ≤ R` thresholds, and the top-X % of `pump_prob` (with its implied
threshold) — n_trades, **precision** (share of real pumps) and its lift over the base rate, win
rate, mean / median realized return, total P&L and a 95 % bootstrap CI of the mean, against the
buy-everything baseline. Read-only: nothing is written but the report.

**Current rule** (champion `solia-20261030T012910179875Z`, 886 797 held-out windows; report in
[`reports/entry-sweep-solia-20261030T012910179875Z.txt`](reports/entry-sweep-solia-20261030T012910179875Z.txt)):
**horizon 15m, `pump_prob ≥ 0.25`, `rug_prob ≤ 0.2`** — the bot's `Bot:Horizon` / `MinPumpProb` /
`MaxRugProb` defaults and the `ReferencePolicy` defaults (`evaluate.py`), kept in step.

| 15m rule | trades | precision (lift) | win rate | mean / median return | mean CI |
|---|---:|---:|---:|---:|---:|
| buy everything | 534 039 | 0.016 (1×) | 0.45 | +0.018 / 0.000 | [+0.017, +0.020] |
| `pump ≥ 0.25, rug ≤ 0.2` (**chosen**) | 672 | 0.384 (24.7×) | 0.59 | +0.504 / +0.094 | [+0.41, +0.60] |
| `pump ≥ 0.3, rug ≤ 0.2` | 186 | 0.500 (32.1×) | 0.60 | +0.574 / +0.112 | [+0.42, +0.73] |
| `pump ≥ 0.25, rug ≤ 0.3` | 1 429 | 0.300 (19.3×) | 0.57 | +0.445 / +0.028 | [+0.37, +0.52] |
| `pump ≥ 0.25`, no rug gate | 4 161 | 0.336 (21.5×) | 0.54 | +0.458 / +0.026 | [+0.41, +0.51] |

Why this one — **predictive quality first, P&L second**:

- **Precision.** 15m has the highest precision among rules with hundreds of trades (38 % of entries
  are real pumps, i.e. a ≥ +100 % move within 15 min); the 1m rules top out at ~22 % and the 5m ones
  at ~32 %. 1h is unusable as is (`pump_prob` max 0.16, rug gates collapse precision).
- **Enough trades.** `pump ≥ 0.3` is more precise (0.50) but on 186 windows — consecutive windows of
  the same token overlap, so the independent events are far fewer; 0.25 keeps 3.6× the trades for
  a precision still 25× the base rate. Its neighbours (rug ≤ 0.3, no rug gate) stay close, so the
  choice is not a lucky grid point.
- **P&L, heavy tails.** The mean is driven by a few large winners (labels capped at +1000 %), hence
  the median (+9 %) and the CI are read alongside it; both clear the baseline by far.
- **Rug gate.** At 15m a *tight* rug gate (≤ 0.1) destroys precision (0.9×): the tokens with almost
  no rug risk are mostly dead ones that don't pump either. 0.2 is the sweet spot.
- **Horizon = position timeout.** 15m is the span of the bot's default 900 s position timeout.

**Confidence.** `confidence = 200 · max(|pump − 0.5|, |rug − 0.5|)` is a margin, not a hit rate: with
both heads mostly near 0 it is mostly high. `rug ≤ 0.2` alone implies `confidence ≥ 60`, so the
bot's `MinConfidence` 60 never blocks an entry the rug gate allows; loosen `MaxRugProb` above 0.2 and
it starts blocking entries whose heads both sit near 0.5 — lower it then.

**Caveats.** The bootstrap treats windows as independent; overlapping windows of one token are not,
so the CIs are optimistic — weigh trade counts accordingly. The returns are realized at the horizon
(unit size, no fees/slippage, no TP/SL), a yardstick for the entry rule, not the bot's P&L.

**Re-run after every promotion.** The probability scale belongs to one model: a new champion (other
data, calibration, loss weights) can move it, and a stale threshold silently stops (or floods)
entries. After a promotion:

```bash
python -m solia_ai.entry_sweep --champion --out /app/models/reports/entry_sweep.json   # + .txt table
#   --pump-thresholds 0.1,0.2,0.3  --rug-thresholds none,0.3,0.2  --top-fractions 0.001,0.01
#   --min-trades 100  --min-confidence 60 (the bot's gate)  --all-rules (print rules under min_trades)
```

Each horizon's `recommended` rule (most precise with ≥ `min_trades` trades whose mean-return CI
clears the baseline) is a starting point, not a deployment: pick by the criteria above, then update
`Bot:Horizon` / `Bot:MinPumpProb` / `Bot:MaxRugProb` (config, or the defaults in `BotOptions` /
`BotHostOptions`) **and** `DEFAULT_POLICY_HORIZON` / `DEFAULT_PUMP_THRESHOLD` /
`DEFAULT_RUG_THRESHOLD` in `evaluate.py` together, so the promotion gate's P&L guard measures the rule
the bot trades.

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

Only the model's columns are read (no JSONB), NUMERIC cast to `float8` in SQL (model-only features then held as
float32, #160), streamed by 200 000 rows with
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

### Training throughput

Measured on the retrain #1 data (2026-10-29: the bounded read, 3 997 258 rows of 16 020 tokens →
3 803 692 windows; RTX 4070, 6 CPUs, `docker-compose.gpu.yml`), with `python -m solia_ai.throughput
--workers 0 2 3 4 --per-item --batches 300` (`solia_ai/throughput.py`: no writes) in the ai-service
container. Model and batch size are the shipped ones (`d_model` 128, 4 layers, batch 256).

| Phase | Before (#152) | After |
|---|---|---|
| Read `features` (bounded, #151) | "a few minutes" | 653 s (unchanged by #152) |
| Dataset build (windows + labels) | ~15 min, 1 core at 100 % — and redone by every candidate | **17 s** (vectorized, ~50×), built once per retrain |
| Training feed, samples/s (GPU utilisation) | 9 400 (57 %) — per-sample loader | **10 360 (70 %)** — batched, 0 workers |
| Candidates trained without a champion | 3 (one per gate seed) | **1** |
| Peak RSS (build + training) | 4.9 GB build, 6.6 GB training | 3.9 GB after the build, 7.4 GB peak (benchmark process, all loaders) |
| Per-epoch progress | none until a candidate ends | every epoch: loss, duration, samples/s, ETA (stdout, `logs`, `/model/status`) |
| Total retrain | > 3 h, interrupted (retrain #1, candidate 1 of 3 still training) | measured at the next real retrain (the per-epoch log gives it) |

Loader workers (same run, batched loader):

| `num_workers` | samples/s | GPU util. | Σ RSS (MB) | Σ PSS (MB) |
|---|---|---|---|---|
| 0 | 10 360 | 70 % | 4 463 | 4 436 |
| 2 | 8 800 | 55 % | 12 755 | 4 695 |
| 3 | 9 470 | 59 % | 16 948 | 3 392 |
| 4 | 8 850 | 58 % | 21 318 | 2 812 |

- **Workers don't multiply memory**: summed RSS counts the shared rows once per process, but the
  summed PSS (shared pages split between their sharers) stays at about one copy — the forked workers
  read the rows copy-on-write, never copy them.
- **Workers don't help either**: gathering a whole batch costs far less than the model's step, so the
  GPU waits on the training step's own launch overhead (~70 % utilisation at batch 256), not on data.
  Hence the shipped `num_workers: 0`. The next levers are a larger batch or mixed precision — both
  model/hyper-parameter changes, to be measured on their own (AMP is not enabled).
- The wall-clock gain of a retrain comes from the build (15 min → 17 s, and no longer rebuilt per
  candidate) and from training one candidate instead of three when there is no champion yet.

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
  is promoted (`initial`) — and since seeds only matter for a paired comparison, a retrain without
  a champion trains **one** candidate (noted in the run); missing/NaN metrics (candidate or
  champion), or runs that are not exactly `windows` distinct windows × `seeds` distinct seeds
  (× 1 seed without a champion) → reject. Only predictive metrics
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

- **Space**: `seq_len`, `d_model`, `n_layers`, `dropout`, `learning_rate`, `weight_decay`, the
  training-window stride (`window_stride_seconds`) and row budget (`max_rows`) — see *Overfitting*
  below — and the labels as one scale per head over every horizon (`pump_threshold_scale`, `rug_threshold_scale` — the term
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
- **Fit diagnostics**: per trial, the best epoch (1-based) and the validation − train loss gap at it
  (`fit/*` in each run's metrics), and for the baseline and the best trial the rug ROC-AUC / PR-AUC
  per walk-forward slice (`drift`: time drift of the rug head).
- **Data**: the same bounded read as a retrain (`--lookback-days` / `--max-rows`, default config
  `data`), widened to the largest `max_rows` of the space.
- **Output**: a JSON report (`--out`) and a **proposed** `config.yaml` diff (`--diff-out`), never
  applied. Trials never save a checkpoint, register a model or touch `is_champion`.

```bash
python -m solia_ai.search --trials 30 --max-minutes 240 --windows 3 --seeds 0 1 2     --out reports/search.json --diff-out reports/search.diff
```

### Overfitting: window stride & regularization (#160)

**Symptom (retrain #2, 4 M rows, RTX 4070).** Validation loss is best at epoch 1 (0.188), then rises every epoch
(0.215, 0.289 … 0.497) while the train loss falls (0.27 → 0.08); early stopping keeps epoch 1 of 9. The read kept
15 055 of 161 531 tokens: the 4 M-row cap is filled by ~250 near-identical 10 s snapshots per token, and every
snapshot was a training window. The model sees few distinct situations many times and memorizes them. The rug
head also drifts across the held-out (15m ROC-AUC 0.924 → 0.886).

**Window stride** (`data.window_stride_seconds`, `solia_ai/dataset.py`). A token's time since its first row is
cut into `stride`-second buckets, and only the first eligible window ending in each bucket is kept: at most one
window per stride of the token's life. Window *ends* are thinned **before labeling**. Nothing else changes: each
window keeps its full 10 s-cadence history (the model and inference are untouched) and its labels. The rule is
deterministic and causal, and train / validation / held-out all go through `config_samples`, so they are thinned
identically. The temporal split and its embargo apply to the thinned windows as before. `0` (default) keeps a
window per snapshot.

**Why the stride alone can't shrink the read.** A window holds its last `seq_len` = 64 rows (640 s), so with a
30–60 s stride (3–6 rows) every row still belongs to some window's history. A real reduction would need a coarser
input cadence, which means a train/serve change (out of scope). The stride cuts the per-*window* cost: labeling,
stored samples and, above all, the **epoch time** (∝ windows: ~1/3 at 30 s, ~1/6 at 60 s). The same epoch budget
therefore covers 3–6× the rows. Memory is per *row*, so the build and the read were also slimmed so that the same
memory holds more rows:

- `build_samples` never copies the frame: rows are filtered and ordered through an index (`lexsort`), and each
  column is gathered already ordered. The samples are bit-for-bit identical (`tests/test_dataset_build.py`).
- `build_feature_matrix` fills its float32 matrix one column at a time, with no `concat` copy of the frame and no
  full float64 intermediate, and only for the rows that hold a sample.
- The bounded read holds every model-only feature as float32 (the model casts them to float32 anyway, with the
  same rounding: `tests/test_features.py`). Labels and eligibility inputs (`price`, `liquidity`,
  `unique_traders`, `volume`, `token_age_sec`) stay float64.

Measured (synthetic frame shaped like the read: 2 000 tokens × 250 snapshots, `tracemalloc`; bytes per row
read, frame + peak of `config_samples`):

| | frame | build peak | total | windows |
|---|---:|---:|---:|---:|
| before #160 | 455 | 1 050 | 1 505 | 480 k |
| after, stride 0 | 331 | 741 | 1 072 | 480 k |
| after, stride 30 s | 331 | 662 | 993 | 162 k |
| after, stride 60 s | 331 | 640 | 971 | 82 k |

At equal memory, `data.max_rows` therefore covers ~1.5× the rows (tokens). Combined with the ~3–6× cheaper epochs,
the row budget can grow with the stride. Re-measure the real RSS with `python -m solia_ai.throughput --max-rows …`
before raising `data.max_rows` in production. The 4 M-row read peaked at ~4 GB of the ~16 GB Docker budget (see
*Bounded training read*).

**Sweep.** Every setting is compared by `solia_ai.search` on the same walk-forward held-out windows × seeds, with
predictive metrics primary (pump / rug PR-AUC per horizon, ECE). A stride or row-budget trial trains on its own
windows but is scored on the baseline's windows. A `max_rows` trial trains on exactly the tokens a read capped
there would keep (whole tokens in `md5(mint)` order, `search.budget_tokens`), out of one frame read at the
largest budget. Two shipped spaces (`solia_ai/sweeps/`, run in the GPU container; pause Windows Update for
overnight runs):

```bash
# Stage 1 — data: stride x row budget vs today's (stride 0, 4 M rows). 4 trials + baseline, ~2 x the
# baseline's epoch cost in all; ~6-7 h with 2 windows x 2 seeds.
python -m solia_ai.search --space overfitting-data --strategy grid --trials 5 --windows 2 --seeds 0 1     --objective pr_auc --out /app/models/reports/overfitting-data.json
# Stage 2 — regularization (dropout, weight decay, learning rate, width) on stage 1's stride / budget,
# set in the config under test (SOLIA_AI_CONFIG) or with --max-rows.
python -m solia_ai.search --space overfitting-regularization --trials 10 --windows 2 --seeds 0 1     --objective pr_auc --out /app/models/reports/overfitting-regularization.json
```

The report gives, per trial, the objective (mean ± std), the best epoch and the val − train gap, and, for the
baseline and the best trial, the rug ROC-AUC / PR-AUC per held-out slice (drift).

**Results & chosen defaults.** *Pending the overnight runs above.* Until they are measured, the shipped config
keeps the pre-#160 behaviour (`window_stride_seconds: 0`, `max_rows` 4 M, `dropout` 0.1, `weight_decay` 0.01,
`learning_rate` 3e-4, `d_model` 128): defaults are measured, not guessed. Record here each stage's table (trial,
PR-AUC per head/horizon, ECE, best epoch, gap), the rug drift per slice, and the chosen values. Also state whether
the best epoch moved past 1 and the gap narrowed, or why not.

## Files (ai-service, built in Sprints 2 & 7)

| File | Role |
|------|------|
| `model.py` | Transformer + prediction heads (pump / rug / return) |
| `dataset.py` | Dataset/DataLoader, sequence windowing (window stride), labeling |
| `features.py` | Shared feature spec / parity with pipeline |
| `train.py` | Initial training entrypoint |
| `candidate.py` | Candidate training for a retrain (warm start from champion or scratch, refreshed time split, never registered) |
| `evaluate.py` | Held-out evaluation (predictive metrics + reference-policy P&L) |
| `entry_sweep.py` | Entry-rule sweep: precision / win rate / P&L per threshold rule on held-out windows (chooses the bot's entry gates) |
| `ablation.py` | Per-feature-group lift across windows & seeds |
| `eligibility.py` | Token eligibility rule (causal, shared with the bot) |
| `eligibility_ablation.py` | Lift of eligibility rules (rule's model vs every-token model on the rule's population) |
| `liquidity.py` | Venue-aware liquidity floor for labels & windows (recorded with the model) |
| `liquidity_report.py` | Held-out metrics & entry rule split by anchor pool liquidity (one or several models, same windows) |
| `search.py` | Hyperparameter, regularization, stride & labeling-threshold search (proposes a `config.yaml` diff); spaces in `sweeps/` |
| `inference.py` | Load champion, run predictions (+ ONNX export) |
| `retrain.py` | Continuous-training orchestrator: trade-count trigger, one run at a time, `training_events` audit, promote/reject, rollback, prune |
| `retrain_service.py` | Retrains inside the API: manual start (`POST /model/retrain`), background trade-count trigger, status progress |
| `throughput.py` | Training throughput benchmark: dataset build time, samples/s per loader setting, GPU use, memory (no writes) |
| `retention.py` | Checkpoint retention policy: keep champion + rollback chain + newest candidates, mark pruned rows |
| `api.py` | FastAPI app (`/predict`, `/model/status`, `/model/retrain`) |
| `config.yaml` | All hyperparameters & thresholds |

## Evaluation metrics

- pump / rug: ROC-AUC, PR-AUC, precision/recall at the operating threshold, calibration (ECE).
- expected return: information coefficient (IC) / correlation, MAE.
- economic: simulated win-rate & P&L on the validation window **via the policy layer** — the metric that ultimately matters.
