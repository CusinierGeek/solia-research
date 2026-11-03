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
- **expected return**: Huber (smooth L1) on the head's bounded output `softplus(x) - 1` (always `> -1`, like
  the `>= -1` target), so training, evaluation and serving share one parametrisation.

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

**The "inverted" 1 h head of retrain #3's candidate (#179): model-side, labels correct.** The candidate
`solia-20261030T203422233817Z` (first warm start under the floor) scored precision 0 at 1 h on `pump ≥ 0.25,
rug ≤ 0.2`. A read-only analysis of its held-out top 1 % 1 h windows (`p ≥ 0.235`, report:
[`reports/inverted-1h-179.txt`](reports/inverted-1h-179.txt), script
[`.py`](reports/inverted-1h-179.py)):

| top 1 % 1 h, rug gate | windows | tokens | 1 h precision |
|---|---:|---:|---:|
| none | 8 307 | 362 | **0.268** (base 0.038, ×7.1) |
| rug ≤ 0.5 | 3 719 | 122 | 0.245 |
| rug ≤ 0.3 | 666 | 28 | 0.027 |
| rug ≤ 0.2 | 417 | **11** | 0.000 |

- **The labels follow the documented rule.** All 8 307 windows were relabeled independently, one window at a
  time: 8 307 / 8 307 match. 73 % of them see their pool drained below the floor within the hour (median first
  drained read at 12 min). The floor removes those dust prices from `pump` as designed: 1 301 pumps that would
  exist without the floor are gone (dust reads up to 1.9·10⁸ × the anchor). `rug` is kept (81 %). The hypothesis
  that the floor drops a spike-then-drain token's post-drain prices while keeping its rug holds, but the labels
  are consistent: such a window is a rug, and a pump only if it pumped while tradeable.
- **The pump head is not inverted.** The pump head alone ranks: the top 1 % hits ×7 the base rate, over 362
  tokens.
- **The 1 h rug head's low tail is where it fails.** Among high-pump windows, `rug ≤ 0.2` isolates 11 tokens.
  353 of the 417 windows (85 %) come from **one token**, `G4vG…pump`: a pumpswap pool of 150–200 SOL with a flat
  price, active for an hour, then emptied in one dump (≈10 600 sells in two snapshots; price ×0.01; liquidity
  202 → 2.0 SOL). 2 SOL is above the 1 SOL floor, so that crash is a counted price: a real rug, labeled right.
  Nothing in the window history announced it.
- **Overfitting signal.** On its own training windows the same score region has 1 h precision 0.58 (mean
  return +2.03); held out, 0.27.
- **Handed to #172.** A rule must also report its number of distinct tokens. A precision built on 11 tokens,
  85 % from one, is an anecdote. Judge the 1 h rug gate on rug calibration (held-out, per token), not on one
  slice. The floor values are not in question.

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

`expected_return` is always finite and `>= -1` (a long position cannot lose more than 100 %): the return
head is bounded by construction, and the contract clamps it as well (NaN -> 0) so a checkpoint saved before
that head — loaded with its legacy raw output — never emits a value the backend would drop.

## Policy layer (not the model)

Buy/sell/hold, position **sizing**, dynamic **take-profit / stop-loss**, and the entry threshold
are **rules**, not model outputs — implemented in the paper-trading bot (Sprints 3–4) and
configurable without retraining. They consume the model's predictions, for example:

- enter when `pump_prob ≥ P` **and** `rug_prob ≤ R` (and liquidity/age gates pass);
- size (`Bot:Sizing`, #219):
  - **`Fixed` (default):** `FixedPositionFraction` (0.1) of the available balance for every entry with upside (`expected_return ≥ MinTakeProfitPct`), capped by `MaxPositionFraction`, the pool cap and the dust floor.
    The 2026-11-02 rule has no rug gate because high-`rug_prob` entries proved good (win rate 0.71). Kelly would read `rug_prob` as the chance of losing the trade and scale by confidence, which sits near 0 on exactly those entries.
    It would therefore refuse or shrink most of them: in session #3's first 40 min, 1 trade out of 10 entry signals, at confidence 8.9.
  - **`Kelly`:** fractional Kelly on `expected_return` (capped at TP) vs `rug_prob` (loss capped at SL), scaled by confidence, capped.
  - A declined entry of a buy signal is logged with its reason (`entry declined: <reason> for <mint> (pump, rug, confidence, expected return)`);
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
Kelly sizing doesn't price the costs in. Latency, network fees and failed fills: "Execution realism" below. Out of
scope: MEV, order splitting.

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

### Execution realism (#175)

Realistic fills price the pool; the bot was still optimistic on **timing** and **network costs**: an order filled at
the very snapshot its prediction was made on, paid no Solana fees and never failed. On a pumping token the seconds an
order spends in flight move the price most, and at the bot's sizes (pool cap 2 %, often < 1 SOL) fixed per-transaction
costs are a material share of each trade. All knobs are `Bot:*`, part of the session fingerprint (changing one opens a
new session, #171), and each one set to 0 disables it — all at 0 is the engine as it was before (regression-tested).

| rule | config (default) | |
|---|---|---|
| **Entry latency** | `EntryLatencyMs` (3 000) | an entry decided at `t` fills at the mint's **first mark at or after `t + latency`**, against that mark's pool; TP/SL are set from that fill mark. Gates, sizing and the pool cap are read at decision time. While pending, the order holds its size and a slot (`MaxConcurrentPositions` counts it; the mint takes no other entry). No mark within `MaxMarkAgeSeconds` after `t + latency` → **abandoned** (no trade, size released, no fee). |
| **Exit latency** | `ExitLatencyMs` (2 000) | a TP/SL/signal/timeout trigger sends the exit; it fills at the first mark at or after `trigger + latency`, with the trigger's reason. Further triggers are ignored meanwhile (no second exit). No mark within `MaxMarkAgeSeconds` after it → the stale-mark rule ("Stale marks on exit" above). |
| **Network fee** | `NetworkFeeSol` (0.0002) | per transaction, any size, filled or failed: base fee 0.000005 + a ~0.0001 priority fee + a ~0.0001 Jito tip (typical retail-bot settings, 2026, approximate). |
| **Account rent** | `AccountRentSol` (0.00203928) | charged once on entry: the rent-exempt minimum of a 165-byte SPL token account, modelled as **never refunded** — closing the account after the sale refunds it but costs another transaction, and bots commonly skip it. Set it to the net of a close to model closing. |
| **Slippage tolerance** | `MaxSlippagePct` (0.15) | each order is quoted at decision time (the fill at the decision mark). An entry filling more than 15 % above its quote — or into a pool with no SOL left — **fails**: no position, the fee is still paid. An exit filling more than 15 % below its quote fails, pays the fee and is re-sent, quoted at what it now sees, one latency later; `MaxExitRetries` (3) bounds it, after which the exit fills at any price — an exit is never dropped. Default from the 10–20 % retail memecoin bots use. |

**Latency breakdown** (the default 3 s entry): the snapshot is aggregated and written (~1 s), the bot polls it
(`Bot:PollInterval` 2 s → ~1 s on average), the ai-service scores it (~0.1 s on GPU), then the transaction is sent and
included (~1 s, 2–3 slots). An exit on a price trigger skips the scoring and is seen through the same feed: ~2 s.

**Accounting.** `pnl = Size · exit / entry − size − entry cost − exit cost`; entry cost = fee + rent, exit cost = one
fee per attempt. A failed entry's fee isn't a trade: it is booked in the engine's realized P&L (`RealizedPnl`,
`ReplayResult.RealizedPnl`) and logged. Sizing uses the balance net of a round trip's fixed costs
(`2 · NetworkFeeSol + AccountRentSol` ≈ 0.0024 SOL), and `MinPositionSize` (now 0.01) must cover them — a floor below
would let the smallest trade lose more than its size (validated; ~24 % of a minimum position at the defaults).

**Visibility.** `logs` (service = bot) get `entry failed` (reason `slippage` / `empty pool`, quote vs fill),
`exit failed` (attempt n) and `entry abandoned`; `position opened` / `closed` carry `entryCost` / `exitCost`. The
engine counts `FailedEntryCount`, `FailedExitCount` and `AbandonedEntryCount` — in `ReplayResult` and in the bot's
`bot stopping` row. They aren't trades, so `/stats/*` (computed from `trades`) don't count them.

**Event clock.** Latency is applied on event time, never wall time, so a `ReplayHarness` run stays reproducible: a
pending order fills on a later tick, or expires when an event (or the live host's idle sweep, `TimeoutSweepSeconds`)
moves the clock past its window. The live feed's `ReorderWindow` (2 s) and `MaxEventLagSeconds` (5 s) don't eat into
the latency: an event applied late keeps its own timestamp for the "mark at or after `t + latency`" test, and a late
prediction is decided at the (later) clock time — so the simulated latency is always at least the configured one.

Out of scope: MEV / sandwiches and same-block competition (could be approximated as extra latency or slippage),
dynamic priority fees, the same latency in the ai-service's reference policy / entry-rule sweep.

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

**Current rule** (champion `solia-20261102T151927246570Z`, promoted by retrain #5 on 2026-11-02; 288 780
held-out windows; report in
[`reports/entry-sweep-solia-20261102T151927246570Z.txt`](reports/entry-sweep-solia-20261102T151927246570Z.txt)):
**horizon 15m, `pump_prob ≥ 0.4`, no rug gate (`MaxRugProb` 1), no confidence gate (`MinConfidence`
0)**. These are the bot's `Bot:Horizon` / `MinPumpProb` / `MaxRugProb` / `MinConfidence` defaults
and the `ReferencePolicy` defaults (`evaluate.py`), kept in step.

| 15m rule | trades | precision (lift) | win rate | mean / median return | mean CI |
|---|---:|---:|---:|---:|---:|
| buy everything | 162 336 | 0.041 (1×) | 0.47 | +0.018 / 0.000 | [+0.016, +0.020] |
| `pump ≥ 0.25, rug ≤ 0.2` (previous rule) | < 100 (below `min_trades`) | — | — | — | — |
| `pump ≥ 0.15, rug ≤ 0.2` (its loosest neighbour) | 144 | 0.125 (3.0×) | 0.50 | +0.094 / 0.000 | [+0.005, +0.21] |
| `pump ≥ 0.25, rug ≤ 0.5` | 1 170 | 0.468 (11.4×) | 0.64 | +0.534 / +0.146 | [+0.45, +0.61] |
| `pump ≥ 0.3`, no rug gate | 2 846 | 0.490 (11.9×) | 0.66 | +0.865 / +0.225 | [+0.80, +0.94] |
| **`pump ≥ 0.4`, no rug gate (chosen)** | **1 113** | **0.605 (14.7×)** | **0.71** | **+1.124 / +0.302** | **[+1.00, +1.24]** |
| `pump ≥ 0.5`, no rug gate | 382 | 0.631 (15.4×) | 0.71 | +1.286 / +0.289 | [+1.05, +1.53] |

Why this one (predictive quality first, P&L second):

- **The previous rule broke on the new champion.** Its rug head sits on a higher scale, so `rug ≤ 0.2`
  left almost nothing, and what it kept was poor. That is why retrain #5's reference-policy P&L lift
  was negative (−3.7) despite pump/5m PR-AUC +0.116.
- **Every rug gate hurts at 15m.**
  - At equal pump threshold, removing the gate raises the precision, the win rate *and* the median return.
  - The tokens about to pump carry high rug probabilities too: volatile tokens pump, then dump.
  - The labeled return already includes the rugs (it is realized at the horizon, floored at −100 %).
- **Precision vs breadth.** `pump ≥ 0.4` is 15× the base rate on 1 113 windows. `pump ≥ 0.5` is barely
  more precise (0.63) on a third of the trades, and `pump ≥ 0.3` gives up 11 points of precision. The
  neighbours stay close, so the choice is not a lucky grid point.
- **P&L, heavy tails.** The mean is driven by a few large winners (labels capped at +1000 %), so the
  median (+30 %) and the CI are read alongside it; both clear the baseline by far.
- **Horizon = position timeout.** 15m is the span of the bot's default 900 s position timeout.
- **No rug exit.** `MaxRugProb` is also the bot's rug exit. At 1 it never fires, which matches the
  sweep's hold-to-horizon returns. The stop-loss and the 900 s timeout still close positions.

**Confidence.** `confidence = 200 · max(|pump − 0.5|, |rug − 0.5|)` is a margin, not a hit rate.
`rug ≤ 0.2` used to imply `confidence ≥ 60`. Without a rug gate, the chosen entries often have both
heads near 0.5 (confidence near 0), so `MinConfidence` is 0 (off). Raise it again only together with
a rug gate ≤ 0.2.

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
  CI: it can veto (`pnl_veto`, CI entirely < 0), never promote. **Entry-selection guard**
  (`promotion.entry_selection`, `solia_ai/entry_selection.py`, #180): averaged metrics count each
  window once, so a few near-duplicate windows of one token can hide a confidently wrong entry rule
  — or fake a good one. Retrain #3's candidate improved ROC-AUC / PR-AUC everywhere, yet its top 1 %
  1 h windows passing the rug gate were 417 windows from 11 tokens, 85 % one token that rugged,
  precision 0 ([#179's finding](reports/inverted-1h-179.txt); the pump head alone ranked well).
  Per horizon, the guard takes the candidate's top `top_fraction` (1 %) held-out windows by
  `pump_prob` **among those passing the reference policy's rug gate** (`rug_prob ≤` the bot's
  `MaxRugProb`) and, averaged over runs, requires precision ≥ `min_top_lift` (1.0) × the horizon's
  base rate and ≥ `min_tokens` (20) distinct tokens with none above `max_token_share` (50 %). A
  selection drawn from fewer than `min_tokens` tokens is *undefined*: reported, neither passed nor
  rejected. The pump-only top-k and the champion's selections are reported, never blocking; all are
  in `training_events.metrics.entry_selection` and in the run metrics (`entry/<h>/…`,
  `entry_pump/<h>/…`) of ablation / search reports. It can reject — the first candidate too — never
  promote; omit the block or set `min_top_lift: 0` to disable it. **Guarded horizons** (`entry_selection.horizons`, default null = all): only those horizons can reject; the others are still measured and reported (`report-only`). The packaged config guards the horizons the bot may trade, `5m` and `15m`. Retrain #4's candidate (pump/5m PR-AUC +0.109, 9/9 runs) was rejected at 1h alone, a horizon the bot never trades, where the rug-gated selection fails in every model (candidate 0.012, champion 0.004, base rate 0.032; #179). No champion → first valid candidate
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
- **Candidate groups stay disabled until they earn it.** The model only sees the groups listed in the config's
  `features.groups` (omitted: `BASELINE_GROUPS`); the other columns are zeroed by a mask stored in the
  model (`feature_mask`, same in training, inference and the ONNX graph, and saved with the checkpoint).
  A group is enabled there only once its ablation lift clears the margin. The packaged config enables
  `momentum` and `concentration` (see the results below).
- **Candidates so far:** `buy_pressure` (FE-T2, #111) — `net_sol_flow`, `buy_volume_ratio`,
  `avg_buy_sol`, `max_buy_sol`, `large_buy_count`; `momentum` (FE-T3, #112) — `return_short`,
  `return_medium`, `return_long`, `price_acceleration`, `ema_slope`, `volume_acceleration`,
  `volume_burst_ratio`, `price_history_sec` (windows: `Pipeline:Momentum{Short,Medium,Long}Seconds`);
  `concentration` (FE-T4, #113) — `unique_traders`, `top_trader_share`, `trader_hhi`,
  `new_trader_ratio`: a holder-concentration proxy from trade signers over the window;
  `burn` (FE-T5, #114) — `burned_liquidity`, % of the AMM pool's LP burned (rug signal);
  `venue` (FE-T5, #114) — the `venue` column one-hot encoded (`VENUE_COLUMNS`);
  `market` (FE-T11, #195) — `MARKET_COLUMNS`: market-wide launch rate (5 min / 1 h), graduations and
  graduation ratio (1 h), SOL buy volume and net SOL flow (5 min / 1 h), active tokens (5 min / 1 h) and
  `mkt_coverage_sec`, the same on every row of a tick (see docs/DATA-PIPELINE.md "Market-regime context");
  `wallet` (FE-T9, #189) — `WALLET_COLUMNS`: freshness of the window's signers (share first seen < 1 h /
  < 24 h, median age), the creator's age and previous launches, and its behaviour on this launch (dev buy in
  SOL and / supply, share sold, time since first sell), plus `wal_coverage_sec` (see docs/DATA-PIPELINE.md
  "Cross-token wallet memory");
  `wallet_reputation` (FE-T10, #190) — `REPUTATION_COLUMNS`: the creator's recorded launches and their rug /
  pump shares (an outcome is visible only after its 1 h horizon plus the label embargo), the first buyers' share
  of rated wallets, mean win rate and P&L per round trip, the smart wallets' share of the window's SOL buys, net
  flow and count, and the launch's bundling (first buys in the creation block / one block); `rep_early_buyers`
  is always written (see docs/DATA-PIPELINE.md "Wallet reputation"). Disabled by default; measured in the
  combined ablation #198, which must also report the distinct creators / wallets behind its top-1 % entries;
  `metadata` (FE-T12, #196) — `METADATA_COLUMNS`: whether the metadata URI is IPFS / Arweave or a launch tool's
  host, website / X / Telegram flags from the token's single live metadata fetch (visible from its completion
  time; NULL in any replay of a window before the fetcher's deployment), and the narrative heat of its normalized
  name / ticker (launches sharing it in the last hour, namesakes graduated in the last 3 h); `meta_coverage_sec`
  is always written (see docs/DATA-PIPELINE.md "Token metadata"). Unknown values are imputed to -1;
  `holders_supply` (FE-T13, #197) — `HOLDERS_SUPPLY_COLUMNS`: holder-side concentration from stream-derived
  positions (top-1 / top-10 share of the circulating supply, the top 10's net flow over the window, wallets above
  2 % of the supply, holder count, and the incomplete-history flag); `hs_holders` and `hs_incomplete` are always
  written (see docs/DATA-PIPELINE.md "Holder supply"). Undefined shares / counts are imputed to -1, the net flow
  to 0.
- **Venue one-hot is encoded at load time, not stored.** `features.py` derives `venue_<name>` from the
  raw `venue` column (training reads it from the DB, `/predict` rows carry it as a string), over a
  fixed category list (`VENUES`: `pumpfun`, `pumpswap`, `raydium-launchpad`, `raydium`,
  `meteora-launchpad`, `meteora`, `bonk`) plus `venue_other` for anything else, NULL included. No
  schema churn when the list changes; appending a venue appends a column (an ordered-superset spec,
  so the champion keeps serving). `DB_FEATURE_COLUMNS` is the stored part the parity tests check.
- **Rows predating a group are refused, not zero-filled.** A NULL in a column the pipeline always writes
  since the group's migration (`ALWAYS_WRITTEN`, e.g. `net_sol_flow`/`large_buy_count` for
  `buy_pressure`, `volume_burst_ratio`/`price_history_sec` for `momentum`, `unique_traders` for `concentration`, `mkt_coverage_sec` for `market`, `wal_coverage_sec` for `wallet`; none for `burn`, whose
  migration 0008 backfills earlier rows from `onchain_signals`, nor `venue`) means "never computed": training, evaluation and ablation raise if an enabled (or
  ablated) group has such rows — re-replay them first. Other NULLs are imputed per `NULL_FILL`
  (`buy_volume_ratio` → 0.5, "no pressure either way"; `volume_burst_ratio` → 1, "steady"; `new_trader_ratio` → −1, "no recent trader", as 0 is "all returning"; `burned_liquidity` → −1, "unknown", as 0% burned is a real rug signal; the non-negative `mkt_*` aggregates → −1, "window not observed yet" (warm-up), as 0 launches / 0 active tokens is a dead market — the signed `mkt_net_sol_flow_*` keep 0 and `mkt_coverage_sec` marks the warm-up; every `wal_*` but `wal_coverage_sec` → −1, "not computable" (warm-up, unknown creator, creator hasn't sold), as 0 is a real reading (no fresh trader, nothing sold yet); else 0 —
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
- **Combinations** (`--combine a b`, repeatable): several candidate groups added together as one
  variant `+a+b`. Lifts of correlated groups don't add up, so the set that would ship is measured
  as a whole, against the same baseline.
- **Bounded read** (#188): the CLI reads exactly what a retrain trains on —
  `read_training_features` over `data.lookback_days` / `data.max_rows` (whole tokens, see
  [Bounded training read](#bounded-training-read)), overridable with `--lookback-days` /
  `--max-rows` — never the whole table. `data.window_stride_seconds` comes from the config, so
  run it with the defaults the variants will ship with (`SOLIA_AI_CONFIG` for a config under test).
- **Resume** (#188): with `--out x.json`, every finished (variant, window, seed) run is appended to
  `x.runs.jsonl` (fsynced, `solia_ai/runlog.py`, the same format as the search's trials file). A
  rerun with the same settings, config, variants and data window skips the runs already there and
  re-reads the window the run started on, so a reboot loses at most the run in progress;
  different settings are refused rather than mixed; `--restart` discards the file.
- Ablation never saves a model checkpoint, registers a model or touches `is_champion`.

#### Results: `momentum` + `concentration` enabled (#188, 2026-11-01)

Both runs use the bounded read (8 M rows of ~77 k tokens out of a 2-day window, stride 60 s) and
compare against the 19-column baseline. Reports: `docs/reports/ablation-188-run1.*` and
`docs/reports/ablation-188-confirm.*`.

- **Run 1**: 2 windows × 2 seeds, every candidate on its own, plus all five together.
  - `momentum` was the only clear gain: pump/1m PR-AUC +0.023 ± 0.007, pump/5m +0.015 ± 0.010.
  - `concentration` came next: pump/5m +0.011 ± 0.017.
  - `buy_pressure`, `burn` and `venue` were ≈ 0 on every head.
  - The five together did no better than `momentum` alone.
- **Confirmation**: 3 windows × 3 seeds; baseline vs `+momentum` vs `+momentum+concentration`.

| Variant | pump/5m PR-AUC (primary) | pump/1m | pump/30s | rug/5m | pump/5m ECE |
|---|---|---|---|---|---|
| +momentum | +0.018 ± 0.014 (not significant) | +0.027 ± 0.011 better | +0.030 ± 0.007 better | +0.005 ± 0.018 | −0.0005 ± 0.0009 |
| **+momentum+concentration** | **+0.020 ± 0.006 better** | +0.031 ± 0.007 better | +0.033 ± 0.006 better | +0.009 ± 0.006 | −0.0012 ± 0.0012 |

- **The pair is the only variant that clears the margin on the primary metric** (mean − std = 0.013 > 0.01).
  - Its lift is also steadier than `momentum` alone: std 0.006 vs 0.014.
  - No rug head regresses: every rug PR-AUC lift is ≥ 0, and rug/30s and rug/1m are +0.012.
  - The raw ECE moves by ≤ 0.0012, not significant. Calibration (Platt) then improves ECE by a further 0.009–0.04.
- **Reference-policy P&L**: +6.8 ± 13.7 against the baseline. That is positive but not significant, as expected for a secondary metric.
- **Decision**:
  - `config.yaml` now enables `features.groups: [price, liquidity, flow, holders, context, momentum, concentration]`.
  - `buy_pressure`, `burn` and `venue` stay computed but masked.
  - The change makes the champion's checkpoint incompatible (its feature mask differs), so the next retrain starts from scratch (`candidate.on_incompatible: scratch`).
  - That retrain then goes through the promotion gate like any challenger.

#### Results: `market` (#195) — pending, measured in #198

Not measured yet. The `market` group is computed by the pipeline but **disabled** (not in
`features.groups`) until it is. It is measured together with `wallet` (#189),
`wallet_reputation` (#190), `metadata` (#196) and `holders_supply` (#197) in one combined
re-replay + ablation (#198), so the history is replayed once for all of them. Protocol:

- **Data**: migration 0013 applied, then a re-replay that starts **≥ 1 h before** the first training
  row (long-window warm-up), so the evaluated rows carry every `mkt_*` aggregate. Rows predating 0013
  are refused for the group (`mkt_coverage_sec` NULL).
- **Variants**: the current defaults (`momentum` + `concentration` enabled) vs the same **+market**.
- **Regimes**: the walk-forward windows should span different activity levels (weekday vs weekend, busy
  vs quiet hours), and the lift is reported **per window**, not only on average: a regime feature that
  only helps in the regime it was trained on is not a gain.
- **Decision rule**: same margin as the other groups. `market` is enabled only if it clears it.

#### Results: `wallet` (#189) — pending, measured in #198

Not measured yet. The `wallet` group is computed by the pipeline but **disabled** until the combined ablation
#198 measures it. Protocol specifics:

- **Data**: migration 0014 applied, then the combined re-replay with **`--warmup-hours 24`**, so the evaluated
  rows carry the history-dependent `wal_*` columns. Rows predating 0014 are refused for the group
  (`wal_coverage_sec` NULL).
- **Variants**: the current defaults vs the same **+wallet**.
- **Decision rule**: same margin as the other groups. `wallet` is enabled only if it clears it.

#### Results: `wallet_reputation` (#190) — pending, measured in #198

Not measured yet. The `wallet_reputation` group is computed by the pipeline but **disabled** until the combined
ablation #198 measures it. Protocol specifics:

- **Data**: migration 0015 applied, then the same combined re-replay with **`--warmup-hours 24`** (the
  reputation look-backs fit in the age cap: launch outcomes need 1 h + 1 h embargo, round trips 18 h + one 6 h
  bucket). Rows predating 0015 are refused for the group (`rep_early_buyers` NULL).
- **Variants**: the current defaults vs the same **+wallet_reputation** (and +wallet +wallet_reputation, since
  both read the same memory).
- **Overfitting check**: report the number of distinct creators and wallets behind the top-1 % entry selection
  (pump heads). A lift carried by a handful of very active wallets is memorization, not a gain.
- **Decision rule**: same margin as the other groups. `wallet_reputation` is enabled only if it clears it.

#### Results: `metadata` (#196) — pending, measured in #198

Not measured yet. The `metadata` group is computed by the pipeline but **disabled** until the combined ablation
#198 measures it. Protocol specifics:

- **Data**: migration 0016 applied, then the combined re-replay with `--warmup-hours 24` (the narrative counts
  need 6 h). Rows predating 0016 are refused for the group (`meta_coverage_sec` NULL).
- **Link flags are live-only.** A replay never fetches: it reads `token_metadata`, which only holds tokens seen
  by live ingestion since the fetcher was deployed. Over a window before that, the three link flags are NULL on
  every row, so #198 measures the stream part (URI kind + narrative heat) on the whole window, and the links only
  on a window fully ingested live after deployment (report both, and the share of rows with a known link).
- **Variants**: the current defaults vs the same **+metadata**.
- **Decision rule**: same margin as the other groups. `metadata` is enabled only if it clears it.

```bash
# from services/ai-service, database from SOLIA_DB_CONNECTION / POSTGRES_* (a replayed history)
python -m solia_ai.ablation                                   # baseline vs dropping each group
python -m solia_ai.ablation --groups flow holders --windows 4 --seeds 0 1 2 3 \
    --margin 0.01 --pnl-margin 0 --out reports/ablation.json  # JSON report + readable summary
# unattended, in the GPU container (resumable: rerun the same command after a reboot)
python -u -m solia_ai.ablation --groups momentum concentration --combine momentum concentration \
    --windows 3 --seeds 0 1 2 --out /app/models/reports/ablation.json
```

#### Results: `holders_supply` (#197) — pending, measured in #198

Not measured yet. The `holders_supply` group is computed by the pipeline but **disabled** until the combined
ablation #198 measures it. Protocol specifics:

- **Data**: migration 0017 applied, then the combined re-replay with `--warmup-hours 24`, so tokens created in the
  warm-up have complete positions. Rows predating 0017 are refused for the group (`hs_holders` NULL).
- **Incomplete history**: report the share of training rows with `hs_incomplete = 1`, and the lift on complete
  tokens only (`hs_incomplete = 0`); a lift that only shows on incomplete tokens is suspicious.
- **Variants**: the current defaults vs the same **+holders_supply** (and **+holders_supply −concentration**, to see
  whether the two concentration views overlap).
- **Decision rule**: same margin as the other groups. `holders_supply` is enabled only if it clears it.

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
  applied. Trials never save a model checkpoint, register a model or touch `is_champion`.
- **Resume (#181)**: with `--out x.json`, each finished trial is appended (and fsynced) to
  `x.trials.jsonl` as it ends. Its first line fingerprints the search: settings (space, strategy,
  trials, seeds, windows, objective, …), the base config and the data read (window bounds, rows,
  tokens). A rerun with the same `--out` re-reads **the window the search started on** (not today's:
  live rows keep arriving) and skips the trials already in the file, so a reboot loses at most the
  trial in progress; time already spent counts against `--max-minutes`. A file of another search
  (any fingerprint difference, or a trial recorded with other parameters) is refused before training,
  naming what differs; `--restart` discards it. A torn last line (killed mid-write) is ignored.
- **Failures**: a candidate trial that raises (e.g. CUDA out of memory) is recorded as failed (report
  `failed`, with its error), never ranked, and the search goes on; a rerun doesn't retry it. A failing
  baseline still stops the search.

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

**Unattended overnight run (#181).** Both stages in one detached, resumable command, from the host
(Git Bash), against the running GPU stack:

```bash
infra/scripts/overnight-sweep.sh --dry-run   # preconditions only
infra/scripts/overnight-sweep.sh             # preflight, then stage 1 -> stage 2, detached
```

- **Preflight** (`python -m solia_ai.overnight preflight`, in the ai-service container) refuses to
  start, listing every problem: a retrain or rollback holds the retrain advisory lock;
  `RETRAIN_SCHEDULER_ENABLED` (ai-service) or `PAPER_BOT_ENABLED` (backend) is true — both share the
  GPU and the database; CUDA isn't visible (CPU image); less than 20 GB free on the models volume; a
  search or overnight run is already running. It then reminds you to **pause Windows Update**. Live
  ingestion stays on (training reads `features`, not the bot).
- **Detached**: `docker exec -d` runs `python -u -m solia_ai.overnight run` in the container, so the
  terminal or the docker client can go away. Log: `/app/models/reports/overnight.log` (appended).
- **Stages**: stage 1 is the `overfitting-data` grid (5 trials); stage 2 the `overfitting-regularization`
  random search (10 trials), 2 windows × seeds 0 1, objective `pr_auc` (flags: `--stage1-trials`,
  `--stage2-trials`, `--windows`, `--seeds`, `--objective`). Stage 2 runs on stage 1's
  `window_stride_seconds` / `max_rows`: the best stage-1 trial's only when its objective lift over
  the baseline is `better` (paired verdict), else the base config's. The choice is written to
  `reports/overfitting-regularization.config.yaml` and given to stage 2 as `SOLIA_AI_CONFIG`;
  `--stride` / `--max-rows` override it (stage 2 then runs even if stage 1 failed).
- **After a reboot** (or a crash, or `docker compose up` recreating the container): run the same
  command again. Each stage resumes from its `<report>.trials.jsonl`; finished trials are not rerun.
  `--restart` starts both stages over.
- **Summary**: `reports/overnight-summary.txt` — per stage the trials finished / failed, what stopped
  it, the training time, the baseline and top-3 scores, the best trial's lift and verdict, stage 2's
  data, and every failure (a stage that stopped or crashed, a failed trial). The run exits 1 when
  anything failed. `python -m solia_ai.overnight summary` rewrites it from the reports.

**Results & chosen defaults (#172, 2026-11-01/02).**

How to read the tables:
- **Objective:** mean PR-AUC over every head × horizon (pump and rug, 30 s → 1 h).
- **Score:** objective mean − std, used for ranking.
- **Gap:** validation − train loss at the best epoch. Negative means the validation loss is below the train loss: dropout is active in training, and validation is unweighted.
- **Lifts:** paired run for run against the baseline, verdict `mean − std > margin` (0.01).

The absolute values of the two stages are not comparable with each other, because their windows differ. Only the lifts within a stage are.

*Stage 1 — data* (`docs/reports/overfitting-data-172-stage1.json`):
- Features as of 2026-10-31, 19 baseline columns.
- Held-out 10-31 03:01 → 16:29 UTC, 2 windows × 2 seeds.

| trial | stride | max_rows | objective | score | best epoch | gap | min / run |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 4 M | 0.4701 ± 0.0378 | 0.4323 | 1.00 | −0.092 | 50 |
| trial-01 | 60 s | 4 M | 0.4645 ± 0.0332 | 0.4313 | 2.75 | −0.096 | 22 |
| trial-02 | 30 s | 4 M | 0.4759 ± 0.0361 | 0.4397 | 2.25 | −0.080 | 26 |
| trial-03 | 30 s | 8 M | 0.4905 ± 0.0357 | 0.4548 | 1.75 | −0.098 | 30 |
| **trial-04** | **60 s** | **8 M** | 0.4904 ± 0.0351 | **0.4552** | **3.25** | −0.091 | **24** |

Best trial-04 vs baseline:
- **Objective:** +0.020 ± 0.011, not significant (mean − std = 0.0095, just under the margin).
- **Pump / rug PR-AUC:** rug/5m +0.025 ± 0.012 (better); pump/5m −0.005 ± 0.027; pump/15m +0.023 ± 0.021.
- **No ECE regression:** pump/5m ECE +0.002 ± 0.002.
- **Policy P&L:** +72 ± 130, not significant.
- **The stride fixes the original symptom:** the best epoch moves from 1 (every baseline run) to 2–3, and a run takes half the time.

*Stage 2 — regularization* (`docs/reports/overfitting-regularization-172-stage2.{json,txt}`, driver `…-stage2-driver.sh`):
- Run on 8 M rows / stride 60, with the groups #188 enabled (`momentum` + `concentration`).
- Window 10-30 19:17 → 11-01 19:17 UTC, held-out 11-01 06:35 → 19:17, 2 windows × 2 seeds, 10 random trials.

| trial | dropout | weight_decay | lr | d_model | objective | score | best epoch | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trial-04 | 0.2 | 0.05 | 3e-4 | 128 | 0.5074 ± 0.0185 | 0.4889 | 2.50 | −0.099 |
| trial-03 | 0.1 | 0.1 | 3e-4 | 128 | 0.5085 ± 0.0203 | 0.4881 | 3.00 | −0.093 |
| **baseline** | 0.1 | 0.01 | 3e-4 | 128 | 0.5054 ± 0.0175 | 0.4879 | 2.25 | −0.106 |
| trial-05 | 0.1 | 0.1 | 3e-4 | 64 | 0.5096 ± 0.0237 | 0.4859 | 5.75 | −0.093 |
| trial-02 | 0.1 | 0.01 | 1e-4 | 64 | 0.5077 ± 0.0221 | 0.4856 | 10.25 | −0.092 |
| trial-08 | 0.1 | 0.1 | 1e-4 | 128 | 0.5097 ± 0.0243 | 0.4853 | 6.25 | −0.087 |
| trial-09 | 0.2 | 0.01 | 1e-4 | 64 | 0.5063 ± 0.0217 | 0.4846 | 10.25 | −0.095 |
| trial-01 | 0.3 | 0.05 | 3e-4 | 64 | 0.5040 ± 0.0206 | 0.4833 | 5.25 | −0.100 |
| trial-06 | 0.2 | 0.1 | 3e-4 | 64 | 0.5049 ± 0.0219 | 0.4830 | 5.50 | −0.099 |
| trial-07 | 0.3 | 0.1 | 3e-4 | 64 | 0.5060 ± 0.0231 | 0.4829 | 6.75 | −0.098 |

Best trial-04 vs baseline:
- **Objective:** +0.002 ± 0.004.
- **Pump / rug PR-AUC:** pump/5m +0.001 ± 0.004; rug/5m +0.002 ± 0.009.
- **ECE:** pump/5m −0.0005 ± 0.0007.
- **Policy P&L:** −0.4 ± 3.6.
- **Verdict:** nothing significant on any metric, so the proposed diff (dropout 0.2, weight decay 0.05) is **not** applied.

Findings:
- **Regularization is not the lever.** All `d_model` 128 trials are within ±0.001 of the baseline's score, and every `d_model` 64 trial scores 0.002–0.005 lower.
- **A lower learning rate (1e-4) only delays the best epoch** (to 6–10) at a higher cost, with no gain.
- **The gap is the same with every setting** (−0.087 to −0.106). Neither dropout nor weight decay changes how the model generalizes from its training period to the next. The held-out slices also differ more from each other than the trials do: rug/5m PR-AUC is 0.665 on slice 0 and 0.713 on slice 1 for the baseline.
- **Where to look next:** more and broader data (#202), recency weighting (#203), market context (#195 / #198) and seed ensembles (#204).

*Chosen defaults.*
- **`data.window_stride_seconds: 60`, `data.max_rows: 8000000` — a documented exception to the margin rule:**
  - the objective lift (+0.020 ± 0.011) falls just short of significance;
  - every metric's lift is ≥ −0.005 in mean, and rug/5m is significantly better;
  - the best epoch moves past 1;
  - each training takes half the time (24 vs 50 min per run, 2 × 2 on the RTX 4070), which halves every retrain, ablation and search;
  - #202 re-decides the data settings rigorously.
- **Model and optimizer stay unchanged** (`dropout` 0.1, `weight_decay` 0.01, `learning_rate` 3e-4, `d_model` 128).
- **Peak RSS of the 8 M-row read at stride 60:** measured with `python -m solia_ai.throughput --max-rows 8000000` (stride 60, 80 478 tokens, 843 k samples): **8.7 GB at peak, during the read** (transient), and 4.5 GB after the build (4.8 GB while training; 3 loader workers add only ~0.1 GB PSS). The ablation, stage 2 and this measurement each ran to completion at 8 M rows. The peak is inside the ~16 GB Docker budget, but with Postgres (~7 GB) the host is close to full during the read: do not raise `max_rows` further before #202 lowers the read's peak. Training throughput: ~7 700 samples/s (RTX 4070).

### Overfitting: training data breadth (#202)

#172 stage 2 found that regularization is not the lever. Every `d_model` 128 trial scored within ±0.001 of the
baseline, with the same validation − train gap. The largest effect so far is data volume (4 M → 8 M rows:
0.4645 → 0.4904). The hypothesis: **more distinct tokens and a longer period** generalize better than more windows
per token (stride 30 vs 60 made no difference).

**Memory: why "every token of 3.5 d" doesn't fit as-is.** *Bounded training read* and #160 measured ≈ 1 000
bytes per row read (frame 331 + `config_samples` peak 640). 3.5 d ≈ 60–70 M rows would need ≈ 65 GB, and the host
has ~8 GB left beside Postgres (15.6 GB, ~7 GB for Postgres). The windows' float32 matrix is
`FEATURE_COLUMNS`-wide (91 columns, 364 B/row) even though the model zeroes every disabled group. Row thinning
can't help either: at `seq_len` 64 × ~10 s snapshots, a window's history (~640 s) is longer than the strides
studied (120–240 s), so almost every row belongs to some kept window.

**Step 1: read and hold only the enabled groups.** Opt-in, research CLIs only:

- `--prune-columns` (`solia_ai.search`, `solia_ai.throughput`; sets `data.prune_columns`). The read selects only
  the enabled groups' stored columns, plus what labels and eligibility need (`price`, `liquidity`,
  `unique_traders`, `volume`, `token_age_sec`; `mint` / `ts` / `venue` as always). It returns the same rows and
  tokens (`solia_ai.db.training_columns`).
- The windows then keep a **narrow matrix** of the enabled columns only (`Samples.columns`): 31 of 91 for the
  current `price, liquidity, flow, holders, context, momentum, concentration`. Windows are still served
  `FEATURE_COLUMNS`-wide, with the other columns 0, which the model's feature mask would make of them anyway. The
  model's outputs are bit-identical (`tests/test_training_breadth.py`). The normalizer is fit on the narrow rows
  (`fit_normalizer(..., columns=)`), so the training rows are never widened. Its statistics equal the full fit
  up to float32 summation order (~1e-7).
- The candidate / retrain refuse `data.prune_columns`, because the champion scored on the same windows may use
  groups the config doesn't enable. Without the flag, nothing changes: same columns, rows, windows and matrix.

Expected effect: ≈ 2.5–3× fewer bytes per row. **Measured (2026-11-03, GPU container, `throughput
--prune-columns --lookback-days 3.5 --max-rows 8000000`): only −12 %**, i.e. 7.06 GB peak RSS for 8.0 M rows
(≈ 880 B/row; RSS after the build 3.99 GB), and 1 006 s of read (≈ 8 k rows/s: `mint = ANY(…) ORDER BY mint, ts`
reads every token's rows at random places of the heap). The peak is the read itself (chunk frames + `concat`) and the
build's temporaries, not the stored columns. Step 1 falls short, hence step 2. That first run also exposed a CUDA
bug in the narrow normalizer fit (#227).

**Step 2: compact training rows** (`solia_ai.rows.TrainingRows`, used by `--prune-columns` in `solia_ai.search` and
`solia_ai.throughput`; `solia_ai.db.read_training_data` picks the reader). The default path is unchanged.

- **The rows are held once**: a narrow `float32` matrix of the enabled columns (124 B/row for the 31 current ones),
  plus a slim frame aligned with it (`mint` as a category, `ts`, `venue`, and float64 `price`, `liquidity` and the
  eligibility inputs, as in the frame read). Rows are sorted by `(mint, ts)`. Measured: **176 B/row** held.
- **The read never holds a frame of every row** (`solia_ai.db.read_training_rows`). The token selection is the same
  as the bounded read's (whole tokens, `md5(mint)` order, `max_rows`), and it gives each kept token's row count. The
  window is then streamed in 1-hour slices, each `ORDER BY ts`. A whole window gets a parallel scan and an on-disk
  sort; an hour walks the `ts` index. Rows of other tokens are dropped on the client. Each chunk goes straight to
  its final rows of the preallocated arrays (token offset + rank in time). NULL counts of the always-written columns
  are kept, so a group whose rows predate it is still refused. A token that gains or loses rows in the window during
  the read is an error.
- **Every Samples shares the matrix** (`Samples.shared_rows`): `build_samples(..., features_matrix=)` indexes it
  instead of copying the windows' rows. A search's reference windows (base stride) and each trial's (its stride) cost
  only their indices and labels. The build runs per block of whole tokens (`dataset.rows_samples`, 4 M rows), so its
  temporaries follow the block, not the read.
- **Streamed normalizer fit** (`SoliaTransformer.fit_normalizer_rows`). Mean and variance are merged per 1 M-row
  chunk in float64 over the rows the training windows cover (`train.training_row_mask`), and the shared matrix is
  never copied. The statistics equal the one-pass fit up to float rounding (tests: `rtol` 1e-5), not bit for bit.
- Same windows, labels and served inputs as the frame path. This holds with and without stride, eligibility and the
  liquidity floor, for any block size (`tests/test_training_rows.py`). The read returns the same frame, matrix and
  bounds as `TrainingRows.from_frame` of the frame read (`tests/test_training_read.py`).

Measured (2026-11-03, same container, live pipeline running, bot and retrain scheduler off):

| read | rows / tokens | read | held | RSS after read | peak RSS | samples (stride 60) |
|---|---:|---:|---:|---:|---:|---:|
| step 1, 3.5 d, `max_rows` 8 M | 8.00 M / 59 k | 1 006 s | — | — | 7.06 GB | 775 k |
| step 2, 3.5 d, `max_rows` 8 M | 8.00 M / 59 k | 1 741 s | 1.41 GB | 3.07 GB | **4.36 GB** | 778 k |
| step 2, 3.5 d, **every token** | **26.0 M / 192 k** | 1 783 s | 4.57 GB | 7.32 GB | **9.21 GB** | 2.52 M |

- **Every whole token of 3.5 d is 26 M rows, not the 60–70 M estimated.** The window holds 75.9 M rows of 242 k
  tokens, but the 49.9 k tokens that started before it (long-lived, mostly migrated) hold two thirds of the rows.
  They are dropped, never cut. With the frame read, those 26 M rows would need ≈ 23 GB.
- The compact read costs ~30 min whatever `max_rows` is, because it parses every row of the window (≈ 43 k rows/s,
  CPU-bound). That is slower than the frame read for a small budget, and ~4× faster for every token.
- The RSS beyond the held rows (≈ 2.7 GB after the read) is the interpreter, the chunk churn and the build's
  windows. The peak adds the CUDA context and a training step.

**Searchable lookback.** `lookback_days` is a search parameter (`data`). When it is in the space, the read covers
its largest value. Every trial, the baseline included with the base `data.lookback_days`, then trains per
walk-forward slice on what a bounded read taken at that slice's training cutoff (`eval_start − embargo`) would
keep: whole tokens whose first row is within the lookback, capped by the trial's `max_rows` (rows up to the cutoff,
`md5(mint)` order). Every trial is scored on the same held-out slices (those of the widest read), so the lifts stay
paired. Without `lookback_days` in the space, nothing changes. The shipped grid is `solia_ai/sweeps/training-breadth.yaml`
(lookback 2 / 3.5 d × stride 120 / 240 s, every token). Its baseline is the current config (`--max-rows 8000000`,
stride 60, momentum + concentration).

**Results & decision.** *Pending.* The memory measurement is above (every token of 3.5 d fits: 9.2 GB peak). The
comparison (`python -m solia_ai.search --space training-breadth --strategy grid --objective pr_auc
--prune-columns`, ~4–6 h of GPU) runs with the retrain scheduler and the bot off, in a container capped at ~14 GB.
Record here the paired table (`mean − std > margin`, primary pump/5m PR-AUC), the decision and its cost, with the
report under `docs/reports/`. `config.yaml` `data.*` changes only if a variant clears the margin.

### Overfitting: recency-weighted loss (#203)

The market drifts between periods (#172 stage 2: no regularization setting moves the gap or the score, and the
held-out slices differ more from each other than the trials do). A longer training period (#202) adds regimes the
model should know about, but also older data that may describe today's market less well. Recency weighting keeps
the history while trusting yesterday more than four days ago.

- `training.recency_half_life_hours` (optional; omitted = off = today's training, the shipped default). Each
  training sample's loss weighs `0.5 ** (age / half_life)`, where `age` runs from the sample's end to the end of
  the training window (the newest training sample). The weights are normalized to mean 1 over the training set,
  so the effective learning rate doesn't move with the half-life (`solia_ai.dataset.recency_weights`).
- The weight is a per-sample factor in `SoliaLoss.label_weight`. Every head and the return use it, and it
  multiplies the label mask, the horizon weights, `pos_weight` and the focal term: they compose, none replaces
  another. Balanced sampling (if on) still picks the batches; the weights then scale their losses.
- **Training loss only.** Early-stopping validation, calibration and evaluation datasets carry no weight and stay
  unweighted. The reported `train_loss` (and so the fit `gap`) is the weighted mean.
- Validation rejects a non-numeric, non-positive or non-finite value, naming the key. The value is searchable
  (`recency_half_life_hours`, shipped grid `solia_ai/sweeps/recency.yaml`: 12 / 24 / 48 h).

**Results & decision.** *Pending.* A paired comparison (`mean − std > margin`, primary pump/5m PR-AUC) on the
longest window #202 retains needs the GPU. It runs after #202's measurements, with the retrain scheduler and the
bot off. Record the table and the decision here. `config.yaml` changes only if a half-life clears the margin.

### Seed ensemble champion (#204)

Scores vary noticeably from seed to seed. Pump/5m PR-AUC has a std of ~0.03 across runs, while the lifts we act on
are ~0.02. Averaging models trained under different seeds reduces that variance, and the champion depends less on
a lucky or unlucky seed.

- `model.ensemble_size` (optional; default 1 = today's single model, bit for bit: same code path, no
  `ensemble_size` in the checkpoint). With N > 1, `solia_ai.train.fit` trains N members one after the other on the
  same data, under seeds `training.seed + k`. The members share the architecture and the feature mask; only the
  seed differs. They are returned as one `SoliaEnsemble`, which has the single model's interface.
- **Averaging: logit space.** The members' pump / rug logits are averaged (the expected return as is). The
  ensemble is then **calibrated once, as one model**: Platt / temperature fit on the averaged logits by the same
  `fit_calibration_window`. Logit averaging was chosen over probability averaging because it keeps calibration
  unchanged: a fit on logits, applied to logits, in training, serving and ONNX alike.
- **Artifact:** one checkpoint. Its `model_state` holds `members.<k>.*`, `ensemble_size` N is recorded beside it,
  and `hyperparams.model.ensemble_size` too. `build_model` rebuilds an ensemble from it. The `Predictor`
  (`/predict`), evaluation and the promotion gate score the ensemble as a whole. The ONNX export holds every
  member in one graph (parity checked like a single model's).
- **Compatibility:**
  - a single-model champion keeps serving unchanged;
  - a warm start from an ensemble of the same size seeds member k from champion member k;
  - a size mismatch (single to ensemble, or another N) is an incompatibility, which `candidate.on_incompatible`
    resolves (scratch or error);
  - an ensemble state loaded into one `SoliaTransformer` raises `IncompatibleCheckpointError` naming the member
    count, rather than a list of missing keys.
- **Fit diagnostics:** the `metrics` are the ensemble's own validation losses. `best_epoch` / `history` (the fit
  gap) are member 0's. Each member's epoch events carry `member` / `members`.
- **Cost:** GPU training time and inference both scale ~linearly with N (N forward passes per batch). The retrain
  scheduler holds its slot (and the orchestrator its advisory lock) for the whole run without a time budget. A
  longer run only delays the next tick, which skips while a run holds the slot.
- Searchable (`ensemble_size`; shipped grid `solia_ai/sweeps/ensemble.yaml`: 3 / 5 against the baseline 1).

**Results & decision.** *Pending.* The paired comparison of sizes 1 / 3 / 5 (`mean − std > margin`, primary
pump/5m PR-AUC) needs the GPU. It runs after #202, on its retained data settings, with the retrain scheduler and
the bot off. Record here the table, the GPU training time per size and the inference latency per batch
(`/predict`, PyTorch and ONNX). `config.yaml` changes only if the gain clears the margin at an acceptable cost.

### GBDT benchmark (#205)

Is the Transformer near what these features allow? #172 stage 2 found it insensitive to its own regularization and
size. A gradient-boosted-trees model trained on the **same windows, labels and splits** answers that. If GBDT ≈
Transformer, the ceiling is the data: put the effort into data and features (#198, #202). If GBDT is clearly
below, the sequence model earns its complexity. If GBDT is clearly above, the model side needs work.
**Measurement only**: nothing is saved, registered, served or promoted.

`python -m solia_ai.benchmark_gbdt [--windows 3] [--seeds 0 1 2] [--lookback-days …] [--max-rows …]
[--prune-columns] [--gbdt-device cpu|gpu] [--gbdt-jobs 6] --out reports/gbdt-205.json`

- **Same slices.** It uses the bounded read, `config_samples` and `walk_forward_windows` (embargo included) of the
  ablation / search. Every (slice, seed) runs the Transformer through `train_and_score` itself, and the GBDT on the
  same indices, so the lifts are paired run for run.
- **Tabular inputs** (`tabular_features`). For each enabled group's column (`features.groups`): the window's last
  row, its mean / min / max over the real rows (padding excluded) and last − first; plus the window length. That
  gives 5 × 31 + 1 = 156 columns for today's groups. The values are the raw inputs as served (trees don't need
  normalization).
- **One LightGBM classifier per head × horizon**, trained on the slice's training windows labeled at that horizon.
  It early-stops on binary log-loss (100 rounds patience, up to 2 000 trees, `num_leaves` 63, `min_child_samples`
  100, 0.8 row / column subsampling, seeded per run) on the **Transformer's own validation split**, never the
  scored slice. It is scored with the ablation's ROC-AUC / PR-AUC / ECE. Its probabilities are LightGBM's own,
  while the Transformer's are Platt-calibrated on the validation split, as served. A head × horizon whose training
  or validation windows hold a single class scores n/a.
- **Report** (JSON + `.txt`): each model's mean ± std per metric, and the paired GBDT − Transformer lift with the
  ablation's verdict (`mean − std > margin`). It also gives the GBDT's best iterations and the **top columns by
  gain**, summed over every head × horizon, which tells where feature work pays.
- `lightgbm` is an optional extra (`pip install -e ".[benchmark]"`). The default install and the serving image
  locks don't carry it (`tests/test_requirements_lock.py`). Without it, the command exits with that hint before
  reading anything.

**Results & interpretation.** *Pending.* The run needs the GPU (Transformer side) and the database read. It goes
after the official post-#172 retrain, with the retrain scheduler and the bot off, on the champion's settings.
Estimate the runtime first: the Transformer side costs what an ablation of the same windows × seeds costs; the GBDT
side, 2 heads × 5 horizons classifiers per run on 6 CPU cores. Record the table and the interpretation here, with
the report under `docs/reports/`.

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
| `search.py` | Hyperparameter, regularization, stride & labeling-threshold search (proposes a `config.yaml` diff; resumable); spaces in `sweeps/` |
| `overnight.py` | Unattended #160 sweep: preflight, stage 1 → stage 2, summary (`infra/scripts/overnight-sweep.sh`) |
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
