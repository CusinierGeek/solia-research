# Findings — can a model find a tradeable edge on Pump.fun?

**Short answer: no, not at the speed of a bot that reacts in seconds.** This page summarizes what was tested between
late October and 10 November 2026, in order, and what each test showed. Details, commands and caveats are in
[`ML-MODEL.md`](ML-MODEL.md); raw reports and the scripts that produced them are in [`reports/`](reports/).

Everything here is **paper trading or offline simulation** on public on-chain data (pumpapi.io's free stream and hourly
replay archives). No real order was ever sent.

## How results were judged

A result only counted if it held up under rules fixed **before** looking at the test data:

- **Costs included**: Pump.fun/PumpSwap fees, network fees, account rent, slippage on the pool's real reserves, and a
  scoring/fill latency of a few seconds. Many backtests seen online skip these; on this market they decide everything.
- **Out-of-sample**: settings are chosen on one period and confirmed, unchanged, on a later one, with an embargo between them.
- **Uncertainty**: means come with 95 % bootstrap confidence intervals, resampled by token (trades on the same token aren't
  independent). The edge criterion asks for a lower bound above 0, at least 300 trades, on at least two separate periods.
- **Pre-registration**: for the last studies, the hypothesis, population and pass/fail threshold were committed to git
  before the test data was read.

## 1. The Transformer: good scores, no edge (Sprints 2–8, feature enrichment)

A multi-horizon Transformer predicted pump and rug probabilities from 10-second feature snapshots.

- On paper it looked strong: ROC AUC around 0.93–0.96 for "pump" at short horizons.
- But the "pump" label (price touches +X % at any time in the horizon) rewards **volatility**, not direction. A high
  `pump_prob` meant the price would move a lot, often both ways: in one bucket, +50 % was touched 42 % of the time and −30 %
  47 % of the time.
- Feature ablations (momentum, holder concentration, buy pressure, market regime, wallet reputation, token metadata), more
  training data, recency weighting and regularization moved the metrics by less than the run-to-run noise.

## 2. The paper bot: a winning backtest that was racing the rug-pullers (#221, #252, sessions 5–6)

An exit simulator (`exit_sim`) replays the bot's decisions with its real fill rules; it reproduced 95 of 95 paper trades.

- A frozen setting (`pump ≥ 0.5`, take-profit +200 %, volatility stop, 15 min timeout) backtested at **+17 % per trade on two
  periods** and beat random entries matched on the same tokens.
- On later, untouched data it gave +2.0 %, +15.1 % and −12.8 %. Paper session 5 made +31.9 % over 57 trades; session 6 lost
  (mean −9 % over 151 trades, 34 % total losses). Net over both sessions: about zero.
- Why: the tokens the model liked were mostly **operator tokens**. On session 5, liquidity was pulled on 47 of 48 traded
  tokens, by the wallet that created the pool on 46. 78 % of pools of ≥ 50 SOL on a full day were drained by their creator,
  a median 6.5 minutes after creation, **in a single transaction**. The bot won only when it left before the drain, and
  neither the creator's history nor the clock predicted when that would happen.

## 3. Tokens whose liquidity can't be pulled: plain rules and LightGBM ("direction A")

Restricting to tokens with ≥ 90 % of their LP burned removes the drain risk.

- All 12 pre-set rule cells (breakouts, buy pressure, several stops) lost on all three periods, every CI below 0 (best:
  −6 to −8 % per trade). They beat random entries (−26 to −53 %), but not the costs.
- One LightGBM attempt (take-profit before stop) ranked well but lost −16 % and −31 % on the test periods: large holders
  dump in one transaction, through the stop.
- Older tokens (≥ 1 h) rarely crash, but they don't pump either.

## 4. A rug-risk score (#260)

A LightGBM score with SHAP explanations, predicting sudden rugs and gradual dumps.

- It **failed its pre-registered target**: the "safe" set it kept still had 7.7 % rugs, covering only 23 % of tokens.
- It does beat the common "LP burned ≥ 90 %" rule: 9.8 % rugs vs 28.8 % at equal coverage. Useful as information, not
  enough to trade on.

## 5. Established tokens (#261)

Tokens that survived 1 to 6 hours, held for hours to days.

- Every mean net return is negative; e.g. entries at 6 h of age, held 24 h: **−28 % [−32, −23]**.
- About 60 % of losses are sudden (no chance to exit). The only clear warning sign, liquidity leaving the pool, comes too
  late to help.

## 6. Winning wallets and copy trading (#269, #271)

Every trade in 92 hours of archives was rebuilt into per-wallet positions (≈ 96 M trades; 1.7 M wallets in the first period alone).

- The market is **negative-sum**: closed positions lost 170 k SOL in one period and 66 k SOL in the next.
- Winners **do exist and persist**: 96 % of one period's top 100 wallets were profitable in the next.
- But their edge is **speed and position**: 15 of the top 20 buy in the same block as the token's creation and hold 3–9
  seconds.
- Copying them 3 s later loses −8.6 % per copy (random wallets: −16 %).
- A pre-registered test on new data copied 1 860 **slow** persistent winners (entering ≥ 8 s after creation, holding
  ≥ 60 s). The leaders kept winning (+7.6 k SOL, 63 % profitable), yet copies lost **−18.2 % [−18.9, −17.6]** over 168 k
  copies, worse than random wallets. Even with no costs and no delay, the median copy made 0 %: the leader's buy moves the
  price, and the follower buys after it and sells into the leader's own exit.

## Conclusion

Pump.fun is a negative-sum market. Its consistent winners profit from being first, in the creation block or within
seconds. The liquidity pools are often controlled by the token's creator, who can empty them in one transaction. A
model trading seconds later, on public data, ends up on the wrong side of both. Every approach above was measured after
costs, on data it was not tuned on, and none produced an edge whose confidence interval stays above zero.

The project stops its search for a trading strategy here. The infrastructure and the methods (replay from free archives,
realistic exit simulation, pre-registered tests, token-clustered bootstrap) remain usable for other questions.
