# Status update: notes and numbers

Raw material for the status update (task 10). Every number below comes from the code
at commit 788e58f plus the plot changes committed with this file. Rerun with:

```
python -m experiments.evaluate --markets 1000 --out results/toy
python -m experiments.evaluate --markets 300 --params calibrated --out results/calibrated
```

Figures, `metrics.json` and the printed `summary.txt` for each run are in
`results/toy/` and `results/calibrated/`, next to this file.

---

## 1. Project in one paragraph

A trading agent for binary prediction markets that keeps a belief over the outcome Y and
whether an informed trader is active (I_t), and trades on it. It is a POMDP. The hidden
state is (Y, I_t); the agent sees the price, public news, other participants' large trades
with wallet features (fresh wallet, wallet age, category concentration, longshot record),
and its own fills. Actions: buy YES or NO (small or large lot), hold, exit, or an exact
order size. The question for the status update: does modelling insiders help, compared
with a buy-and-hold baseline and an agent that assumes no insiders exist?

## 2. What is built

| Component | Where | Owner | Notes |
|---|---|---|---|
| Market simulator | `market/sim.py`, `market/flow.py` | Aarav | log-odds price: impact + news + insider drift + noise; marked-Poisson large trades from retail, insider and copy-bot wallets; cash budget with solvency check |
| Polymarket data pipeline | `polymarket/data.py`, `download.py`, `selection.py` | Aarav | pulls 11 markets (3 calibration, 8 incident) from a 46 GB Hugging Face dataset by reading only the needed row groups |
| Flagged wallets | `polymarket/flagged_wallets.csv` | Aarav | 21 wallets named by journalists or on-chain analysts, with sources and a confidence label |
| Simulator calibration | `polymarket/calibrate.py` → `market/calibrated.json` | Aarav | price dynamics, impact, trade rates and sizes; 1 step = 1 hour |
| Wallet observation model | `polymarket/history.py`, `observation.py` → `market/wallet_profiles.json` | Aarav | P(wallet features \| retail / insider / copy-bot), flagged wallets as weak labels |
| Exact Bayes filter | `agent/belief.py` | Anish | 2x2 belief over (Y, I_t); wallet type summed out |
| Particle filter | `agent/particle.py` | Anish's task, done by Aarav | bootstrap filter over (Y, I_t), systematic resampling; matches the exact filter |
| Buy-and-hold baseline | `agent/policies.py` | Anish's task, done by Aarav | side from p0, half the budget, hold to resolution (as in the proposal) |
| QMDP planner | `agent/qmdp.py` | Anish | finite-horizon value iteration on the fully observed problem, weighted by the belief |
| Evaluation harness | `experiments/evaluate.py`, `plots.py` | Anish | return, Sharpe, adverse-selection loss, detection delay and lead, false positives; plots |
| Tests | `tests/` (49 tests) | both | accounting, budget, filter calibration, particle vs exact filter, solver, calibration, observation model |

Policies compared: buy-and-hold; QMDP with an insider-blind filter (assumes no insider
ever appears); QMDP with the insider-aware filter (exact, and particle with 2000
particles); an oracle that sees the true outcome and insider status (upper bound).

## 3. Data and calibration

- **Source:** Hugging Face `SII-WANGZJ/Polymarket_data` (MIT licence), every Polymarket
  CLOB fill from 2022-11-21 to 2026-10-04.
- **Markets:** 3 calibration markets plus 8 incident markets with publicly flagged
  wallets, 11 in all. 11,317 quiet hours (no insider window) used for price dynamics.
- **Units:** 1 step = 1 hour; 1 simulator contract = 1,000 real contracts; a large trade
  is a taker order of at least 2,000 contracts.

Fitted parameters:

| Parameter | Value | From |
|---|---|---|
| noise_sd | 0.054 / hour | quiet-hour log-odds variance, corrected for bid-ask bounce |
| news_mean, news_weight | 0.010, 0.021 | drift toward the outcome in quiet hours (not significant: 0.0002 ± 0.0007) |
| insider_drift | 0.027 / hour | median over 8 incidents, first flagged fill to reveal |
| impact | 0.00065 per sim contract | median slope of hourly log-odds on net taker flow |
| uninformed large orders | 3.3 / hour | median over markets, last 72 h before the end |
| insider_rate | 0.79 / hour | flagged wallets' winning-side large orders |
| copy_follow_rate | 0.29 / hour | other wallets' winning-side minus losing-side orders in insider windows |

Design choices (not estimated): 72-hour horizon; onset rate set so half of simulated
markets get an insider; copy-bots are 1/7 of uninformed flow; budget = 6 contracts' worth.

Findings worth mentioning:
- Large orders must be defined by contract count, not USD. USD-large orders lean 90/10
  toward the favourite (a dollar buys more longshot contracts); contract-large orders
  split about 50/50, as the simulator assumes.
- Activity surges near resolution with or without insiders (e.g. 9 → 49 large orders an
  hour). So the follower rate is measured as same-side minus opposite-side flow.
- In three incidents the price passed 0.95 hours before the documented announcement.
- Many flagged wallets traded mostly with limit orders, so insider orders are counted
  whether the wallet was maker or taker.
- Data gotcha: the dataset's `usd_amount` records the counterparty's cash on some
  fills. One flagged wallet's $32.5k stake showed as $329k.

## 4. Wallet observation model

- **Features**, computed only from the wallet's trades in other markets before the order:
  - fresh (no earlier trades anywhere; stands in for the proposal's funding-source
    feature, which the data lacks);
  - for wallets that are not fresh: wallet age, share of earlier markets in the same
    category, and longshot record (bets bought below 20c: any, how many, win rate).
- **Weak labels:**
  - insider: flagged wallets' winning-side orders before the reveal, weighted by flag
    confidence;
  - copy-bot: same-side order within 60 s of the same other wallet, at least 3 times;
  - retail: everyone else.
- **Fit:** supervised, weighted per wallet. Semi-supervised EM was tried and dropped:
  on simulated data its likelihood fell as it iterated and it put 2.6-4x too many orders
  in the insider type.
- **Evaluation:** leave one incident out (fit without an incident's wallets, then score
  them):

| Held-out incident (market id) | AUC, insider vs other large orders |
|---|---|
| 532742 | 0.98 |
| 916440 | 0.97 |
| 554010 | 0.96 |
| 560868 | 0.95 |
| 1706788 | 0.93 |
| 1198423 | 0.89 |
| 1419011 | 0.84 |
| 635246 | 0.63 |
| **median** | **0.94** (in-sample 0.90) |

- **Effect on the agent:** with trade size as the only wallet feature, calibrated false
  positives were 29% of insider-free markets and the delay 5 hours. With the full
  features they are 11% and 2 hours.

## 5. First results

### Calibrated simulator, 300 markets (151 with an insider)

| Policy | Mean return ($) | Sharpe | Adverse-selection loss | Trades / market |
|---|---|---|---|---|
| buy-and-hold | −0.06 ± 0.12 | −0.03 | 0.00 | 1.0 |
| QMDP, insider-blind | −0.49 ± 0.15 | −0.19 | 1.55 | 11.6 |
| QMDP | **+1.41 ± 0.14** | **0.59** | 0.18 | 18.1 |
| QMDP, particle filter | +1.38 ± 0.14 | 0.57 | 0.18 | 18.1 |
| oracle | +2.69 ± 0.06 | 2.48 | 0.00 | 7.1 |

| Detection (flag when P(insider) > 0.5) | Exact filter | Particle filter |
|---|---|---|
| insiders detected | 94.7% | 95.4% |
| median delay after onset | 2 h | 2 h |
| median lead before resolution | 43 h | 42.5 h |
| insider-free markets flagged at least once | 11.4% | 12.1% |
| insider-free hours flagged | 0.4% | 0.4% |

Return split by whether the market had an insider:

| Policy | Insider markets | Insider-free markets | Share of markets with a loss |
|---|---|---|---|
| buy-and-hold | −0.15 ± 0.17 | +0.04 ± 0.16 | 37% |
| QMDP, insider-blind | −1.38 ± 0.14 | +0.42 ± 0.23 | 69% |
| QMDP | +2.38 ± 0.11 | +0.42 ± 0.23 | 27% |
| oracle | +2.65 ± 0.08 | +2.73 ± 0.09 | 0% |

### Toy simulator (placeholder parameters), 1000 markets (625 with an insider)

| Policy | Mean return ($) | Sharpe | Adverse-selection loss | Trades / market |
|---|---|---|---|---|
| buy-and-hold | −0.12 ± 0.06 | −0.06 | 0.00 | 1.0 |
| QMDP, insider-blind | −0.22 ± 0.06 | −0.12 | 0.73 | 12.9 |
| QMDP | +1.06 ± 0.06 | 0.60 | 0.12 | 14.8 |
| QMDP, particle filter | +1.05 ± 0.06 | 0.59 | 0.11 | 14.9 |
| oracle | +2.51 ± 0.03 | 2.55 | 0.00 | 8.4 |

Detection: 96% of insiders, median delay 2 steps, lead 28 steps; 31% of insider-free
markets flagged at least once (1.3% of steps). The particle filter matches within a
percentage point.
- Split by insider presence: QMDP +1.30 with an insider and +0.65 without; insider-blind
  −0.76 and +0.68.

### Figures (`results/`)

1. **`calibrated/belief_traces.png`**: P(insider active), P(YES wins) and the market
   price over 72 hours for four markets.
   - The markets are the first three in seed order with an insider starting after hour
     5, plus the first insider-free market. They were not hand-picked.
   - The dashed line marks the true insider onset.
   - Shows the belief jumping within a few hours of onset while the price lags.
   - Panel 2 shows a false alarm near hour 25 that resolves, then detection after onset.
2. **`calibrated/onset_aligned.png`**: P(insider active) around the true onset across all
   151 insider markets.
   - Lines: median for each filter; band: middle half of markets (exact filter).
   - Dotted line: the median in insider-free markets.
   - The median crosses 0.5 about 2 hours after onset and passes 0.9 by hour 3; the
     slowest quarter of markets passes 0.5 between hours 4 and 5.
   - The exact and particle-filter curves lie on top of each other.
3. **`calibrated/returns.png`**: return per market for each policy (box: middle half;
   ◆ mean; white bar: median).
   - Buy-and-hold's box is wide because its outcome is roughly win or lose a fixed stake.
   - The insider-blind agent's median is negative.
4. `toy/*.png`: the same three figures for the toy simulator.

The task list asked for two plots: the belief trace (1) and the return distribution
(3). Figure 2 is an extra summary of all markets that backs up figure 1.

### What the results say

- Modelling insiders is what makes the agent profitable. In insider markets, the
  insider-blind agent loses −1.38: it reads the insider's push as noise and trades
  against it (adverse-selection loss 1.55). The insider-aware agent makes +2.38 in the
  same markets, 90% of the oracle's +2.65. It gets there by flagging the insider about 2
  hours in and then following.
- The two filters reach the same decisions. The particle filter matches the exact
  filter's returns and detection within noise, so the proposal's particle filter works.
  The exact filter is a cheaper way to compute the same belief while the state is 2x2.
- Buy-and-hold makes about zero (slightly negative from impact): the opening price is
  fair by construction.

## 6. Caveats to state honestly

1. **Upper bound.** The agent knows the simulator's dynamics and wallet profiles exactly.
   Replaying real markets will be harder.
2. **Part of QMDP's return is not about insiders.** In insider-free markets, both QMDP
   agents make +0.42 (calibrated) and the blind one makes the same.
   - In the simulator, the price takes a random-walk noise term that carries no
     information. The agent observes the news directly, so it can trade against noise
     moves.
   - This edge likely does not exist in real markets, where news is not observed
     separately.
   - The insider-attributable effect is the comparison within insider markets
     (+2.38 vs −1.38), or the aware-minus-blind gap in the overall mean (+1.90).
3. **Base rate.** Half the simulated markets have an insider; in real markets far fewer
   do. Then the 11% of insider-free markets flagged at least once matters more than the
   95% detection rate.
4. **Small, biased labels.** The insider wallet profile rests on 19 wallets from 8
   incidents notable enough to be reported. One incident (635246) gets AUC 0.63 when
   held out.
5. **Repeat orders.** One wallet's repeat orders are treated as independent trades, so a
   single real wallet can be overcounted.
6. **Time-varying flow.** Real activity surges before resolution. The simulator's
   uninformed rate is constant, so replay on real markets needs a time-varying rate or
   the surge will look like an insider.
7. **Budget.** The QMDP planner ignores cash. With the default budget (6), it only binds
   after losses, and adding it changed QMDP's Sharpe by 0.01.

## 7. Deviations from the proposal

| Proposal | Now | Why |
|---|---|---|
| Extend ABIDES with binary contracts and Avellaneda–Stoikov market makers | own lightweight simulator, calibrated to Polymarket | see below |
| Particle filter | both a particle filter and the exact filter; they agree | the hidden state (Y, I_t) has 4 values, so the exact sum is cheap; the particle filter is ready for a larger state |
| Funding source as a wallet feature | "fresh wallet" (no earlier trades) | the dataset has addresses but no funding transfers |
| Maker-only trading | the agent takes liquidity at the post-impact price | simpler; fees are 0 in the simulator. Filling at the pre-impact price let a policy profit from its own impact |
| — | insider never stops trading once active | simplification; a natural extension for the particle filter |

**Why ABIDES is deferred.** These are suggested points; check that you agree.
- The planners call the simulator's `step` as a generative model, very many times:
  QMDP value iteration, and POMCPOW will call it millions of times. Our `step` runs at
  about 7,700 steps/s in pure Python. ABIDES simulates an order book message by message
  at nanosecond resolution, orders of magnitude slower per market-hour.
- Our Polymarket data is fills, not order-book snapshots or message traffic, so it cannot
  pin down the market-maker and order-book parameters ABIDES needs. Hourly price
  dynamics, impact and trade rates are what the data supports, and those are what the
  current simulator uses.
- Adding binary contracts and Avellaneda–Stoikov market makers to ABIDES is a large
  engineering job that does not change the POMDP the agent solves.
- The current simulator exposes the same interface (`reset`, `step`, `Observation`), so
  ABIDES can replace it later without touching the agent.

## 8. Next steps

1. **POMCPOW** (the proposal's main comparison): does planning ahead for uncertainty beat
   QMDP? QMDP assumes everything is known after one step, so it never waits just to learn
   more; POMCPOW can. It can reuse `market.sim.step` as its generative model and
   `ParticleFilter` for beliefs.
2. **Replay flagged incidents** on real Polymarket trades. This needs:
   - an adapter from real trades to `Observation`;
   - a time-varying uninformed rate;
   - merging a wallet's repeat orders.
3. **2026 midterm markets.** After they resolve, look for markets where the agent put
   high insider probability before a surprising outcome (exploratory only). Note that the
   current dataset ends 2026-10-04, so this needs newer data.
4. Possible simulator extensions: insiders who stop trading, several insiders, and a cash
   or risk term in the planner.

## 9. Division of work so far

- **Aarav:** simulator and order flow, Polymarket data and flagged wallets, calibration,
  wallet observation model, budget; also the particle filter and buy-and-hold (tasks 5–6).
- **Anish:** market interface, exact belief filter, QMDP solver, evaluation harness
  (initial commit).

## 10. References used so far

- Proposal references: Sunberg & Kochenderfer 2018 (POMCPOW); Byrd et al. 2020 (ABIDES);
  Avellaneda & Stoikov 2008.
- Data: SII-WANGZJ/Polymarket_data on Hugging Face.
- Flagged-wallet sources: listed per wallet in `polymarket/flagged_wallets.csv`.
- Roll (1984) for the bid-ask bounce correction in the noise estimate.
