# Trading Against Insiders

A trading agent for binary prediction markets that keeps a belief over whether
an informed trader is active, and uses it to decide when to trade, follow, or
stay out. Final project for AA228/CS238 (Stanford, Fall 2026).

The trader's problem is a POMDP. The hidden state is the outcome and whether
an insider is trading; the agent observes the price, public news, other
participants' large trades with the wallet features behind them, and its own
fills. Actions are buy YES, buy NO (small or
large lot), hold, or exit, or an `Order` for an exact number of contracts. The
agent starts with a cash budget and its orders are shrunk to stay solvent.

## Layout

```
market/sim.py          shared interface (MarketParams, MarketState, Observation, Action,
                       reset, step) and the toy simulator
market/calibration.py  MarketParams fitted to Polymarket data (from market/calibrated.json)
market/flow.py         large trades from retail, insider, and copy-bot wallets; wallet
                       features; exact likelihood with trader type marginalized out
agent/belief.py        exact Bayes filter over (outcome, insider active); shared observation likelihood
agent/particle.py      particle filter over the same pair (the proposal's belief update)
agent/qmdp.py          QMDP solver: finite-horizon value iteration on the fully observed problem
agent/policies.py      buy-and-hold, QMDP (exact or particle filter), insider-blind QMDP, oracle
polymarket/data.py     pulls markets and trades from the SII-WANGZJ/Polymarket_data
                       Hugging Face dataset, converted to the YES perspective
polymarket/selection.py  the markets we pull (calibration and incident)
polymarket/calibrate.py  estimates the simulator's parameters; writes market/calibrated.json
polymarket/history.py  each large trader's history across all markets (11 GB scan)
polymarket/categories.py  market categories from keyword rules
polymarket/observation.py  fits P(wallet features | trader type); writes market/wallet_profiles.json
polymarket/flagged_wallets.csv  publicly flagged wallets with sources (weak insider labels)
experiments/evaluate.py  runs all policies on the same markets, prints metrics, writes plots
results/               evaluation outputs (figures, metrics.json, summary.txt) and
                       status_update_notes.md: numbers, figures and caveats for the status update
tests/                 accounting, budget, filter calibration, particle vs exact filter, solver checks
```

## Running

```
pip install numpy matplotlib pytest
pytest
python -m experiments.evaluate --markets 1000 --out results/toy
python -m experiments.evaluate --markets 300 --params calibrated --out results/calibrated
# --particles N sets the particle filter's size (default 2000; 0 skips it)
```

## Polymarket data

```
pip install duckdb pandas pyarrow huggingface_hub
python -m polymarket.download      # about 10 min the first time; writes data/ (gitignored)
```

The source is [SII-WANGZJ/Polymarket_data](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data)
(MIT licence): every Polymarket CLOB fill from 2022-11-21 to 2026-10-04. The download
fetches `markets.parquet` (0.5 GB), indexes where each market's rows sit in the 46 GB
`trades.parquet`, and reads only those row groups. Output:

- `data/polymarket/markets.json`: the selected markets, with `outcome` = 1 if answer1 won.
- `data/polymarket/trades/<id>.parquet`: one row per fill, with `time`, `maker`, `taker`,
  `token`, `taker_side` (+1 if the taker bought YES or sold NO), `yes_price`, `contracts`,
  and `taker_usd` / `maker_usd`, each party's stake on the side it took.
- `data/polymarket/prices/<id>.parquet`: hourly last price, VWAP, contracts, USD volume,
  fill count.

Caveats:
- Use `taker_usd` / `maker_usd` for trade sizes, not the raw `usd_amount`. When a YES
  buyer is matched with a NO buyer, the raw row records the NO buyer's cash, so one
  flagged wallet's $32.5k stake shows up as $329k.
- `end_date` in `markets.parquet` is not always the resolution time ("US strikes Iran
  by February 28, 2026?" says Jan 30). Use the last fill or the price path instead.
- The dataset has wallet addresses but no funding source, wallet creation date, market
  category, or insider labels.

### Flagged wallets

`polymarket/flagged_wallets.csv` lists 21 wallets that journalists or on-chain analysts
named as suspected insiders, each with its sources, how the address was obtained, and
a confidence label (`strong`, `multi_source`, `single_source`, `weak`). These are weak
labels: only one is tied to filed charges. The ACDC report's 152 wallets are not public.
`polymarket.flagged.flagged_activity()` summarizes each wallet's fills in the downloaded
markets.

## Interface for the simulator

- `reset(params, rng) -> (MarketState, Observation)` and
  `step(state, action, rng, params) -> (MarketState, Observation, reward)`.
- `step` must be a pure function of its arguments; planners call it on sampled states.
- Policies see only `Observation`. `Observation.trades` lists other participants'
  large trades (side, contracts, and wallet features: fresh wallet, age,
  category share, longshot record); the trader type stays hidden. `MarketState.flow_types` holds
  the true types for evaluation. `score` is the insider log-likelihood ratio of
  those trades, kept as a one-number summary.
- A learned wallet-feature model plugs into `BeliefFilter(flow_loglik=...)` (or
  `ParticleFilter`), a
  function `(obs, params) -> log P(trades | outcome, insider)` as a 2x2 array.
  The default, `market.flow.flow_loglik`, is exact for the simulator.
- Large trades are a sample of the flow, not an extra price driver: the price
  equation is unchanged, and `insider_drift` is the net impact of informed flow.
- Fills happen at the post-impact price. Filling at the pre-impact price lets a
  policy profit from its own impact.
- The filters and QMDP use the stub's dynamics. If the simulator's dynamics change,
  `agent.belief.observation_loglik` and `QMDPSolver._expected_value` must change with them.
- Budget: `MarketParams.budget` (default 6, equal to `max_position`) is the starting
  cash. A trade is shrunk until we can pay out either way (`cash >= 0` and
  `cash + position >= 0`). `Observation.cash` reports it. QMDP plans without it; the
  budget only binds after losses.

## Policies and filters

- Buy-and-hold, as in the proposal: from the opening price p0, buy YES if p0 >= 0.5
  and NO otherwise, spending `fraction` (default 0.5) of the budget, and hold to
  resolution. At p0 = 0.7 with a budget of 6 that is 4 YES contracts.
- `ParticleFilter` is a bootstrap filter over (outcome, insider active) with the wallet
  type summed out inside the flow likelihood, resampled systematically when the effective
  sample size falls below half. Since the hidden pair has only four values,
  `BeliefFilter` computes the same posterior exactly; tests check that the two agree
  (mean gap under 0.02 with 4000 particles). The particle filter is the one to extend
  when the hidden state grows. Because the outcome never changes, resampling could lose
  one outcome entirely; a floor keeps at least 0.1% of particles on each.
- `QMDPPolicy(params, solver, particles=N)` uses the particle filter instead of the
  exact one.

## Current results (toy simulator, uncalibrated, 1000 markets)

| Policy | Mean return ($) | Sharpe | Adverse-selection loss | Trades |
|---|---|---|---|---|
| buy-and-hold | -0.12 ± 0.06 | -0.06 | 0.00 | 1.0 |
| QMDP, insider-blind | -0.22 ± 0.06 | -0.12 | 0.73 | 12.9 |
| QMDP | +1.06 ± 0.06 | 0.60 | 0.12 | 14.8 |
| QMDP, particle filter (2000) | +1.05 ± 0.06 | 0.59 | 0.11 | 14.9 |
| oracle | +2.51 ± 0.03 | 2.55 | 0.00 | 8.4 |

The insider-aware filter flags 96% of insiders, with a median delay of 2 steps
and 28 steps of lead before resolution. 31% of insider-free markets are
flagged at least once (1.3% of steps) at a 0.5 threshold. The particle filter's
detection numbers match to within a percentage point.

These numbers come from a simulator whose dynamics and wallet profiles the
agent knows exactly, with placeholder parameters that have not yet been fit to
Polymarket data; they are an upper bound on what to expect from a calibrated
one.

## Calibration

`python -m polymarket.calibrate` fits the simulator to the 11 downloaded markets and writes
`market/calibrated.json` (committed; `market.calibration.calibrated_params()` loads it).
One step is one hour, one simulator contract is 1,000 real contracts, and a large trade is
a taker order of at least 2,000 contracts. The method is in the module docstring.

| Parameter | Value | From |
|---|---|---|
| `noise_sd` | 0.054 / hour | quiet-hour log-odds variance, Roll-corrected, minus the news share |
| `news_mean`, `news_weight` | 0.010, 0.021 | drift toward the outcome in quiet hours (0.0002 ± 0.0007 / hour: not significant; `news_mean` < 0.028 at 95%) |
| `insider_drift` | 0.027 / hour | median over 8 incidents of drift from first flagged fill to reveal |
| `impact` | 0.00065 / sim contract | median OLS slope of hourly log-odds on net taker flow |
| uninformed large orders | 3.3 / hour | median over markets, last 72 h before the market's end |
| `insider_rate` | 0.79 / hour | flagged-wallet large orders on the winning side in the window |
| `copy_follow_rate` | 0.29 / hour | other wallets' winning-side minus losing-side large orders in the window |
| trade sizes | median ~5.6k (uninformed), ~7.7k (insider) contracts | lognormal fits |

Design choices, not estimates: a 72-hour horizon; `onset_rate` set so half of simulated
markets have an insider by the end; copy-bots as 1/7 of uninformed flow. Wallet
features come from the observation model below.

Findings that shaped the method:
- Use a contract threshold, not USD. $1k buys ten times more contracts on a longshot than
  on a favourite, so USD-large orders lean 90/10 toward the favourite; contract-large
  orders split about 50/50 at every price, as the simulator assumes.
- Activity surges near resolution with or without insiders (Mamdani: 9 to 49 large orders
  an hour; Super Bowl: 0.1 to 101). Inside insider windows both sides surge about 5x, so
  the follower rate is measured as same-side minus opposite-side flow. The simulator's
  uninformed rate is constant, so a filter replayed on real markets will need a
  time-varying rate to avoid reading the surge as an insider.
- In three incidents the price hit 0.95+ hours before the documented announcement, so the
  reveal is the earlier of the two.
- Many flagged wallets traded with limit orders (taker on 6-30% of fills for some), so
  insider orders are counted in either role.

## Wallet observation model

`python -m polymarket.history` scans each large trader's activity across every market
(about 11 GB of `trades.parquet` columns over HTTP; resumable). `python -m
polymarket.observation` then fits P(wallet features | retail / insider / copy-bot) and
writes `market/wallet_profiles.json`, which `calibrated_params()` picks up.

Features (`market/flow.py`), all computed from the wallet's trades in *other* markets
before the order, so an insider's winning bet never feeds its own features:

- **fresh**: no earlier trades anywhere else. The dataset has no funding source, so this
  stands in for the proposal's "funding source" feature.
- for wallets that are not fresh: **age** (days since first trade elsewhere),
  **category share** (smoothed share of earlier markets in this market's category, from
  keyword rules in `polymarket/categories.py`), and **longshot record** (resolved bets
  bought below 20c: whether any, how many, smoothed win rate).

Weak labels: flagged wallets' winning-side orders before the reveal are insiders,
weighted by flag confidence; wallets that place a same-side order within 60 s of the
same other wallet at least 3 times are copy-bots; everyone else is treated as retail.
Each wallet counts once per market. 35,073 large orders from 9,816 wallets; 195 insider
orders from 19 wallets; 146 copy-bot wallets.

| Type | Median size | Fresh | Median age | Category share | Has longshots | Win rate |
|---|---|---|---|---|---|---|
| retail | 6.2k | 10% | 63 d | 0.34 | 87% | 0.11 |
| insider | 14.0k | 48% | 4 d | 0.75 | 56% | 0.31 |
| copy-bot | 6.3k | 2% | 141 d | 0.31 | 93% | 0.11 |

Leave-one-incident-out AUC for telling flagged orders from other large orders in the same
market: median 0.94 (0.63 for Google/AlphaRaccoon, an established wallet; 0.84-0.98 for
the rest).

Semi-supervised EM was tried and dropped: on simulated data its likelihood fell as it
iterated and it put 2.6-4x too many orders in the insider type.

## Calibrated results (300 markets, fitted wallet profiles)

| Policy | Mean return ($) | Sharpe | Adverse-selection loss | Trades |
|---|---|---|---|---|
| buy-and-hold | -0.06 ± 0.12 | -0.03 | 0.00 | 1.0 |
| QMDP, insider-blind | -0.49 ± 0.15 | -0.19 | 1.55 | 11.6 |
| QMDP | +1.41 ± 0.14 | 0.59 | 0.18 | 18.1 |
| QMDP, particle filter (2000) | +1.38 ± 0.14 | 0.57 | 0.18 | 18.1 |
| oracle | +2.69 ± 0.06 | 2.48 | 0.00 | 7.1 |

The filter flags 95% of insiders with a median delay of 2 hours and 43 hours of lead;
11% of insider-free markets are flagged at least once (0.4% of hours); the particle
filter gives 95% and 12%. With trade size as
the only wallet feature, false positives rise to 29% and the delay to 5 hours.

Caveats: the insider profile rests on 19 wallets from incidents notable enough to be
reported; one wallet's orders are treated as independent trades, so replaying real
markets will overcount repeat orders; and the uninformed rate should vary with time to
resolution before replay.
