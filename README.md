# Trading Against Insiders

A trading agent for binary prediction markets that keeps a belief over whether
an informed trader is active, and uses it to decide when to trade, follow, or
stay out. Final project for AA228/CS238 (Stanford, Fall 2026).

The trader's problem is a POMDP. The hidden state is the outcome and whether
an insider is trading; the agent observes the price, public news, other
participants' large trades with the wallet features behind them, and its own
fills. Actions are buy YES, buy NO (small or
large lot), hold, or exit.

## Layout

```
market/sim.py          shared interface (MarketParams, MarketState, Observation, Action,
                       reset, step) and the toy simulator
market/flow.py         large trades from retail, insider, and copy-bot wallets; wallet
                       features; exact likelihood with trader type marginalized out
agent/belief.py        exact Bayes filter over (outcome, insider active)
agent/qmdp.py          QMDP solver: finite-horizon value iteration on the fully observed problem
agent/policies.py      buy-and-hold, QMDP, insider-blind QMDP, oracle
experiments/evaluate.py  runs all policies on the same markets, prints metrics, writes plots
tests/                 accounting, filter calibration, solver sanity checks
```

## Running

```
pip install numpy matplotlib pytest
pytest
python -m experiments.evaluate --markets 1000    # writes results/
```

## Interface for the simulator

- `reset(params, rng) -> (MarketState, Observation)` and
  `step(state, action, rng, params) -> (MarketState, Observation, reward)`.
- `step` must be a pure function of its arguments; planners call it on sampled states.
- Policies see only `Observation`. `Observation.trades` lists other participants'
  large trades (side, USD size, wallet age, fresh funding, category share,
  longshot record); the trader type stays hidden. `MarketState.flow_types` holds
  the true types for evaluation. `score` is the insider log-likelihood ratio of
  those trades, kept as a one-number summary.
- A learned wallet-feature model plugs into `BeliefFilter(flow_loglik=...)`, a
  function `(obs, params) -> log P(trades | outcome, insider)` as a 2x2 array.
  The default, `market.flow.flow_loglik`, is exact for the simulator.
- Large trades are a sample of the flow, not an extra price driver: the price
  equation is unchanged, and `insider_drift` is the net impact of informed flow.
- Fills happen at the post-impact price. Filling at the pre-impact price lets a
  policy profit from its own impact.
- The filter and QMDP use the stub's dynamics. If the simulator's dynamics change,
  `BeliefFilter.update` and `QMDPSolver._expected_value` must change with them.

## Current results (toy simulator, uncalibrated, 1000 markets)

| Policy | Mean return ($) | Sharpe | Adverse-selection loss | Trades |
|---|---|---|---|---|
| buy-and-hold | -0.07 ± 0.04 | -0.05 | 0.00 | 1.0 |
| QMDP, insider-blind | -0.27 ± 0.06 | -0.15 | 0.72 | 13.1 |
| QMDP | +1.02 ± 0.06 | 0.59 | 0.07 | 14.4 |
| oracle | +2.51 ± 0.03 | 2.56 | 0.00 | 9.0 |

The insider-aware filter flags 96% of insiders, with a median delay of 2 steps
and 29 steps of lead before resolution. 23% of insider-free markets are
flagged at least once (1.1% of steps) at a 0.5 threshold.

These numbers come from a simulator whose dynamics and wallet profiles the
agent knows exactly, with placeholder parameters that have not yet been fit to
Polymarket data; they are an upper bound on what to expect from a calibrated
one.
