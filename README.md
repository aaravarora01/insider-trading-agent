# Trading Against Insiders

A trading agent for binary prediction markets that keeps a belief over whether
an informed trader is active, and uses it to decide when to trade, follow, or
stay out. Final project for AA228/CS238 (Stanford, Fall 2026).

The trader's problem is a POMDP. The hidden state is the outcome and whether
an insider is trading; the agent observes the price, public news, an
order-flow score, and its own fills. Actions are buy YES, buy NO (small or
large lot), hold, or exit.

## Layout

```
market/sim.py          shared interface (MarketParams, MarketState, Observation, Action,
                       reset, step) and a stub simulator
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
- Policies see only `Observation`. `score` is the order-flow summary; a learned
  wallet-feature model plugs into `BeliefFilter(score_loglik=...)`.
- Fills happen at the post-impact price. Filling at the pre-impact price lets a
  policy profit from its own impact.
- The filter and QMDP use the stub's dynamics. If the simulator's dynamics change,
  `BeliefFilter.update` and `QMDPSolver._expected_value` must change with them.

## Current results (stub simulator, 1000 markets)

| Policy | Mean return ($) | Sharpe | Adverse-selection loss | Trades |
|---|---|---|---|---|
| buy-and-hold | -0.07 ± 0.04 | -0.05 | 0.00 | 1.0 |
| QMDP, insider-blind | -0.25 ± 0.06 | -0.14 | 0.82 | 13.2 |
| QMDP | +1.10 ± 0.05 | 0.64 | 0.11 | 14.8 |
| oracle | +2.52 ± 0.03 | 2.56 | 0.00 | 8.4 |

The insider-aware filter flags 97% of insiders, with a median delay of 2 steps
and 29 steps of lead before resolution. 36% of insider-free markets are
flagged at least once (1.7% of steps) at a 0.5 threshold.

These numbers come from a stub simulator whose dynamics the agent knows
exactly; they are an upper bound on what to expect from a calibrated one.
