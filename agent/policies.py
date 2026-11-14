"""Trading policies.

Each policy implements:

    reset(view)    start of a market
    update(view)   after every step, including the last
    act() -> Action

`view` is the `Observation`, or the full `MarketState` for policies with
`full_information = True` (only the oracle).
"""
from __future__ import annotations

import numpy as np

from agent.belief import BeliefFilter
from agent.qmdp import QMDPSolver
from market.sim import Action, MarketParams, MarketState, Observation, logit


class BuyAndHold:
    """Buy one large lot of the side favored at the start and hold it."""

    name = "buy-and-hold"
    full_information = False

    def reset(self, obs: Observation) -> None:
        self.order = Action.BUY_YES_LARGE if obs.price >= 0.5 else Action.BUY_NO_LARGE

    def update(self, obs: Observation) -> None:
        pass

    def act(self) -> Action:
        order, self.order = self.order, Action.HOLD
        return order


class QMDPPolicy:
    """Pick argmax_a sum_{outcome, insider} b(outcome, insider) Q(..., a)."""

    full_information = False

    def __init__(self, params: MarketParams, solver: QMDPSolver, model_insiders: bool = True):
        self.solver = solver
        self.filter = BeliefFilter(params, model_insiders=model_insiders)
        self.name = "QMDP" if model_insiders else "QMDP (insider-blind)"

    def reset(self, obs: Observation) -> None:
        self.obs = obs
        self.position = 0
        self.beliefs = [self.filter.reset(obs)]

    def update(self, obs: Observation) -> None:
        self.obs = obs
        self.position += obs.executed
        self.beliefs.append(self.filter.update(obs))

    def act(self) -> Action:
        q = self.solver.q_values(self.obs.t, float(logit(self.obs.price)), self.position)
        return Action(int(np.argmax(np.einsum("yi,yia->a", self.filter.belief, q))))


class Oracle:
    """Upper bound: acts on the true outcome and insider status."""

    name = "oracle"
    full_information = True

    def __init__(self, solver: QMDPSolver):
        self.solver = solver

    def reset(self, state: MarketState) -> None:
        self.state = state

    def update(self, state: MarketState) -> None:
        self.state = state

    def act(self) -> Action:
        s = self.state
        q = self.solver.q_values(s.t, s.log_odds, s.position)
        return Action(int(np.argmax(q[s.outcome, int(s.insider)])))
