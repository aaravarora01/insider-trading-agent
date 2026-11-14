"""Trading policies.

Each policy implements:

    reset(view)    start of a market
    update(view)   after every step, including the last
    act() -> Action | Order

`view` is the `Observation`, or the full `MarketState` for policies with
`full_information = True` (only the oracle).
"""
from __future__ import annotations

import numpy as np

from agent.belief import BeliefFilter
from agent.particle import ParticleFilter
from agent.qmdp import QMDPSolver
from market.sim import Action, MarketParams, MarketState, Observation, Order, logit


class BuyAndHold:
    """The proposal's baseline: decide once from the opening price p0.

    Buy YES if p0 >= 0.5, otherwise NO, spending `fraction` of the starting
    budget, and hold to resolution. A YES contract costs p0 and a NO contract
    1 - p0; the order is whole contracts, so it spends at most that fraction
    (the simulator's position limit and impact can shrink it further).
    """

    name = "buy-and-hold"
    full_information = False

    def __init__(self, params: MarketParams, fraction: float = 0.5):
        self.params = params
        self.fraction = fraction

    def reset(self, obs: Observation) -> None:
        side = 1 if obs.price >= 0.5 else -1
        cost = obs.price if side == 1 else 1.0 - obs.price
        self.order = Order(side * int(self.fraction * self.params.budget // cost))

    def update(self, obs: Observation) -> None:
        pass

    def act(self) -> Action | Order:
        order, self.order = self.order, Action.HOLD
        return order


class QMDPPolicy:
    """Pick argmax_a sum_{outcome, insider} b(outcome, insider) Q(..., a)."""

    full_information = False

    def __init__(self, params: MarketParams, solver: QMDPSolver, model_insiders: bool = True,
                 particles: int | None = None, seed: int = 0):
        """`particles=None` tracks the belief with the exact filter; a number
        uses a particle filter with that many particles."""
        self.solver = solver
        if particles is None:
            self.filter = BeliefFilter(params, model_insiders=model_insiders)
            self.name = "QMDP" if model_insiders else "QMDP (insider-blind)"
        else:
            self.filter = ParticleFilter(params, particles, model_insiders=model_insiders,
                                         seed=seed)
            self.name = ("QMDP (particle filter)" if model_insiders
                         else "QMDP (insider-blind, particle filter)")

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
