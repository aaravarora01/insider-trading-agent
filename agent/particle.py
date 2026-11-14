"""Particle filter over the hidden pair (outcome Y, insider active I_t).

The proposal's belief update: a bootstrap particle filter. Each particle is a
pair (y, i). Per step, an inactive particle's insider switches on with
probability `onset_rate`, every particle is weighted by
P(obs | y, i) from `agent.belief.observation_loglik` (wallet types are summed
out inside the flow likelihood, so they never need to be sampled), and the
set is resampled systematically when the effective sample size falls below
half the particles.

Because (Y, I_t) has only four values, `agent.belief.BeliefFilter` computes
the same posterior exactly; it is the Rao-Blackwellized limit of this filter,
and the tests check that the two agree. The particle filter is the version to
extend when the hidden state grows (insiders who stop trading, several
insiders, continuous wallet types), where exact sums no longer fit.

One known weakness: Y never changes, so once resampling has removed every
particle with one outcome, that outcome's probability stays zero even if later
evidence favours it. With enough particles this is rare; `prob_floor` keeps a
small share of particles on each outcome to guard against it.
"""
from __future__ import annotations

import numpy as np

from agent.belief import FlowLogLikelihood, observation_loglik, simulator_flow_loglik
from market.sim import MarketParams, Observation, logit


class ParticleFilter:
    """Same interface as BeliefFilter: reset, update, belief, prob_yes, prob_insider."""

    def __init__(self, params: MarketParams, particles: int = 2000,
                 model_insiders: bool = True,
                 flow_loglik: FlowLogLikelihood = simulator_flow_loglik,
                 prob_floor: float = 1e-3, seed: int = 0):
        self.params = params
        self.n = particles
        self.onset_rate = params.onset_rate if model_insiders else 0.0
        self.flow_loglik = flow_loglik
        self.prob_floor = prob_floor
        self.rng = np.random.default_rng(seed)
        self.resamples = 0

    def reset(self, obs: Observation) -> np.ndarray:
        self.y = (self.rng.random(self.n) < obs.price).astype(int)
        self.i = np.zeros(self.n, dtype=int)
        self.logw = np.zeros(self.n)
        self.log_odds = float(logit(obs.price))
        return self.belief

    def update(self, obs: Observation) -> np.ndarray:
        inactive = self.i == 0
        self.i[inactive & (self.rng.random(self.n) < self.onset_rate)] = 1

        table = observation_loglik(obs, self.log_odds, self.params, self.flow_loglik)
        self.logw += table[self.y, self.i]
        self.logw -= self.logw.max()
        self.log_odds = float(logit(obs.price))

        if self.effective_size < self.n / 2:
            self._resample()
        return self.belief

    @property
    def weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    @property
    def effective_size(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w ** 2))

    def _resample(self) -> None:
        """Systematic resampling, keeping at least `prob_floor` of particles on each outcome."""
        w = self.weights
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        y, i = self.y[idx], self.i[idx]
        logw = np.zeros(self.n)

        keep = max(int(np.ceil(self.prob_floor * self.n)), 1)
        for outcome in (0, 1):
            mass = w[self.y == outcome].sum()
            if (y == outcome).sum() >= keep or mass <= 0:
                continue
            # Resampling (nearly) dropped this outcome. Top it up to `keep`
            # particles drawn from its old ones, then reweight both outcomes
            # so each keeps exactly its pre-resampling probability.
            missing = keep - (y == outcome).sum()
            old = np.flatnonzero(self.y == outcome)
            src = self.rng.choice(old, size=missing, p=w[old] / mass)
            slots = self.rng.choice(np.flatnonzero(y != outcome), size=missing, replace=False)
            y[slots], i[slots] = outcome, self.i[src]
            mine = y == outcome
            logw[mine] = np.log(mass / mine.sum())
            logw[~mine] = np.log((1.0 - mass) / (~mine).sum())
        self.y, self.i, self.logw = y, i, logw
        self.resamples += 1

    @property
    def belief(self) -> np.ndarray:
        w = self.weights
        b = np.zeros((2, 2))
        np.add.at(b, (self.y, self.i), w)
        return b

    @property
    def prob_yes(self) -> float:
        return float(self.belief[1].sum())

    @property
    def prob_insider(self) -> float:
        return float(self.belief[:, 1].sum())
