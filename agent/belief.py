"""Exact Bayes filter over the hidden pair (outcome, insider active).

Both are binary and everything else is observed, so the belief is a 2x2 table
b[outcome, insider]. Wallet types are summed out inside the flow likelihood.
`agent.particle.ParticleFilter` is the sampling version the proposal
describes; it uses the same `observation_loglik` and converges to this filter.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from market import flow
from market.sim import MarketParams, Observation, logit

# log P(other participants' trades | outcome, insider), indexed [outcome, insider].
FlowLogLikelihood = Callable[[Observation, MarketParams], np.ndarray]

SIGN = np.array([[-1.0], [1.0]])     # rows: outcome NO, YES
ACTIVE = np.array([[0.0, 1.0]])      # columns: insider inactive, active


def normal_logpdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd)


def simulator_flow_loglik(obs: Observation, params: MarketParams) -> np.ndarray:
    """The simulator's own flow model, with wallet type marginalized out."""
    return flow.flow_loglik(obs.trades, params.flow)


def observation_loglik(obs: Observation, prev_log_odds: float, params: MarketParams,
                       flow_loglik: FlowLogLikelihood = simulator_flow_loglik) -> np.ndarray:
    """log P(obs | outcome, insider) as a 2x2 array [outcome, insider].

    The price move not explained by our own trade or the news is the insider's
    drift plus noise; the news and the other participants' trades are scored
    directly.
    """
    p = params
    unexplained = (float(logit(obs.price)) - prev_log_odds
                   - p.impact * obs.executed
                   - p.news_weight * obs.news)
    return (normal_logpdf(obs.news, p.news_mean * SIGN, p.news_sd)
            + normal_logpdf(unexplained, p.insider_drift * ACTIVE * SIGN, p.noise_sd)
            + flow_loglik(obs, p))


class BeliefFilter:
    """Tracks b[outcome, insider].

    `model_insiders=False` gives the insider-blind ablation: it assumes no
    insider ever appears, so all price moves are read as news or noise.
    `flow_loglik` is where a learned wallet-feature model plugs in.
    """

    def __init__(self, params: MarketParams, model_insiders: bool = True,
                 flow_loglik: FlowLogLikelihood = simulator_flow_loglik):
        self.params = params
        self.onset_rate = params.onset_rate if model_insiders else 0.0
        self.flow_loglik = flow_loglik

    def reset(self, obs: Observation) -> np.ndarray:
        self.belief = np.array([[1.0 - obs.price, 0.0],
                                [obs.price, 0.0]])
        self.log_odds = float(logit(obs.price))
        return self.belief

    def update(self, obs: Observation) -> np.ndarray:
        predicted = self.belief.copy()
        switched_on = predicted[:, 0] * self.onset_rate
        predicted[:, 0] -= switched_on
        predicted[:, 1] += switched_on

        loglik = observation_loglik(obs, self.log_odds, self.params, self.flow_loglik)
        posterior = predicted * np.exp(loglik - loglik.max())
        self.belief = posterior / posterior.sum()
        self.log_odds = float(logit(obs.price))
        return self.belief

    @property
    def prob_yes(self) -> float:
        return float(self.belief[1].sum())

    @property
    def prob_insider(self) -> float:
        return float(self.belief[:, 1].sum())
