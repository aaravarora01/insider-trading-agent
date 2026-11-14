"""Exact Bayes filter over the hidden pair (outcome, insider active).

Both are binary and everything else is observed, so the belief is a 2x2 table
b[outcome, insider]. A particle filter is not needed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from market.sim import MarketParams, Observation, logit

ScoreLogLikelihood = Callable[[float, int, MarketParams], float]

SIGN = np.array([[-1.0], [1.0]])     # rows: outcome NO, YES
ACTIVE = np.array([[0.0, 1.0]])      # columns: insider inactive, active


def normal_logpdf(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd)


def gaussian_score(score: float, insider: int, params: MarketParams) -> float:
    return normal_logpdf(score, params.score_mean[insider], params.score_sd)


class BeliefFilter:
    """Tracks b[outcome, insider].

    `model_insiders=False` gives the insider-blind ablation: it assumes no
    insider ever appears, so all price moves are read as news or noise.
    `score_loglik` is where a learned wallet-feature model plugs in.
    """

    def __init__(self, params: MarketParams, model_insiders: bool = True,
                 score_loglik: ScoreLogLikelihood = gaussian_score):
        self.params = params
        self.onset_rate = params.onset_rate if model_insiders else 0.0
        self.score_loglik = score_loglik

    def reset(self, obs: Observation) -> np.ndarray:
        self.belief = np.array([[1.0 - obs.price, 0.0],
                                [obs.price, 0.0]])
        self.log_odds = float(logit(obs.price))
        return self.belief

    def update(self, obs: Observation) -> np.ndarray:
        p = self.params

        predicted = self.belief.copy()
        switched_on = predicted[:, 0] * self.onset_rate
        predicted[:, 0] -= switched_on
        predicted[:, 1] += switched_on

        log_odds = float(logit(obs.price))
        unexplained = (log_odds - self.log_odds
                       - p.impact * obs.executed
                       - p.news_weight * obs.news)
        score = np.array([[self.score_loglik(obs.score, i, p) for i in (0, 1)]])
        loglik = (normal_logpdf(obs.news, p.news_mean * SIGN, p.news_sd)
                  + normal_logpdf(unexplained, p.insider_drift * ACTIVE * SIGN, p.noise_sd)
                  + score)

        posterior = predicted * np.exp(loglik - loglik.max())
        self.belief = posterior / posterior.sum()
        self.log_odds = log_odds
        return self.belief

    @property
    def prob_yes(self) -> float:
        return float(self.belief[1].sum())

    @property
    def prob_insider(self) -> float:
        return float(self.belief[:, 1].sum())
