"""QMDP: solve the fully observed problem, then weight its Q-values by the belief."""
from __future__ import annotations

import itertools

import numpy as np

from market.sim import (Action, MarketParams, executed_quantity, fill_price,
                        outcome_sign, trade_cash)


class QMDPSolver:
    """Finite-horizon value iteration over (t, outcome, insider, log-odds, position).

    Log-odds live on a grid. The expectation over next-step log-odds uses
    Gauss-Hermite quadrature with linear interpolation between grid points.
    """

    def __init__(self, params: MarketParams, model_insiders: bool = True,
                 grid_size: int = 121, grid_limit: float = 7.0,
                 quadrature_points: int = 11):
        self.params = params
        self.onset_rate = params.onset_rate if model_insiders else 0.0
        self.grid = np.linspace(-grid_limit, grid_limit, grid_size)
        self.positions = np.arange(-params.max_position, params.max_position + 1)
        nodes, weights = np.polynomial.hermite_e.hermegauss(quadrature_points)
        self.nodes = nodes
        self.weights = weights / weights.sum()
        self.step_sd = np.hypot(params.news_weight * params.news_sd, params.noise_sd)
        self.q_table = self._solve()

    def q_values(self, t: int, log_odds: float, position: int) -> np.ndarray:
        """Q[outcome, insider, action] at a log-odds value between grid points."""
        k = position + self.params.max_position
        j = int(np.clip(np.searchsorted(self.grid, log_odds) - 1, 0, len(self.grid) - 2))
        frac = np.clip((log_odds - self.grid[j]) / (self.grid[j + 1] - self.grid[j]), 0.0, 1.0)
        q_t = self.q_table[t]
        return (1 - frac) * q_t[:, :, j, k] + frac * q_t[:, :, j + 1, k]

    def _solve(self) -> np.ndarray:
        p = self.params
        n_x, n_q = len(self.grid), len(self.positions)
        q_table = np.empty((p.horizon, 2, 2, n_x, n_q, len(Action)))

        settlement = self.positions[None, :] * np.array([0, 1])[:, None]   # [outcome, position]
        value = np.broadcast_to(settlement[:, None, None, :], (2, 2, n_x, n_q))

        for t in reversed(range(p.horizon)):
            for outcome, insider, k, action in itertools.product((0, 1), (0, 1), range(n_q), Action):
                quantity = executed_quantity(int(self.positions[k]), action, p)
                reward = trade_cash(quantity, fill_price(self.grid, quantity, p), p)
                future = self._expected_value(value, outcome, insider, k + quantity, quantity)
                q_table[t, outcome, insider, :, k, action] = reward + future
            value = q_table[t].max(axis=-1)
        return q_table

    def _expected_value(self, value, outcome, insider, next_k, quantity) -> np.ndarray:
        p = self.params
        s = outcome_sign(outcome)
        transitions = [(1, 1.0)] if insider else [(0, 1.0 - self.onset_rate), (1, self.onset_rate)]
        total = np.zeros_like(self.grid)
        for next_insider, prob in transitions:
            if prob == 0.0:
                continue
            mean = (self.grid + p.impact * quantity + p.news_weight * p.news_mean * s
                    + p.insider_drift * next_insider * s)
            points = mean[:, None] + self.step_sd * self.nodes[None, :]
            total += prob * np.interp(points, self.grid, value[outcome, next_insider, :, next_k]) @ self.weights
        return total
