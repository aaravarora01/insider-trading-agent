"""Binary prediction market in which an informed trader may become active.

This module is the interface between the market side and the agent side:

    state, obs = reset(params, rng)
    state, obs, reward = step(state, action, rng, params)

Agents see only `Observation`. The hidden parts of `MarketState` (the outcome
and when the insider became active) are for planners that sample states and
for evaluation.

Dynamics are in log-odds, x = logit(price). With s = +1 if YES wins and -1
otherwise:

    x' = x + impact * dq + news_weight * news + insider_drift * insider * s + noise
    news  ~ N(news_mean * s, news_sd^2)
    noise ~ N(0, noise_sd^2)

Each step also reports the large trades placed by other participants (retail,
the insider, copy-bots) with the wallet features behind them; see
`market.flow`. `score` summarizes them as a log-likelihood ratio for an active
insider.

An inactive insider becomes active with probability `onset_rate` per step and
stays active. Setting news_weight = 2 * news_mean / news_sd^2 makes the price
the exact posterior given public news, so news alone carries no edge.

Accounting: buying a NO contract at YES price p costs 1 - p and pays
1 - outcome, which is the same profit and loss as selling a YES contract at p.
We therefore track one signed position (negative means NO). Trading dq
contracts at price p pays -dq * p, and the position settles for
position * outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum

import numpy as np

from market.flow import FlowParams, LargeTrade, flow_loglik, insider_score, sample_flow


@dataclass(frozen=True)
class MarketParams:
    horizon: int = 50
    onset_rate: float = 0.02
    insider_drift: float = 0.15
    news_mean: float = 0.15
    news_sd: float = 1.0
    news_weight: float = 0.30
    noise_sd: float = 0.15
    flow: FlowParams = field(default_factory=FlowParams)
    impact: float = 0.02
    fee: float = 0.0
    small_lot: int = 1
    large_lot: int = 3
    max_position: int = 6
    initial_price_range: tuple[float, float] = (0.2, 0.8)


class Action(IntEnum):
    HOLD = 0
    BUY_YES_SMALL = 1
    BUY_YES_LARGE = 2
    BUY_NO_SMALL = 3
    BUY_NO_LARGE = 4
    EXIT = 5


@dataclass(frozen=True)
class MarketState:
    t: int
    log_odds: float
    position: int
    outcome: int
    onset: int | None  # step at which the insider became active
    flow_types: tuple[int, ...] = ()  # true TraderType of each trade in the last Observation

    @property
    def price(self) -> float:
        return float(sigmoid(self.log_odds))

    @property
    def insider(self) -> bool:
        return self.onset is not None


@dataclass(frozen=True)
class Observation:
    t: int
    price: float
    news: float
    score: float     # insider log-likelihood ratio of `trades`
    executed: int    # signed contracts we traded on the last step
    done: bool
    trades: tuple[LargeTrade, ...] = ()  # other participants' large trades this step


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p) - np.log1p(-p)


def outcome_sign(outcome: int) -> int:
    return 1 if outcome == 1 else -1


def executed_quantity(position: int, action: Action, params: MarketParams) -> int:
    """Signed contracts traded by `action`, after the position limit."""
    requested = {
        Action.HOLD: 0,
        Action.BUY_YES_SMALL: params.small_lot,
        Action.BUY_YES_LARGE: params.large_lot,
        Action.BUY_NO_SMALL: -params.small_lot,
        Action.BUY_NO_LARGE: -params.large_lot,
        Action.EXIT: -position,
    }[Action(action)]
    limit = params.max_position
    return int(np.clip(position + requested, -limit, limit)) - position


def fill_price(log_odds, quantity: int, params: MarketParams):
    """Our order moves the price before it fills, so we pay our own impact."""
    return sigmoid(log_odds + params.impact * quantity)


def trade_cash(quantity: int, price, params: MarketParams):
    return -quantity * price - params.fee * abs(quantity)


def reset(params: MarketParams, rng: np.random.Generator) -> tuple[MarketState, Observation]:
    price = rng.uniform(*params.initial_price_range)
    outcome = int(rng.random() < price)
    state = MarketState(t=0, log_odds=float(logit(price)), position=0,
                        outcome=outcome, onset=None)
    obs = Observation(t=0, price=price, news=0.0, score=0.0, executed=0, done=False)
    return state, obs


def step(state: MarketState, action: Action, rng: np.random.Generator,
         params: MarketParams) -> tuple[MarketState, Observation, float]:
    quantity = executed_quantity(state.position, action, params)
    reward = float(trade_cash(quantity, fill_price(state.log_odds, quantity, params), params))
    position = state.position + quantity

    onset = state.onset
    if onset is None and rng.random() < params.onset_rate:
        onset = state.t + 1
    insider = onset is not None

    s = outcome_sign(state.outcome)
    news = rng.normal(params.news_mean * s, params.news_sd)
    log_odds = (state.log_odds
                + params.impact * quantity
                + params.news_weight * news
                + params.insider_drift * insider * s
                + rng.normal(0.0, params.noise_sd))
    trades, types = sample_flow(state.outcome, insider, params.flow, rng)
    score = insider_score(flow_loglik(trades, params.flow))

    t = state.t + 1
    done = t >= params.horizon
    if done:
        reward += position * state.outcome

    next_state = replace(state, t=t, log_odds=log_odds, position=position, onset=onset,
                         flow_types=types)
    obs = Observation(t=t, price=float(sigmoid(log_odds)), news=news, score=score,
                      executed=quantity, done=done, trades=trades)
    return next_state, obs, reward
