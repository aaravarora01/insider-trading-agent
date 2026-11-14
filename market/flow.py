"""Large trades from other participants, and the wallet features we see on them.

Each step, three kinds of trader place large trades:

    retail   uninformed; YES or NO with equal probability
    insider  only while the insider is active; always trades toward the outcome
    copybot  a background stream on random sides, plus, while the insider is
             active, a stream that copies the insider's side

Each (trader type, side) pair is an independent Poisson stream, so the trades
in a step form a marked Poisson process. We observe each trade's side, its
size, and the wallet behind it, but not the trader type. Wallet features,
all measurable from public trade history:

    fresh           no trades in any other market before this one
    and, for wallets that are not fresh:
    age             days since the wallet's first trade in another market
    category_share  share of its earlier markets in this market's category
    longshots       how many earlier bets it placed at a price below 0.2 that
                    have resolved, and its smoothed win rate on them

Fresh wallets have no history, so the other features are undefined for them
rather than counted as extra evidence. Features are independent given the
trader type. `polymarket.observation` fits the profiles to Polymarket data.

Because the process is Poisson, the likelihood of a step's trades given the
outcome Y and insider status I has a closed form with the type marginalized
out:

    log P(trades | Y, I) = -sum_{k, side} rate[k, side | Y, I]
                           + sum_j log sum_k rate[k, side_j | Y, I] f_k(x_j)

up to a term that does not depend on (Y, I). `flow_loglik` computes it.

The price equation in `market.sim` is unchanged: `insider_drift` is the net
price impact of informed flow, and `noise_sd` covers everything else,
including trades too small to report. The large trades we observe are a
sample of that flow whose mix depends on (Y, I).

Copy-bots react within the step. With a time step of roughly an hour this is
a fair approximation, and it keeps the likelihood Markov in (Y, I_t).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class TraderType(IntEnum):
    RETAIL = 0
    INSIDER = 1
    COPYBOT = 2


@dataclass(frozen=True)
class WalletProfile:
    """Distribution of trade size and wallet features for one trader type."""

    size_log_mean: float        # log contracts (payout exposure, same on either side)
    size_log_sd: float
    fresh_prob: float           # no trades in any other market before this one
    # The rest describe wallets that are not fresh.
    age_log_mean: float         # log days since first trade in another market
    age_log_sd: float
    category_a: float           # Beta(a, b) share of earlier markets in this category
    category_b: float
    longshot_prob: float        # has at least one resolved longshot bet
    longshot_log_mean: float    # log number of resolved longshot bets, if any
    longshot_log_sd: float
    win_a: float                # Beta(a, b) smoothed win rate on those bets
    win_b: float


# Placeholder values for the toy simulator; `polymarket.observation` fits real
# ones. No single feature separates the types: fresh wallets lean insider, but
# plenty of retail wallets are fresh too.
RETAIL = WalletProfile(size_log_mean=math.log(800), size_log_sd=1.0, fresh_prob=0.15,
                       age_log_mean=math.log(120), age_log_sd=1.3,
                       category_a=2.0, category_b=4.0, longshot_prob=0.6,
                       longshot_log_mean=math.log(6), longshot_log_sd=1.0,
                       win_a=2.0, win_b=12.0)
INSIDER = WalletProfile(size_log_mean=math.log(2400), size_log_sd=1.0, fresh_prob=0.45,
                        age_log_mean=math.log(30), age_log_sd=1.5,
                        category_a=3.0, category_b=2.5, longshot_prob=0.5,
                        longshot_log_mean=math.log(4), longshot_log_sd=1.0,
                        win_a=3.0, win_b=6.0)
COPYBOT = WalletProfile(size_log_mean=math.log(1200), size_log_sd=0.8, fresh_prob=0.10,
                        age_log_mean=math.log(150), age_log_sd=1.0,
                        category_a=1.5, category_b=3.0, longshot_prob=0.8,
                        longshot_log_mean=math.log(15), longshot_log_sd=1.0,
                        win_a=2.5, win_b=12.0)


@dataclass(frozen=True)
class FlowParams:
    retail_rate: float = 3.0         # large trades per step
    insider_rate: float = 0.5        # while active
    copy_base_rate: float = 0.5      # copying uninformed wallets, random side
    copy_follow_rate: float = 0.5    # copying the insider, while active
    profiles: tuple[WalletProfile, ...] = field(
        default_factory=lambda: (RETAIL, INSIDER, COPYBOT))  # indexed by TraderType


@dataclass(frozen=True)
class LargeTrade:
    """One observed trade. `side` is +1 for YES and -1 for NO.

    For a fresh wallet, `age_days`, `category_share` and `longshot_win_rate` are
    NaN and `longshot_bets` is 0; `longshot_win_rate` is also NaN with no bets.
    The win rate is smoothed: (wins + 0.5) / (bets + 1).
    """

    side: int
    contracts: float
    fresh: bool
    age_days: float
    category_share: float
    longshot_bets: int
    longshot_win_rate: float


def side_index(side: int) -> int:
    return 1 if side > 0 else 0


def flow_rates(outcome: int, insider: bool, flow: FlowParams) -> np.ndarray:
    """Poisson rate of each stream, indexed [trader type, side index]."""
    rates = np.zeros((len(TraderType), 2))
    rates[TraderType.RETAIL] = flow.retail_rate / 2
    rates[TraderType.COPYBOT] = flow.copy_base_rate / 2
    if insider:
        rates[TraderType.INSIDER, outcome] += flow.insider_rate
        rates[TraderType.COPYBOT, outcome] += flow.copy_follow_rate
    return rates


# Rates for every hidden state, indexed [outcome, insider, trader type, side index].
def _rate_table(flow: FlowParams) -> np.ndarray:
    return np.array([[flow_rates(y, bool(i), flow) for i in (0, 1)] for y in (0, 1)])


def sample_wallet(profile: WalletProfile, side: int,
                  rng: np.random.Generator) -> LargeTrade:
    """Draw one trade. Always uses the same random numbers, whatever the branch."""
    p = profile
    u = rng.random(2)
    contracts, age, bets = rng.lognormal([p.size_log_mean, p.age_log_mean, p.longshot_log_mean],
                                         [p.size_log_sd, p.age_log_sd, p.longshot_log_sd])
    category, win_rate = rng.beta([p.category_a, p.win_a], [p.category_b, p.win_b])
    fresh, has_longshots = bool(u[0] < p.fresh_prob), bool(u[1] < p.longshot_prob)
    if fresh:
        return LargeTrade(side, float(contracts), True, math.nan, math.nan, 0, math.nan)
    if not has_longshots:
        return LargeTrade(side, float(contracts), False, float(age), float(category), 0, math.nan)
    return LargeTrade(side, float(contracts), False, float(age), float(category),
                      max(int(round(bets)), 1), float(win_rate))


def sample_flow(outcome: int, insider: bool, flow: FlowParams,
                rng: np.random.Generator) -> tuple[tuple[LargeTrade, ...], tuple[int, ...]]:
    """Draw one step of large trades. Returns (trades, true trader types).

    Trades are shuffled so their order says nothing about who placed them.
    """
    counts = rng.poisson(flow_rates(outcome, insider, flow))
    labelled = [(sample_wallet(flow.profiles[k], 2 * s - 1, rng), k)
                for k in TraderType for s in (0, 1) for _ in range(counts[k, s])]
    order = rng.permutation(len(labelled))
    trades = tuple(labelled[i][0] for i in order)
    types = tuple(int(labelled[i][1]) for i in order)
    return trades, types


def features(trades) -> dict[str, np.ndarray]:
    """Column arrays from LargeTrade objects (or anything with the same attributes)."""
    return {
        "contracts": np.array([t.contracts for t in trades], dtype=float),
        "fresh": np.array([t.fresh for t in trades], dtype=bool),
        "age_days": np.array([t.age_days for t in trades], dtype=float),
        "category_share": np.array([t.category_share for t in trades], dtype=float),
        "longshot_bets": np.array([t.longshot_bets for t in trades], dtype=float),
        "longshot_win_rate": np.array([t.longshot_win_rate for t in trades], dtype=float),
    }


def _lognormal_logpdf(x, log_mean, log_sd):
    z = (np.log(x) - log_mean) / log_sd
    return -0.5 * z ** 2 - np.log(log_sd) - np.log(x) - 0.5 * math.log(2 * math.pi)


def _beta_logpdf(x, a, b):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - log_norm


def feature_loglik(x: dict[str, np.ndarray], profile: WalletProfile) -> np.ndarray:
    """log f_k(features) per trade under one trader type, from `features(...)` arrays."""
    p = profile
    fresh = x["fresh"]
    with np.errstate(invalid="ignore", divide="ignore"):
        history = (math.log1p(-p.fresh_prob)
                   + _lognormal_logpdf(np.maximum(x["age_days"], 1 / 24),
                                       p.age_log_mean, p.age_log_sd)
                   + _beta_logpdf(x["category_share"], p.category_a, p.category_b))
        bets = x["longshot_bets"]
        longshots = np.where(
            bets > 0,
            math.log(p.longshot_prob)
            + _lognormal_logpdf(np.maximum(bets, 1), p.longshot_log_mean, p.longshot_log_sd)
            + _beta_logpdf(x["longshot_win_rate"], p.win_a, p.win_b),
            math.log1p(-p.longshot_prob))
    wallet = np.where(fresh, math.log(p.fresh_prob), history + longshots)
    return _lognormal_logpdf(x["contracts"], p.size_log_mean, p.size_log_sd) + wallet


def wallet_loglik(trades: tuple[LargeTrade, ...], profile: WalletProfile) -> np.ndarray:
    """log f_k(features) for each trade under one trader type. Side is excluded."""
    return feature_loglik(features(trades), profile)


def flow_loglik(trades: tuple[LargeTrade, ...], flow: FlowParams) -> np.ndarray:
    """log P(trades | outcome, insider) with trader type marginalized out.

    Returns an array indexed [outcome, insider].
    """
    rates = _rate_table(flow)
    total = -rates.sum(axis=(2, 3))
    if not trades:
        return total
    wallet = np.stack([wallet_loglik(trades, prof) for prof in flow.profiles])   # [type, trade]
    sides = np.array([side_index(t.side) for t in trades])
    with np.errstate(divide="ignore"):
        log_rates = np.log(rates[..., sides])                                    # [y, i, type, trade]
    per_trade = np.logaddexp.reduce(log_rates + wallet, axis=2)                  # [y, i, trade]
    return total + per_trade.sum(axis=-1)


def type_posterior(trades: tuple[LargeTrade, ...], outcome: int, insider: bool,
                   flow: FlowParams) -> np.ndarray:
    """P(trader type | trade, outcome, insider), indexed [trade, type]."""
    if not trades:
        return np.zeros((0, len(TraderType)))
    rates = flow_rates(outcome, insider, flow)
    wallet = np.stack([wallet_loglik(trades, prof) for prof in flow.profiles], axis=1)
    sides = np.array([side_index(t.side) for t in trades])
    with np.errstate(divide="ignore"):
        logits = np.log(rates[:, sides].T) + wallet
    logits -= logits.max(axis=1, keepdims=True)
    post = np.exp(logits)
    return post / post.sum(axis=1, keepdims=True)


def insider_score(loglik: np.ndarray) -> float:
    """Log-likelihood ratio for an active insider, outcome weighted 50/50.

    With no insider the flow is symmetric, so loglik[0, 0] == loglik[1, 0].
    """
    return float(np.logaddexp(loglik[0, 1], loglik[1, 1]) - math.log(2) - loglik[0, 0])
