"""Large trades from other participants, and the wallet features we see on them.

Each step, three kinds of trader place large trades:

    retail   uninformed; YES or NO with equal probability
    insider  only while the insider is active; always trades toward the outcome
    copybot  a background stream on random sides, plus, while the insider is
             active, a stream that copies the insider's side

Each (trader type, side) pair is an independent Poisson stream, so the trades
in a step form a marked Poisson process. We observe each trade's side, its
size, and the wallet behind it (age, whether it was freshly funded, how
concentrated it is in this market's category, and its record on longshot
bets). We do not observe the trader type.

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

    size_log_mean: float        # log USD
    size_log_sd: float
    age_log_mean: float         # log days since the wallet first traded
    age_log_sd: float
    new_funding_prob: float     # funded from a fresh source just before trading
    category_a: float           # Beta(a, b) share of past volume in this category
    category_b: float
    longshot_bets_mean: float   # Poisson mean of past bets placed at price < 0.2
    longshot_win_prob: float    # chance each of those bets won


# Placeholder values until they are fit to Polymarket data. They are chosen so
# that no single feature separates the types: a fresh wallet is more likely to
# be an insider, but plenty of retail wallets are fresh too.
RETAIL = WalletProfile(size_log_mean=math.log(400), size_log_sd=1.0,
                       age_log_mean=math.log(120), age_log_sd=1.3,
                       new_funding_prob=0.15, category_a=2.0, category_b=4.0,
                       longshot_bets_mean=6.0, longshot_win_prob=0.10)
INSIDER = WalletProfile(size_log_mean=math.log(1200), size_log_sd=1.0,
                        age_log_mean=math.log(30), age_log_sd=1.5,
                        new_funding_prob=0.45, category_a=3.0, category_b=2.5,
                        longshot_bets_mean=4.0, longshot_win_prob=0.30)
COPYBOT = WalletProfile(size_log_mean=math.log(600), size_log_sd=0.8,
                        age_log_mean=math.log(150), age_log_sd=1.0,
                        new_funding_prob=0.10, category_a=1.5, category_b=3.0,
                        longshot_bets_mean=12.0, longshot_win_prob=0.14)


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
    """One observed trade. `side` is +1 for YES and -1 for NO."""

    side: int
    usd_size: float
    wallet_age_days: float
    new_funding: bool
    category_share: float
    longshot_bets: int
    longshot_wins: int


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
    bets = int(rng.poisson(profile.longshot_bets_mean))
    return LargeTrade(
        side=side,
        usd_size=float(rng.lognormal(profile.size_log_mean, profile.size_log_sd)),
        wallet_age_days=float(rng.lognormal(profile.age_log_mean, profile.age_log_sd)),
        new_funding=bool(rng.random() < profile.new_funding_prob),
        category_share=float(rng.beta(profile.category_a, profile.category_b)),
        longshot_bets=bets,
        longshot_wins=int(rng.binomial(bets, profile.longshot_win_prob)),
    )


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


_lgamma = np.vectorize(math.lgamma, otypes=[float])


def _features(trades: tuple[LargeTrade, ...]) -> dict[str, np.ndarray]:
    return {
        "size": np.array([t.usd_size for t in trades]),
        "age": np.array([t.wallet_age_days for t in trades]),
        "new_funding": np.array([t.new_funding for t in trades], dtype=bool),
        "category": np.clip([t.category_share for t in trades], 1e-9, 1 - 1e-9),
        "bets": np.array([t.longshot_bets for t in trades], dtype=float),
        "wins": np.array([t.longshot_wins for t in trades], dtype=float),
    }


def _lognormal_logpdf(x, log_mean, log_sd):
    z = (np.log(x) - log_mean) / log_sd
    return -0.5 * z ** 2 - np.log(log_sd) - np.log(x) - 0.5 * math.log(2 * math.pi)


def wallet_loglik(trades: tuple[LargeTrade, ...], profile: WalletProfile) -> np.ndarray:
    """log f_k(features) for each trade under one trader type. Side is excluded."""
    x = _features(trades)
    p = profile
    funding = np.where(x["new_funding"], math.log(p.new_funding_prob),
                       math.log1p(-p.new_funding_prob))
    log_beta = math.lgamma(p.category_a) + math.lgamma(p.category_b) \
        - math.lgamma(p.category_a + p.category_b)
    category = ((p.category_a - 1) * np.log(x["category"])
                + (p.category_b - 1) * np.log1p(-x["category"]) - log_beta)
    bets, wins = x["bets"], x["wins"]
    longshots = (bets * math.log(p.longshot_bets_mean) - p.longshot_bets_mean
                 - _lgamma(wins + 1) - _lgamma(bets - wins + 1)
                 + wins * math.log(p.longshot_win_prob)
                 + (bets - wins) * math.log1p(-p.longshot_win_prob))
    return (_lognormal_logpdf(x["size"], p.size_log_mean, p.size_log_sd)
            + _lognormal_logpdf(x["age"], p.age_log_mean, p.age_log_sd)
            + funding + category + longshots)


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
