"""Run every policy on the same simulated markets and report metrics.

    python -m experiments.evaluate --markets 1000
    python -m experiments.evaluate --markets 300 --params calibrated

The simulator draws the same random numbers whatever the policy does, so a
given seed produces the same market for every policy apart from our own
price impact.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from agent.policies import BuyAndHold, Oracle, QMDPPolicy
from agent.qmdp import QMDPSolver
from experiments import plots
from market import sim
from market.calibration import calibrated_params
from market.sim import MarketParams, outcome_sign


@dataclass(frozen=True)
class Trade:
    t: int
    quantity: int
    price: float
    insider_active: bool


@dataclass
class Episode:
    outcome: int
    onset: int | None
    reward: float = 0.0
    prices: list[float] = field(default_factory=list)
    insider: list[bool] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    beliefs: np.ndarray | None = None


def _view(policy, state, obs):
    return state if policy.full_information else obs


def run_episode(policy, params: MarketParams, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    state, obs = sim.reset(params, rng)
    policy.reset(_view(policy, state, obs))
    prices, insider, trades, total = [obs.price], [state.insider], [], 0.0

    while not obs.done:
        prior = state
        state, obs, reward = sim.step(state, policy.act(), rng, params)
        policy.update(_view(policy, state, obs))
        total += reward
        prices.append(obs.price)
        insider.append(state.insider)
        if obs.executed:
            price = float(sim.fill_price(prior.log_odds, obs.executed, params))
            trades.append(Trade(prior.t, obs.executed, price, prior.insider))

    beliefs = getattr(policy, "beliefs", None)
    return Episode(outcome=state.outcome, onset=state.onset, reward=total,
                   prices=prices, insider=insider, trades=trades,
                   beliefs=None if beliefs is None else np.array(beliefs))


def adverse_selection_loss(episode: Episode) -> float:
    """Loss on trades made against an active insider, valued at resolution."""
    insider_side = outcome_sign(episode.outcome)
    return sum(abs(tr.quantity) * abs(episode.outcome - tr.price)
               for tr in episode.trades
               if tr.insider_active and np.sign(tr.quantity) == -insider_side)


def trading_metrics(episodes: list[Episode]) -> dict:
    returns = np.array([e.reward for e in episodes])
    sd = returns.std(ddof=1)
    return {
        "mean_return": returns.mean(),
        "stderr_return": sd / np.sqrt(len(returns)),
        "sharpe": returns.mean() / sd if sd > 0 else float("nan"),
        "adverse_selection_loss": np.mean([adverse_selection_loss(e) for e in episodes]),
        "trades_per_market": np.mean([len(e.trades) for e in episodes]),
    }


def detection_metrics(episodes: list[Episode], threshold: float) -> dict:
    """Flag a market when P(insider active) exceeds `threshold`."""
    delays, leads = [], []
    clean_markets = flagged_clean_markets = 0
    inactive_steps = flagged_inactive_steps = 0

    for e in episodes:
        flagged = e.beliefs[:, :, 1].sum(axis=1) > threshold
        active = np.array(e.insider)
        inactive_steps += (~active).sum()
        flagged_inactive_steps += (flagged & ~active).sum()

        if e.onset is None:
            clean_markets += 1
            flagged_clean_markets += flagged.any()
            continue
        hits = np.flatnonzero(flagged[e.onset:])
        if hits.size:
            delays.append(hits[0])
            leads.append(len(flagged) - 1 - (e.onset + hits[0]))
        else:
            delays.append(np.nan)

    delays = np.array(delays, dtype=float)
    return {
        "markets_with_insider": len(delays),
        "detected_share": np.mean(~np.isnan(delays)),
        "median_delay_steps": np.nanmedian(delays),
        "median_lead_steps": np.median(leads),
        "false_positive_rate_markets": flagged_clean_markets / clean_markets,
        "false_positive_rate_steps": flagged_inactive_steps / inactive_steps,
    }


def print_table(results: dict[str, dict]) -> None:
    header = f"{'policy':<22}{'return':>16}{'sharpe':>8}{'adverse':>9}{'trades':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        ret = f"{m['mean_return']:+.2f} ± {m['stderr_return']:.2f}"
        print(f"{name:<22}{ret:>16}{m['sharpe']:>8.2f}"
              f"{m['adverse_selection_loss']:>9.2f}{m['trades_per_market']:>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markets", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--params", choices=["toy", "calibrated"], default="toy",
                        help="toy defaults, or the fit to Polymarket data in market/calibrated.json")
    args = parser.parse_args()

    params = MarketParams() if args.params == "toy" else calibrated_params()
    aware = QMDPSolver(params, model_insiders=True)
    blind = QMDPSolver(params, model_insiders=False)
    policies = [
        BuyAndHold(),
        QMDPPolicy(params, blind, model_insiders=False),
        QMDPPolicy(params, aware),
        Oracle(aware),
    ]

    seeds = range(args.seed * args.markets, (args.seed + 1) * args.markets)
    episodes = {p.name: [run_episode(p, params, s) for s in seeds] for p in policies}
    trading = {name: trading_metrics(eps) for name, eps in episodes.items()}
    detection = detection_metrics(episodes["QMDP"], args.threshold)

    print_table(trading)
    print()
    for key, value in detection.items():
        print(f"{key:<30}{value:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"params": asdict(params), "markets": args.markets,
              "trading": trading, "detection": detection}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    plots.belief_traces(episodes["QMDP"], args.out / "belief_traces.png")
    plots.return_distributions(episodes, args.out / "returns.png")


if __name__ == "__main__":
    main()
