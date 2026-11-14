"""Estimate the simulator's parameters from Polymarket trades.

    python -m polymarket.calibrate        # writes market/calibrated.json

One simulator step is one hour; one simulator contract is CONTRACT_SIZE real
contracts. Prices are hourly last YES prices, in log-odds.

Price dynamics. Without an insider the simulator moves log-odds by
news_weight * news + noise, with news ~ N(news_mean * s, 1) and
news_weight = 2 * news_mean, so the price is the posterior given public news.
That ties drift to variance: per step, the drift toward the true outcome is
d = 2 * news_mean^2 and the variance from news is 2d. From quiet hours we
measure d = mean(s * dx) and the permanent variance V (Roll-corrected for
bid-ask bounce), and get

    news_mean = sqrt(d / 2),   news_weight = 2 * news_mean,   noise_sd^2 = V - 2d.

Insider. In each incident market the insider window runs from the hour of the
first fill by a flagged wallet (label other than "weak") on the winning side
to the public reveal: the earlier of the documented announcement time in
`polymarket.selection` and the first hour the price reaches 0.95 on the
winning side. Leaks move prices a lot but rarely to 0.95, and once the price
is there the outcome is public in effect (in three incidents it got there
hours before the documented announcement). insider_drift is
the median over incidents of (drift toward the outcome in the window) - d.

Impact. OLS of hourly dx on net taker flow (YES contracts) in quiet hours,
per market; we keep the median slope.

Flow. A large trade is a taker order (fills sharing taker, block, and side)
of at least LARGE_CONTRACTS contracts. We threshold on contracts, not USD:
$1k buys ten times more contracts on a 10c longshot than on a 90c favourite,
so a USD threshold makes large orders lean toward the favourite, while with a
contract threshold uninformed large orders split about 50/50 at every price,
as the simulator assumes. Quiet-hour rates and sizes give the uninformed flow;
flagged-wallet orders in the window give the insider's.

Activity surges near resolution whether or not anyone is informed (large
orders in the Mamdani market go from 9/hour a month out to 49/hour in the
last 12 hours; the Super Bowl market from 0.1 to 101 during the game). So the
uninformed rate is measured over the last HORIZON hours before each market's
end (the onset for incidents, the last hour before the price settles at 0.95
otherwise), and followers are measured inside the insider window as large
orders from other wallets on the winning side minus those on the losing side.
Uninformed flow splits evenly, so the difference cancels the surge.

Quiet hours: before any onset, with both prices in [0.05, 0.95].

What the data cannot pin down is recorded under "design_choices": the horizon,
how often markets have an insider, the copy-bot share of uninformed flow, and
the wallet features other than size (task 4).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from polymarket.data import OUT_DIR, load_prices, load_trades
from polymarket.flagged import load_flagged
from polymarket.selection import MARKETS

LARGE_CONTRACTS = 2000.0
CONTRACT_SIZE = 1000          # real contracts per simulator contract
PRICE_RANGE = (0.05, 0.95)
REVEAL_PRICE = 0.95
HORIZON = 72                  # hours: the last three days before resolution
INSIDER_SHARE = 0.5           # share of simulated markets with an insider by the end
COPY_SHARE = 1 / 7            # share of uninformed large orders from copy-bots (task 4)
OUTPUT = Path(__file__).resolve().parent.parent / "market" / "calibrated.json"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p) - np.log1p(-p)


def taker_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """Merge fills from one taker order (same taker, block, and side)."""
    g = trades.groupby(["taker", "block_number", "taker_side"], sort=False)
    return g.agg(time=("time", "min"), contracts=("contracts", "sum"),
                 usd=("taker_usd", "sum")).reset_index().rename(
        columns={"taker": "wallet", "taker_side": "side"})


def wallet_orders(trades: pd.DataFrame, wallets: set[str]) -> pd.DataFrame:
    """Orders by the given wallets in either role, with each wallet's own side and stake."""
    as_taker = trades[trades["taker"].isin(wallets)].assign(
        wallet=lambda d: d["taker"], side=lambda d: d["taker_side"], usd=lambda d: d["taker_usd"])
    as_maker = trades[trades["maker"].isin(wallets)].assign(
        wallet=lambda d: d["maker"], side=lambda d: -d["taker_side"], usd=lambda d: d["maker_usd"])
    g = pd.concat([as_taker, as_maker]).groupby(["wallet", "block_number", "side"], sort=False)
    return g.agg(time=("time", "min"), contracts=("contracts", "sum"),
                 usd=("usd", "sum")).reset_index()


def onset_wallets(trades: pd.DataFrame, flagged: pd.DataFrame, market_id: str,
                  sign: int) -> set[str]:
    """Flagged wallets that end up net long the winning side by a large trade.

    Wallets that traded both ways and exited, or bet the other side, do not
    mark the onset. Strong and multi-source labels take priority; single-source
    labels are used only when a market has none of those.
    """
    here = flagged[flagged["market_ids"].map(lambda ids: market_id in ids)]
    for labels in (("strong", "multi_source"), ("single_source",)):
        candidates = set(here.loc[here["label"].isin(labels), "address"])
        orders = wallet_orders(trades, candidates)
        net = (orders["side"] * orders["contracts"]).groupby(orders["wallet"]).sum() * sign
        winners = set(net.index[net >= LARGE_CONTRACTS])
        if winners:
            return winners
    return set()


@dataclass
class Market:
    id: str
    role: str
    outcome: int
    trades: pd.DataFrame
    hours: pd.DataFrame          # per hour: p, x, dx (move into this hour), flow, quiet, window
    onset: pd.Timestamp | None
    reveal: pd.Timestamp | None
    reveal_from: str | None
    end: pd.Timestamp            # onset, or the last hour before the price settles

    @property
    def sign(self) -> int:
        return 1 if self.outcome == 1 else -1


def prepare(meta: dict, flagged: pd.DataFrame, reveal_utc: str | None = None,
            out_dir: Path = OUT_DIR) -> Market:
    trades = load_trades(meta["id"], out_dir)
    prices = load_prices(meta["id"], out_dir).set_index("hour")
    hours = pd.DataFrame({"p": prices["last"], "x": logit(prices["last"].to_numpy())},
                         index=prices.index)
    hours["dx"] = hours["x"].diff()
    flow = (trades["taker_side"] * trades["contracts"]).groupby(trades["time"].dt.floor("1h")).sum()
    hours["flow"] = flow.reindex(hours.index, fill_value=0.0)
    sign = 1 if meta["outcome"] == 1 else -1

    onset = reveal = reveal_from = None
    insiders = onset_wallets(trades, flagged, meta["id"], sign)
    if insiders:
        orders = wallet_orders(trades, insiders)
        onset = orders.loc[orders["side"] == sign, "time"].min().floor("1h")
    if onset is not None:
        after = hours.loc[onset:]
        on_side = after["p"] if sign > 0 else 1 - after["p"]
        hit = on_side >= REVEAL_PRICE
        if hit.any():
            reveal, reveal_from = hit.idxmax(), "price"
        if reveal_utc and (reveal is None or pd.Timestamp(reveal_utc).floor("1h") < reveal):
            reveal, reveal_from = pd.Timestamp(reveal_utc).floor("1h"), "documented"

    lo, hi = PRICE_RANGE
    in_range = hours["p"].between(lo, hi) & hours["p"].shift().between(lo, hi)
    before = hours.index < onset if onset is not None else np.ones(len(hours), bool)
    hours["quiet"] = in_range & before
    # dx at hour h is the move from h-1 to h. The window's moves start after the
    # onset hour and stop before the reveal hour, whose move is the news itself.
    hours["window"] = False
    if reveal is not None:
        hours.loc[(hours.index > onset) & (hours.index < reveal), "window"] = True
    on_side = hours["p"] if sign > 0 else 1 - hours["p"]
    end = onset if onset is not None else on_side.index[on_side < REVEAL_PRICE].max()
    return Market(meta["id"], meta["role"], int(meta["outcome"]), trades, hours,
                  onset, reveal, reveal_from, end)


def price_dynamics(markets: list[Market]) -> dict:
    drift, dxs, lag = [], [], []
    for m in markets:
        quiet = m.hours["quiet"]
        drift.append(m.sign * m.hours.loc[quiet, "dx"])
        dxs.append(m.hours.loc[quiet, "dx"])
        both = quiet & quiet.shift(fill_value=False)
        lag.append(m.hours.loc[both, "dx"] * m.hours["dx"].shift()[both])
    drift, dxs, lag = pd.concat(drift), pd.concat(dxs), pd.concat(lag)
    d = float(drift.mean())
    total_var = float(np.mean(dxs ** 2))
    bounce = float(lag.mean())
    permanent_var = total_var + 2 * min(bounce, 0.0)
    news_mean = math.sqrt(max(d, 0.0) / 2)
    noise_var = permanent_var - 4 * news_mean ** 2
    return {
        "quiet_hours": int(len(dxs)),
        "drift_toward_outcome": d,
        "drift_stderr": float(drift.std(ddof=1) / math.sqrt(len(drift))),
        "news_mean_95_upper": math.sqrt(max(d + 1.96 * drift.std(ddof=1) / math.sqrt(len(drift)),
                                            0.0) / 2),
        "total_var": total_var,
        "lag1_autocov": bounce,
        "permanent_var": permanent_var,
        "news_mean": news_mean,
        "news_weight": 2 * news_mean,
        "noise_sd": math.sqrt(max(noise_var, 0.0)),
    }


def impact(markets: list[Market]) -> dict:
    slopes = {}
    for m in markets:
        q = m.hours[m.hours["quiet"]]
        if len(q) >= 50 and q["flow"].var() > 0:
            slopes[m.id] = float(np.polyfit(q["flow"], q["dx"], 1)[0])
    return {"per_contract_by_market": slopes,
            "per_contract": float(np.median(list(slopes.values())))}


def lognormal_fit(contracts: pd.Series) -> dict:
    logs = np.log(contracts)
    return {"n": int(len(contracts)), "log_mean": float(logs.mean()),
            "log_sd": float(logs.std(ddof=1)), "median": float(np.exp(logs.median()))}


def uninformed_flow(markets: list[Market]) -> dict:
    """Large taker orders in quiet hours during the last HORIZON hours before each market's end."""
    rates, sizes = {}, []
    for m in markets:
        recent = (m.hours.index >= m.end - pd.Timedelta(hours=HORIZON)) & (m.hours.index < m.end)
        quiet_hours = m.hours.index[m.hours["quiet"] & recent]
        if len(quiet_hours) < 24:
            continue
        orders = taker_orders(m.trades)
        large = orders[(orders["contracts"] >= LARGE_CONTRACTS)
                       & orders["time"].dt.floor("1h").isin(quiet_hours)]
        rates[m.id] = {"quiet_hours": len(quiet_hours),
                       "per_hour": len(large) / len(quiet_hours),
                       "yes_share": float((large["side"] > 0).mean()) if len(large) else None}
        sizes.append(large["contracts"])
    return {"by_market": rates,
            "per_hour": float(np.median([r["per_hour"] for r in rates.values()])),
            "size": lognormal_fit(pd.concat(sizes))}


def insider_flow(markets: list[Market], flagged: pd.DataFrame, d: float) -> dict:
    wallets = set(flagged.loc[flagged["label"] != "weak", "address"])
    rows, insider_sizes, follower_sizes = [], [], []
    for m in markets:
        window = m.hours[m.hours["window"]]
        if m.reveal is None or len(window) == 0:
            continue
        hours = len(window) + 1          # onset hour through the hour before the reveal

        def in_window(t):
            return (t >= m.onset) & (t < m.reveal)

        flagged_orders = wallet_orders(m.trades, wallets)
        flagged_orders = flagged_orders[in_window(flagged_orders["time"])
                                        & (flagged_orders["contracts"] >= LARGE_CONTRACTS)
                                        & (flagged_orders["side"] == m.sign)]
        insider_sizes.append(flagged_orders["contracts"])

        others = taker_orders(m.trades)
        others = others[~others["wallet"].isin(wallets) & (others["contracts"] >= LARGE_CONTRACTS)
                        & in_window(others["time"])]
        same, opposite = others[others["side"] == m.sign], others[others["side"] != m.sign]
        before = taker_orders(m.trades)
        before = before[(before["contracts"] >= LARGE_CONTRACTS)
                        & (before["time"] >= m.onset - (m.reveal - m.onset))
                        & (before["time"] < m.onset)]
        follower_sizes.append(same["contracts"])

        rows.append({
            "market_id": m.id, "onset": m.onset.isoformat(), "reveal": m.reveal.isoformat(),
            "reveal_from": m.reveal_from, "window_hours": hours,
            "price_at_onset": float(m.hours.loc[m.onset, "p"]),
            "price_before_reveal": float(m.hours["p"].shift().loc[m.reveal]),
            "drift": float(m.sign * window["dx"].mean()),
            "flagged_orders": len(flagged_orders),
            "flagged_per_hour": len(flagged_orders) / hours,
            "same_side_per_hour": len(same) / hours,
            "opposite_side_per_hour": len(opposite) / hours,
            "both_sides_per_hour_before_onset": len(before) / hours,
        })
    inc = pd.DataFrame(rows)
    inc["excess_follow_per_hour"] = inc["same_side_per_hour"] - inc["opposite_side_per_hour"]
    inc["activity_ratio"] = ((inc["same_side_per_hour"] + inc["opposite_side_per_hour"])
                             / inc["both_sides_per_hour_before_onset"].where(
                                 inc["both_sides_per_hour_before_onset"] > 0))
    return {
        "incidents": inc.to_dict("records"),
        "insider_drift": float(inc["drift"].median() - d),
        "insider_drift_pooled": float(
            sum(r.drift * (r.window_hours - 1) for r in inc.itertuples())
            / sum(r.window_hours - 1 for r in inc.itertuples()) - d),
        "insider_rate": float(inc["flagged_per_hour"].median()),
        "copy_follow_rate": float(max(inc["excess_follow_per_hour"].median(), 0.0)),
        "insider_size": lognormal_fit(pd.concat(insider_sizes)),
        "follower_size": lognormal_fit(pd.concat(follower_sizes)),
        "window_hours_median": float(inc["window_hours"].median()),
        "activity_ratio_median": float(inc["activity_ratio"].median()),
    }


def to_params(est: dict) -> dict:
    """Map the estimates onto MarketParams fields (see market/calibration.py)."""
    price, flow, ins = est["price"], est["uninformed_flow"], est["insider"]
    uninformed = flow["per_hour"]

    def size(fit):
        return {"size_log_mean": fit["log_mean"], "size_log_sd": fit["log_sd"]}

    return {
        "horizon": HORIZON,
        "onset_rate": 1 - (1 - INSIDER_SHARE) ** (1 / HORIZON),
        "insider_drift": max(ins["insider_drift"], 0.0),
        "news_mean": price["news_mean"],
        "news_sd": 1.0,
        "news_weight": price["news_weight"],
        "noise_sd": price["noise_sd"],
        "impact": est["impact"]["per_contract"] * CONTRACT_SIZE,
        "fee": 0.0,
        "flow": {
            "retail_rate": uninformed * (1 - COPY_SHARE),
            "copy_base_rate": uninformed * COPY_SHARE,
            "insider_rate": ins["insider_rate"],
            "copy_follow_rate": ins["copy_follow_rate"],
            "profiles": {"retail": size(flow["size"]), "insider": size(ins["insider_size"]),
                         "copybot": size(ins["follower_size"])},
        },
    }


def calibrate(out_dir: Path = OUT_DIR) -> dict:
    flagged = load_flagged()
    reveals = {m["id"]: m.get("reveal_utc") for m in MARKETS}
    meta = [m for m in json.loads((out_dir / "markets.json").read_text())
            if m["outcome"] in (0, 1)]
    markets = [prepare(m, flagged, reveals.get(m["id"]), out_dir) for m in meta]
    est = {
        "markets": [{"id": m.id, "role": m.role, "quiet_hours": int(m.hours["quiet"].sum())}
                    for m in markets],
        "price": price_dynamics(markets),
        "impact": impact(markets),
        "uninformed_flow": uninformed_flow(markets),
    }
    est["insider"] = insider_flow(markets, flagged, est["price"]["drift_toward_outcome"])
    return {
        "source": "SII-WANGZJ/Polymarket_data; see polymarket/calibrate.py",
        "units": {"step": "1 hour", "contract_size": CONTRACT_SIZE,
                  "large_trade_contracts": LARGE_CONTRACTS},
        "design_choices": {
            "horizon_hours": HORIZON, "insider_share": INSIDER_SHARE, "copy_share": COPY_SHARE,
            "wallet_features": "only trade size is calibrated; age, funding, category and "
                               "longshot record keep their placeholder values until task 4",
        },
        "params": to_params(est),
        "estimates": est,
    }


def main() -> None:
    report = calibrate()
    OUTPUT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("units", "params")}, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
