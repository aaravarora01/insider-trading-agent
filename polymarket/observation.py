"""Fit the wallet observation model P(wallet features | retail / insider / copy-bot).

    python -m polymarket.observation      # writes market/wallet_profiles.json

Needs `python -m polymarket.history` first.

Population: every order of at least LARGE_CONTRACTS contracts in the downloaded
markets (taker orders, plus flagged wallets' orders in either role).

Features are point-in-time: each order sees only the wallet's activity in
*other* markets before the order (`market.flow` lists them). A longshot bet
counts once its market has resolved (last fill) before the order, so an
insider's winning bet never leaks into its own features.

Weak labels:
- insider: a flagged wallet's orders on the winning side of its flagged market,
  before the reveal, weighted by the flag's confidence (LABEL_WEIGHT).
- copy-bot: a wallet that places a same-side order within FOLLOW_SECONDS of
  the *same* other wallet's order at least FOLLOW_MIN_PAIRS times. Requiring a
  repeated leader keeps busy markets, where some large order lands within a
  minute by chance, from labelling everyone a copy-bot.
- everyone else is treated as retail. Some are unflagged insiders, but
  insiders are a small minority of large orders, so the bias is small.

Labelled orders are weighted so each wallet counts once per market: a wallet
with 30 orders should not define what an insider looks like.

We tried semi-supervised EM (unlabeled orders free to be any type). On data
drawn from the simulator it drifted: the likelihood fell as it iterated (the
Beta fits are moment-matched, not exact maximum likelihood) and the insider
share came out 2.6-4x too high. With ~20 labelled insider wallets the
supervised fit is the safer choice.

The fit is scored by leave-one-incident-out AUC: drop one incident's labels,
fit, and rank that market's flagged orders against its other large orders by
log f_insider - log f_retail.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from market.flow import TraderType, WalletProfile, feature_loglik
from polymarket.calibrate import LARGE_CONTRACTS, prepare, taker_orders, wallet_orders
from polymarket.categories import market_categories
from polymarket.data import OUT_DIR, _outcome, load_trades, markets_path
from polymarket.flagged import load_flagged
from polymarket.history import MARKET_LAST_FILL, WALLET_MARKETS
from polymarket.selection import MARKETS

FOLLOW_SECONDS = 60
FOLLOW_MIN_PAIRS = 3
LABEL_WEIGHT = {"strong": 1.0, "multi_source": 0.9, "single_source": 0.7}
COPY_WEIGHT = 0.8
OUTPUT = Path(__file__).resolve().parent.parent / "market" / "wallet_profiles.json"
TYPES = [t.name.lower() for t in TraderType]
FEATURES = ["contracts", "fresh", "age_days", "category_share", "longshot_bets",
            "longshot_win_rate"]


# ---------------------------------------------------------------- orders and labels

def epoch_seconds(times: pd.Series) -> np.ndarray:
    """Unix seconds, whatever the datetime resolution (pandas 3 defaults to microseconds)."""
    return ((times - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy()


def large_orders(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    flagged = set(load_flagged()["address"])
    frames = []
    for meta in json.loads((out_dir / "markets.json").read_text()):
        trades = load_trades(meta["id"], out_dir)
        takers = taker_orders(trades)
        orders = pd.concat([takers[~takers["wallet"].isin(flagged)],
                            wallet_orders(trades, flagged)])
        orders = orders[orders["contracts"] >= LARGE_CONTRACTS]
        frames.append(orders.assign(market_id=meta["id"], outcome=meta["outcome"]))
    orders = pd.concat(frames, ignore_index=True)
    return orders.sort_values(["market_id", "time"], kind="stable").reset_index(drop=True)


def reveal_times(out_dir: Path = OUT_DIR) -> dict[str, pd.Timestamp]:
    flagged = load_flagged()
    documented = {m["id"]: m.get("reveal_utc") for m in MARKETS}
    out = {}
    for meta in json.loads((out_dir / "markets.json").read_text()):
        if meta["role"] == "incident":
            m = prepare(meta, flagged, documented.get(meta["id"]), out_dir)
            if m.reveal is not None:
                out[meta["id"]] = m.reveal
    return out


def insider_weights(orders: pd.DataFrame, reveals: dict[str, pd.Timestamp]) -> np.ndarray:
    weight = np.zeros(len(orders))
    for w in load_flagged().itertuples():
        if w.label not in LABEL_WEIGHT:
            continue
        for market_id in w.market_ids:
            if market_id not in reveals:
                continue
            o = orders
            hit = ((o["wallet"] == w.address) & (o["market_id"] == market_id)
                   & (o["side"] == np.where(o["outcome"] == 1, 1, -1))
                   & (o["time"] < reveals[market_id]))
            weight[hit.to_numpy()] = LABEL_WEIGHT[w.label]
    return weight


def copybot_wallets(orders: pd.DataFrame, max_lag: int = 30) -> tuple[set[str], pd.DataFrame]:
    """Wallets that repeatedly follow the same leader within FOLLOW_SECONDS."""
    pairs = []
    for _, g in orders.groupby(["market_id", "side"], sort=False):
        g = g.sort_values("time")
        t = epoch_seconds(g["time"])
        wallet = g["wallet"].to_numpy()
        order = g.index.to_numpy()
        for lag in range(1, min(max_lag, len(g))):
            follows = (t[lag:] - t[:-lag] <= FOLLOW_SECONDS) & (wallet[lag:] != wallet[:-lag])
            if not follows.any():
                break
            pairs.append(pd.DataFrame({"leader": wallet[:-lag][follows],
                                       "follower": wallet[lag:][follows],
                                       "follower_order": order[lag:][follows]}))
    if not pairs:
        return set(), pd.DataFrame(columns=["leader", "follower", "n"])
    # Each follower order counts once per leader, however many of the leader's
    # orders it trails.
    pairs = pd.concat(pairs).drop_duplicates()
    counts = pairs.groupby(["leader", "follower"]).size().rename("n").reset_index()
    repeat = counts[counts["n"] >= FOLLOW_MIN_PAIRS]
    flagged = set(load_flagged()["address"])
    return set(repeat["follower"]) - flagged, repeat


# ---------------------------------------------------------------- point-in-time features

def order_features(orders: pd.DataFrame) -> pd.DataFrame:
    categories = market_categories()
    history = pd.read_parquet(WALLET_MARKETS)
    history = history[history["wallet"].isin(set(orders["wallet"]))]
    resolved = pd.read_parquet(MARKET_LAST_FILL).set_index("market_id")["last_fill"]
    outcome_prices = duckdb.execute(
        f"select id, outcome_prices from '{markets_path()}' where id in "
        f"(select unnest($ids))", {"ids": history["market_id"].unique().tolist()}).df()
    outcomes = outcome_prices.set_index("id")["outcome_prices"].map(_outcome)

    history = history.assign(
        category=history["market_id"].map(categories).fillna("other"),
        resolved=history["market_id"].map(resolved),
        outcome=history["market_id"].map(outcomes))
    o = orders.assign(oid=np.arange(len(orders)),
                      t=epoch_seconds(orders["time"]),
                      category=orders["market_id"].map(categories).fillna("other"))

    con = duckdb.connect()
    con.register("o", o[["oid", "wallet", "market_id", "t", "category"]])
    con.register("h", history)
    agg = con.execute("""
        select o.oid,
               count(h.market_id) as n_prior,
               min(h.first_fill) as first_prior,
               count(h.market_id) filter (where h.category = o.category) as same_category,
               -- A missing longshot time or outcome is NULL; coalesce each term so one
               -- NULL does not drop the whole market from the sum.
               coalesce(sum(coalesce((h.longshot_yes < o.t and h.resolved < o.t
                                      and h.outcome in (0, 1))::int, 0)
                          + coalesce((h.longshot_no < o.t and h.resolved < o.t
                                      and h.outcome in (0, 1))::int, 0)), 0) as bets,
               coalesce(sum(coalesce((h.longshot_yes < o.t and h.resolved < o.t
                                      and h.outcome = 1)::int, 0)
                          + coalesce((h.longshot_no < o.t and h.resolved < o.t
                                      and h.outcome = 0)::int, 0)), 0) as wins
        from o left join h
          on h.wallet = o.wallet and h.market_id <> o.market_id and h.first_fill < o.t
        group by o.oid order by o.oid""").df()

    fresh = agg["n_prior"].to_numpy() == 0
    bets = agg["bets"].to_numpy(dtype=float)
    return pd.DataFrame({
        "contracts": orders["contracts"].to_numpy(),
        "fresh": fresh,
        "age_days": np.where(fresh, np.nan, (o["t"].to_numpy() - agg["first_prior"].to_numpy())
                             / 86400),
        "category_share": np.where(fresh, np.nan, (agg["same_category"] + 0.5)
                                   / (agg["n_prior"] + 1)),
        "longshot_bets": np.where(fresh, 0, bets),
        "longshot_win_rate": np.where(bets > 0, (agg["wins"] + 0.5) / (bets + 1), np.nan),
        "prior_markets": agg["n_prior"].to_numpy(),
    }, index=orders.index)


# ---------------------------------------------------------------- fitting

def _wmean(v, w):
    return float(np.sum(w * v) / np.sum(w))


def _wsd(v, w):
    m = _wmean(v, w)
    return max(math.sqrt(_wmean((v - m) ** 2, w)), 0.1)


def _beta(v, w):
    m = min(max(_wmean(v, w), 0.02), 0.98)
    var = max(_wmean((v - m) ** 2, w), 1e-4)
    k = max(m * (1 - m) / var - 1, 0.5)
    return max(m * k, 0.3), max((1 - m) * k, 0.3)


def fit_profile(x: pd.DataFrame, w: np.ndarray, fallback: WalletProfile | None = None,
                min_weight: float = 3.0) -> WalletProfile:
    """Weighted maximum likelihood (method of moments for the Beta parts)."""
    fresh = x["fresh"].to_numpy()
    hist = ~fresh
    bets = x["longshot_bets"].to_numpy()
    has = hist & (bets > 0)
    out = {
        "size_log_mean": _wmean(np.log(x["contracts"]), w),
        "size_log_sd": _wsd(np.log(x["contracts"]), w),
        "fresh_prob": min(max(_wmean(fresh, w), 0.01), 0.99),
    }
    if w[hist].sum() >= min_weight or fallback is None:
        age = np.log(np.maximum(x["age_days"].to_numpy()[hist], 1 / 24))
        a, b = _beta(x["category_share"].to_numpy()[hist], w[hist])
        out |= {"age_log_mean": _wmean(age, w[hist]), "age_log_sd": _wsd(age, w[hist]),
                "category_a": a, "category_b": b,
                "longshot_prob": min(max(_wmean(bets[hist] > 0, w[hist]), 0.01), 0.99)}
    else:
        out |= {k: getattr(fallback, k) for k in
                ("age_log_mean", "age_log_sd", "category_a", "category_b", "longshot_prob")}
    if w[has].sum() >= min_weight or fallback is None:
        lb = np.log(bets[has])
        a, b = _beta(x["longshot_win_rate"].to_numpy()[has], w[has])
        out |= {"longshot_log_mean": _wmean(lb, w[has]), "longshot_log_sd": _wsd(lb, w[has]),
                "win_a": a, "win_b": b}
    else:
        out |= {k: getattr(fallback, k) for k in
                ("longshot_log_mean", "longshot_log_sd", "win_a", "win_b")}
    return WalletProfile(**out)


def _arrays(x: pd.DataFrame) -> dict[str, np.ndarray]:
    return {k: x[k].to_numpy(dtype=bool if k == "fresh" else float) for k in FEATURES}


def type_loglik(x: pd.DataFrame, profiles: list[WalletProfile]) -> np.ndarray:
    arrays = _arrays(x)
    return np.stack([feature_loglik(arrays, p) for p in profiles], axis=1)


def fit(x: pd.DataFrame, labels: np.ndarray) -> list[WalletProfile]:
    """Supervised fit. labels[i, k] is the weight of label k on order i; rows with
    no label are retail. Weights only matter relative to others of the same type."""
    pooled = fit_profile(x, np.ones(len(x)))
    weights = labels.copy()
    weights[labels.sum(axis=1) == 0, TraderType.RETAIL] = 1.0
    return [fit_profile(x, weights[:, k] + 1e-12, pooled) for k in TraderType]


def per_wallet(weight: np.ndarray, orders: pd.DataFrame) -> np.ndarray:
    """Divide each labelled order's weight by its wallet's labelled orders in that market."""
    labelled = weight > 0
    groups = orders.loc[labelled, ["wallet", "market_id"]]
    count = groups.groupby(["wallet", "market_id"])["wallet"].transform("size")
    out = weight.copy()
    out[labelled] = weight[labelled] / count.to_numpy()
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def leave_one_incident_out(orders, x, labels) -> dict[str, float]:
    insider = labels[:, TraderType.INSIDER] > 0
    out = {}
    for market_id in sorted(orders.loc[insider, "market_id"].unique()):
        here = (orders["market_id"] == market_id).to_numpy()
        held = labels.copy()
        held[here & insider] = 0
        profiles = fit(x, held)
        ll = type_loglik(x[here], profiles)
        score = ll[:, TraderType.INSIDER] - ll[:, TraderType.RETAIL]
        out[market_id] = auc(score[insider[here]], score[~insider[here]])
    return out


# ---------------------------------------------------------------- main

def build(out_dir: Path = OUT_DIR) -> dict:
    orders = large_orders(out_dir)
    x = order_features(orders)
    labels = np.zeros((len(orders), len(TraderType)))
    labels[:, TraderType.INSIDER] = per_wallet(insider_weights(orders, reveal_times(out_dir)),
                                               orders)
    copybots, pairs = copybot_wallets(orders)
    is_copy = orders["wallet"].isin(copybots).to_numpy() & (labels.sum(axis=1) == 0)
    labels[is_copy, TraderType.COPYBOT] = per_wallet(np.where(is_copy, COPY_WEIGHT, 0.0),
                                                     orders)[is_copy]

    evaluation = leave_one_incident_out(orders, x, labels)
    profiles = fit(x, labels)

    ll = type_loglik(x, profiles)
    insider_llr = ll[:, TraderType.INSIDER] - ll[:, TraderType.RETAIL]
    labelled = labels[:, TraderType.INSIDER] > 0
    summary = x.assign(type=np.select([labelled, is_copy], ["insider", "copybot"], "unlabeled"))
    return {
        "method": "supervised, weak labels weighted per wallet; see polymarket/observation.py",
        "profiles": {name: asdict(p) for name, p in zip(TYPES, profiles)},
        "evaluation": {
            "leave_one_incident_out_auc": evaluation,
            "median_auc": float(np.nanmedian(list(evaluation.values()))),
            "in_sample_auc": auc(insider_llr[labelled], insider_llr[~labelled]),
        },
        "data": {
            "large_orders": int(len(orders)),
            "wallets": int(orders["wallet"].nunique()),
            "insider_labelled_orders": int(labelled.sum()),
            "insider_labelled_wallets": int(orders.loc[labelled, "wallet"].nunique()),
            "copybot_wallets": len(copybots),
            "copybot_labelled_orders": int(is_copy.sum()),
            "copy_pairs": int(len(pairs)),
            "feature_means_by_label": summary.groupby("type").agg(
                orders=("contracts", "size"), median_contracts=("contracts", "median"),
                fresh=("fresh", "mean"), median_age_days=("age_days", "median"),
                category_share=("category_share", "mean"),
                has_longshots=("longshot_bets", lambda b: float((b > 0).mean())),
                win_rate=("longshot_win_rate", "mean")).round(3).to_dict("index"),
        },
    }


def main() -> None:
    report = build()
    OUTPUT.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({k: report[k] for k in ("evaluation", "data")}, indent=2, default=float))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
