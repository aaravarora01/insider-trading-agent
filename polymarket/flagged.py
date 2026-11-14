"""Publicly flagged wallets, used as weak insider labels.

`flagged_wallets.csv` lists wallets that journalists or on-chain analysts
named as suspected insiders. Columns:

- `label`: how much to trust the flag.
    strong         legal action, or the ACDC report's own funding analysis
    multi_source   reported by several independent analysts or outlets
    single_source  one analyst firm
    weak           an established multi-market trader; likely noisy
- `address_from`: how the address was obtained.
    published        printed in full by the source
    username_lookup  source gave a username; resolved via Polymarket's profile search
    prefix_match     source gave a prefix; matched in the market's trades with the
                     reported amounts
- `market_ids`: the markets in `polymarket.selection` (semicolon-separated)
  where the flagged trades happened.

None of these is a confirmed insider except where charges were filed. The
ACDC report's full list of 152 wallets is not public.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymarket.data import OUT_DIR, load_trades

CSV = Path(__file__).with_name("flagged_wallets.csv")
LABELS = ("strong", "multi_source", "single_source", "weak")


def load_flagged() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    df["address"] = df["address"].str.lower()
    df["market_ids"] = df["market_ids"].str.split(";")
    return df


def wallet_activity(trades: pd.DataFrame, address: str) -> dict:
    """One wallet's fills in one market: YES contracts bought, cost, and timing."""
    taker, maker = trades["taker"] == address, trades["maker"] == address
    rows = trades[taker | maker]
    if rows.empty:
        return {"fills": 0}
    side = rows["taker_side"].where(taker[rows.index], -rows["taker_side"])
    stake = rows["taker_usd"].where(taker[rows.index], rows["maker_usd"])
    return {
        "fills": len(rows),
        "net_yes_contracts": float((side * rows["contracts"]).sum()),
        "usd_long_yes": float(stake[side > 0].sum()),
        "usd_long_no": float(stake[side < 0].sum()),
        "first_fill": rows["time"].min(),
        "last_fill": rows["time"].max(),
        "taker_share": float(taker[rows.index].mean()),
    }


def flagged_activity(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Activity of every flagged wallet in each of its downloaded markets."""
    rows, cache = [], {}
    for w in load_flagged().itertuples():
        for market_id in w.market_ids:
            if not (out_dir / "trades" / f"{market_id}.parquet").exists():
                continue
            if market_id not in cache:
                cache[market_id] = load_trades(market_id, out_dir)
            rows.append({"address": w.address, "username": w.username, "label": w.label,
                         "market_id": market_id,
                         **wallet_activity(cache[market_id], w.address)})
    return pd.DataFrame(rows)
