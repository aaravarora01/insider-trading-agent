"""Trading history of the wallets that place large orders in our markets.

    python -m polymarket.history       # resumable; about 11 GB over HTTP

Wallet features must be computed from what a wallet did *before* a trade, so we
need each wallet's activity across all markets, not just ours. We stream seven
columns of `trades.parquet` (skipping the 26 GB `transaction_hash`), keep fills
involving a wallet of interest, and reduce each row group to one row per
(wallet, market):

    first_fill, last_fill, fills       when the wallet traded this market
    longshot_yes, longshot_no          first time it bought YES (or NO) at a
                                       price below LONGSHOT_PRICE, if ever

We also record each market's last fill over all wallets, a stand-in for its
resolution time. Results go to data/raw/history/ (one file per row group) and
are combined into data/raw/wallet_markets.parquet and market_last_fill.parquet.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from polymarket.calibrate import LARGE_CONTRACTS, taker_orders
from polymarket.data import OUT_DIR, RAW_DIR, _open_trades, load_trades
from polymarket.flagged import load_flagged

LONGSHOT_PRICE = 0.2
COLUMNS = ["timestamp", "market_id", "maker", "taker", "price", "nonusdc_side", "taker_direction"]
PARTS = RAW_DIR / "history"
WALLETS = RAW_DIR / "wallets_of_interest.parquet"
WALLET_MARKETS = RAW_DIR / "wallet_markets.parquet"
MARKET_LAST_FILL = RAW_DIR / "market_last_fill.parquet"


def wallets_of_interest(out_dir: Path = OUT_DIR) -> list[str]:
    """Takers of large orders in the downloaded markets, plus every flagged wallet."""
    if WALLETS.exists():
        return pd.read_parquet(WALLETS)["wallet"].tolist()
    wallets = set(load_flagged()["address"])
    for meta in json.loads((out_dir / "markets.json").read_text()):
        orders = taker_orders(load_trades(meta["id"], out_dir))
        wallets |= set(orders.loc[orders["contracts"] >= LARGE_CONTRACTS, "wallet"])
    pd.DataFrame({"wallet": sorted(wallets)}).to_parquet(WALLETS, index=False)
    return sorted(wallets)


def _reduce(table: pa.Table, wallets: pa.Array) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_fill = (table.group_by("market_id").aggregate([("timestamp", "max")])
                 .rename_columns(["market_id", "last_fill"]).to_pandas())

    is_maker = pc.is_in(table["maker"], wallets)
    is_taker = pc.is_in(table["taker"], wallets)
    rows = table.filter(pc.or_(is_maker, is_taker)).to_pandas()
    if rows.empty:
        return pd.DataFrame(), last_fill

    yes_token = rows["nonusdc_side"] == "token1"
    yes_price = rows["price"].where(yes_token, 1 - rows["price"])
    taker_long_yes = (rows["taker_direction"] == "BUY") == yes_token
    parts = []
    for role, long_yes in (("taker", taker_long_yes), ("maker", ~taker_long_yes)):
        mine = rows[role].isin(set(wallets.to_pylist()))
        own_price = yes_price.where(long_yes, 1 - yes_price)
        longshot = own_price < LONGSHOT_PRICE
        parts.append(pd.DataFrame({
            "wallet": rows.loc[mine, role],
            "market_id": rows.loc[mine, "market_id"],
            "time": rows.loc[mine, "timestamp"],
            "longshot_yes": rows["timestamp"].where(longshot & long_yes)[mine],
            "longshot_no": rows["timestamp"].where(longshot & ~long_yes)[mine],
        }))
    fills = pd.concat(parts)
    per_wallet = fills.groupby(["wallet", "market_id"]).agg(
        first_fill=("time", "min"), last_fill=("time", "max"), fills=("time", "size"),
        longshot_yes=("longshot_yes", "min"), longshot_no=("longshot_no", "min")).reset_index()
    return per_wallet, last_fill


def scan(workers: int = 16) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    wallets = pa.array(wallets_of_interest())
    with _open_trades() as f:
        metadata = pq.ParquetFile(f).metadata
    todo = [g for g in range(metadata.num_row_groups)
            if not (PARTS / f"wallets_{g:04d}.parquet").exists()]
    print(f"{len(wallets)} wallets; {len(todo)} of {metadata.num_row_groups} row groups to read",
          flush=True)

    def work(group: int) -> int:
        # Exact range reads: the requests are latency-bound, so many small
        # parallel reads beat fewer large buffered ones.
        with _open_trades(cache_type="none") as f:
            table = pq.ParquetFile(f, metadata=metadata).read_row_group(group, columns=COLUMNS)
        per_wallet, last_fill = _reduce(table, wallets)
        last_fill.to_parquet(PARTS / f"markets_{group:04d}.parquet", index=False)
        tmp = PARTS / f"wallets_{group:04d}.tmp"
        per_wallet.to_parquet(tmp, index=False)
        tmp.rename(PARTS / f"wallets_{group:04d}.parquet")   # written last: marks the group done
        return group

    with ThreadPoolExecutor(workers) as pool:
        for i, future in enumerate(as_completed(pool.submit(work, g) for g in todo), 1):
            future.result()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} row groups", flush=True)
    combine()


def combine() -> None:
    per_wallet = pd.concat(pd.read_parquet(p) for p in sorted(PARTS.glob("wallets_*.parquet")))
    per_wallet = per_wallet.groupby(["wallet", "market_id"]).agg(
        first_fill=("first_fill", "min"), last_fill=("last_fill", "max"), fills=("fills", "sum"),
        longshot_yes=("longshot_yes", "min"), longshot_no=("longshot_no", "min")).reset_index()
    per_wallet.to_parquet(WALLET_MARKETS, index=False)
    last = pd.concat(pd.read_parquet(p) for p in sorted(PARTS.glob("markets_*.parquet")))
    last.groupby("market_id")["last_fill"].max().reset_index().to_parquet(MARKET_LAST_FILL,
                                                                           index=False)
    print(f"wrote {WALLET_MARKETS} ({len(per_wallet):,} wallet-market rows)")


if __name__ == "__main__":
    scan()
