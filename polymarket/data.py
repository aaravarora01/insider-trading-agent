"""Download Polymarket markets and trades from the SII-WANGZJ/Polymarket_data dataset.

The dataset (Hugging Face, MIT licence) covers every CLOB fill from 2022-11-21
to 2026-10-04. `trades.parquet` is 46 GB with no row-group statistics, so we
cannot ask the server for one market's rows directly. Instead:

1. `markets.parquet` (0.5 GB) is downloaded once.
2. `trades_index.parquet` records, for every market, the first and last row of
   `trades.parquet` that mentions it. Building it reads only the `market_id`
   column (50 MB over HTTP, about 90 s).
3. For the markets we want, we read just the row groups in that range, skip
   the 26 GB `transaction_hash` column, and keep the matching rows.

Trades are converted to the YES perspective: `answer1`/`token1` is YES. Each
row is one fill between a maker and a taker; `taker_side` is +1 if the taker
bought YES (or sold NO) and -1 otherwise, and `yes_price` is the fill price
of the YES token. Buying NO at q is the same bet as selling YES at 1 - q.

The raw `usd_amount` is the cash on the token named in the row, which is not
always the taker's own stake: when a YES buyer is matched with a NO buyer, the
row records the NO buyer's side. `taker_usd` and `maker_usd` are each party's
stake on the side they took: contracts x YES price if long YES, contracts x
(1 - YES price) if long NO. Use them for trade sizes.
"""
from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = "SII-WANGZJ/Polymarket_data"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
HF_PATH = f"datasets/{REPO}"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "polymarket"

TRADE_COLUMNS = ["timestamp", "block_number", "market_id", "maker", "taker", "price",
                 "usd_amount", "token_amount", "taker_direction", "nonusdc_side"]


def markets_path() -> Path:
    path = RAW_DIR / "markets.parquet"
    if not path.exists():
        import urllib.request
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE_URL}/markets.parquet", tmp)
        tmp.rename(path)
    return path


def trades_index_path() -> Path:
    path = RAW_DIR / "trades_index.parquet"
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"""
            copy (select market_id, count(*) as n_trades,
                         min(file_row_number) as first_row, max(file_row_number) as last_row
                  from read_parquet('{BASE_URL}/trades.parquet', file_row_number=true)
                  group by market_id)
            to '{path}'""")
    return path


def load_markets(ids: list[str] | None = None) -> pd.DataFrame:
    """Market metadata joined with trade counts. `outcome` is 1 if answer1 won."""
    where = "" if ids is None else "where m.id in (select unnest($ids))"
    df = duckdb.execute(f"""
        select m.*, ix.n_trades, ix.first_row, ix.last_row
        from '{markets_path()}' m left join '{trades_index_path()}' ix on ix.market_id = m.id
        {where}""", {"ids": ids} if ids is not None else {}).df()
    df["outcome"] = df["outcome_prices"].map(_outcome)
    return df


def _outcome(outcome_prices: str | None) -> float:
    """1 if answer1 won, 0 if answer2 won, NaN if unresolved or split."""
    try:
        first, second = (float(x) for x in ast.literal_eval(outcome_prices))
    except (ValueError, SyntaxError, TypeError):
        return np.nan
    if first >= 0.99 and second <= 0.01:
        return 1.0
    if second >= 0.99 and first <= 0.01:
        return 0.0
    return np.nan


def _open_trades():
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(f"{HF_PATH}/trades.parquet", block_size=8 * 2**20)


def _row_group_starts(metadata: pq.FileMetaData) -> np.ndarray:
    sizes = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    return np.concatenate([[0], np.cumsum(sizes)])


def row_groups_for(first_row: int, last_row: int, starts: np.ndarray) -> range:
    lo = int(np.searchsorted(starts, first_row, side="right")) - 1
    hi = int(np.searchsorted(starts, last_row, side="right"))
    return range(lo, hi)


def fetch_trades(market_ids: list[str], workers: int = 8) -> dict[str, pd.DataFrame]:
    """Raw fills for each market, read from only the row groups that hold them."""
    markets = load_markets(market_ids).set_index("id")
    missing = set(market_ids) - set(markets.index[markets["n_trades"].notna()])
    if missing:
        raise ValueError(f"no trades in the dataset for markets {sorted(missing)}")

    with _open_trades() as f:
        metadata = pq.ParquetFile(f).metadata
    starts = _row_group_starts(metadata)
    groups = sorted({g for _, m in markets.iterrows()
                     for g in row_groups_for(int(m.first_row), int(m.last_row), starts)})
    wanted = pa.array(list(market_ids))

    def read(group: int) -> pa.Table:
        with _open_trades() as f:
            table = pq.ParquetFile(f, metadata=metadata).read_row_group(group, columns=TRADE_COLUMNS)
        return table.filter(pc.is_in(table["market_id"], wanted))

    print(f"reading {len(groups)} row groups for {len(market_ids)} markets")
    with ThreadPoolExecutor(workers) as pool:
        tables = list(pool.map(read, groups))
    trades = pa.concat_tables(tables).to_pandas()

    out = {}
    for market_id, rows in trades.groupby("market_id"):
        expected = int(markets.loc[market_id, "n_trades"])
        if len(rows) != expected:
            raise RuntimeError(f"market {market_id}: got {len(rows)} fills, index says {expected}")
        out[market_id] = rows
    return out


def to_yes_perspective(raw: pd.DataFrame) -> pd.DataFrame:
    is_yes = (raw["nonusdc_side"] == "token1").to_numpy()
    buys = (raw["taker_direction"] == "BUY").to_numpy()
    df = pd.DataFrame({
        "time": pd.to_datetime(raw["timestamp"].astype("int64"), unit="s", utc=True),
        "block_number": raw["block_number"].astype("int64"),
        "maker": raw["maker"],
        "taker": raw["taker"],
        "token": np.where(is_yes, "YES", "NO"),
        "taker_side": np.where(buys == is_yes, 1, -1).astype("int8"),
        "yes_price": np.where(is_yes, raw["price"], 1.0 - raw["price"]),
        "contracts": raw["token_amount"],
        "usd_amount": raw["usd_amount"],
    })
    taker_long_yes = df["taker_side"] > 0
    df["taker_usd"] = df["contracts"] * np.where(taker_long_yes, df["yes_price"], 1 - df["yes_price"])
    df["maker_usd"] = df["contracts"] * np.where(taker_long_yes, 1 - df["yes_price"], df["yes_price"])
    return df.sort_values(["block_number", "time"], kind="stable").reset_index(drop=True)


def hourly_prices(trades: pd.DataFrame) -> pd.DataFrame:
    """Last and contract-weighted YES price per hour, with volume and fill count."""
    g = trades.set_index("time").groupby(pd.Grouper(freq="1h"))
    weighted = (trades["yes_price"] * trades["contracts"]).groupby(
        trades["time"].dt.floor("1h")).sum()
    out = pd.DataFrame({
        "last": g["yes_price"].last(),
        "contracts": g["contracts"].sum(),
        "usd_volume": (g["taker_usd"].sum() + g["maker_usd"].sum()),
        "fills": g["yes_price"].size(),
    })
    out["vwap"] = weighted.reindex(out.index) / out["contracts"].where(out["contracts"] > 0)
    out["last"] = out["last"].ffill()
    return out.rename_axis("hour").reset_index()


def market_record(row: dict, role: str, note: str) -> dict:
    return {
        "id": row["id"], "question": row["question"], "role": role, "note": note,
        "answer1": row["answer1"], "answer2": row["answer2"], "outcome": row["outcome"],
        "event_title": row["event_title"], "created_at": row["created_at"].isoformat(),
        "end_date": row["end_date"].isoformat(), "volume_usd": row["volume"],
        "n_trades": int(row["n_trades"]),
    }


def download(selection: list[dict], out_dir: Path = OUT_DIR, force: bool = False) -> None:
    (out_dir / "trades").mkdir(parents=True, exist_ok=True)
    (out_dir / "prices").mkdir(parents=True, exist_ok=True)

    ids = [m["id"] for m in selection]
    todo = [i for i in ids if force or not (out_dir / "trades" / f"{i}.parquet").exists()]
    if todo:
        for market_id, raw in fetch_trades(todo).items():
            trades = to_yes_perspective(raw)
            trades.to_parquet(out_dir / "trades" / f"{market_id}.parquet", index=False)
            hourly_prices(trades).to_parquet(out_dir / "prices" / f"{market_id}.parquet", index=False)
            print(f"  {market_id}: {len(trades):,} fills")

    markets = load_markets(ids).set_index("id")
    records = [market_record({**markets.loc[m["id"]].to_dict(), "id": m["id"]},
                             m["role"], m["note"]) for m in selection]
    (out_dir / "markets.json").write_text(json.dumps(records, indent=2, default=str))


def load_trades(market_id: str, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "trades" / f"{market_id}.parquet")


def load_prices(market_id: str, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "prices" / f"{market_id}.parquet")
