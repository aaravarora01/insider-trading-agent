"""Pull trades and hourly prices for the markets in `polymarket.selection`.

    python -m polymarket.download            # skips markets already downloaded
    python -m polymarket.download --force

Writes data/polymarket/{markets.json, trades/<id>.parquet, prices/<id>.parquet}.
"""
from __future__ import annotations

import argparse

from polymarket.data import download
from polymarket.selection import MARKETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing markets")
    args = parser.parse_args()
    download(MARKETS, force=args.force)


if __name__ == "__main__":
    main()
