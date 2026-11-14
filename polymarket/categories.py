"""Market categories from keyword rules on the question and event title.

The dataset has no category field. Rules are checked in order and the first
match wins, so specific ones come first: "Will Trump win the Nobel Peace
Prize?" is an award, not politics; "Maduro out?" is geopolitics, not politics.
This is a heuristic; it only needs to be consistent, since the feature is the
share of a wallet's earlier markets in the same category as the current one.
"""
from __future__ import annotations

import re

import duckdb
import pandas as pd

from polymarket.data import RAW_DIR, markets_path

CACHE = RAW_DIR / "market_categories.parquet"

RULES: list[tuple[str, str]] = [
    ("culture", r"nobel|oscar|grammy|emmy|golden globe|award|searched|year in search|"
                r"time person of the year|box office|album|billboard|spotify|eurovision"),
    ("sports", r"\bnba\b|\bnfl\b|\bmlb\b|\bnhl\b|\bufc\b|\bfifa\b|world cup|super bowl|"
               r"premier league|champions league|la liga|serie a|bundesliga|ligue 1|\bepl\b|"
               r"stanley cup|wimbledon|us open|french open|australian open|grand prix|\bf1\b|"
               r"formula 1|olympic|ncaa|march madness|cricket|boxing|tennis|golf|\bpga\b|"
               r"masters|nascar|esports|league of legends|\blol\b|dota|counter-strike|\bcs2\b|"
               r"valorant|\bvs\.? |o/u \d|spread|championship|college football|\bafc\b|\bnfc\b|"
               r"\bmvp\b|playoff|touchdown|\bgoal|heisman"),
    ("crypto", r"bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|\bxrp\b|doge|crypto|token|"
               r"airdrop|\bfdv\b|memecoin|coinbase|binance|microstrategy|stablecoin|"
               r"up or down|\baxiom\b|pump\.fun|hyperliquid|polymarket|megaeth|public sale"),
    ("geopolitics", r"\bwar\b|strike|military|invade|invasion|ceasefire|\biran|israel|russia|"
                    r"ukrain|gaza|hamas|hezbollah|taiwan|\bnato\b|missile|nuclear|venezuela|"
                    r"maduro|khamenei|putin|zelensk|north korea|houthi|troops|\bcoup\b|"
                    r"sanction|syria|yemen|attack|forces|netanyahu|xi jinping|hormuz|cartel"),
    ("economics", r"\bfed\b|federal reserve|interest rate|\bbps\b|\bcpi\b|inflation|\bgdp\b|"
                  r"unemployment|jobs report|payroll|recession|tariff|s&p|nasdaq|\bdow\b|"
                  r"stock|earnings|\bipo\b|treasury|crude oil|\boil\b|\blng\b|market cap|"
                  r"largest company"),
    ("politics", r"election|president|senate|congress|governor|mayor|primary|nominee|democrat|"
                 r"republican|\bgop\b|trump|biden|harris|vance|\bvote|parliament|"
                 r"prime minister|cabinet|impeach|approval rating|supreme court|shutdown|"
                 r"executive order|pardon|poll|deport|minister|chancellor|\bpope\b|senator|"
                 r"leader of|leader out|government|coalition|signed into law"),
    ("tech", r"\bai\b|openai|chatgpt|\bgpt|gemini|google|apple|tesla|spacex|nvidia|microsoft|"
             r"\bmeta\b|iphone|anthropic|claude|grok|\bxai\b|starship|launch|release"),
    ("culture", r"\bsay\b|tweet|\bpost\b|youtube|mrbeast|taylor swift|kanye|drake|movie|"
                r"film|song|celebrity|wedding|pregnan|divorce|netflix|tiktok|stranger things|"
                r"season \d"),
    ("weather", r"temperature|hurricane|earthquake|weather|climate|rain|snow|\bnasa\b|covid|"
                r"measles|pandemic"),
]
OTHER = "other"
_COMPILED = [(name, re.compile(pattern)) for name, pattern in RULES]


def categorize(text: str) -> str:
    text = text.lower()
    for name, pattern in _COMPILED:
        if pattern.search(text):
            return name
    return OTHER


def market_categories() -> pd.Series:
    """Category of every market in the dataset, indexed by market id (cached)."""
    if not CACHE.exists():
        markets = duckdb.sql(f"select id, question, event_title from '{markets_path()}'").df()
        markets["category"] = categorize_markets(markets)
        markets[["id", "category"]].to_parquet(CACHE, index=False)
    return pd.read_parquet(CACHE).set_index("id")["category"]


def categorize_markets(markets: pd.DataFrame) -> pd.Series:
    """Category per market from `question` and `event_title`, indexed like `markets`."""
    text = (markets["question"].fillna("") + " | " + markets["event_title"].fillna("")).str.lower()
    out = pd.Series(OTHER, index=markets.index)
    unset = pd.Series(True, index=markets.index)
    for name, pattern in _COMPILED:
        hit = unset & text.str.contains(pattern)
        out[hit] = name
        unset &= ~hit
    return out
