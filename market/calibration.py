"""Simulator parameters fitted to Polymarket data.

Two committed files, so using them needs no data download:

- `market/calibrated.json` (`python -m polymarket.calibrate`): price dynamics,
  impact, trade rates, and trade sizes.
- `market/wallet_profiles.json` (`python -m polymarket.observation`): the full
  wallet-feature profile of each trader type. Where present it replaces the
  size-only profiles from calibrated.json.

Anything neither file sets (lot sizes, position limit, starting prices) keeps
its default. One step is one hour and one contract is
`units.contract_size` real contracts.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from market.flow import COPYBOT, INSIDER, RETAIL, FlowParams, WalletProfile
from market.sim import MarketParams

PATH = Path(__file__).with_name("calibrated.json")
PROFILES_PATH = Path(__file__).with_name("wallet_profiles.json")
TYPES = (("retail", RETAIL), ("insider", INSIDER), ("copybot", COPYBOT))


def calibrated_params(path: Path = PATH, profiles_path: Path = PROFILES_PATH) -> MarketParams:
    fitted = dict(json.loads(path.read_text())["params"])
    flow = dict(fitted.pop("flow"))
    profiles = flow.pop("profiles")
    if profiles_path.exists():
        full = json.loads(profiles_path.read_text())["profiles"]
        flow["profiles"] = tuple(WalletProfile(**full[name]) for name, _ in TYPES)
    else:
        flow["profiles"] = tuple(replace(default, **profiles[name]) for name, default in TYPES)
    return MarketParams(**fitted, flow=FlowParams(**flow))


def load_report(path: Path = PATH) -> dict:
    return json.loads(path.read_text())
