"""Simulator parameters fitted to Polymarket data.

`market/calibrated.json` is written by `python -m polymarket.calibrate` and
committed, so using it needs no data download. Its "params" section overrides
the toy defaults in `MarketParams`; anything it leaves out (lot sizes, position
limit, starting prices, wallet features other than size) keeps its default.
One step is one hour and one contract is `units.contract_size` real contracts.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from market.flow import COPYBOT, INSIDER, RETAIL, FlowParams
from market.sim import MarketParams

PATH = Path(__file__).with_name("calibrated.json")


def calibrated_params(path: Path = PATH) -> MarketParams:
    fitted = dict(json.loads(path.read_text())["params"])
    flow = dict(fitted.pop("flow"))
    profiles = flow.pop("profiles")
    flow["profiles"] = tuple(replace(default, **profiles[name]) for name, default in
                             (("retail", RETAIL), ("insider", INSIDER), ("copybot", COPYBOT)))
    return MarketParams(**fitted, flow=FlowParams(**flow))


def load_report(path: Path = PATH) -> dict:
    return json.loads(path.read_text())
