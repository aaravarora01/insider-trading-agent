import numpy as np
import pandas as pd
import pytest

from market import sim
from market.flow import FlowParams
from market.sim import Action, MarketParams
from polymarket.calibrate import Market, price_dynamics

TRUE = MarketParams(horizon=200, onset_rate=0.0, news_mean=0.05, news_weight=0.10,
                    noise_sd=0.10, initial_price_range=(0.3, 0.7),
                    flow=FlowParams(retail_rate=0.0, copy_base_rate=0.0))


def simulated_market(seed: int) -> Market:
    rng = np.random.default_rng(seed)
    state, obs = sim.reset(TRUE, rng)
    prices = [obs.price]
    while not obs.done:
        state, obs, _ = sim.step(state, Action.HOLD, rng, TRUE)
        prices.append(obs.price)
    hours = pd.DataFrame({"p": prices}, index=pd.date_range("2025-01-01", periods=len(prices),
                                                            freq="1h", tz="UTC"))
    hours["x"] = np.log(hours["p"]) - np.log1p(-hours["p"])
    hours["dx"] = hours["x"].diff()
    hours["quiet"] = hours["p"].between(0.05, 0.95) & hours["p"].shift().between(0.05, 0.95)
    return Market(str(seed), "calibration", state.outcome, pd.DataFrame(), hours,
                  onset=None, reveal=None, reveal_from=None, end=hours.index[-1])


def test_price_dynamics_recovers_news_and_noise():
    est = price_dynamics([simulated_market(s) for s in range(400)])
    assert est["news_mean"] == pytest.approx(TRUE.news_mean, rel=0.15)
    assert est["news_weight"] == pytest.approx(TRUE.news_weight, rel=0.15)
    assert est["noise_sd"] == pytest.approx(TRUE.noise_sd, rel=0.15)
    assert abs(est["lag1_autocov"]) < 1e-3     # no bid-ask bounce in the simulator


def test_calibrated_params_load_and_run():
    from market.calibration import calibrated_params, load_report

    params = calibrated_params()
    report = load_report()
    assert params.horizon == report["params"]["horizon"]
    assert params.flow.retail_rate == pytest.approx(report["params"]["flow"]["retail_rate"])
    assert params.news_weight == pytest.approx(2 * params.news_mean)
    # Wallet profiles come from wallet_profiles.json when it exists.
    from market.calibration import PROFILES_PATH
    import json
    fitted = json.loads(PROFILES_PATH.read_text())["profiles"]["insider"]
    assert params.flow.profiles[1].age_log_mean == pytest.approx(fitted["age_log_mean"])
    assert params.flow.profiles[1].fresh_prob == pytest.approx(fitted["fresh_prob"])

    rng = np.random.default_rng(0)
    state, obs = sim.reset(params, rng)
    while not obs.done:
        state, obs, _ = sim.step(state, Action.HOLD, rng, params)
    assert obs.t == params.horizon
