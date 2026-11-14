import numpy as np
import pytest

from market import sim
from market.sim import Action, MarketParams, executed_quantity, trade_cash

PARAMS = MarketParams()


def test_buying_no_matches_selling_yes():
    # Buy NO at YES price 0.6 (cost 0.4), sell it back at YES price 0.5 (receive 0.5).
    pnl = trade_cash(-1, 0.6, PARAMS) + trade_cash(+1, 0.5, PARAMS)
    assert pnl == pytest.approx(-0.4 + 0.5)


def test_position_limit_and_exit():
    limit = PARAMS.max_position
    assert executed_quantity(limit - 1, Action.BUY_YES_LARGE, PARAMS) == 1
    assert executed_quantity(-limit, Action.BUY_NO_SMALL, PARAMS) == 0
    assert executed_quantity(-4, Action.EXIT, PARAMS) == 4


def test_round_trip_pays_impact():
    rng = np.random.default_rng(0)
    state, _ = sim.reset(PARAMS, rng)
    bought = sim.fill_price(state.log_odds, 3, PARAMS)
    sold = sim.fill_price(state.log_odds + 3 * PARAMS.impact, -3, PARAMS)
    assert trade_cash(3, bought, PARAMS) + trade_cash(-3, sold, PARAMS) < 0


def test_market_path_does_not_depend_on_actions():
    def path(action):
        rng = np.random.default_rng(7)
        state, obs = sim.reset(PARAMS, rng)
        news = []
        while not obs.done:
            state, obs, _ = sim.step(state, action, rng, PARAMS)
            news.append(obs.news)
        return news, state.onset

    assert path(Action.HOLD) == path(Action.BUY_YES_LARGE)


def test_settlement_pays_position_times_outcome():
    rng = np.random.default_rng(3)
    state, obs = sim.reset(PARAMS, rng)
    while not obs.done:
        action = Action.BUY_NO_LARGE if obs.t == 0 else Action.HOLD
        state, obs, reward = sim.step(state, action, rng, PARAMS)
    assert state.position == -PARAMS.large_lot
    assert reward == pytest.approx(state.position * state.outcome)


def test_order_trades_exact_quantity():
    params = MarketParams()
    assert executed_quantity(0, sim.Order(4), params) == 4
    assert executed_quantity(0, sim.Order(-2), params) == -2
    assert executed_quantity(4, sim.Order(5), params) == params.max_position - 4


def test_trades_are_shrunk_to_stay_solvent():
    params = MarketParams(max_position=100, budget=2.0, impact=0.0)
    x = float(sim.logit(0.5))
    # Two dollars buys four contracts at 0.50, on either side.
    assert executed_quantity(0, sim.Order(10), params, cash=2.0, log_odds=x) == 4
    assert executed_quantity(0, sim.Order(-10), params, cash=2.0, log_odds=x) == -4
    # Selling back a long position is always allowed.
    assert executed_quantity(4, Action.EXIT, params, cash=0.0, log_odds=x) == -4
    # Broke and flat: nothing trades.
    assert executed_quantity(0, Action.BUY_YES_LARGE, params, cash=0.0, log_odds=x) == 0


def test_cash_is_tracked_and_never_goes_negative():
    params = MarketParams(budget=1.0)
    rng = np.random.default_rng(0)
    state, obs = sim.reset(params, rng)
    total = 0.0
    while not obs.done:
        state, obs, reward = sim.step(state, Action.BUY_YES_LARGE, rng, params)
        total += reward
        assert obs.cash >= -1e-9 and obs.cash + state.position >= -1e-9
    assert np.isclose(obs.cash, params.budget + total)
