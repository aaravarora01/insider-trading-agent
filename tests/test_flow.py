import dataclasses

import numpy as np
import pytest

from market import sim
from market.flow import (LargeTrade, TraderType, flow_loglik,
                         flow_rates, sample_flow, side_index, type_posterior)
from market.sim import Action, MarketParams

PARAMS = MarketParams()
FLOW = PARAMS.flow


def draw(outcome, insider, steps, seed=0):
    rng = np.random.default_rng(seed)
    trades, types = [], []
    for _ in range(steps):
        tr, ty = sample_flow(outcome, insider, FLOW, rng)
        trades.extend(tr)
        types.extend(ty)
    return trades, np.array(types)


def test_trader_type_is_not_observable():
    fields = {f.name for f in dataclasses.fields(LargeTrade)}
    assert not fields & {"type", "trader_type", "insider"}


@pytest.mark.parametrize("outcome", [0, 1])
def test_insider_trades_toward_the_outcome_only_while_active(outcome):
    trades, types = draw(outcome, insider=False, steps=500)
    assert TraderType.INSIDER not in types

    trades, types = draw(outcome, insider=True, steps=500)
    sides = np.array([t.side for t in trades])
    assert np.all(sides[types == TraderType.INSIDER] == 2 * outcome - 1)


def test_trade_counts_match_rates():
    steps = 4000
    for insider in (False, True):
        trades, types = draw(1, insider, steps, seed=int(insider))
        sides = np.array([side_index(t.side) for t in trades])
        expected = flow_rates(1, insider, FLOW) * steps
        for k in TraderType:
            for s in (0, 1):
                count = np.sum((types == k) & (sides == s))
                assert count == pytest.approx(expected[k, s], abs=4 * np.sqrt(expected[k, s]) + 1)


def test_type_posterior_is_calibrated():
    # If the wallet densities or rates were wrong, the posterior would not
    # match the empirical type frequencies.
    trades, types = draw(1, insider=True, steps=3000, seed=5)
    post = type_posterior(tuple(trades), 1, True, FLOW)
    for k in TraderType:
        truth = types == k
        assert post[:, k].mean() == pytest.approx(truth.mean(), abs=0.01)
        confident = post[:, k] > 0.7
        if confident.sum() > 50:
            assert truth[confident].mean() == pytest.approx(post[confident, k].mean(), abs=0.05)


def test_flow_carries_no_outcome_information_without_an_insider():
    rng = np.random.default_rng(2)
    trades, _ = sample_flow(1, True, FLOW, rng)
    ll = flow_loglik(trades, FLOW)
    assert ll[0, 0] == pytest.approx(ll[1, 0])
    assert np.all(np.isfinite(ll))


def test_empty_step_is_evidence_against_an_insider():
    ll = flow_loglik((), FLOW)
    assert ll[0, 0] > ll[0, 1]
    assert ll[0, 1] == pytest.approx(ll[1, 1])


def test_score_separates_insider_markets():
    scores = {True: [], False: []}
    rng = np.random.default_rng(9)
    for insider in (True, False):
        for _ in range(2000):
            trades, _ = sample_flow(int(rng.random() < 0.5), insider, FLOW, rng)
            ll = flow_loglik(trades, FLOW)
            scores[insider].append(np.logaddexp(ll[0, 1], ll[1, 1]) - np.log(2) - ll[0, 0])
    assert np.mean(scores[True]) > 0 > np.mean(scores[False])


def test_flow_does_not_depend_on_actions():
    def trades(action):
        rng = np.random.default_rng(11)
        state, obs = sim.reset(PARAMS, rng)
        out = []
        while not obs.done:
            state, obs, _ = sim.step(state, action, rng, PARAMS)
            out.append((obs.trades, state.flow_types, obs.score))
        return out

    assert trades(Action.HOLD) == trades(Action.BUY_NO_LARGE)


def test_flow_types_line_up_with_trades():
    rng = np.random.default_rng(4)
    state, obs = sim.reset(PARAMS, rng)
    assert obs.trades == () and state.flow_types == ()
    while not obs.done:
        state, obs, _ = sim.step(state, Action.HOLD, rng, PARAMS)
        assert len(obs.trades) == len(state.flow_types)
        if not state.insider:
            assert TraderType.INSIDER not in state.flow_types

