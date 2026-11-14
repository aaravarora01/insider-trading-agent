import numpy as np
import pytest

from agent.qmdp import QMDPSolver
from market.sim import Action, MarketParams, logit

PARAMS = MarketParams()


@pytest.fixture(scope="module")
def solver():
    return QMDPSolver(PARAMS)


def test_buys_the_known_winner_when_it_is_cheap(solver):
    q = solver.q_values(0, float(logit(0.3)), 0)
    assert np.argmax(q[1, 0]) == Action.BUY_YES_LARGE
    assert np.argmax(q[0, 0]) == Action.BUY_NO_LARGE


def test_flat_position_at_fair_price_is_worth_little(solver):
    # One step before resolution, with no edge and no position, no action
    # should be worth much.
    t = PARAMS.horizon - 1
    q = solver.q_values(t, float(logit(0.5)), 0)
    belief = np.array([[0.5, 0.0], [0.5, 0.0]])
    assert np.einsum("yi,yia->a", belief, q).max() < 0.1
