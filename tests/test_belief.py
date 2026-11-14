import numpy as np

from agent.belief import BeliefFilter
from market import sim
from market.sim import Action, MarketParams

PARAMS = MarketParams()


def simulate(filter_, markets, seed=1):
    rng = np.random.default_rng(seed)
    prob_insider, insider, prob_yes, outcome = [], [], [], []
    for _ in range(markets):
        state, obs = sim.reset(PARAMS, rng)
        filter_.reset(obs)
        while not obs.done:
            state, obs, _ = sim.step(state, Action.HOLD, rng, PARAMS)
            filter_.update(obs)
            prob_insider.append(filter_.prob_insider)
            insider.append(state.insider)
        prob_yes.append(filter_.prob_yes)
        outcome.append(state.outcome)
    return (np.array(prob_insider), np.array(insider),
            np.array(prob_yes), np.array(outcome))


def test_insider_belief_is_calibrated():
    prob, truth, _, _ = simulate(BeliefFilter(PARAMS), markets=300)
    assert abs(prob.mean() - truth.mean()) < 0.03
    confident = prob > 0.8
    assert truth[confident].mean() > 0.9


def test_outcome_belief_is_calibrated():
    _, _, prob_yes, outcome = simulate(BeliefFilter(PARAMS), markets=300)
    assert abs(prob_yes.mean() - outcome.mean()) < 0.05


def test_insider_blind_filter_never_suspects_an_insider():
    prob, _, _, _ = simulate(BeliefFilter(PARAMS, model_insiders=False), markets=20)
    assert np.all(prob == 0.0)
