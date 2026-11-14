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


def test_buy_and_hold_spends_a_fraction_of_the_budget():
    from agent.policies import BuyAndHold
    params = MarketParams(budget=10.0, max_position=100)
    policy = BuyAndHold(params, fraction=0.5)
    obs = sim.Observation(t=0, price=0.7, news=0.0, score=0.0, executed=0, done=False)
    policy.reset(obs)
    assert policy.act() == sim.Order(7)        # 5 / 0.70 -> 7 YES contracts
    assert policy.act() == Action.HOLD
    policy.reset(sim.Observation(t=0, price=0.2, news=0.0, score=0.0, executed=0, done=False))
    assert policy.act() == sim.Order(-6)       # 5 / 0.80 -> 6 NO contracts
