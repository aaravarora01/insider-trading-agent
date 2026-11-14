import numpy as np

from agent.belief import BeliefFilter
from agent.particle import ParticleFilter
from market import sim
from market.sim import Action, MarketParams

PARAMS = MarketParams()


def run_both(particles, markets, seed=3):
    """Max gap between the particle filter and the exact filter over every step."""
    rng = np.random.default_rng(seed)
    gaps_insider, gaps_yes = [], []
    for k in range(markets):
        exact, pf = BeliefFilter(PARAMS), ParticleFilter(PARAMS, particles, seed=k)
        state, obs = sim.reset(PARAMS, rng)
        exact.reset(obs), pf.reset(obs)
        while not obs.done:
            state, obs, _ = sim.step(state, Action.HOLD, rng, PARAMS)
            exact.update(obs), pf.update(obs)
            gaps_insider.append(abs(exact.prob_insider - pf.prob_insider))
            gaps_yes.append(abs(exact.prob_yes - pf.prob_yes))
    return np.array(gaps_insider), np.array(gaps_yes)


def test_particle_filter_matches_exact_filter():
    insider, yes = run_both(particles=4000, markets=20)
    assert insider.mean() < 0.02 and yes.mean() < 0.02
    assert np.quantile(insider, 0.99) < 0.12


def test_more_particles_track_the_exact_filter_more_closely():
    few, _ = run_both(particles=200, markets=10)
    many, _ = run_both(particles=4000, markets=10)
    assert many.mean() < few.mean()


def test_belief_is_a_distribution():
    pf = ParticleFilter(PARAMS, 500)
    rng = np.random.default_rng(0)
    state, obs = sim.reset(PARAMS, rng)
    pf.reset(obs)
    while not obs.done:
        state, obs, _ = sim.step(state, Action.HOLD, rng, PARAMS)
        b = pf.update(obs)
        assert b.shape == (2, 2) and np.all(b >= 0) and np.isclose(b.sum(), 1.0)


def test_resampling_keeps_both_outcomes():
    pf = ParticleFilter(PARAMS, 1000, prob_floor=0.01)
    pf.reset(sim.Observation(t=0, price=0.5, news=0.0, score=0.0, executed=0, done=False))
    # Evidence strongly for YES: every NO particle is down-weighted.
    pf.logw[pf.y == 0] -= 12.0
    mass_no = pf.weights[pf.y == 0].sum()
    pf._resample()
    assert (pf.y == 0).sum() >= 10
    assert np.isclose(pf.weights[pf.y == 0].sum(), mass_no)


def test_insider_blind_particle_filter():
    pf = ParticleFilter(PARAMS, 300, model_insiders=False)
    rng = np.random.default_rng(0)
    state, obs = sim.reset(PARAMS, rng)
    pf.reset(obs)
    while not obs.done:
        state, obs, _ = sim.step(state, Action.HOLD, rng, PARAMS)
        pf.update(obs)
        assert pf.prob_insider == 0.0
