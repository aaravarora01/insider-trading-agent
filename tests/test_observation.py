import dataclasses

import numpy as np
import pandas as pd
import pytest

from market.flow import COPYBOT, INSIDER, RETAIL, TraderType, features, sample_wallet
from polymarket.observation import FEATURES, auc, copybot_wallets, fit, fit_profile, per_wallet


def sample(profile, n, seed):
    rng = np.random.default_rng(seed)
    trades = [sample_wallet(profile, 1, rng) for _ in range(n)]
    return pd.DataFrame(features(trades))[FEATURES]


@pytest.mark.parametrize("profile", [RETAIL, INSIDER, COPYBOT])
def test_fit_profile_recovers_the_generating_profile(profile):
    est = fit_profile(sample(profile, 20000, seed=1), np.ones(20000))
    for name in ("size_log_mean", "size_log_sd", "fresh_prob", "age_log_mean", "age_log_sd",
                 "longshot_prob"):
        assert getattr(est, name) == pytest.approx(getattr(profile, name), rel=0.05, abs=0.02)
    for a, b in (("category_a", "category_b"), ("win_a", "win_b")):
        mean = getattr(profile, a) / (getattr(profile, a) + getattr(profile, b))
        est_mean = getattr(est, a) / (getattr(est, a) + getattr(est, b))
        assert est_mean == pytest.approx(mean, abs=0.02)


def test_supervised_fit_with_weak_labels():
    # 9000 retail and 1000 insider orders; 200 insiders are labelled and the
    # rest are (wrongly) treated as retail.
    x = pd.concat([sample(RETAIL, 9000, 2), sample(INSIDER, 1000, 3)], ignore_index=True)
    labels = np.zeros((len(x), 3))
    labels[9000:9200, TraderType.INSIDER] = 1.0
    retail, insider, _ = fit(x, labels)
    assert insider.fresh_prob == pytest.approx(INSIDER.fresh_prob, abs=0.08)
    assert insider.size_log_mean == pytest.approx(INSIDER.size_log_mean, abs=0.2)
    # Unlabelled insiders pull retail a little toward them, but not far.
    assert retail.fresh_prob == pytest.approx(RETAIL.fresh_prob, abs=0.04)
    assert retail.size_log_mean == pytest.approx(RETAIL.size_log_mean, abs=0.15)


def test_per_wallet_weights():
    orders = pd.DataFrame({"wallet": ["a", "a", "a", "b", "c"], "market_id": ["1"] * 5})
    w = per_wallet(np.array([1.0, 1.0, 1.0, 0.9, 0.0]), orders)
    assert w.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.9, 0.0])


def test_auc():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_copybot_needs_a_repeated_leader():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    # 0xbot follows 0xlead within 20 s, four times; 0xone follows once.
    for i in range(4):
        rows += [("0xlead", t0 + pd.Timedelta(hours=i)),
                 ("0xbot", t0 + pd.Timedelta(hours=i, seconds=20))]
    rows += [("0xone", t0 + pd.Timedelta(hours=10)), ("0xlead", t0 + pd.Timedelta(hours=10, seconds=-5))]
    orders = pd.DataFrame(rows, columns=["wallet", "time"]).assign(market_id="1", side=1)
    bots, pairs = copybot_wallets(orders)
    assert bots == {"0xbot"}
