import numpy as np
import pandas as pd
import pytest

from polymarket.data import _outcome, hourly_prices, row_groups_for, to_yes_perspective


def raw_fills():
    return pd.DataFrame({
        "timestamp": [1_750_000_000, 1_750_000_060, 1_750_003_700, 1_750_003_800],
        "block_number": [10, 11, 12, 13],
        "market_id": ["1"] * 4,
        "maker": ["0xm"] * 4,
        "taker": ["0xt"] * 4,
        "price": [0.30, 0.70, 0.40, 0.65],
        "usd_amount": [30.0, 70.0, 40.0, 65.0],
        "token_amount": [100.0] * 4,
        "taker_direction": ["BUY", "BUY", "SELL", "SELL"],
        "nonusdc_side": ["token1", "token2", "token1", "token2"],
    })


def test_yes_perspective():
    df = to_yes_perspective(raw_fills())
    # Buy YES, buy NO, sell YES, sell NO.
    assert df["taker_side"].tolist() == [1, -1, -1, 1]
    assert df["yes_price"].tolist() == pytest.approx([0.30, 0.30, 0.40, 0.35])
    assert df["token"].tolist() == ["YES", "NO", "YES", "NO"]
    assert str(df["time"].dt.tz) == "UTC"
    # Each party's stake on its own side: 100 contracts at 0.30 or 0.70.
    assert df["taker_usd"].tolist() == pytest.approx([30.0, 70.0, 60.0, 35.0])
    assert (df["taker_usd"] + df["maker_usd"]).tolist() == pytest.approx([100.0] * 4)


def test_hourly_prices():
    prices = hourly_prices(to_yes_perspective(raw_fills()))
    assert len(prices) == 2
    first, second = prices.iloc[0], prices.iloc[1]
    assert first["last"] == pytest.approx(0.30)
    assert first["fills"] == 2 and first["contracts"] == pytest.approx(200.0)
    assert first["usd_volume"] == pytest.approx(200.0)
    assert first["vwap"] == pytest.approx(0.30)
    assert second["vwap"] == pytest.approx((0.40 + 0.35) / 2)


@pytest.mark.parametrize("prices, expected", [
    ("['1', '0']", 1.0), ("['0', '1']", 0.0), ("['0.9985', '0.0015']", 1.0),
    ("['0.5', '0.5']", np.nan), (None, np.nan), ("garbage", np.nan),
])
def test_outcome(prices, expected):
    assert _outcome(prices) == pytest.approx(expected, nan_ok=True)


def test_row_groups_cover_the_range():
    starts = np.array([0, 100, 200, 300])  # three groups of 100 rows
    assert list(row_groups_for(0, 99, starts)) == [0]
    assert list(row_groups_for(99, 100, starts)) == [0, 1]
    assert list(row_groups_for(150, 299, starts)) == [1, 2]


def test_flagged_wallets_file_is_well_formed():
    from polymarket.flagged import LABELS, load_flagged
    from polymarket.selection import MARKETS

    df = load_flagged()
    assert df["address"].str.fullmatch(r"0x[0-9a-f]{40}").all()
    assert df["address"].is_unique
    assert set(df["label"]) <= set(LABELS)
    assert set(df["address_from"]) <= {"published", "username_lookup", "prefix_match"}
    assert df["sources"].str.startswith("https://").all()
    selected = {m["id"] for m in MARKETS}
    in_selection = df["market_ids"].map(lambda ids: bool(set(ids) & selected))
    assert in_selection[df["label"] != "weak"].all()


def test_wallet_activity_counts_both_roles():
    from polymarket.flagged import wallet_activity

    trades = to_yes_perspective(raw_fills().assign(taker=["0xa", "0xb", "0xb", "0xb"],
                                                   maker=["0xb", "0xa", "0xc", "0xc"]))
    act = wallet_activity(trades, "0xa")
    # Taker buying YES at 0.30, then maker on the other side of a NO buy: long YES at 0.30.
    assert act["fills"] == 2
    assert act["net_yes_contracts"] == pytest.approx(200.0)
    assert act["usd_long_yes"] == pytest.approx(60.0)
    assert act["usd_long_no"] == 0.0
    assert act["taker_share"] == pytest.approx(0.5)
