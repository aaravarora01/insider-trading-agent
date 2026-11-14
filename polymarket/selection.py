"""The markets we pull. `role` is "calibration" (ordinary markets for fitting the
simulator) or "incident" (markets with publicly reported suspected insider
trading, for replay). Incidents are exploratory: a surprising outcome is not
proof of insider trading.

IDs are `markets.parquet` ids (= `trades.market_id`). `reveal_utc` is when the
outcome was announced, where news coverage documents it. Calibration takes the
earlier of this and the first hour the price reaches 0.95.
"""

MARKETS = [
    # Calibration: long-running markets in different categories.
    {"id": "538932", "role": "calibration",
     "note": "Election: Mamdani wins 2025 NYC mayoral race (YES). $143M volume, Apr-Nov 2025."},
    {"id": "570361", "role": "calibration",
     "note": "Macro: Fed cuts 25 bps at the Dec 2025 meeting (YES). $55M, Jul-Dec 2025."},
    {"id": "1269423", "role": "calibration",
     "note": "Sports: Super Bowl LX, Seahawks vs. Patriots (answer1 Seahawks won). $36M, 2 weeks."},

    # Incidents.
    {"id": "916440", "role": "incident",
     "note": "Maduro out by Jan 31, 2026 (YES). US operation on Jan 3, 2026; widely reported "
             "large bets placed hours before.",
     "reveal_utc": "2026-01-03T06:00Z",
     "reveal_note": "First explosions in Caracas reported around 02:00 local time."},
    {"id": "560868", "role": "incident",
     "note": "Machado wins the 2025 Nobel Peace Prize (YES). Price jumped the night before the "
             "announcement; the Nobel Institute investigated a possible leak.",
     "reveal_utc": "2025-10-10T09:00Z", "reveal_note": "Announced at 11:00 CEST."},
    {"id": "532742", "role": "incident",
     "note": "Israel military action against Iran before July 2025 (YES). ACDC's 'Orca 0xc0a' "
             "bought in the days before the June 13 strikes.",
     "reveal_utc": "2025-06-12T23:30Z",
     "reveal_note": "Strikes began around 03:00 Tehran time on June 13."},
    {"id": "554010", "role": "incident",
     "note": "US military action against Iran by Saturday, June 21, 2025 (YES). Fresh wallets "
             "bet hours before the strikes on Fordow.",
     "reveal_utc": "2025-06-21T23:50Z",
     "reveal_note": "Trump announced the strikes at about 19:50 ET."},
    {"id": "635246", "role": "incident",
     "note": "d4vd is Google's #1 searched person of 2025 (YES). AlphaRaccoon won 22 of 23 Google "
             "markets; DOJ and CFTC filed charges in May 2026."},
    {"id": "1198423", "role": "incident",
     "note": "US strikes Iran by Feb 28, 2026 (YES). Bubblemaps flagged a cluster of fresh "
             "wallets that netted ~$1M hours before the strikes."},
    {"id": "1419011", "role": "incident",
     "note": "Axiom accused of insider trading (YES). A new wallet bet $66k two days before "
             "ZachXBT's exposé; he traced its funds to an Axiom user."},
    {"id": "1706788", "role": "incident",
     "note": "US x Iran ceasefire by April 7, 2026 (YES). New wallets bought hours before the "
             "ceasefire announcement.",
     "reveal_utc": "2026-04-07T22:32Z", "reveal_note": "Per Cointelegraph, Apr 8, 2026."},
]
