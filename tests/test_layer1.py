"""Offline checks for Layer 1.

No network needed: builds synthetic bars in exactly the shapes yfinance
returns / fetch_bars emits, then exercises bar normalization, the hybrid
merge, the trade generator, and the trades loader end to end.

Run:  python tests/test_layer1.py
"""

import datetime as dt
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bars as bars_mod
import fetch_and_gen
import trades_io

PASSED = 0


def ok(condition, label):
    global PASSED
    assert condition, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


# ---------------------------------------------------------- synthetic tapes


def fake_yf_download(symbol, day, interval_min):
    """A frame shaped like yf.download output: tz-aware index, MultiIndex cols."""
    idx = pd.date_range(f"{day} 08:00", f"{day} 17:55",  # incl. pre/post market
                        freq=f"{interval_min}min").tz_localize("US/Eastern")
    n = len(idx)
    rng = np.random.default_rng(abs(hash((symbol, str(day), interval_min))) % 2**32)
    close = 100 + np.cumsum(rng.normal(0.01, 0.12, n))
    opn = np.r_[close[0], close[:-1]]
    high = np.maximum(opn, close) + rng.uniform(0.01, 0.2, n)
    low = np.minimum(opn, close) - rng.uniform(0.01, 0.2, n)
    df = pd.DataFrame(
        {"Adj Close": close, "Close": close, "High": high, "Low": low,
         "Open": opn, "Volume": rng.integers(1000, 90000, n).astype(float)},
        index=idx,
    )
    df.columns = pd.MultiIndex.from_product([df.columns, [symbol]], names=["Price", "Ticker"])
    return df


def synthetic_merged_bars(symbols, n_sessions=60, seed=11):
    """Bars in fetch_bars' output format: ~60 sessions of 5m, last 7 with 1m."""
    days = pd.bdate_range(end=pd.Timestamp("2026-07-31"), periods=n_sessions)
    frames = []
    for sym_i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + sym_i)
        px = 80.0 + 40 * sym_i
        for d_i, day in enumerate(days):
            interval = 1 if d_i >= n_sessions - 7 else 5
            idx = pd.date_range(f"{day.date()} 09:30", f"{day.date()} 15:59",
                                freq=f"{interval}min")
            idx = idx[idx.time < dt.time(16, 0)]
            n = len(idx)
            drift = rng.normal(0.002, 0.03)  # sessions trend a little, both ways
            close = px + np.cumsum(rng.normal(drift, 0.10, n))
            opn = np.r_[close[0] + rng.normal(0, 0.05), close[:-1]]
            high = np.maximum(opn, close) + rng.uniform(0.01, 0.15, n)
            low = np.minimum(opn, close) - rng.uniform(0.01, 0.15, n)
            frames.append(pd.DataFrame({
                "instrument": sym, "datetime": idx, "open": opn, "high": high,
                "low": low, "close": close,
                "volume": rng.integers(1000, 60000, n),
            }))
            px = close[-1] + rng.normal(0, 0.3)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------- tests


def test_normalize_and_merge():
    print("bars: normalization + hybrid merge")
    day = "2026-07-30"
    raw5 = fake_yf_download("TEST", day, 5)
    raw1 = fake_yf_download("TEST", day, 1)
    n5 = bars_mod._normalize(raw5, "TEST", "5m")
    n1 = bars_mod._normalize(raw1, "TEST", "1m")

    ok(list(n5.columns) == bars_mod.BARS_COLUMNS + ["_interval"], "normalized columns")
    times = n5["datetime"].dt.time
    ok(times.min() >= dt.time(9, 30) and times.max() < dt.time(16, 0),
       "regular session only (09:30–16:00)")
    ok(n5["datetime"].dt.tz is None, "naive Eastern timestamps")
    ok(len(n5) == 78, "78 five-minute session bars")

    merged = bars_mod._merge([n5, n1])
    ok(list(merged.columns) == bars_mod.BARS_COLUMNS, "merged column set")
    ok(len(merged) == 390, "1m bars win: one bar per session minute")
    at_930 = merged[merged["datetime"] == pd.Timestamp(f"{day} 09:30")]
    want = n1[n1["datetime"] == pd.Timestamp(f"{day} 09:30")]["close"].iloc[0]
    ok(float(at_930["close"].iloc[0]) == float(want), "collision kept the finer bar")
    ok(merged["datetime"].is_monotonic_increasing, "sorted by datetime")
    ok(not merged.duplicated(["instrument", "datetime"]).any(), "no duplicate bars")

    empty, warnings = bars_mod._merge([]), None
    ok(empty.empty and list(empty.columns) == bars_mod.BARS_COLUMNS, "empty merge is safe")


def test_symbol_mapping():
    print("bars: symbol mapping")
    ok(bars_mod.to_yahoo_symbol("ES 03-26") == "ES=F", "NinjaTrader futures -> ES=F")
    ok(bars_mod.to_yahoo_symbol("MES 06-25") == "MES=F", "micro futures -> MES=F")
    ok(bars_mod.to_yahoo_symbol("SPY") == "SPY", "equities pass through")


def test_generator_and_roundtrip(tmpdir):
    print("generator: winning momentum trader from the bars")
    bars = synthetic_merged_bars(["SPY", "QQQ", "NVDA", "TSLA"], n_sessions=60)
    trades = fetch_and_gen.generate_trades(bars, seed=7)

    ok(len(trades) >= 300, f"300+ trades (got {len(trades)})")
    per_day = trades.groupby(trades["Entry time"].dt.normalize()).size()
    ok(len(per_day) >= 55, f"trades across ~60 sessions (got {len(per_day)})")
    ok(per_day.max() <= 10 and per_day.mean() >= 4,
       f"believable daily cadence (avg {per_day.mean():.1f}, max {per_day.max()})")
    net = float(trades["Profit"].sum() - trades["Commission"].sum())
    ok(net > 0, f"net winner (${net:,.2f})")
    longs = (trades["Market pos."] == "Long").mean()
    ok(longs >= 0.75, f"long-biased ({longs:.0%} long)")

    # baked-in leaks
    ok((trades["Exit name"] == "Manual").sum() >= 10, "some 'Manual' exits")
    late = (trades["Entry time"].dt.time >= dt.time(15, 30)).sum()
    ok(late >= 1, f"a few late-session entries ({late})")
    qty, prev_p = trades["Qty"].to_numpy(), trades["Profit"].to_numpy()
    jumps = sum(1 for i in range(1, len(qty))
                if prev_p[i - 1] < 0 and qty[i] >= 2 * np.median(qty))
    ok(jumps >= 5, f"oversizing after losses ({jumps} occurrences)")

    # every price/timestamp is a real bar
    ok(fetch_and_gen.verify_trades(trades, bars) == [],
       "every entry is a bar open, every exit a bar close")

    print("trades_io: NinjaTrader round trip")
    path = os.path.join(tmpdir, "trades.csv")
    fetch_and_gen.write_trades(trades, path)
    loaded = trades_io.load_trades(path)
    ok(len(loaded) == len(trades), "round trip keeps all trades")
    ok(list(loaded.columns) == trades_io.COLUMN_ORDER, "exact NinjaTrader columns")
    ok(loaded["Entry time"].dtype.kind == "M", "timestamps parsed")
    ok(fetch_and_gen.verify_trades(loaded, bars) == [], "verifies after reload too")


def test_trades_io_edges(tmpdir):
    print("trades_io: edge cases")
    good = os.path.join(tmpdir, "min.csv")
    pd.DataFrame({
        "Instrument": ["SPY", "SPY"], "Market pos.": ["Long", "short"],
        "Qty": [100, 200], "Entry price": ["$450.10", "451.00"],
        "Exit price": ["$451.20", "($449.85)"],
        "Entry time": ["2026-07-30 09:35:00", "7/30/2026 10:15 AM"],
        "Exit time": ["2026-07-30 09:55:00", "7/30/2026 10:40 AM"],
        "Profit": ["$110.00", "($230.00)"],
    }).to_csv(good, index=False)
    t = trades_io.load_trades(good)
    ok(float(t["Profit"].iloc[1]) == -230.0, "currency + parentheses negatives parsed")
    ok(float(t["Exit price"].iloc[1]) == -449.85, "parenthesized price parsed")
    ok(t["Market pos."].tolist() == ["Long", "Short"], "position normalized")
    ok(t["Trade number"].tolist() == [1, 2], "Trade number backfilled")
    ok(t["Account"].iloc[0] == "Sim101" and t["Commission"].iloc[0] == 0.0,
       "optional columns backfilled")
    ok(abs(float(t["Cum. net profit"].iloc[1]) - (-120.0)) < 1e-9,
       "Cum. net profit backfilled as running sum")

    bad = os.path.join(tmpdir, "bad.csv")
    pd.DataFrame({"Instrument": ["SPY"], "Qty": [1]}).to_csv(bad, index=False)
    try:
        trades_io.load_trades(bad)
        ok(False, "missing columns should raise")
    except trades_io.TradesFileError as e:
        msg = str(e)
        ok("missing required column" in msg and "\n" not in msg,
           "bad file -> one clear error line")

    try:
        trades_io.load_trades(os.path.join(tmpdir, "nope.csv"))
        ok(False, "unreadable file should raise")
    except trades_io.TradesFileError as e:
        ok("Could not read" in str(e), "unreadable file -> one clear error line")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_normalize_and_merge()
        test_symbol_mapping()
        test_generator_and_roundtrip(tmpdir)
        test_trades_io_edges(tmpdir)
    print(f"\nall {PASSED} checks passed")


if __name__ == "__main__":
    main()
