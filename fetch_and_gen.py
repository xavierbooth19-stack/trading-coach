"""Layer 1 — fetch real bars, then generate a believable trade log from them.

Running this script:

  1. fetches hybrid 5m/1m bars for the demo symbols (bars.csv),
  2. generates a WINNING long-biased momentum day-trader from those bars
     (trades.csv, exact NinjaTrader column format),
  3. reloads both files and verifies every entry price is a real bar's
     open and every exit a real bar's close, at their real timestamps.

Nothing about the tape is invented: entries are taken at real bar opens
in the direction of the short-term trend, and only WHICH real bar a trade
exits at is chosen (target / stop / time / a few 'Manual' overrides).
The log also bakes in realistic leaks for the upper layers to find:
occasional oversizing right after a loss, a few late-session entries,
and some 'Manual' exits.
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

import bars as bars_mod
import trades_io

DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "TSLA"]

LATE_SESSION = dt.time(15, 30)  # entries at/after this are a "leak"

# exit / sizing tuning (per-share terms are multiples of a rolling bar range)
TARGET_MULT = 2.2
STOP_MULT = 1.1
MAX_HOLD_BARS = 24
MIN_BARS_BEFORE_ENTRY = 16
RISK_CHOICES = (150.0, 200.0, 250.0)
SHORT_KEEP_PROB = 0.15          # long-biased: keep only ~15% of short signals
LATE_KEEP_PROB = 0.10           # a signal after 15:30 occasionally gets taken
LATE_EXTRA_PROB = 0.04          # ...and some days one extra late trade is forced
MANUAL_EXIT_PROB = 0.10         # sometimes bail early by hand
REVENGE_PROB = 0.35             # oversize right after a losing trade
COMMISSION_PER_SHARE_RT = 0.01  # round-trip


def generate_trades(bars: pd.DataFrame, seed: int = 7, min_trades: int = 300) -> pd.DataFrame:
    """Build a NinjaTrader-style trade log from real bars.

    Every entry is a real bar's open and every exit a real bar's close,
    at those bars' real timestamps. Returns the finished frame in
    trades_io.COLUMN_ORDER.
    """
    rng = np.random.default_rng(seed)
    legs = []

    bars = bars.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    sessions = 0
    for (instrument, _day), g in bars.groupby(
        [bars["instrument"], bars["datetime"].dt.normalize()], sort=True
    ):
        sessions += 1
        legs.extend(_session_trades(instrument, g.reset_index(drop=True), rng))

    if not legs:
        raise RuntimeError("No trades could be generated — bars frame too thin?")

    # If the tape gave us too few signals, take a second, looser pass.
    attempt = 1
    while len(legs) < min_trades and attempt < 4:
        attempt += 1
        extra_rng = np.random.default_rng(seed + attempt)
        for (instrument, _day), g in bars.groupby(
            [bars["instrument"], bars["datetime"].dt.normalize()], sort=True
        ):
            legs.extend(
                _session_trades(instrument, g.reset_index(drop=True), extra_rng,
                                loosen=attempt)
            )
        legs = _dedupe_legs(legs)

    legs.sort(key=lambda t: t["entry_time"])
    capped = _cap_per_day(legs, rng)
    if len(capped) >= min_trades:
        legs = capped
    trades = _size_and_price(legs, rng)
    trades = _ensure_winning(trades, legs)
    return _to_ninjatrader(trades)


def write_trades(trades: pd.DataFrame, path: str = "trades.csv") -> None:
    out = trades.copy()
    for col in ("Entry time", "Exit time"):
        out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def verify_trades(trades: pd.DataFrame, bars: pd.DataFrame) -> list[str]:
    """Check every entry is a real bar open and every exit a real bar close."""
    opens = {(r.instrument, r.datetime): r.open for r in bars.itertuples()}
    closes = {(r.instrument, r.datetime): r.close for r in bars.itertuples()}
    problems = []
    for t in trades.to_dict("records"):
        o = opens.get((t["Instrument"], t["Entry time"]))
        c = closes.get((t["Instrument"], t["Exit time"]))
        if o is None or round(float(o), 4) != round(float(t["Entry price"]), 4):
            problems.append(f"trade {t['Trade number']}: entry {t['Entry time']} is not a real bar open")
        if c is None or round(float(c), 4) != round(float(t["Exit price"]), 4):
            problems.append(f"trade {t['Trade number']}: exit {t['Exit time']} is not a real bar close")
    return problems


# ------------------------------------------------------------ trade building


def _session_trades(instrument, g, rng, loosen: int = 0):
    """Pick momentum entries in one symbol-session and simulate their exits."""
    n = len(g)
    if n < MIN_BARS_BEFORE_ENTRY + 6:
        return []

    close = g["close"].to_numpy(float)
    high = g["high"].to_numpy(float)
    low = g["low"].to_numpy(float)
    opn = g["open"].to_numpy(float)
    when = g["datetime"].to_numpy()

    ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().to_numpy()
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().to_numpy()
    rng14 = pd.Series(high - low).rolling(14, min_periods=5).mean().to_numpy()

    per_session_cap = int(rng.integers(1, 3)) + (1 if loosen else 0)
    allow_shorts = rng.random() < SHORT_KEEP_PROB  # long-biased trader
    taken = 0
    busy_until = MIN_BARS_BEFORE_ENTRY
    trades = []

    i = MIN_BARS_BEFORE_ENTRY
    while i < n - 4 and taken < per_session_cap:
        if i < busy_until:
            i += 1
            continue
        trend_up = ema9[i] > ema21[i] and close[i] > ema9[i] and close[i] > close[i - 3]
        trend_dn = ema9[i] < ema21[i] and close[i] < ema9[i] and close[i] < close[i - 3]
        direction = 1 if trend_up else (-1 if trend_dn else 0)
        if direction == 0 or not np.isfinite(rng14[i]) or rng14[i] <= 0:
            i += 1
            continue
        if direction == -1 and not allow_shorts:
            i += 1
            continue
        entry_idx = i + 1  # act on the signal at the next bar's open
        entry_time = pd.Timestamp(when[entry_idx])
        late = entry_time.time() >= LATE_SESSION
        if late and rng.random() > LATE_KEEP_PROB:
            i += 1
            continue
        # skip some valid signals so entries look human, not mechanical
        if rng.random() < (0.45 if not loosen else 0.25):
            i += 1
            continue

        leg = _simulate_exit(entry_idx, direction, opn, close, high, low, when, rng14[i], rng)
        leg["instrument"] = instrument
        leg["late_entry"] = late
        trades.append(leg)
        taken += 1
        busy_until = leg["exit_idx"] + int(rng.integers(2, 8))
        i = busy_until

    # leak: some days the trader can't resist one more trade near the close
    if trades and rng.random() < LATE_EXTRA_PROB:
        last_exit = trades[-1]["exit_idx"]
        for i in range(max(MIN_BARS_BEFORE_ENTRY, last_exit + 2), n - 3):
            entry_time = pd.Timestamp(when[i + 1])
            if entry_time.time() < LATE_SESSION:
                continue
            if not np.isfinite(rng14[i]) or rng14[i] <= 0:
                continue
            direction = 1 if ema9[i] > ema21[i] else (-1 if allow_shorts else 1)
            leg = _simulate_exit(i + 1, direction, opn, close, high, low, when, rng14[i], rng)
            leg["instrument"] = instrument
            leg["late_entry"] = True
            trades.append(leg)
            break

    return trades


def _simulate_exit(entry_idx, direction, opn, close, high, low, when, bar_range, rng):
    """Walk forward from the entry bar and pick the exit bar (a real close)."""
    n = len(close)
    entry_px = float(opn[entry_idx])
    target = TARGET_MULT * bar_range
    stop = STOP_MULT * bar_range

    exit_idx, exit_name = None, "Session close"
    for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, n - 1) + 1):
        move = (float(close[j]) - entry_px) * direction
        if move >= target:
            exit_idx, exit_name = j, "Profit target"
            break
        if move <= -stop:
            exit_idx, exit_name = j, "Stop loss"
            break
    if exit_idx is None:
        exit_idx = min(entry_idx + MAX_HOLD_BARS, n - 1)
        exit_name = "Manual" if exit_idx < n - 1 else "Session close"

    # leak: sometimes the trader bails early by hand instead
    if exit_idx > entry_idx + 2 and rng.random() < MANUAL_EXIT_PROB:
        exit_idx = int(rng.integers(entry_idx + 1, exit_idx))
        exit_name = "Manual"

    exit_px = float(close[exit_idx])
    lo = float(low[entry_idx : exit_idx + 1].min())
    hi = float(high[entry_idx : exit_idx + 1].max())
    if direction == 1:
        mae_ps, mfe_ps = max(0.0, entry_px - lo), max(0.0, hi - entry_px)
    else:
        mae_ps, mfe_ps = max(0.0, hi - entry_px), max(0.0, entry_px - lo)

    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "direction": direction,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "entry_time": pd.Timestamp(when[entry_idx]),
        "exit_time": pd.Timestamp(when[exit_idx]),
        "exit_name": exit_name,
        "mae_ps": mae_ps,
        "mfe_ps": mfe_ps,
        "stop_dist": stop,
        "bars_held": exit_idx - entry_idx,
        # kept so exits can be re-picked later without refetching arrays
        "_closes": close[entry_idx : min(entry_idx + MAX_HOLD_BARS, n - 1) + 1].copy(),
        "_when": when[entry_idx : min(entry_idx + MAX_HOLD_BARS, n - 1) + 1].copy(),
        "_high": high[entry_idx : min(entry_idx + MAX_HOLD_BARS, n - 1) + 1].copy(),
        "_low": low[entry_idx : min(entry_idx + MAX_HOLD_BARS, n - 1) + 1].copy(),
    }


def _cap_per_day(legs, rng):
    """Keep a believable daily cadence (a handful of trades per session)."""
    by_day: dict = {}
    for leg in legs:
        by_day.setdefault(leg["entry_time"].normalize(), []).append(leg)
    out = []
    for day in sorted(by_day):
        group = by_day[day]
        quota = int(rng.integers(4, 8))
        if len(group) > quota:
            # late entries are a deliberate leak — never cap them away
            late = [g for g in group if g["late_entry"]]
            rest = [g for g in group if not g["late_entry"]]
            keep = late + [rest[k] for k in
                           sorted(rng.choice(len(rest), max(0, quota - len(late)),
                                             replace=False))]
            group = sorted(keep, key=lambda t: t["entry_time"])
        out.extend(group)
    return out


def _dedupe_legs(legs):
    seen, out = set(), []
    for leg in legs:
        key = (leg["instrument"], leg["entry_time"])
        if key not in seen:
            seen.add(key)
            out.append(leg)
    return out


def _size_and_price(legs, rng):
    """Chronological pass: position size, revenge-sizing leak, P&L, costs."""
    rows = []
    prev_profit = 0.0
    for leg in legs:
        risk = float(rng.choice(RISK_CHOICES))
        qty = int(np.clip(round(risk / max(leg["stop_dist"], 0.01)), 5, 2000))
        oversized = False
        if prev_profit < 0 and rng.random() < REVENGE_PROB:
            qty *= int(rng.integers(2, 4))  # leak: doubling/tripling up after a loss
            oversized = True
        profit = round((leg["exit_px"] - leg["entry_px"]) * leg["direction"] * qty, 2)
        rows.append({**leg, "qty": qty, "profit": profit, "oversized": oversized})
        prev_profit = profit
    return rows


def _ensure_winning(trades, legs):
    """Guarantee the demo trader nets positive.

    If the simulated log isn't already a winner, re-pick the exit bar for
    the worst losers as the best real close inside their allowed holding
    window (the trader 'managed those trades better'). Prices and
    timestamps stay real bars throughout.
    """
    def total(ts):
        return sum(t["profit"] for t in ts)

    goal = 0.02 * sum(abs(t["profit"]) for t in trades) + 1.0
    if total(trades) > goal:
        return trades

    for t in sorted(trades, key=lambda t: t["profit"]):
        if t["profit"] >= 0:
            break
        closes, when = t["_closes"], t["_when"]
        if len(closes) < 2:
            continue
        moves = (closes[1:] - t["entry_px"]) * t["direction"]
        best = int(np.argmax(moves)) + 1
        t["exit_px"] = float(closes[best])
        t["exit_time"] = pd.Timestamp(when[best])
        t["exit_idx"] = t["entry_idx"] + best
        t["bars_held"] = best
        t["exit_name"] = "Manual"
        t["profit"] = round((t["exit_px"] - t["entry_px"]) * t["direction"] * t["qty"], 2)
        hi = float(t["_high"][1 : best + 1].max())
        lo = float(t["_low"][1 : best + 1].min())
        if t["direction"] == 1:
            t["mae_ps"], t["mfe_ps"] = max(0.0, t["entry_px"] - lo), max(0.0, hi - t["entry_px"])
        else:
            t["mae_ps"], t["mfe_ps"] = max(0.0, hi - t["entry_px"]), max(0.0, t["entry_px"] - lo)
        if total(trades) > goal:
            break
    return trades


def _to_ninjatrader(trades) -> pd.DataFrame:
    rows = []
    cum = 0.0
    for i, t in enumerate(trades, start=1):
        commission = round(t["qty"] * COMMISSION_PER_SHARE_RT, 2)
        cum = round(cum + t["profit"] - commission, 2)
        mfe = round(t["mfe_ps"] * t["qty"], 2)
        rows.append({
            "Trade number": i,
            "Instrument": t["instrument"],
            "Account": "Sim101",
            "Strategy": "Momentum",
            "Market pos.": "Long" if t["direction"] == 1 else "Short",
            "Qty": t["qty"],
            "Entry price": round(t["entry_px"], 4),
            "Exit price": round(t["exit_px"], 4),
            "Entry time": t["entry_time"],
            "Exit time": t["exit_time"],
            "Exit name": t["exit_name"],
            "Profit": t["profit"],
            "Cum. net profit": cum,
            "Commission": commission,
            "MAE": round(t["mae_ps"] * t["qty"], 2),
            "MFE": mfe,
            "ETD": round(max(0.0, mfe - t["profit"]), 2),
            "Bars": t["bars_held"],
        })
    return pd.DataFrame(rows)[trades_io.COLUMN_ORDER]


# -------------------------------------------------------------------- script


def main(argv=None):
    symbols = (argv or sys.argv[1:]) or DEFAULT_SYMBOLS
    print(f"Fetching bars for {', '.join(symbols)} ...")
    bars, warnings = bars_mod.fetch_bars(symbols)
    for w in warnings:
        print(f"WARNING: {w}")
    if bars.empty:
        sys.exit("No bars fetched — cannot generate trades.")
    days = bars["datetime"].dt.normalize().nunique()
    print(f"bars.csv: {len(bars):,} bars, {bars['instrument'].nunique()} symbols, {days} sessions")

    trades = generate_trades(bars)
    write_trades(trades, "trades.csv")

    reloaded = trades_io.load_trades("trades.csv")
    problems = verify_trades(reloaded, bars_mod.load_bars("bars.csv"))
    net = float(reloaded["Profit"].sum() - reloaded["Commission"].sum())
    wins = int((reloaded["Profit"] > 0).sum())
    print(f"trades.csv: {len(reloaded)} trades over {days} sessions, "
          f"win rate {wins / len(reloaded):.0%}, net P&L ${net:,.2f}")
    if problems:
        print(f"VERIFY FAILED ({len(problems)} mismatches):")
        for p in problems[:10]:
            print("  " + p)
        sys.exit(1)
    print("verify: every entry is a real bar open, every exit a real bar close ✔")


if __name__ == "__main__":
    main()
