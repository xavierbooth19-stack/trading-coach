"""Layer 3 — the rule replay: what your rules would have done,
trade by trade, on the real bars. Conservative by design.

The exact assumptions in ASSUMPTIONS below are printed on the dashboard.
Eligibility reuses metrics.adherence_violations, so the auditor and the
replay can never disagree on what was "outside your plan".
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import metrics
import rules as rules_mod

# Printed verbatim on the dashboard later — keep in sync with the code.
ASSUMPTIONS = [
    "Exits resolve on bars strictly AFTER the entry bar — no intra-entry-bar lookahead.",
    "If one bar contains both the stop and the target, the STOP is assumed to fill first (pessimistic).",
    "Stops are gap-aware: a bar that opens through the stop fills at the open (worse than the stop), then slippage.",
    "Targets are resting limit orders at the exact price — never gap-improved.",
    "Slippage (slippage_bps) is charged on stop and market/time exits; limit fills get none.",
    "Each replayed trade is charged the SAME commission as the real trade.",
    "exit_style is modeled explicitly: fixed; breakeven (stop to entry at +1R); trailing "
    "(stop trails by the stop distance, no target); time (hold to session end).",
    "Plan eligibility first: trades violating direction, session or size are 'outside your "
    "plan, not taken' and contribute 0 to the disciplined benchmark.",
    "Anything still open at session_end closes at the last close of the session.",
]


def replay(trades: pd.DataFrame, bars: pd.DataFrame, rules=None):
    """Replay every trade under the rules on the real bars.

    Returns (per_trade, aggregate): a DataFrame with one row per real
    trade (rule exit price / time / kind + P&L, actual P&L, difference)
    and an aggregate dict (actual vs rule-managed vs signed difference,
    exit-kind counts, and the assumption list).
    """
    rules, errors = rules_mod.normalize_rules(rules or rules_mod.DEFAULT_RULES)
    if errors:
        raise ValueError("Bad rules for replay: " + "; ".join(errors))
    if trades.empty:
        raise ValueError("No trades to replay")

    sessions = _index_bars(bars)
    session_end = dt.datetime.strptime(rules["session_end"], "%H:%M").time()

    rows = []
    for trade in trades.sort_values("Entry time", kind="stable").to_dict("records"):
        rows.append(_replay_one(trade, sessions, rules, session_end))
    per_trade = pd.DataFrame(rows)

    replayed = per_trade[~per_trade["rule_exit_kind"].isin(["not_taken", "no_data"])]
    actual_net = float((per_trade["actual_pnl"]).sum())
    rule_net = float(per_trade["rule_pnl"].sum())
    aggregate = {
        "actual_net": actual_net,
        "rule_net": rule_net,
        "difference": rule_net - actual_net,
        "n_trades": int(len(per_trade)),
        "n_replayed": int(len(replayed)),
        "n_not_taken": int((per_trade["rule_exit_kind"] == "not_taken").sum()),
        "n_no_data": int((per_trade["rule_exit_kind"] == "no_data").sum()),
        "exit_kinds": per_trade["rule_exit_kind"].value_counts().to_dict(),
        "rules": rules,
        "assumptions": ASSUMPTIONS,
    }
    return per_trade, aggregate


# -------------------------------------------------------------- one trade


def _replay_one(trade, sessions, rules, session_end):
    entry_time = pd.Timestamp(trade["Entry time"])
    qty = int(trade["Qty"])
    side = 1 if str(trade["Market pos."]).strip().lower() == "long" else -1
    entry = float(trade["Entry price"])
    commission = float(trade["Commission"])
    actual_net = float(trade["Profit"]) - commission

    base = {
        "trade_number": int(trade["Trade number"]),
        "instrument": str(trade["Instrument"]),
        "entry_time": entry_time,
        "side": "Long" if side == 1 else "Short",
        "qty": qty,
        "entry_price": entry,
        "actual_pnl": actual_net,
    }

    # Rule 8 — plan eligibility first.
    violations = metrics.adherence_violations(
        trade["Market pos."], entry_time, qty, rules
    )
    if violations:
        return {**base, "rule_exit_price": None, "rule_exit_time": None,
                "rule_exit_kind": "not_taken",
                "reason": "outside your plan, not taken: " +
                          "; ".join(v["text"] for v in violations),
                "rule_pnl": 0.0, "difference": 0.0 - actual_net}

    day = sessions.get((str(trade["Instrument"]), entry_time.normalize()))
    if day is None:
        return {**base, "rule_exit_price": None, "rule_exit_time": None,
                "rule_exit_kind": "no_data", "reason": "no bars for this session",
                "rule_pnl": 0.0, "difference": 0.0 - actual_net}

    when, opn, high, low, close = day
    in_session = when.time < session_end

    # Rule 1 — resolve only on bars strictly after the entry bar.
    after = (when > entry_time.to_datetime64()) & in_session
    if not after.any():
        # Entered on the last bar of the session: rule 9 close-out at the
        # session's last close (the entry bar's own close).
        last = np.flatnonzero(in_session)
        if last.size == 0:
            return {**base, "rule_exit_price": None, "rule_exit_time": None,
                    "rule_exit_kind": "no_data", "reason": "no session bars",
                    "rule_pnl": 0.0, "difference": 0.0 - actual_net}
        i = last[-1]
        price = _slip(float(close[i]), side, rules["slippage_bps"])
        return _finish(base, price, when[i], "session_close", entry, side, qty,
                       commission, actual_net)

    idx = np.flatnonzero(after)
    price, exit_when, kind = _simulate(
        entry, side, rules,
        opn[idx], high[idx], low[idx], close[idx], when[idx],
    )
    return _finish(base, price, exit_when, kind, entry, side, qty, commission, actual_net)


def _finish(base, price, exit_when, kind, entry, side, qty, commission, actual_net):
    gross = (price - entry) * side * qty
    rule_pnl = gross - commission
    return {**base, "rule_exit_price": float(price),
            "rule_exit_time": pd.Timestamp(exit_when),
            "rule_exit_kind": kind, "reason": "",
            "rule_pnl": round(rule_pnl, 2),
            "difference": round(rule_pnl - actual_net, 2)}


# -------------------------------------------------------------- simulator


def _simulate(entry, side, rules, opn, high, low, close, when):
    """Walk the bars and return (exit_price, exit_time, kind).

    side is +1 long / -1 short. Favorable/adverse extremes and stop /
    target comparisons are all expressed through `side` so one code path
    serves both directions.
    """
    style = rules["exit_style"]
    slip_bps = rules["slippage_bps"]
    stop_dist = rules["stop_pct"] / 100.0 * entry
    n = len(close)

    # Rule 7 — time style: pure hold to session end.
    if style == "time":
        return _slip(float(close[n - 1]), side, slip_bps), when[n - 1], "time"

    stop = entry - side * stop_dist
    target = None
    if style in ("fixed", "breakeven") and rules["target_pct"] is not None:
        target = entry + side * rules["target_pct"] / 100.0 * entry
    breakeven_armed = False

    for i in range(n):
        bar_open, bar_close = float(opn[i]), float(close[i])
        favorable = float(high[i]) if side == 1 else float(low[i])
        adverse = float(low[i]) if side == 1 else float(high[i])
        stop_kind = ("breakeven_stop" if breakeven_armed and stop == entry
                     else "trailing_stop" if style == "trailing" else "stop")

        # Rule 3 — gap through the stop: fill at the open, then slippage.
        if side * (bar_open - stop) <= 0:
            return _slip(bar_open, side, slip_bps), when[i], stop_kind
        # Rule 2 — stop checked before target (pessimistic).
        if side * (adverse - stop) <= 0:
            return _slip(stop, side, slip_bps), when[i], stop_kind
        # Rule 4 — target is a resting limit at the exact price.
        if target is not None and side * (favorable - target) >= 0:
            return float(target), when[i], "target"

        # End-of-bar state updates take effect on the NEXT bar.
        if style == "breakeven" and not breakeven_armed:
            if side * (favorable - (entry + side * stop_dist)) >= 0:  # +1R reached
                breakeven_armed = True
                stop = entry
        elif style == "trailing":
            candidate = favorable - side * stop_dist
            stop = max(stop, candidate) if side == 1 else min(stop, candidate)

    # Rule 9 — still open at session end: close at the last close.
    return _slip(float(close[n - 1]), side, slip_bps), when[n - 1], "session_close"


def _slip(price, side, slippage_bps):
    """Rule 5 — adverse slippage on stop and market/time exits."""
    return price * (1.0 - side * slippage_bps / 10000.0)


def _index_bars(bars):
    """{(instrument, session day): (when, open, high, low, close)} arrays."""
    bars = bars.sort_values(["instrument", "datetime"])
    sessions = {}
    for (instrument, day), g in bars.groupby(
        [bars["instrument"], bars["datetime"].dt.normalize()], sort=False
    ):
        sessions[(str(instrument), day)] = (
            pd.DatetimeIndex(g["datetime"]),
            g["open"].to_numpy(float),
            g["high"].to_numpy(float),
            g["low"].to_numpy(float),
            g["close"].to_numpy(float),
        )
    return sessions
