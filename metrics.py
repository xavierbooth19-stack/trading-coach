"""Layer 3 — the shared metrics library.

Every number that both the engine and the dashboard show comes from
here, so they can never disagree. All functions are pure and take plain
sequences/scalars; missing data comes back as None, never a crash.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd


# ------------------------------------------------------------ core stats


def win_rate(profits):
    profits = _arr(profits)
    return None if profits.size == 0 else float((profits > 0).mean())


def avg_win(profits):
    profits = _arr(profits)
    wins = profits[profits > 0]
    return None if wins.size == 0 else float(wins.mean())


def avg_loss(profits):
    """Mean losing P&L — returned as a negative number."""
    profits = _arr(profits)
    losses = profits[profits < 0]
    return None if losses.size == 0 else float(losses.mean())


def payoff_ratio(profits):
    """Average win divided by the size of the average loss."""
    w, l = avg_win(profits), avg_loss(profits)
    if w is None or l is None or l == 0:
        return None
    return float(w / abs(l))


def profit_factor(profits):
    """Gross winning P&L divided by gross losing P&L."""
    profits = _arr(profits)
    gross_win = float(profits[profits > 0].sum())
    gross_loss = float(abs(profits[profits < 0].sum()))
    if gross_loss == 0:
        return None if gross_win == 0 else math.inf
    return gross_win / gross_loss


def expectancy(profits):
    """Average P&L per trade, in dollars."""
    profits = _arr(profits)
    return None if profits.size == 0 else float(profits.mean())


def r_dollars(entry_price, qty, stop_pct):
    """1R in dollars: stop_pct% of entry price, times quantity."""
    return float(stop_pct) / 100.0 * float(entry_price) * float(qty)


def expectancy_in_r(profits, r_values):
    """Average P&L per trade measured in R (profit / that trade's 1R)."""
    profits, r_values = _arr(profits), _arr(r_values)
    mask = r_values > 0
    if not mask.any():
        return None
    return float((profits[mask] / r_values[mask]).mean())


def core_stats(profits, r_values=None):
    """The headline bundle both the engine and the dashboard print."""
    profits = _arr(profits)
    return {
        "n_trades": int(profits.size),
        "net_pnl": float(profits.sum()) if profits.size else 0.0,
        "win_rate": win_rate(profits),
        "avg_win": avg_win(profits),
        "avg_loss": avg_loss(profits),
        "payoff_ratio": payoff_ratio(profits),
        "profit_factor": profit_factor(profits),
        "expectancy_usd": expectancy(profits),
        "expectancy_r": expectancy_in_r(profits, r_values) if r_values is not None else None,
    }


# ----------------------------------------------------------- quant stats


def std_error(profits):
    """Standard error of the per-trade P&L mean."""
    profits = _arr(profits)
    if profits.size < 2:
        return None
    return float(profits.std(ddof=1) / math.sqrt(profits.size))


def sharpe_per_trade(profits):
    """Mean per-trade P&L over its standard deviation (per-trade Sharpe)."""
    profits = _arr(profits)
    if profits.size < 2:
        return None
    sd = profits.std(ddof=1)
    return None if sd == 0 else float(profits.mean() / sd)


def sqn(profits):
    """System Quality Number: sqrt(N) x per-trade Sharpe (Van Tharp)."""
    s = sharpe_per_trade(profits)
    profits = _arr(profits)
    return None if s is None else float(s * math.sqrt(profits.size))


def kelly_fraction(profits):
    """Kelly: p - (1-p)/payoff — the bankroll fraction that maximizes growth."""
    p, payoff = win_rate(profits), payoff_ratio(profits)
    if p is None or payoff is None or payoff == 0:
        return None
    return float(p - (1 - p) / payoff)


def equity_curve(profits):
    """Cumulative P&L, trade by trade."""
    return np.cumsum(_arr(profits))


def max_drawdown(profits):
    """Deepest peak-to-trough fall of the cumulative equity curve, in $ (<= 0)."""
    profits = _arr(profits)
    if profits.size == 0:
        return 0.0
    equity = np.cumsum(profits)
    peak = np.maximum.accumulate(np.maximum(equity, 0))
    return float(min(0.0, (equity - peak).min()))


# ------------------------------------------------------------------ luck


def pnl_without_top(profits, n):
    """Total P&L with the n biggest winners removed."""
    profits = _arr(profits)
    if profits.size == 0:
        return 0.0
    top = np.sort(profits)[::-1][: max(0, int(n))]
    return float(profits.sum() - top[top > 0].sum())


def top_decile_concentration_pct(profits):
    """The top decile of trades' share of GROSS WINNING P&L, bounded 0-100."""
    profits = _arr(profits)
    if profits.size == 0:
        return 0.0
    gross_win = float(profits[profits > 0].sum())
    if gross_win <= 0:
        return 0.0
    n_top = max(1, math.ceil(profits.size * 0.10))
    top = np.sort(profits)[::-1][:n_top]
    share = float(top[top > 0].sum()) / gross_win * 100.0
    return float(min(100.0, max(0.0, share)))


# ------------------------------------------------------------------ time


def hold_minutes(entry_times, exit_times):
    """Per-trade hold time in minutes."""
    entry = pd.to_datetime(pd.Series(entry_times))
    exit_ = pd.to_datetime(pd.Series(exit_times))
    return ((exit_ - entry).dt.total_seconds() / 60.0).to_numpy()


# ------------------------------------------------------- plan adherence

# Shared by the auditor (violation report) and the rule replay
# (eligibility, rule 8) so the two can never disagree on what counts
# as "outside your plan".


def adherence_violations(position, entry_time, qty, rules):
    """Violations of direction / session window / max size.

    Returns a list of {"code": "direction"|"session"|"size", "text": ...};
    empty list = the trade was inside the plan. A 'Manual' exit is NOT
    checked here — it's an info line, not a violation.
    """
    violations = []
    position = str(position).strip().lower()

    direction = rules["direction"]
    if direction == "long_only" and position == "short":
        violations.append({"code": "direction", "text": "short trade in a long-only plan"})
    elif direction == "short_only" and position == "long":
        violations.append({"code": "direction", "text": "long trade in a short-only plan"})

    t = pd.Timestamp(entry_time).time()
    start = _parse_hhmm(rules["session_start"])
    end = _parse_hhmm(rules["session_end"])
    if not (start <= t < end):
        violations.append({
            "code": "session",
            "text": (f"entry at {t.strftime('%H:%M')} outside the session window "
                     f"{rules['session_start']}-{rules['session_end']}"),
        })

    if int(qty) > int(rules["max_size"]):
        violations.append({
            "code": "size",
            "text": f"size {int(qty)} above the max of {int(rules['max_size'])}",
        })

    return violations


def _parse_hhmm(text):
    return dt.datetime.strptime(str(text), "%H:%M").time()


def _arr(values):
    return np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
