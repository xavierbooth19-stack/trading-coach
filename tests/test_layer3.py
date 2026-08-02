"""Offline checks for Layer 3 (metrics, behavioral audit, rule replay).

No network needed. Fill mechanics are verified against hand-computed
scenarios (gap fills, stop-before-target, breakeven arming, trailing,
slippage math to the tick), then the whole engine runs end to end on the
Layer-1 synthetic tape.

Run:  python tests/test_layer3.py
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auditor_engine
import discipline
import fetch_and_gen
import metrics
from test_layer1 import synthetic_merged_bars

PASSED = 0


def ok(condition, label):
    global PASSED
    assert condition, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


def approx(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


# ------------------------------------------------------------- fixtures


BASE_RULES = {
    "direction": "both", "stop_pct": 1.0, "target_pct": 2.0,
    "session_start": "09:30", "session_end": "16:00",
    "max_size": 1000, "exit_style": "fixed", "slippage_bps": 2,
}


def make_bars(ohlc_rows):
    """Bars for TST on 2026-07-30 at fixed times; rows are (o, h, l, c)."""
    times = ["10:00", "10:05", "10:10", "15:55"][: len(ohlc_rows)]
    return pd.DataFrame([{
        "instrument": "TST",
        "datetime": pd.Timestamp(f"2026-07-30 {t}"),
        "open": o, "high": h, "low": l, "close": c, "volume": 1000,
    } for t, (o, h, l, c) in zip(times, ohlc_rows)])


def make_trade(side="Long", qty=100, entry=100.0, entry_time="2026-07-30 10:00"):
    return pd.DataFrame([{
        "Trade number": 1, "Instrument": "TST", "Account": "Sim101",
        "Strategy": "Momentum", "Market pos.": side, "Qty": qty,
        "Entry price": entry, "Exit price": 101.0,
        "Entry time": pd.Timestamp(entry_time),
        "Exit time": pd.Timestamp("2026-07-30 10:30"),
        "Exit name": "Manual", "Profit": 100.0, "Cum. net profit": 99.0,
        "Commission": 1.0, "MAE": 20.0, "MFE": 150.0, "ETD": 50.0, "Bars": 6,
    }])


def replay_one(bars, trade, **rule_overrides):
    rules = {**BASE_RULES, **rule_overrides}
    per_trade, agg = discipline.replay(trade, bars, rules)
    return per_trade.iloc[0], agg


# ---------------------------------------------------------------- tests


def test_metrics():
    print("metrics: identities on a hand-made ledger")
    profits = [10, -5, 20, -10, 30]
    ok(approx(metrics.win_rate(profits), 0.6), "win rate")
    ok(approx(metrics.avg_win(profits), 20.0), "avg win")
    ok(approx(metrics.avg_loss(profits), -7.5), "avg loss (negative)")
    ok(approx(metrics.payoff_ratio(profits), 20 / 7.5), "payoff = avg win / |avg loss|")
    ok(approx(metrics.profit_factor(profits), 4.0), "profit factor = gross win / gross loss")
    ok(approx(metrics.expectancy(profits), 9.0), "expectancy in $")
    ok(approx(metrics.expectancy_in_r([50, -25], [50, 50]), 0.25),
       "expectancy in R uses per-trade 1R")
    ok(approx(metrics.r_dollars(200.0, 100, 0.5), 100.0), "1R = stop_pct% x entry x qty")
    ok(approx(metrics.pnl_without_top(profits, 1), 15.0), "P&L without top 1")
    ok(approx(metrics.pnl_without_top(profits, 3), -15.0), "P&L without top 3")
    ok(approx(metrics.top_decile_concentration_pct(profits), 50.0),
       "top decile share of gross winning P&L")
    ok(metrics.top_decile_concentration_pct([-1, -2]) == 0.0, "all losers -> 0% concentration")
    ok(metrics.win_rate([]) is None and metrics.pnl_without_top([], 3) == 0.0,
       "empty ledger degrades to None/0, never crashes")

    v = metrics.adherence_violations("Short", "2026-07-30 09:31", 100,
                                     {**BASE_RULES, "direction": "long_only"})
    ok([x["code"] for x in v] == ["direction"], "direction violation detected")
    v = metrics.adherence_violations("Long", "2026-07-30 16:00", 2000, BASE_RULES)
    ok(sorted(x["code"] for x in v) == ["session", "size"],
       "session-end entry and oversize both flagged")
    ok(metrics.adherence_violations("Long", "2026-07-30 09:30", 100, BASE_RULES) == [],
       "entry exactly at session_start is inside the plan")


def test_fill_mechanics():
    print("discipline: hand-verified fill mechanics (long, stop 99 / target 102)")
    entry_bar = (100.0, 100.6, 98.0, 100.2)  # dips through the stop INTRA-entry-bar

    # A: next bar holds both stop and target -> stop first, slipped 2bps
    row, _ = replay_one(make_bars([entry_bar, (100.5, 102.5, 98.9, 101.0)]), make_trade())
    ok(row["rule_exit_kind"] == "stop", "stop before target in the same bar")
    ok(approx(row["rule_exit_price"], 99 * (1 - 0.0002), 1e-9), "stop fill = stop price - 2bps")
    ok(str(row["rule_exit_time"]) == "2026-07-30 10:05:00",
       "entry-bar stop dip ignored (no intra-entry-bar lookahead)")
    ok(approx(row["rule_pnl"], (99 * 0.9998 - 100) * 100 - 1.0, 1e-6),
       "stop P&L = (fill - entry) x qty - same commission")

    # B: gap through the stop -> fills at the open, worse than the stop
    row, _ = replay_one(make_bars([entry_bar, (98.0, 99.5, 97.8, 99.2)]), make_trade())
    ok(row["rule_exit_kind"] == "stop" and approx(row["rule_exit_price"], 98 * 0.9998, 1e-9),
       "gap-through stop fills at the open, then slippage")

    # C: clean target hit -> exact limit price, no slippage
    row, _ = replay_one(make_bars([entry_bar, (100.5, 101.0, 99.5, 100.8),
                                   (100.9, 102.4, 100.5, 101.9)]), make_trade())
    ok(row["rule_exit_kind"] == "target" and approx(row["rule_exit_price"], 102.0, 1e-12),
       "target is a resting limit: exact price, no slippage")

    # C2: bar gaps OPEN above the target -> still fills at 102, not the better open
    row, _ = replay_one(make_bars([entry_bar, (103.0, 103.5, 102.5, 103.2)]), make_trade())
    ok(row["rule_exit_kind"] == "target" and approx(row["rule_exit_price"], 102.0, 1e-12),
       "targets are never gap-improved")

    # D: nothing hit -> session close at last close, slipped (market exit)
    row, _ = replay_one(make_bars([entry_bar, (100.4, 101.0, 99.6, 100.7),
                                   (100.7, 101.2, 99.9, 100.5),
                                   (100.5, 101.1, 99.8, 100.9)]), make_trade())
    ok(row["rule_exit_kind"] == "session_close"
       and approx(row["rule_exit_price"], 100.9 * 0.9998, 1e-9),
       "open at session_end closes at the last close, slipped")

    # E: breakeven arms at +1R (101), stop moves to entry for LATER bars
    row, _ = replay_one(make_bars([entry_bar, (100.5, 101.2, 99.5, 100.9),
                                   (100.4, 100.6, 99.8, 100.1)]),
                        make_trade(), exit_style="breakeven", target_pct=None)
    ok(row["rule_exit_kind"] == "breakeven_stop"
       and approx(row["rule_exit_price"], 100.0 * 0.9998, 1e-9),
       "breakeven: +1R arms, next-bar dip exits at entry (slipped)")

    # F: trailing stop follows the high by the stop distance
    row, _ = replay_one(make_bars([entry_bar, (100.5, 103.0, 100.2, 102.8),
                                   (102.8, 102.9, 101.9, 102.0)]),
                        make_trade(), exit_style="trailing")
    ok(row["rule_exit_kind"] == "trailing_stop"
       and approx(row["rule_exit_price"], 102.0 * 0.9998, 1e-9),
       "trailing: stop = high - stop distance, exits on the pullback")

    # G: time style ignores stop and target, holds to session end
    row, _ = replay_one(make_bars([entry_bar, (100.5, 102.5, 98.5, 101.5),
                                   (101.5, 102.0, 100.9, 101.8)]),
                        make_trade(), exit_style="time")
    ok(row["rule_exit_kind"] == "time" and approx(row["rule_exit_price"], 101.8 * 0.9998, 1e-9),
       "time style holds to session end regardless of stop/target")

    # H: short side mirrors everything
    row, _ = replay_one(make_bars([(100.0, 100.4, 99.4, 99.8), (100.2, 101.5, 99.9, 101.2)]),
                        make_trade(side="Short"))
    ok(row["rule_exit_kind"] == "stop" and approx(row["rule_exit_price"], 101 * 1.0002, 1e-9),
       "short stop at 101 fills with adverse (upward) slippage")


def test_eligibility_and_aggregate():
    print("discipline: plan eligibility + aggregate")
    bars = make_bars([(100.0, 100.6, 99.2, 100.2), (100.5, 102.5, 98.9, 101.0)])

    row, agg = replay_one(bars, make_trade(qty=5000))
    ok(row["rule_exit_kind"] == "not_taken" and "outside your plan" in row["reason"],
       "oversized trade -> outside your plan, not taken")
    ok(row["rule_pnl"] == 0.0 and agg["rule_net"] == 0.0,
       "not-taken trades contribute 0 to the benchmark")

    row, _ = replay_one(bars, make_trade(side="Short"), direction="long_only")
    ok(row["rule_exit_kind"] == "not_taken", "wrong-direction trade not taken")

    row, _ = replay_one(bars, make_trade(entry_time="2026-07-30 15:35"),
                        session_end="15:30")
    ok(row["rule_exit_kind"] == "not_taken", "late entry not taken under earlier session_end")

    per_trade, agg = discipline.replay(make_trade(), bars, BASE_RULES)
    ok(approx(agg["difference"], agg["rule_net"] - agg["actual_net"], 1e-9),
       "difference = rule-managed minus actual")
    ok(len(agg["assumptions"]) == 9, "all 9 printed assumptions ship with the aggregate")
    ok(approx(agg["actual_net"], 99.0, 1e-9), "actual side is net of the real commission")


def test_full_pipeline():
    print("engine: full pipeline on the synthetic tape")
    bars = synthetic_merged_bars(["SPY", "QQQ"], n_sessions=30)
    trades = fetch_and_gen.generate_trades(bars, seed=7, min_trades=120)
    rules = {**BASE_RULES, "direction": "long_only", "session_end": "15:30",
             "max_size": int(trades["Qty"].median() * 2)}

    findings = auditor_engine.audit(trades, rules)
    for key in ("meta", "core", "luck", "revenge", "disposition",
                "pnl_by_hour", "pnl_by_instrument", "adherence"):
        ok(key in findings, f"findings has '{key}'")
    core = findings["core"]
    ok(0 < core["win_rate"] < 1 and core["profit_factor"] > 0, "core stats in range")
    ok(core["expectancy_r"] is not None, "expectancy reported in R too")
    luck = findings["luck"]["pnl_without_top"]
    ok(luck["1"] >= luck["3"] >= luck["5"], "removing more top trades never raises P&L")
    ok(0 <= findings["luck"]["top_decile_concentration_pct"] <= 100, "concentration bounded")
    ok(findings["revenge"]["oversized_trades"], "generator's revenge sizing is caught")
    ok(findings["adherence"]["counts"]["session"] > 0, "late-session leak flagged")
    ok(findings["adherence"]["counts"]["direction"] > 0, "shorts flagged under long_only")
    ok(findings["adherence"]["manual_exits"]["count"] > 0
       and "not a plan violation" in findings["adherence"]["manual_exits"]["note"],
       "manual exits are an info line, not a violation")
    ok(findings["disposition"]["mfe_capture"] is not None
       and 0 < findings["disposition"]["mfe_capture"] <= 1.0,
       "MFE capture = realized / MFE, sane range")
    ok(sum(findings["pnl_by_instrument"].values()) is not None
       and abs(sum(findings["pnl_by_instrument"].values()) - findings["meta"]["gross_pnl"]) < 0.01,
       "per-instrument P&L sums to the total (shared metrics, no drift)")

    per_trade, agg = discipline.replay(trades, bars, rules)
    ok(len(per_trade) == len(trades), "one replay row per real trade")
    ok(agg["n_not_taken"] == findings["adherence"]["n_violating_trades"],
       "replay eligibility == auditor violations (shared metrics.py)")
    ok(agg["n_no_data"] == 0, "every in-plan trade found its bars")
    replayed = per_trade[~per_trade["rule_exit_kind"].isin(["not_taken", "no_data"])]
    ok(set(replayed["rule_exit_kind"]) <= {"stop", "target", "breakeven_stop",
                                           "trailing_stop", "session_close", "time"},
       "exit kinds from the documented set")
    ok(approx(agg["rule_net"], float(per_trade["rule_pnl"].sum()), 1e-6),
       "aggregate equals the sum of per-trade rows")

    per_trade2, agg2 = discipline.replay(trades, bars, rules)
    ok(per_trade.equals(per_trade2) and agg2["rule_net"] == agg["rule_net"],
       "replay is fully deterministic")

    exit_after_entry = replayed["rule_exit_time"] > replayed["entry_time"]
    ok(bool(exit_after_entry.all() or
            (replayed.loc[~exit_after_entry, "rule_exit_kind"] == "session_close").all()),
       "exits resolve strictly after the entry bar (except last-bar close-outs)")


def main():
    test_metrics()
    test_fill_mechanics()
    test_eligibility_and_aggregate()
    test_full_pipeline()
    print(f"\nall {PASSED} checks passed")


if __name__ == "__main__":
    main()
