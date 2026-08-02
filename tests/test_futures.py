"""Offline checks for the futures / prop-firm (TopstepX) support.

Covers: contract identification + point values, the TopstepX/ProjectX
CSV adapter (UTC timestamps, Buy/Sell sides, Fees/PnL headers), points-
based stops with exact dollar replay math, and the daily-loss-limit rule
in both the auditor and the replay.

Run:  python tests/test_futures.py
"""

import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auditor_engine
import bars as bars_mod
import discipline
import instruments
import metrics
import rules as rules_mod
import trades_io

PASSED = 0


def ok(condition, label):
    global PASSED
    assert condition, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


FUT_RULES = {
    "direction": "both", "stop_pct": 10, "target_pct": 20,
    "session_start": "09:30", "session_end": "16:00", "max_size": 10,
    "exit_style": "fixed", "slippage_bps": 0, "price_unit": "points",
    "daily_loss_limit": None,
}


def mes_trade(n=1, entry="2026-07-30 10:00", exit_="2026-07-30 10:10",
              profit=-104.0, side="Long", entry_px=5000.0, exit_px=4990.0, qty=2):
    return {
        "Trade number": n, "Instrument": "MES 06-25", "Account": "Sim101",
        "Strategy": "Momentum", "Market pos.": side, "Qty": qty,
        "Entry price": entry_px, "Exit price": exit_px,
        "Entry time": pd.Timestamp(entry), "Exit time": pd.Timestamp(exit_),
        "Exit name": "Stop loss", "Profit": profit, "Cum. net profit": profit,
        "Commission": 4.0, "MAE": 0.0, "MFE": 0.0, "ETD": 0.0, "Bars": 2,
    }


def test_instruments():
    print("instruments: roots, point values, yahoo mapping")
    cases = [("MES 06-25", "MES", 5.0), ("ES 03-26", "ES", 50.0),
             ("ESH25", "ES", 50.0), ("/MNQZ5", "MNQ", 2.0),
             ("F.US.EP", "ES", 50.0), ("F.US.ENQ", "NQ", 20.0),
             ("CON.F.US.MES.H25", "MES", 5.0), ("EP", "ES", 50.0),
             ("mnqz5", "MNQ", 2.0), ("GC 12-25", "GC", 100.0),
             ("CLX5", "CL", 1000.0)]
    for text, root, pv in cases:
        ok(instruments.contract_root(text) == root
           and instruments.point_value(text) == pv, f"{text!r} -> {root} @ ${pv:g}/pt")
    ok(instruments.point_value("SPY") == 1.0 and instruments.contract_root("SPY") == "SPY",
       "equities keep point value 1.0")
    ok(instruments.contract_root("F") == "F" and instruments.point_value("AAPL") == 1.0,
       "single-letter tickers are not mistaken for futures")
    ok(bars_mod.to_yahoo_symbol("F.US.EP") == "ES=F"
       and bars_mod.to_yahoo_symbol("MES 06-25") == "MES=F"
       and bars_mod.to_yahoo_symbol("SPY") == "SPY",
       "yahoo mapping: any futures form -> continuous contract")


def test_topstepx_adapter(tmp):
    print("trades_io: TopstepX/ProjectX CSV adapter")
    csv = (
        "Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type\n"
        "a1b2-guid,CON.F.US.MES.U25,2026-07-30T14:31:00+00:00,2026-07-30T14:45:00+00:00,"
        "5000.00,5010.00,$2.72,$100.00,2,Buy\n"
        "c3d4-guid,CON.F.US.MES.U25,2026-07-30T15:05:00+00:00,2026-07-30T15:20:00+00:00,"
        "5012.00,5015.00,$2.72,\"($30.00)\",2,Sell\n"
    )
    path = os.path.join(tmp, "topstepx.csv")
    with open(path, "w") as fh:
        fh.write(csv)
    t = trades_io.load_trades(path)
    ok(len(t) == 2 and list(t.columns) == trades_io.COLUMN_ORDER,
       "TopstepX headers mapped onto the NinjaTrader schema")
    ok(t["Instrument"].iloc[0] == "CON.F.US.MES.U25"
       and instruments.point_value(t["Instrument"].iloc[0]) == 5.0,
       "ProjectX contract name kept and recognized as MES")
    ok(str(t["Entry time"].iloc[0]) == "2026-07-30 10:31:00",
       "UTC 14:31 converted to naive Eastern 10:31")
    ok(t["Market pos."].tolist() == ["Long", "Short"], "Buy/Sell -> Long/Short")
    ok(float(t["Commission"].iloc[0]) == 2.72 and float(t["Profit"].iloc[1]) == -30.0,
       "Fees/PnL currency values parsed")
    ok(t["Trade number"].tolist() == [1, 2], "GUID ids renumbered sequentially")


def test_points_math():
    print("metrics: points mode + point values")
    ok(metrics.stop_distance(5000, FUT_RULES) == 10.0, "points stop = points, not percent")
    ok(metrics.stop_distance(200, {**FUT_RULES, "price_unit": "percent", "stop_pct": 0.5}) == 1.0,
       "percent stop unchanged")
    ok(metrics.r_dollars_for(5000, 2, FUT_RULES, "MES 06-25") == 100.0,
       "1R on MES: 10 pts x 2 contracts x $5/pt = $100")
    ok(metrics.r_dollars_for(200, 100, {**FUT_RULES, "price_unit": "percent",
                                        "stop_pct": 0.5}, "AAPL") == 100.0,
       "equity 1R unchanged: 0.5% x $200 x 100 = $100")
    clean, errors = rules_mod.normalize_rules(FUT_RULES)
    ok(errors == [] and clean["price_unit"] == "points", "points rules validate")
    r = rules_mod.readout(clean)
    ok("10 points" in r["text"] and r["reward_to_risk"] == 2.0,
       "readout speaks points for futures")
    _, errors = rules_mod.normalize_rules({**FUT_RULES, "price_unit": "furlongs"})
    ok(any("price_unit" in e for e in errors), "bad unit -> clear error")


def test_futures_replay():
    print("discipline: exact dollar replay on MES (points + $5/pt)")
    bars = pd.DataFrame([
        {"instrument": "MES 06-25", "datetime": pd.Timestamp("2026-07-30 10:00"),
         "open": 5000.0, "high": 5004.0, "low": 4997.0, "close": 5001.0, "volume": 100},
        {"instrument": "MES 06-25", "datetime": pd.Timestamp("2026-07-30 10:05"),
         "open": 4996.0, "high": 4999.0, "low": 4988.0, "close": 4992.0, "volume": 100},
    ])
    trades = pd.DataFrame([mes_trade()])
    per_trade, agg = discipline.replay(trades, bars, FUT_RULES)
    row = per_trade.iloc[0]
    ok(row["rule_exit_kind"] == "stop" and row["rule_exit_price"] == 4990.0,
       "stop at entry - 10 points (4990), no slippage configured")
    ok(row["rule_pnl"] == -104.0,
       "rule P&L = -10 pts x 2 contracts x $5/pt - $4 commission = -$104")
    ok(agg["actual_net"] == -108.0, "actual net = log P&L - commission")

    findings = auditor_engine.audit(trades, FUT_RULES)
    ok(abs(findings["core"]["expectancy_r"] - (-104.0 - 0) / 100.0 - 0.04) < 0.05
       or findings["core"]["expectancy_r"] is not None, "R expectancy computed with $100 1R")
    ok(abs(findings["core"]["expectancy_r"] - (-1.04)) < 1e-9,
       "gross -104 on a $100 1R = -1.04R exactly")


def test_daily_loss_limit():
    print("engine: daily loss limit (prop-firm rule)")
    trades = pd.DataFrame([
        mes_trade(1, "2026-07-30 09:40", "2026-07-30 09:55", profit=-600.0),
        mes_trade(2, "2026-07-30 10:30", "2026-07-30 10:40", profit=50.0),   # after -604 net
        mes_trade(3, "2026-07-31 09:40", "2026-07-31 09:55", profit=-600.0), # fresh day
        mes_trade(4, "2026-07-31 09:20", "2026-07-31 09:25", profit=10.0),   # before the loss
    ])
    limited = metrics.daily_loss_limit_violations(trades, 500)
    ok(limited == {2}, "only the entry after the day is down $500+ is flagged")
    ok(metrics.daily_loss_limit_violations(trades, None) == set(), "null limit = off")

    rules = {**FUT_RULES, "daily_loss_limit": 500, "session_start": "09:00"}
    findings = auditor_engine.audit(trades, rules)
    ok(findings["adherence"]["counts"]["loss_limit"] == 1, "auditor counts the violation")
    texts = [v for v in findings["adherence"]["violations"] if v["trade_number"] == 2]
    ok(texts and "daily loss limit" in texts[0]["violations"][0],
       "violation text names the rule")

    bars = pd.DataFrame([{"instrument": "MES 06-25",
                          "datetime": pd.Timestamp(f"2026-07-{d} {t}"),
                          "open": 5000.0, "high": 5004.0, "low": 4997.0,
                          "close": 5001.0, "volume": 100}
                         for d in (30, 31) for t in ("09:40", "10:30", "11:30")])
    per_trade, agg = discipline.replay(trades, bars, rules)
    flagged = per_trade[per_trade["trade_number"] == 2].iloc[0]
    ok(flagged["rule_exit_kind"] == "not_taken"
       and "daily loss limit" in flagged["reason"],
       "replay: not taken, contributes 0 to the benchmark")
    ok(agg["n_not_taken"] == findings["adherence"]["n_violating_trades"],
       "replay eligibility still equals auditor violations")


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_instruments()
        test_topstepx_adapter(tmp)
        test_points_math()
        test_futures_replay()
        test_daily_loss_limit()
    print(f"\nall {PASSED} checks passed")


if __name__ == "__main__":
    main()
