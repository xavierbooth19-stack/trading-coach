"""Offline checks for Layer 4 (dash_data.json + the dashboard page).

Runs the full pipeline on the Layer-1 synthetic tape, builds the
dashboard, and verifies both the data contract and the rendered page.

Run:  python tests/test_layer4.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import build_dashboard
import fetch_and_gen
import rules as rules_mod
import strategy_ai
from test_layer1 import synthetic_merged_bars

PASSED = 0


def ok(condition, label):
    global PASSED
    assert condition, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


def build_fixture(tmpdir):
    bars = synthetic_merged_bars(["SPY", "QQQ"], n_sessions=30)
    trades = fetch_and_gen.generate_trades(bars, seed=7, min_trades=120)

    bars_path = os.path.join(tmpdir, "bars.csv")
    trades_path = os.path.join(tmpdir, "trades.csv")
    rules_path = os.path.join(tmpdir, "rules.json")
    playbook_path = os.path.join(tmpdir, "playbook.json")
    bars.to_csv(bars_path, index=False)
    fetch_and_gen.write_trades(trades, trades_path)

    rules = {**rules_mod.DEFAULT_RULES, "direction": "long_only",
             "session_end": "15:30", "max_size": int(trades["Qty"].median() * 2)}
    strategy_ai.approve(rules, "Momentum", "I buy strength after a strong open.",
                        rules_path=rules_path, playbook_path=playbook_path)

    data = build_dashboard.build(
        trades_path=trades_path, bars_path=bars_path,
        rules_path=rules_path, playbook_path=playbook_path,
        out_json=os.path.join(tmpdir, "dash_data.json"),
        out_html=os.path.join(tmpdir, "dashboard.html"),
        template_path=os.path.join(REPO, "dashboard_template.html"),
    )
    return data, tmpdir


def test_data_contract(data, tmpdir):
    print("dash_data: contract")
    for key in ("verdict", "stepper", "coach", "kpis", "quant", "discipline",
                "leaks", "pnl_by_hour", "equity", "style", "trades", "days",
                "bars", "rules", "meta"):
        ok(key in data, f"dash_data has '{key}'")

    with open(os.path.join(tmpdir, "dash_data.json")) as fh:
        roundtrip = json.load(fh)
    ok(roundtrip["verdict"] == data["verdict"], "dash_data.json round-trips")

    ok("remove the top 3" in data["verdict"] and "$" in data["verdict"],
       "verdict is concentration-framed")

    st = data["stepper"]
    ok(st["strategy"]["color"] == "purple" and st["audit"]["color"] == "cyan"
       and st["coaching"]["color"] == "purple", "stepper colors: AI purple, engine cyan")
    ok("Momentum" in st["strategy"]["detail"], "approved strategy shows in the stepper")
    ok(data["coach"]["markdown"] is None, "coach panel starts empty")

    ok(len(data["kpis"]) == 6, "six KPI tiles")
    ids = [k["id"] for k in data["kpis"]]
    ok(ids == ["net", "wo3", "rules", "winpf", "expr", "conc"], "KPI tiles in spec order")
    ok(sum(1 for k in data["kpis"] if k["spark"]) >= 4, "sparklines where natural")

    keys = [s["key"] for s in data["quant"]["stats"]]
    ok(keys == ["expectancy", "sqn", "sharpe", "payoff", "maxdd", "kelly"],
       "quant grid: expectancy±SE, SQN, Sharpe, payoff, max DD, kelly")
    ok(all(s["meaning"] for s in data["quant"]["stats"]), "each stat has its one-line meaning")
    ok(data["quant"]["read"].startswith("Read:"), "plain-english 'Read:' sentence present")

    ok(len(data["discipline"]["assumptions"]) == 9, "all 9 replay assumptions shipped")
    ok(data["discipline"]["difference"] == data["discipline"]["rule"] - data["discipline"]["actual"]
       or abs(data["discipline"]["difference"]
              - (data["discipline"]["rule"] - data["discipline"]["actual"])) < 1e-6,
       "discipline difference is the signed rule-minus-actual")


def test_trades_and_days(data):
    print("dash_data: trade cards + replay days")
    cards = data["trades"]
    ok(len(cards) == data["meta"]["n_trades"], "one card per trade")
    c = cards[0]
    for f in ("side", "qty", "r_multiple", "style", "rule", "flags", "date"):
        ok(f in c, f"card has '{f}'")
    ok(any(t["oversized"] for t in cards), "oversized flag present somewhere")
    ok(any(t["manual"] for t in cards), "manual flag present somewhere")
    ok(any(t["outside_plan"] for t in cards), "outside-plan flag present somewhere")
    ok(all(t["rule"]["kind"] == "not_taken" for t in cards if t["outside_plan"]),
       "outside-plan cards carry the not-taken rule verdict")
    replayed = [t for t in cards if t["rule"]["kind"] not in ("not_taken", "no_data")]
    ok(replayed and all(t["rule"]["price"] is not None for t in replayed),
       "replayed cards carry rule exit price/time/kind")

    days = data["days"]
    ok(set(days) == {"SPY", "QQQ"}, "days keyed by instrument")
    spy = days["SPY"]
    ok(any(d["has_1m"] for d in spy) and any(not d["has_1m"] for d in spy),
       "1m enabled only on days that actually have 1-minute data")
    one_min_days = {d["date"] for d in spy if d["has_1m"]}
    bars_day = data["bars"]["SPY"][sorted(one_min_days)[0]]
    ok(len(bars_day) > 300, "1m day ships ~390 native bars")
    ok(len(bars_day[0]) == 5 and ":" in bars_day[0][0], "bar rows are [HH:MM,o,h,l,c]")

    eq = data["equity"]
    ok(len(eq["actual"]) == len(cards) == len(eq["without_top3"]),
       "equity curves have one point per trade")
    ok(eq["without_top3"][-1][1] < eq["actual"][-1][1], "without-top-3 curve ends lower")

    ok(data["leaks"] == sorted(data["leaks"], key=lambda l: l["dollars"]),
       "leaks ranked worst first")
    ok(all("$" not in l["name"] and isinstance(l["dollars"], float) for l in data["leaks"]),
       "each leak carries its dollar figure")
    names = " ".join(l["name"] for l in data["leaks"])
    ok("Revenge" in names and "session" in names, "classic leaks present")

    style = data["style"]
    ok(style["declared"] == "Momentum", "declared family from the playbook")
    ok(sum(f["n"] for f in style["families"]) == len(cards), "family table covers every trade")
    ok(any(f["small_n"] for f in style["families"]) or all(f["n"] >= 10 for f in style["families"]),
       "small-N rows flagged for greying")
    ok(style["match"] in (True, False, None) and "declared" in style["banner"],
       "declared-vs-observed banner ready")
    ok(set(style["axis_direction"]) <= {"continuation", "counter-trend", "unclassified"},
       "direction axis buckets")


def test_html(tmpdir, data):
    print("dashboard.html: rendered page")
    with open(os.path.join(tmpdir, "dashboard.html"), encoding="utf-8") as fh:
        html = fh.read()
    ok("__DASH_DATA__" not in html, "data token replaced")
    ok(json.dumps(data["verdict"])[1:-1][:40] in html, "dash data embedded inline")
    for needle in ("TRADE AUDITOR", "AI COACHING", "JetBrains Mono", "TRADE REPLAY",
                   "CONSTELLATION", "DISCIPLINE COST", "REPLAY ASSUMPTIONS",
                   "renderCoach", 'data-tf="1"', "PREV DAY"):
        ok(needle in html, f"page contains {needle!r}")
    ok(html.count("DETERMINISTIC") >= 5 and html.count("purple") > 3,
       "cyan/purple deterministic-vs-AI split visible")
    ok("cdn" not in html.lower().replace("fonts.googleapis", ""),
       "no CDN scripts — page is self-contained (fonts degrade gracefully)")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        data, tmpdir = build_fixture(tmpdir)
        test_data_contract(data, tmpdir)
        test_trades_and_days(data)
        test_html(tmpdir, data)
    print(f"\nall {PASSED} checks passed")


if __name__ == "__main__":
    main()
