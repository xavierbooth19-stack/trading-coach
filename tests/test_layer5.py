"""Offline checks for Layer 5 (Flask app, wizard endpoints, coach).

No network and no key: Yahoo fetches and the AI client are stubbed; the
whole wizard flow (upload -> strategy -> run -> dashboard -> coach) runs
through the Flask test client inside a temp working directory.

Run:  python tests/test_layer5.py
"""

import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import ai_client
import bars as bars_mod
import coaching_layer
import fetch_and_gen
from test_layer1 import synthetic_merged_bars
from test_layer2 import clear_keys

PASSED = 0


def ok(condition, label):
    global PASSED
    assert condition, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


def main():
    clear_keys()
    bars = synthetic_merged_bars(["SPY", "QQQ"], n_sessions=30)
    trades = fetch_and_gen.generate_trades(bars, seed=7, min_trades=120)

    import app as app_mod
    client = app_mod.app.test_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        oldcwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            run_all(client, bars, trades)
        finally:
            os.chdir(oldcwd)
            clear_keys()
    print(f"\nall {PASSED} checks passed")


def run_all(client, bars, trades):
    print("wizard pages")
    r = client.get("/")
    ok(r.status_code == 200 and b"TRADE AUDITOR" in r.data and b"PLAIN ENGLISH (AI)" in r.data,
       "GET / serves the setup wizard")
    r = client.get("/dashboard")
    ok(r.status_code == 302, "GET /dashboard before a run redirects to the wizard")

    print("step 1: data")
    # sample path with Yahoo stubbed
    real_fetch = bars_mod.fetch_bars
    bars_mod.fetch_bars = lambda symbols, **kw: (bars.copy(), ["SPY: 5m history clipped (stub)"])
    try:
        r = client.post("/api/data/sample", json={})
        j = r.get_json()
    finally:
        bars_mod.fetch_bars = real_fetch
    ok(j["ok"] and j["n_trades"] >= 120 and j["warnings"], "sample path generates trades + surfaces warnings")
    ok(os.path.exists("trades.csv"), "sample writes trades.csv")

    # upload path: no file / bad file / good file with one un-fetchable instrument
    r = client.post("/api/data/upload", data={})
    ok(not r.get_json()["ok"] and "No file" in r.get_json()["error"], "upload without file -> clear error")

    bad = io.BytesIO(b"Instrument,Qty\nSPY,100\n")
    r = client.post("/api/data/upload", data={"file": (bad, "bad.csv")},
                    content_type="multipart/form-data")
    ok("missing required column" in r.get_json()["error"], "invalid CSV -> trades_io's clear error")

    buf = io.StringIO()
    out = trades.copy()
    for col in ("Entry time", "Exit time"):
        out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(buf, index=False)
    spy_only = bars[bars["instrument"] == "SPY"].copy()  # QQQ bars "un-fetchable"
    real_fbt = bars_mod.fetch_bars_for_trades
    bars_mod.fetch_bars_for_trades = lambda t, save_path=None: (spy_only, ["QQQ: no usable bars at any resolution"])
    try:
        r = client.post("/api/data/upload",
                        data={"file": (io.BytesIO(buf.getvalue().encode()), "trades.csv")},
                        content_type="multipart/form-data")
        j = r.get_json()
    finally:
        bars_mod.fetch_bars_for_trades = real_fbt
    ok(j["ok"] and j["n_trades"] == len(trades), "valid upload accepted with exact trade count")
    ok(j["missing"] and all(m.startswith("QQQ") for m in j["missing"]),
       "trades whose bars can't be fetched are listed")
    ok("100% of the trades" in j["note"], "continues anyway — audit still covers all trades")
    ok(j["warnings"], "fetch warnings surfaced")

    # restore full bars for the run step (as if fetch had succeeded for both)
    bars.to_csv("bars.csv", index=False)

    print("step 2: strategy endpoints")
    j = client.get("/api/presets").get_json()
    ok(len(j) >= 12 and "momentum" in j, "preset grid served")
    j = client.post("/api/readout", json={"stop_pct": 0.5, "target_pct": 1.0}).get_json()
    ok("2.0R" in j["text"], "live readout endpoint")
    j = client.post("/api/strategies/save",
                    json={"name": "wizard test", "rules": {"stop_pct": 0.4}}).get_json()
    ok(j["ok"] and "wizard test" in j["strategies"], "library save")
    j = client.post("/api/strategies/load", json={"name": "wizard test"}).get_json()
    ok(j["ok"] and j["rules"]["stop_pct"] == 0.4, "library one-click load")

    r = client.post("/translate", json={"family": "Momentum", "playbook": "I buy strength"})
    ok(r.status_code == 503 and "key" in r.get_json()["error"].lower(),
       "keyless /translate -> greyed out with a note")

    os.environ["AI_API_KEY"] = "sk-ant-test"
    real_complete = ai_client.complete
    ai_client.complete = lambda s, u, max_tokens=4096: json.dumps({
        "direction": "long_only", "stop_pct": 0.5, "target_pct": 1.0,
        "session_start": "09:30", "session_end": "15:30", "max_size": 800,
        "exit_style": "fixed", "slippage_bps": 2})
    try:
        j = client.post("/translate", json={"family": "Momentum",
                                            "playbook": "I buy the first pullback"}).get_json()
    finally:
        ai_client.complete = real_complete
        clear_keys()
    ok(j["ok"] and j["rules"]["max_size"] == 800, "/translate drafts editable rules (stubbed model)")

    print("step 3: run + dashboard")
    r = client.post("/run", json={"rules": {"stop_pct": -1}})
    ok("Fix the rules" in r.get_json()["error"], "bad rules -> clear error before running")

    rules = {"direction": "long_only", "stop_pct": 0.5, "target_pct": 1.0,
             "session_start": "09:30", "session_end": "15:30",
             "max_size": int(trades["Qty"].median() * 2), "exit_style": "fixed",
             "slippage_bps": 2}
    j = client.post("/run", json={"rules": rules, "family": "Momentum",
                                  "playbook": "I buy the first pullback"}).get_json()
    ok(j["ok"] and j["redirect"] == "/dashboard", "run -> redirect to /dashboard")
    ok("$" in j["headline"]["verdict"] and j["headline"]["trades"] > 0,
       "headline numbers returned (and printed to the terminal)")
    ok(os.path.exists("dash_data.json") and os.path.exists("dashboard.html"),
       "engine + build_dashboard produced the payload")
    ok(os.path.exists("playbook.json"), "AI-path run stores the playbook for the coach")

    r = client.get("/dashboard")
    ok(r.status_code == 200 and b"TRADE AUDITOR" in r.data and b"AI COACHING" in r.data,
       "GET /dashboard serves the Layer-4 page with the fresh payload")

    print("coach")
    r = client.post("/coach")
    ok(r.status_code == 503 and "set a key to enable coaching" in r.get_json()["error"],
       "keyless coach button -> 'set a key to enable coaching'")

    payload = coaching_layer.build_payload()
    for key in ("sample_size_note", "headline", "luck", "leaks_ranked_worst_first",
                "discipline", "worst_rule_deviation_trades", "style_match",
                "trader_playbook_free_text"):
        ok(key in payload, f"distilled payload has '{key}'")
    blob = json.dumps(payload)
    ok(len(blob) < 20000, f"payload is a few KB ({len(blob)} bytes), never the bars")
    ok("bars" not in payload and '"o":' not in blob and "09:3" not in json.dumps(payload["luck"]),
       "no price bars in the coach payload")
    ok(len(payload["worst_rule_deviation_trades"]) <= 5
       and payload["worst_rule_deviation_trades"][0]["gave_up_vs_rule"]
           >= payload["worst_rule_deviation_trades"][-1]["gave_up_vs_rule"],
       "five worst rule-deviation trades, worst first")
    ok(payload["trader_playbook_free_text"] == "I buy the first pullback",
       "user's free-text playbook included")

    os.environ["AI_API_KEY"] = "sk-ant-test"
    captured = {}
    def fake_complete(system, user, max_tokens=4096):
        captured["system"], captured["user"] = system, user
        return "# Coach\n- **leak**: manual exits\n\nVERDICT: promising but tail-heavy."
    ai_client.complete = fake_complete
    try:
        j = client.post("/coach").get_json()
    finally:
        ai_client.complete = real_complete
        clear_keys()
    ok(j["ok"] and "VERDICT:" in j["markdown"], "coach returns markdown for the Coach's Read panel")
    ok("blunt trading performance coach" in captured["system"]
       and "cite a number" in captured["system"]
       and "ICT order blocks" in captured["system"]
       and 'prefixed exactly "VERDICT:"' in captured["system"],
       "coach system prompt: blunt, cite-numbers, ICT honesty, one-line verdict")
    ok("FINDINGS JSON" in captured["user"] and "leaks_ranked_worst_first" in captured["user"],
       "distilled findings are what the model sees")


if __name__ == "__main__":
    main()
