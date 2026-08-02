"""Layer 5 — the Flask app that ties the auditor together (port 5050).

One command (python run.py), fully functional with zero API keys: the
sample pipeline, uploads, presets/manual strategy setup, the engine, and
the dashboard all run deterministically. Only the plain-english strategy
path (/translate) and the coach (/coach) need a key, and both degrade
with one clear message.

Data files (bars.csv, trades.csv, rules.json, dash_data.json,
dashboard.html) live in the process working directory.
"""

from __future__ import annotations

import io
import os
import tempfile

from flask import Flask, jsonify, redirect, request, send_file

import ai_client
import bars as bars_mod
import build_dashboard
import coaching_layer
import fetch_and_gen
import rules as rules_mod
import strategy_ai
import trades_io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE_DIR, "dashboard_template.html")
PORT = 5050

app = Flask(__name__)


def _err(message, status=400):
    return jsonify({"ok": False, "error": str(message)}), status


# ------------------------------------------------------------------ pages


@app.get("/")
def wizard():
    return send_file(os.path.join(BASE_DIR, "setup.html"))


@app.get("/dashboard")
def dashboard():
    path = os.path.join(os.getcwd(), "dashboard.html")
    if not os.path.exists(path):
        return redirect("/")
    return send_file(path)


# ------------------------------------------------------------ step 1: data


@app.post("/api/data/sample")
def data_sample():
    """Run fetch_and_gen fresh from live Yahoo: real bars + sample trader."""
    symbols = (request.get_json(silent=True) or {}).get("symbols") or fetch_and_gen.DEFAULT_SYMBOLS
    bars, warnings = bars_mod.fetch_bars(symbols)
    if bars.empty:
        return _err("No bars could be fetched from Yahoo — check your connection. "
                    + " | ".join(warnings), 502)
    trades = fetch_and_gen.generate_trades(bars)
    fetch_and_gen.write_trades(trades, "trades.csv")
    return jsonify({
        "ok": True, "source": "sample",
        "n_trades": int(len(trades)),
        "symbols": sorted(bars["instrument"].unique().tolist()),
        "sessions": int(bars["datetime"].astype(str).str[:10].nunique()),
        "warnings": warnings, "missing": [],
        "note": "sample trader generated from real bars — every price is a real bar's open/close",
    })


@app.post("/api/data/upload")
def data_upload():
    """Validate a NinjaTrader CSV, then fetch bars for exactly its symbols/dates."""
    if "file" not in request.files or not request.files["file"].filename:
        return _err("No file uploaded — choose your NinjaTrader CSV export first")
    upload = request.files["file"]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        upload.save(tmp.name)
        tmp_path = tmp.name
    try:
        trades = trades_io.load_trades(tmp_path)
    except trades_io.TradesFileError as exc:
        os.unlink(tmp_path)
        return _err(exc)
    os.unlink(tmp_path)

    fetched, warnings = bars_mod.fetch_bars_for_trades(trades, save_path=None)
    if not fetched.empty:
        fetched.to_csv("bars.csv", index=False)

    # Which trades have no bars for their session? The replay will skip
    # them ('no_data') — the behavioral audit still covers 100% of trades.
    covered = set()
    if not fetched.empty:
        covered = set(zip(fetched["instrument"],
                          fetched["datetime"].astype(str).str[:10]))
    missing = sorted({
        f"{row['Instrument']} {str(row['Entry time'])[:10]}"
        for _, row in trades.iterrows()
        if (row["Instrument"], str(row["Entry time"])[:10]) not in covered
    })

    out = trades.copy()
    for col in ("Entry time", "Exit time"):
        out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv("trades.csv", index=False)

    return jsonify({
        "ok": True, "source": "upload",
        "n_trades": int(len(trades)),
        "symbols": sorted(trades["Instrument"].unique().tolist()),
        "sessions": int(trades["Entry time"].dt.normalize().nunique()),
        "warnings": warnings,
        "missing": missing[:40],
        "note": (f"{len(missing)} trade-session(s) have no fetchable bars — the rule replay "
                 "will mark them 'no data', the behavioral audit still covers 100% of the trades")
                if missing else "bars fetched for every trade session",
    })


# -------------------------------------------------------- step 2: strategy


@app.get("/api/presets")
def presets():
    return jsonify({name: {"label": p["label"], "blurb": p["blurb"],
                           "rules": rules_mod.get_preset(name)}
                    for name, p in rules_mod.PRESETS.items()})


@app.post("/api/readout")
def readout():
    return jsonify(rules_mod.readout(request.get_json(silent=True) or {}))


@app.get("/api/strategies")
def strategies_list():
    return jsonify({"strategies": sorted(rules_mod.list_strategies())})


@app.post("/api/strategies/save")
def strategies_save():
    body = request.get_json(silent=True) or {}
    try:
        rules_mod.save_strategy(body.get("name", ""), body.get("rules") or {})
    except ValueError as exc:
        return _err(exc)
    return jsonify({"ok": True, "strategies": sorted(rules_mod.list_strategies())})


@app.post("/api/strategies/load")
def strategies_load():
    body = request.get_json(silent=True) or {}
    try:
        loaded = rules_mod.load_strategy(body.get("name", ""))
    except (KeyError, ValueError) as exc:
        return _err(str(exc).strip("'\""))
    return jsonify({"ok": True, "rules": loaded})


@app.post("/translate")
def translate():
    """Plain-english path: family + free text -> draft rules for review."""
    body = request.get_json(silent=True) or {}
    if not strategy_ai.is_available():
        return _err(strategy_ai.unavailable_note(), 503)
    try:
        drafted, warnings = strategy_ai.draft_rules(
            body.get("family", ""), body.get("playbook", ""))
    except strategy_ai.StrategyAIError as exc:
        return _err(exc, 502)
    return jsonify({"ok": True, "rules": drafted, "warnings": warnings})


# ------------------------------------------------------------- step 3: run


@app.post("/run")
def run_pipeline():
    """Approve rules, run the engine, build the dashboard, report headline."""
    body = request.get_json(silent=True) or {}
    clean, errors = rules_mod.normalize_rules(body.get("rules") or {})
    if errors:
        return _err("Fix the rules first: " + "; ".join(errors))
    if not os.path.exists("trades.csv"):
        return _err("No trade log yet — finish step 1 (sample or upload) first")
    if not os.path.exists("bars.csv"):
        return _err("No bars yet — finish step 1 first")

    if body.get("family") and body.get("playbook"):
        strategy_ai.approve(clean, body["family"], body["playbook"])
    else:
        rules_mod.save_rules(clean)

    try:
        data = build_dashboard.build(template_path=TEMPLATE)
    except Exception as exc:
        return _err(f"Engine failed: {exc}", 500)

    headline = {
        "verdict": data["verdict"],
        "net": data["meta"]["net_pnl"],
        "trades": data["meta"]["n_trades"],
        "win_rate": data["kpis"][3]["value"],
        "rules_minus_actual": data["discipline"]["difference"],
        "worst_leak": (data["leaks"][0]["name"] + " " +
                       build_dashboard._usd(data["leaks"][0]["dollars"]))
                      if data["leaks"] else "none",
    }
    print("\n" + "-" * 72)
    print("AUDIT COMPLETE")
    print(f"  {data['verdict']}")
    print(f"  trades: {headline['trades']}   net: {build_dashboard._usd(headline['net'])}   "
          f"win·pf: {headline['win_rate']}")
    print(f"  rules − actual: {build_dashboard._usd(headline['rules_minus_actual'])}   "
          f"worst leak: {headline['worst_leak']}")
    print(f"  dashboard: http://127.0.0.1:{PORT}/dashboard")
    print("-" * 72 + "\n")
    return jsonify({"ok": True, "redirect": "/dashboard", "headline": headline})


# ----------------------------------------------------------------- coach


@app.post("/api/coach")
@app.post("/coach")
def coach():
    if not ai_client.is_configured():
        return _err("set a key to enable coaching — the deterministic audit is complete without it", 503)
    if not os.path.exists("dash_data.json"):
        return _err("Run the audit first", 400)
    try:
        markdown = coaching_layer.coach()
    except ai_client.AIClientError as exc:
        return _err(exc, 502)
    return jsonify({"ok": True, "markdown": markdown})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT)
