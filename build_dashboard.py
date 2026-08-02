"""Layer 4 — compute dash_data.json from the engine's findings and bake
it into the terminal-style dashboard page.

build() runs the Layer-3 engine (audit + rule replay), derives everything
the page needs — verdict, KPI tiles, quant stats, per-trade cards, per-day
candle data, style read, leaks — writes dash_data.json, and renders
dashboard.html by embedding that JSON into dashboard_template.html so the
page opens straight from disk. Deterministic and keyless: the only AI
parts of the page (strategy step, coach panel) are clearly tagged and
optional.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

import auditor_engine
import bars as bars_mod
import discipline
import metrics
import rules as rules_mod
import strategy_ai
import trades_io

TEMPLATE_PATH = "dashboard_template.html"
SMALL_N = 10          # family-table rows under this many trades are greyed out
TREND_SPAN = 20       # EMA span used to classify continuation vs counter-trend

CONTINUATION_FAMILIES = ("momentum", "breakout", "trend", "pullback", "gap",
                         "opening range", "day trading", "swing")
COUNTER_FAMILIES = ("mean reversion", "fade", "contrarian", "vwap")


def build(trades_path="trades.csv", bars_path="bars.csv",
          rules_path=rules_mod.RULES_PATH, playbook_path=strategy_ai.PLAYBOOK_PATH,
          out_json="dash_data.json", out_html="dashboard.html",
          template_path=TEMPLATE_PATH):
    """Compute dash_data.json and render dashboard.html. Returns the dict."""
    trades = trades_io.load_trades(trades_path)
    bars = bars_mod.load_bars(bars_path)
    rules = rules_mod.load_rules(rules_path) or dict(rules_mod.DEFAULT_RULES)
    playbook = strategy_ai.load_playbook(playbook_path)

    findings = auditor_engine.audit(trades, rules)
    per_trade, aggregate = discipline.replay(trades, bars, rules)

    data = compute_dash_data(trades, bars, rules, playbook, findings, per_trade, aggregate)

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
        fh.write("\n")
    if template_path and os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as fh:
            html = fh.read()
        html = html.replace("__DASH_DATA__", json.dumps(data, default=str))
        with open(out_html, "w", encoding="utf-8") as fh:
            fh.write(html)
    return data


def compute_dash_data(trades, bars, rules, playbook, findings, per_trade, aggregate):
    t = trades.sort_values("Entry time", kind="stable").reset_index(drop=True)
    net = (t["Profit"] - t["Commission"]).to_numpy(float)
    styles = _observed_styles(t, bars)
    cards = _trade_cards(t, net, styles, per_trade, findings, rules)

    return {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": _verdict(findings),
        "stepper": _stepper(rules, playbook),
        "coach": {"available": strategy_ai.is_available(), "markdown": None},
        "kpis": _kpis(t, net, findings, per_trade, aggregate),
        "quant": _quant(net, findings),
        "discipline": {
            "actual": aggregate["actual_net"],
            "rule": aggregate["rule_net"],
            "difference": aggregate["difference"],
            "n_not_taken": aggregate["n_not_taken"],
            "exit_kinds": aggregate["exit_kinds"],
            "assumptions": aggregate["assumptions"],
        },
        "leaks": _leaks(t, net, findings, cards),
        "pnl_by_hour": findings["pnl_by_hour"],
        "equity": _equity(t, net),
        "style": _style_panel(playbook, cards, net),
        "trades": cards,
        "days": _days(bars, cards),
        "bars": _bars_payload(bars),
        "rules": rules,
        "meta": findings["meta"],
    }


# ----------------------------------------------------------------- pieces


def _verdict(findings):
    total = findings["meta"]["net_pnl"]
    without3 = findings["luck"]["pnl_without_top"]["3"]
    conc = findings["luck"]["top_decile_concentration_pct"]
    updown = "up" if total >= 0 else "down"
    return (f"This account is {updown} ${abs(total):,.0f} — remove the top 3 trades "
            f"and it's ${without3:,.0f}; the top decile of trades produced "
            f"{conc:.0f}% of the winning P&L.")


def _stepper(rules, playbook):
    return {
        "strategy": {
            "label": "Strategy", "tag": "AI", "color": "purple",
            "state": "done" if playbook else ("done" if rules else "pending"),
            "detail": (f"{playbook['family']} (approved)" if playbook
                       else "rules set manually / preset"),
        },
        "audit": {"label": "Audit", "tag": "DETERMINISTIC", "color": "cyan",
                  "state": "done", "detail": "engine replayed every trade"},
        "coaching": {"label": "Coaching", "tag": "AI · OPTIONAL", "color": "purple",
                     "state": "pending", "detail": "press the coach button"},
    }


def _kpis(t, net, findings, per_trade, aggregate):
    equity = metrics.equity_curve(net)
    profits = t["Profit"].to_numpy(float)
    top3 = set(np.argsort(profits)[::-1][:3][profits[np.argsort(profits)[::-1][:3]] > 0])
    net_wo3 = np.array([0.0 if i in top3 else net[i] for i in range(len(net))])
    ordered = per_trade.sort_values("entry_time", kind="stable")
    cum_diff = (ordered["rule_pnl"].to_numpy(float)
                - ordered["actual_pnl"].to_numpy(float)).cumsum()

    win = pd.Series((profits > 0).astype(float))
    rolling_wr = win.rolling(20, min_periods=5).mean().dropna().tolist()
    r_vals = np.array([metrics.r_dollars_for(p, q, findings["rules"], inst)
                       for p, q, inst in zip(t["Entry price"], t["Qty"], t["Instrument"])])
    r_mult = pd.Series(np.where(r_vals > 0, profits / np.where(r_vals > 0, r_vals, 1), 0.0))
    rolling_r = r_mult.rolling(20, min_periods=5).mean().dropna().tolist()

    core = findings["core"]
    conc = findings["luck"]["top_decile_concentration_pct"]
    return [
        {"id": "net", "label": "NET P&L", "value": _usd(float(net.sum())),
         "sub": f"{len(t)} trades", "color": "auto", "raw": float(net.sum()),
         "spark": _thin(equity.tolist())},
        {"id": "wo3", "label": "WITHOUT TOP 3", "value": _usd(float(net_wo3.sum())),
         "sub": "luck check", "color": "auto", "raw": float(net_wo3.sum()),
         "spark": _thin(np.cumsum(net_wo3).tolist())},
        {"id": "rules", "label": "RULES − ACTUAL", "value": _usd(aggregate["difference"]),
         "sub": "disciplined benchmark", "color": "auto", "raw": aggregate["difference"],
         "spark": _thin(cum_diff.tolist())},
        {"id": "winpf", "label": "WIN RATE · PROFIT FACTOR",
         "value": f"{core['win_rate']:.0%} · {core['profit_factor']:.2f}",
         "sub": "rolling 20 below", "color": "cyan", "raw": core["win_rate"],
         "spark": _thin(rolling_wr)},
        {"id": "expr", "label": "EXPECTANCY (R)",
         "value": f"{core['expectancy_r']:+.2f}R" if core["expectancy_r"] is not None else "—",
         "sub": f"{_usd(core['expectancy_usd'])} per trade", "color": "auto",
         "raw": core["expectancy_r"] or 0.0, "spark": _thin(rolling_r)},
        {"id": "conc", "label": "CONCENTRATION", "value": f"{conc:.0f}%",
         "sub": "top decile share of winning P&L",
         "color": "amber" if conc > 50 else "cyan", "raw": conc, "spark": []},
    ]


def _quant(net, findings):
    exp, se = metrics.expectancy(net), metrics.std_error(net)
    stats = [
        {"key": "expectancy", "label": "EXPECTANCY ± SE",
         "value": f"{_usd(exp)} ± {_usd(se)}",
         "meaning": "average P&L per trade, with the statistical wobble of that average"},
        {"key": "sqn", "label": "SQN", "value": _num(metrics.sqn(net)),
         "meaning": "system quality: √N × per-trade Sharpe; under ~1.6 is hard to trade"},
        {"key": "sharpe", "label": "SHARPE / TRADE", "value": _num(metrics.sharpe_per_trade(net)),
         "meaning": "average trade divided by trade-to-trade volatility"},
        {"key": "payoff", "label": "PAYOFF", "value": _num(metrics.payoff_ratio(net)),
         "meaning": "average winner divided by average loser"},
        {"key": "maxdd", "label": "MAX DRAWDOWN", "value": _usd(metrics.max_drawdown(net)),
         "meaning": "deepest fall of the equity curve from its peak"},
        {"key": "kelly", "label": "KELLY", "value": _pct(metrics.kelly_fraction(net)),
         "meaning": "bankroll fraction that maximizes growth — most traders size well below it"},
    ]
    sqn_v = metrics.sqn(net) or 0.0
    conc = findings["luck"]["top_decile_concentration_pct"]
    quality = ("statistically solid" if sqn_v >= 2 else
               "positive but statistically thin" if sqn_v >= 1 else
               "not distinguishable from luck yet")
    read = (f"Read: the edge is {quality} (SQN {sqn_v:.2f}), and {conc:.0f}% of the "
            f"winning P&L sits in the top decile of trades — "
            f"{'the tail is carrying this account' if conc > 50 else 'gains are reasonably spread out'}.")
    return {"stats": stats, "read": read}


def _equity(t, net):
    times = t["Entry time"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    profits = t["Profit"].to_numpy(float)
    order = np.argsort(profits)[::-1][:3]
    top3 = set(order[profits[order] > 0])
    wo3 = np.array([0.0 if i in top3 else net[i] for i in range(len(net))])
    return {
        "actual": [[times[i], float(v)] for i, v in enumerate(np.cumsum(net))],
        "without_top3": [[times[i], float(v)] for i, v in enumerate(np.cumsum(wo3))],
    }


def _observed_styles(t, bars):
    """Per-trade observed style: continuation vs counter-trend x hold bucket."""
    sessions = discipline._index_bars(bars)
    styles = []
    for row in t.to_dict("records"):
        entry_time = pd.Timestamp(row["Entry time"])
        day = sessions.get((str(row["Instrument"]), entry_time.normalize()))
        with_trend = None
        if day is not None:
            when, _, _, _, close = day
            before = close[when < entry_time.to_datetime64()]
            if before.size >= 5:
                ema = pd.Series(before).ewm(span=TREND_SPAN, adjust=False).mean().iloc[-1]
                trend_up = before[-1] > ema
                is_long = str(row["Market pos."]).strip().lower() == "long"
                with_trend = trend_up if is_long else (not trend_up)
        hold_min = (pd.Timestamp(row["Exit time"]) - entry_time).total_seconds() / 60.0
        bucket = "scalp <10m" if hold_min < 10 else "intraday 10-60m" if hold_min <= 60 else "extended >60m"
        axis = ("continuation" if with_trend else
                "counter-trend" if with_trend is not None else "unclassified")
        styles.append({"axis": axis, "hold": bucket,
                       "family": f"{axis} · {bucket}", "hold_min": hold_min})
    return styles


def _trade_cards(t, net, styles, per_trade, findings, rules):
    replay_by_n = {int(r["trade_number"]): r for r in per_trade.to_dict("records")}
    oversized_n = {o["trade_number"] for o in findings["revenge"]["oversized_trades"]}
    violating = {v["trade_number"]: v["violations"] for v in findings["adherence"]["violations"]}

    cards = []
    for i, row in enumerate(t.to_dict("records")):
        n = int(row["Trade number"])
        r_dollar = metrics.r_dollars_for(row["Entry price"], row["Qty"], rules, row["Instrument"])
        rep = replay_by_n.get(n, {})
        manual = str(row["Exit name"]).strip().lower() == "manual"
        flags = []
        if n in violating:
            flags.append("outside plan: " + "; ".join(violating[n]))
        if n in oversized_n:
            flags.append("oversized after a loss")
        if manual:
            flags.append("manual exit")
        cards.append({
            "n": n,
            "instrument": str(row["Instrument"]),
            "date": pd.Timestamp(row["Entry time"]).strftime("%Y-%m-%d"),
            "side": str(row["Market pos."]),
            "qty": int(row["Qty"]),
            "entry_time": str(row["Entry time"]),
            "exit_time": str(row["Exit time"]),
            "entry_price": float(row["Entry price"]),
            "exit_price": float(row["Exit price"]),
            "profit": float(net[i]),
            "gross": float(row["Profit"]),
            "r_multiple": float(row["Profit"] / r_dollar) if r_dollar > 0 else None,
            "style": styles[i]["family"],
            "exit_name": str(row["Exit name"]),
            "flags": flags,
            "outside_plan": n in violating,
            "oversized": n in oversized_n,
            "manual": manual,
            "rule": {
                "kind": rep.get("rule_exit_kind"),
                "price": rep.get("rule_exit_price"),
                "time": str(rep["rule_exit_time"]) if rep.get("rule_exit_time") is not None else None,
                "pnl": rep.get("rule_pnl"),
                "difference": rep.get("difference"),
                "reason": rep.get("reason", ""),
            },
        })
    return cards


def _style_panel(playbook, cards, net):
    declared = playbook["family"] if playbook else "not declared"
    families = {}
    axis_dir, axis_hold = {}, {}
    for card, pnl in zip(cards, net):
        fam = card["style"]
        families.setdefault(fam, []).append(float(pnl))
        axis = fam.split(" · ")[0]
        hold = fam.split(" · ")[1]
        axis_dir[axis] = axis_dir.get(axis, 0) + 1
        axis_hold[hold] = axis_hold.get(hold, 0) + 1

    table = []
    for fam, pnls in sorted(families.items(), key=lambda kv: -len(kv[1])):
        table.append({
            "family": fam, "n": len(pnls),
            "win_rate": metrics.win_rate(pnls),
            "expectancy": metrics.expectancy(pnls),
            "small_n": len(pnls) < SMALL_N,
        })
    observed = table[0]["family"] if table else "unknown"

    match = None
    d = declared.lower()
    if any(k in d for k in CONTINUATION_FAMILIES):
        match = observed.startswith("continuation")
    elif any(k in d for k in COUNTER_FAMILIES):
        match = observed.startswith("counter-trend")
    banner = (f"declared '{declared}' — observed mostly '{observed}': "
              + ("MATCH — you trade the way you say you do" if match
                 else "MISMATCH — the tape says otherwise" if match is not None
                 else "no strong read"))
    return {"declared": declared, "observed_dominant": observed, "match": match,
            "banner": banner, "families": table,
            "axis_direction": axis_dir, "axis_hold": axis_hold}


def _leaks(t, net, findings, cards):
    """Ranked dollar figures for the classic leaks (negative = it cost you)."""
    leaks = []

    over = findings["revenge"]["oversized_trades"]
    if over:
        median = findings["revenge"]["median_qty"] or 1.0
        excess = sum(o["profit"] * max(0.0, (o["qty"] - median) / o["qty"]) for o in over)
        leaks.append({"name": "Revenge sizing (excess-size P&L)", "dollars": excess,
                      "detail": f"{len(over)} oversized trades right after a loss"})

    by_code = {"session": [], "direction": [], "size": [], "loss_limit": []}
    for v in findings["adherence"]["violations"]:
        idx = int(np.flatnonzero(t["Trade number"].to_numpy() == v["trade_number"])[0])
        for text in v["violations"]:
            code = ("session" if "session window" in text
                    else "loss_limit" if "daily loss limit" in text
                    else "direction" if "plan" in text else "size")
            by_code[code].append(net[idx])
    labels = {"session": "Entries outside the session window",
              "direction": "Trades against your declared direction",
              "size": "Positions above your max size",
              "loss_limit": "Entries after hitting the daily loss limit"}
    for code, pnls in by_code.items():
        if pnls:
            leaks.append({"name": labels[code], "dollars": float(np.sum(pnls)),
                          "detail": f"{len(pnls)} trades"})

    manual_diff = sum(c["rule"]["difference"] for c in cards
                      if c["manual"] and c["rule"]["difference"] is not None
                      and c["rule"]["kind"] not in ("not_taken", "no_data"))
    if any(c["manual"] for c in cards):
        leaks.append({"name": "Manual exits vs your rules",
                      "dollars": -float(manual_diff),
                      "detail": "what hand-exits gave up against the rule exit"})

    hours = findings["pnl_by_hour"]
    if hours:
        worst = min(hours.items(), key=lambda kv: kv[1])
        if worst[1] < 0:
            leaks.append({"name": f"Worst hour ({worst[0]})", "dollars": float(worst[1]),
                          "detail": "total P&L of entries in that hour"})

    return sorted(leaks, key=lambda l: l["dollars"])


def _days(bars, cards):
    """Per instrument: the sessions, whether real 1m data exists, trade count."""
    trades_per = {}
    for c in cards:
        trades_per[(c["instrument"], c["date"])] = trades_per.get((c["instrument"], c["date"]), 0) + 1
    out = {}
    for (instrument, day), g in bars.groupby(
        [bars["instrument"], bars["datetime"].dt.normalize()], sort=True
    ):
        when = g["datetime"].sort_values()
        spacing = when.diff().dropna().min()
        has_1m = bool(pd.notna(spacing) and spacing <= pd.Timedelta(minutes=1))
        key = day.strftime("%Y-%m-%d")
        out.setdefault(str(instrument), []).append({
            "date": key, "has_1m": has_1m,
            "n_trades": trades_per.get((str(instrument), key), 0),
        })
    return out


def _bars_payload(bars):
    """{instrument: {date: [["HH:MM", o, h, l, c], ...]}} at native resolution."""
    payload = {}
    for (instrument, day), g in bars.groupby(
        [bars["instrument"], bars["datetime"].dt.normalize()], sort=True
    ):
        g = g.sort_values("datetime")
        rows = [[ts.strftime("%H:%M"), round(o, 4), round(h, 4), round(l, 4), round(c, 4)]
                for ts, o, h, l, c in zip(g["datetime"], g["open"], g["high"],
                                          g["low"], g["close"])]
        payload.setdefault(str(instrument), {})[day.strftime("%Y-%m-%d")] = rows
    return payload


# ---------------------------------------------------------------- helpers


def _usd(v):
    if v is None:
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}" if abs(v) >= 100 else f"{sign}${abs(v):,.2f}"


def _num(v):
    return "—" if v is None else f"{v:.2f}"


def _pct(v):
    return "—" if v is None else f"{v:.0%}"


def _thin(values, max_points=80):
    """Downsample a series for sparklines."""
    values = [float(v) for v in values]
    if len(values) <= max_points:
        return values
    idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
    return [values[i] for i in idx]


if __name__ == "__main__":
    result = build()
    print(f"dash_data.json + dashboard.html written — {result['meta']['n_trades']} trades, "
          f"{len(result['trades'])} cards, verdict: {result['verdict']}")
