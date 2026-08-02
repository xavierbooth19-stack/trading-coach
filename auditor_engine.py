"""Layer 3 — the behavioral audit, from the trade log alone.

audit(trades, rules) reads a Layer-1 trades frame and produces one
findings dict that every other layer reuses (dashboard, coach). All
numbers come from metrics.py. Deterministic, keyless, no bars needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import metrics
import rules as rules_mod

OVERSIZE_FACTOR = 2.0  # a post-loss trade at >= 2x the median size is "oversized"


def audit(trades: pd.DataFrame, rules=None) -> dict:
    """The full behavioral audit as one findings dict."""
    rules, errors = rules_mod.normalize_rules(rules or rules_mod.DEFAULT_RULES)
    if errors:
        raise ValueError("Bad rules for audit: " + "; ".join(errors))
    if trades.empty:
        raise ValueError("No trades to audit")

    t = trades.sort_values("Entry time", kind="stable").reset_index(drop=True)
    profits = t["Profit"].to_numpy(float)
    qty = t["Qty"].to_numpy(float)
    entry_px = t["Entry price"].to_numpy(float)
    r_values = np.array([metrics.r_dollars_for(p, q, rules, inst)
                         for p, q, inst in zip(entry_px, qty, t["Instrument"])])

    findings = {
        "meta": _meta(t),
        "rules": rules,
        "core": metrics.core_stats(profits, r_values),
        "luck": _luck(profits),
        "revenge": _revenge(t, profits, qty),
        "disposition": _disposition(t, profits),
        "pnl_by_hour": _pnl_by(t, t["Entry time"].dt.strftime("%H:00")),
        "pnl_by_instrument": _pnl_by(t, t["Instrument"]),
        "adherence": _adherence(t, rules),
    }
    return findings


# ---------------------------------------------------------------- pieces


def _meta(t):
    return {
        "n_trades": int(len(t)),
        "first_entry": str(t["Entry time"].min()),
        "last_exit": str(t["Exit time"].max()),
        "instruments": sorted(t["Instrument"].unique().tolist()),
        "sessions": int(t["Entry time"].dt.normalize().nunique()),
        "gross_pnl": float(t["Profit"].sum()),
        "commission": float(t["Commission"].sum()),
        "net_pnl": float((t["Profit"] - t["Commission"]).sum()),
    }


def _luck(profits):
    return {
        "pnl_without_top": {str(n): metrics.pnl_without_top(profits, n) for n in (1, 3, 5)},
        "top_decile_concentration_pct": metrics.top_decile_concentration_pct(profits),
    }


def _revenge(t, profits, qty):
    """Sizing right after a same-day loss vs the rest of the time."""
    after_loss = _after_same_day_loss(t)
    median_qty = float(np.median(qty))

    oversized = []
    for i in np.flatnonzero(after_loss & (qty >= OVERSIZE_FACTOR * median_qty)):
        oversized.append({
            "trade_number": int(t["Trade number"].iloc[i]),
            "instrument": str(t["Instrument"].iloc[i]),
            "entry_time": str(t["Entry time"].iloc[i]),
            "qty": int(qty[i]),
            "profit": float(profits[i]),
        })

    n_after, n_other = int(after_loss.sum()), int((~after_loss).sum())
    avg_after = float(qty[after_loss].mean()) if n_after else None
    avg_other = float(qty[~after_loss].mean()) if n_other else None
    ratio = (avg_after / avg_other) if avg_after and avg_other else None
    return {
        "avg_qty_after_same_day_loss": avg_after,
        "avg_qty_otherwise": avg_other,
        "size_ratio": ratio,
        "n_trades_after_loss": n_after,
        "median_qty": median_qty,
        "oversized_trades": oversized,
    }


def _after_same_day_loss(t):
    """True where the most recently closed same-day trade was a loss."""
    flags = np.zeros(len(t), dtype=bool)
    entries = t["Entry time"].to_numpy()
    exits = t["Exit time"].to_numpy()
    profits = t["Profit"].to_numpy(float)
    days = t["Entry time"].dt.normalize().to_numpy()
    for i in range(len(t)):
        prior = np.flatnonzero((days == days[i]) & (exits <= entries[i]))
        if prior.size:
            last = prior[np.argmax(exits[prior])]
            flags[i] = profits[last] < 0
    return flags


def _disposition(t, profits):
    """Do winners get cut short while losers get room?"""
    hold = metrics.hold_minutes(t["Entry time"], t["Exit time"])
    winners, losers = profits > 0, profits < 0
    mfe = t["MFE"].to_numpy(float)
    win_mfe = float(mfe[winners].sum())
    return {
        "avg_hold_winners_min": float(hold[winners].mean()) if winners.any() else None,
        "avg_hold_losers_min": float(hold[losers].mean()) if losers.any() else None,
        "mfe_capture": (float(profits[winners].sum()) / win_mfe) if win_mfe > 0 else None,
        "avg_giveback_etd": float(t["ETD"].to_numpy(float).mean()),
    }


def _pnl_by(t, keys):
    grouped = t.groupby(np.asarray(keys))["Profit"].sum()
    return {str(k): float(v) for k, v in grouped.items()}


def _adherence(t, rules):
    """Per-trade plan adherence: direction, session window, max size.

    'Manual' exits are reported as a separate info line — a manual exit
    is not a violation by itself.
    """
    violations = []
    counts = {"direction": 0, "session": 0, "size": 0, "loss_limit": 0}
    loss_limited = metrics.daily_loss_limit_violations(t, rules["daily_loss_limit"])
    for i in range(len(t)):
        found = metrics.adherence_violations(
            t["Market pos."].iloc[i], t["Entry time"].iloc[i], t["Qty"].iloc[i], rules
        )
        if int(t["Trade number"].iloc[i]) in loss_limited:
            found = found + [{"code": "loss_limit",
                              "text": "entered after hitting your daily loss limit"}]
        if found:
            violations.append({
                "trade_number": int(t["Trade number"].iloc[i]),
                "instrument": str(t["Instrument"].iloc[i]),
                "entry_time": str(t["Entry time"].iloc[i]),
                "violations": [v["text"] for v in found],
            })
            for v in found:
                counts[v["code"]] += 1

    manual = int((t["Exit name"].astype(str).str.strip().str.lower() == "manual").sum())
    return {
        "violations": violations,
        "counts": counts,
        "n_violating_trades": len(violations),
        "n_compliant_trades": int(len(t) - len(violations)),
        "manual_exits": {
            "count": manual,
            "note": "Info only — a manual exit is not a plan violation by itself.",
        },
    }
