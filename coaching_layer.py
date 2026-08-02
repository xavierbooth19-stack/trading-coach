"""Layer 5 — the AI coach.

Builds a distilled findings JSON (a few KB, NEVER the price bars), sends
it to the model through Layer 2's provider client with a blunt-coach
system prompt, prints the report to the terminal, and returns the
markdown for the dashboard's Coach's Read panel.

Keyless mode: coach() raises ai_client.AIClientError with a clear line;
every deterministic feature runs without it.
"""

from __future__ import annotations

import json

import ai_client
import metrics
import strategy_ai

MAX_WORST_TRADES = 5

SYSTEM_PROMPT = """You are a blunt trading performance coach reviewing a deterministic audit of a real trade log.

Hard rules:
- Every claim you make must cite a number that appears in the findings JSON. Quote it.
- Do not invent, extrapolate, or round-trip numbers that are not in the findings.
- You cannot verify discretionary setup validity (ICT order blocks, fair value gaps, liquidity sweeps, supply/demand zones) from this data alone. If the trader's playbook leans on those, say exactly that rather than pretending to judge it.
- Respect the sample-size note: call out any conclusion that rests on thin data.
- Be direct and specific: name the behavior, the dollar cost, and the single change that fixes it.
- Write in markdown with a few short sections; no preamble.
- End with a one-line verdict on its own line, prefixed exactly "VERDICT:"."""


def build_payload(dash_path="dash_data.json", playbook_path=strategy_ai.PLAYBOOK_PATH):
    """The distilled findings dict sent to the coach. No bars, ever."""
    with open(dash_path, "r", encoding="utf-8") as fh:
        dash = json.load(fh)
    playbook = strategy_ai.load_playbook(playbook_path)

    profits = [t["profit"] for t in dash["trades"]]
    n = len(profits)
    se = metrics.std_error(profits)
    sqn = metrics.sqn(profits)

    replayed = [t for t in dash["trades"]
                if t["rule"]["kind"] not in ("not_taken", "no_data")
                and t["rule"]["difference"] is not None]
    worst = sorted(replayed, key=lambda t: -t["rule"]["difference"])[:MAX_WORST_TRADES]

    payload = {
        "sample_size_note": (
            f"{n} trades over {dash['meta']['sessions']} sessions. Standard error of the "
            f"per-trade mean is {'$%.0f' % se if se is not None else 'n/a'}; SQN "
            f"{'%.2f' % sqn if sqn is not None else 'n/a'}. Treat any split with fewer "
            f"than 10 trades as anecdote, not evidence."),
        "verdict_line": dash["verdict"],
        "headline": {k["label"]: k["value"] for k in dash["kpis"]},
        "quant": {s["label"]: s["value"] for s in dash["quant"]["stats"]},
        "quant_read": dash["quant"]["read"],
        "luck": {
            "pnl_without_top": {str(k): metrics.pnl_without_top(profits, k) for k in (1, 3, 5)},
            "top_decile_concentration_pct": metrics.top_decile_concentration_pct(profits),
        },
        "leaks_ranked_worst_first": dash["leaks"],
        "discipline": {
            "actual_net": dash["discipline"]["actual"],
            "rule_managed_net": dash["discipline"]["rule"],
            "difference_rule_minus_actual": dash["discipline"]["difference"],
            "trades_outside_plan_not_taken": dash["discipline"]["n_not_taken"],
            "rule_exit_kinds": dash["discipline"]["exit_kinds"],
        },
        "worst_rule_deviation_trades": [{
            "trade": t["n"], "instrument": t["instrument"], "date": t["date"],
            "side": t["side"], "qty": t["qty"], "actual_pnl": t["profit"],
            "rule_pnl": t["rule"]["pnl"], "gave_up_vs_rule": t["rule"]["difference"],
            "rule_exit_kind": t["rule"]["kind"], "flags": t["flags"],
        } for t in worst],
        "style_match": {
            "declared": dash["style"]["declared"],
            "observed_dominant": dash["style"]["observed_dominant"],
            "match": dash["style"]["match"],
            "banner": dash["style"]["banner"],
            "families_top": dash["style"]["families"][:5],
        },
        "trader_playbook_free_text": (playbook or {}).get("playbook", "(none provided)"),
        "declared_family": (playbook or {}).get("family", "(not declared)"),
    }
    assert "bars" not in payload
    return payload


def coach(dash_path="dash_data.json", playbook_path=strategy_ai.PLAYBOOK_PATH):
    """Run the coach: distill -> model -> print to terminal -> markdown.

    Raises ai_client.AIClientError (one clear line) when no key is set or
    the provider call fails.
    """
    payload = build_payload(dash_path, playbook_path)
    user = ("FINDINGS JSON (the only numbers you may cite):\n"
            + json.dumps(payload, indent=1)
            + "\n\nWrite the coaching report now.")
    markdown = ai_client.complete(SYSTEM_PROMPT, user, max_tokens=2000)

    print("\n" + "=" * 72)
    print("AI COACH'S READ")
    print("=" * 72)
    print(markdown)
    print("=" * 72 + "\n")
    return markdown
