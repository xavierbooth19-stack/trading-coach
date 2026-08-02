"""Layer 2 — the plain-english path (the purple AI step).

The user picks a trading family from ~15 choices and types how they
trade in their own words. Family + text go to the model with a system
prompt that demands STRICT JSON matching the rules schema — no prose,
no backticks. We parse it, render every field into an EDITABLE review
form, and nothing runs until the user clicks approve. Approving locks
and saves the rules, and the free-text playbook is stored too — the
coach gets it later for context. This step happens once per strategy.

Keyless: is_available() is False, the UI greys the path out with
unavailable_note(), and presets + manual entry still work.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import ai_client
import rules as rules_mod

PLAYBOOK_PATH = "playbook.json"

FAMILIES = [
    "Momentum",
    "Mean reversion",
    "Breakout",
    "Trend following",
    "Pullback",
    "Scalping",
    "Swing",
    "Day trading",
    "Opening range",
    "VWAP trading",
    "ICT / price action",
    "Order flow",
    "Gap trading",
    "Fade / contrarian",
    "News / event",
]

SYSTEM_PROMPT = """You convert a trader's plain-english description of how they trade into one strict JSON object of mechanical rules.

Output EXACTLY one JSON object and NOTHING else: no prose, no explanations, no markdown, no backticks.

The object must have exactly these keys:
- "direction": "long_only" | "short_only" | "both"
- "stop_pct": number > 0 — stop distance as a percent of entry price (e.g. 0.5 means 0.5%)
- "target_pct": number > 0, or null when the trader has no fixed target (trails, scales out, exits on time)
- "session_start": "HH:MM" 24-hour US/Eastern (e.g. "09:30")
- "session_end": "HH:MM" 24-hour US/Eastern (e.g. "16:00")
- "max_size": integer >= 1 — the largest position size in shares/contracts
- "exit_style": "fixed" | "breakeven" | "trailing" | "time"
- "slippage_bps": number >= 0 — assumed slippage in basis points (use 2 if the trader doesn't say)
- "price_unit": "percent" | "points" — "points" when the trader thinks in futures points/ticks (ES, NQ, MES...); then stop_pct/target_pct are point distances, not percents
- "daily_loss_limit": positive dollar amount, or null — set it when the trader mentions a daily loss limit / max daily drawdown (common for prop-firm evaluations like Topstep)

Rules for filling values:
- Infer sensible values from the description plus the stated trading family; when the trader is silent on a field, choose a typical value for that family.
- Futures traders describing stops in points or ticks get price_unit "points" with the distances in points (4 ticks on ES = 1 point). Percent-thinking equity traders get "percent"; convert dollars to percent when the entry price makes that possible.
- Times like "the first hour" mean 09:30-10:30 Eastern. "The open" starts 09:30. "The close" ends 16:00.
- If the trader only goes long, "long_only"; only shorts, "short_only"; otherwise "both".
- "trailing" when they ride winners with a moving stop; "breakeven" when they move the stop to entry; "time" when exits are on the clock; else "fixed"."""


class StrategyAIError(RuntimeError):
    """One clear line about why the plain-english step failed."""


def is_available():
    return ai_client.is_configured()


def unavailable_note():
    return ("AI strategy setup is off — no API key found. Add AI_API_KEY (or "
            "ANTHROPIC_API_KEY / OPENAI_API_KEY), or fill in ai_config.py. "
            "Presets and manual entry work fine without it.")


def draft_rules(family, playbook_text):
    """Family + free text -> (rules, warnings) for the editable review form.

    The rules come back schema-complete and validated; warnings list any
    fields the model got wrong that were replaced with defaults (they'll
    be visible and editable in the form either way).
    """
    if family not in FAMILIES:
        raise StrategyAIError(f"Unknown trading family '{family}'")
    playbook_text = str(playbook_text or "").strip()
    if not playbook_text:
        raise StrategyAIError("Describe how you trade first — a sentence or two is enough")
    if not is_available():
        raise StrategyAIError(unavailable_note())

    user = f"Trading family: {family}\n\nHow I trade, in my own words:\n{playbook_text}"
    try:
        raw_text = ai_client.complete(SYSTEM_PROMPT, user)
    except ai_client.AIClientError as exc:
        raise StrategyAIError(str(exc)) from None

    parsed = parse_strict_json(raw_text)
    rules, errors = rules_mod.normalize_rules(parsed)
    warnings = [f"AI value replaced with default — {e}" for e in errors]
    return rules, warnings


def approve(rules, family, playbook_text,
            rules_path=rules_mod.RULES_PATH, playbook_path=PLAYBOOK_PATH):
    """The user clicked approve: lock and save the (possibly edited) rules.

    Persists rules.json and stores the free-text playbook + family in
    playbook.json for the coach to use later. Raises ValueError with one
    clear line if the edited rules don't validate.
    """
    saved = rules_mod.save_rules(rules, path=rules_path)
    playbook = {
        "family": family,
        "playbook": str(playbook_text or "").strip(),
        "rules": saved,
        "approved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "locked": True,
    }
    with open(playbook_path, "w", encoding="utf-8") as fh:
        json.dump(playbook, fh, indent=2)
        fh.write("\n")
    return saved


def load_playbook(path=PLAYBOOK_PATH):
    """The stored playbook for the coach, or None if never approved."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def parse_strict_json(text):
    """Parse the model's reply, tolerating stray fences or prose.

    The prompt demands bare JSON, but parse defensively: strip markdown
    fences, then fall back to the first balanced {...} block.
    """
    text = str(text or "").strip()
    fenced = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    for candidate in (text, fenced, _first_json_object(fenced or text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise StrategyAIError("The AI reply was not valid JSON — try rewording your description")


def _first_json_object(text):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
