"""Layer 2 — the strategy: structured rules the engine applies mechanically.

One JSON object describes a strategy:

    { direction, stop_pct, target_pct, session_start, session_end,
      max_size, exit_style, slippage_bps }

This module owns the schema (validation + defaults), persistence of the
active rules (rules.json), a named-strategy library (strategies.json),
a dozen one-click presets, and the plain-english live readout
(reward-to-risk + breakeven win rate) recomputed as the user types.
Everything here is deterministic and runs fine without any AI key.
"""

from __future__ import annotations

import datetime as dt
import json
import os

RULES_PATH = "rules.json"
STRATEGIES_PATH = "strategies.json"

DIRECTIONS = ("long_only", "short_only", "both")
EXIT_STYLES = ("fixed", "breakeven", "trailing", "time")

DEFAULT_RULES = {
    "direction": "long_only",
    "stop_pct": 0.5,
    "target_pct": 1.0,        # null = no fixed target
    "session_start": "09:30",
    "session_end": "16:00",   # "HH:MM" Eastern
    "max_size": 500,
    "exit_style": "fixed",    # fixed | breakeven | trailing | time
    "slippage_bps": 2,
    "price_unit": "percent",  # percent (equities) | points (futures)
    "daily_loss_limit": None, # $ — stop entering once the day is down this much (null = off)
}

PRICE_UNITS = ("percent", "points")

RULE_FIELDS = list(DEFAULT_RULES)


# ------------------------------------------------------------- validation


def normalize_rules(raw):
    """Coerce and validate a raw rules dict.

    Returns (rules, errors): a complete schema-shaped dict (missing fields
    filled from DEFAULT_RULES) and a list of plain-english error strings.
    Empty errors means the rules are good to run.
    """
    raw = dict(raw or {})
    rules = dict(DEFAULT_RULES)
    errors: list[str] = []

    direction = str(raw.get("direction", rules["direction"])).strip().lower()
    if direction in DIRECTIONS:
        rules["direction"] = direction
    else:
        errors.append(f"direction must be one of {', '.join(DIRECTIONS)} (got '{direction}')")

    unit = str(raw.get("price_unit", rules["price_unit"])).strip().lower()
    if unit in PRICE_UNITS:
        rules["price_unit"] = unit
    else:
        errors.append(f"price_unit must be one of {', '.join(PRICE_UNITS)} (got '{unit}')")
        unit = "percent"
    stop_cap = 50 if unit == "percent" else 10000       # points can be large (e.g. NQ)
    unit_word = "percent of entry price" if unit == "percent" else "price points"

    stop = _to_float(raw.get("stop_pct", rules["stop_pct"]))
    if stop is None or not 0 < stop <= stop_cap:
        errors.append(f"stop_pct must be a number between 0 and {stop_cap} ({unit_word})")
    else:
        rules["stop_pct"] = stop

    target_raw = raw.get("target_pct", rules["target_pct"])
    if target_raw is None or (isinstance(target_raw, str) and target_raw.strip().lower() in ("", "null", "none")):
        rules["target_pct"] = None
    else:
        target = _to_float(target_raw)
        if target is None or not 0 < target <= stop_cap * 2:
            errors.append(f"target_pct must be null (no fixed target) or a number between 0 and {stop_cap * 2}")
        else:
            rules["target_pct"] = target

    limit_raw = raw.get("daily_loss_limit", rules["daily_loss_limit"])
    if limit_raw is None or (isinstance(limit_raw, str) and limit_raw.strip().lower() in ("", "null", "none")):
        rules["daily_loss_limit"] = None
    else:
        limit = _to_float(limit_raw)
        if limit is None or limit <= 0:
            errors.append("daily_loss_limit must be null (off) or a positive dollar amount")
        else:
            rules["daily_loss_limit"] = limit

    start = _to_time(raw.get("session_start", rules["session_start"]))
    end = _to_time(raw.get("session_end", rules["session_end"]))
    if start is None:
        errors.append('session_start must look like "HH:MM" (24h Eastern), e.g. "09:30"')
    if end is None:
        errors.append('session_end must look like "HH:MM" (24h Eastern), e.g. "16:00"')
    if start is not None and end is not None:
        if start >= end:
            errors.append("session_start must be earlier than session_end")
        else:
            rules["session_start"] = start.strftime("%H:%M")
            rules["session_end"] = end.strftime("%H:%M")

    size = _to_float(raw.get("max_size", rules["max_size"]))
    if size is None or size < 1 or size != int(size):
        errors.append("max_size must be a whole number of shares/contracts, at least 1")
    else:
        rules["max_size"] = int(size)

    exit_style = str(raw.get("exit_style", rules["exit_style"])).strip().lower()
    if exit_style in EXIT_STYLES:
        rules["exit_style"] = exit_style
    else:
        errors.append(f"exit_style must be one of {', '.join(EXIT_STYLES)} (got '{exit_style}')")

    slip = _to_float(raw.get("slippage_bps", rules["slippage_bps"]))
    if slip is None or not 0 <= slip <= 100:
        errors.append("slippage_bps must be a number between 0 and 100 (default 2)")
    else:
        rules["slippage_bps"] = slip

    return rules, errors


def _to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # reject NaN


def _to_time(value):
    text = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


# ------------------------------------------------------------ persistence


def save_rules(rules, path=RULES_PATH):
    """Persist the active rules (validates first; raises on bad rules)."""
    clean, errors = normalize_rules(rules)
    if errors:
        raise ValueError("Cannot save rules: " + "; ".join(errors))
    _write_json(path, clean)
    return clean


def load_rules(path=RULES_PATH):
    """Load the active rules, or None if none have been saved yet."""
    if not os.path.exists(path):
        return None
    clean, errors = normalize_rules(_read_json(path))
    return None if errors else clean


# ------------------------------------------------- named-strategy library


def save_strategy(name, rules, path=STRATEGIES_PATH):
    """Save the current rules under a name for one-click reload later."""
    name = str(name).strip()
    if not name:
        raise ValueError("Strategy name cannot be empty")
    clean, errors = normalize_rules(rules)
    if errors:
        raise ValueError(f"Cannot save strategy '{name}': " + "; ".join(errors))
    library = _read_json(path) if os.path.exists(path) else {}
    library[name] = clean
    _write_json(path, library)
    return clean


def load_strategy(name, path=STRATEGIES_PATH):
    library = list_strategies(path)
    if name not in library:
        raise KeyError(f"No saved strategy named '{name}'")
    clean, errors = normalize_rules(library[name])
    if errors:
        raise ValueError(f"Saved strategy '{name}' is invalid: " + "; ".join(errors))
    return clean


def list_strategies(path=STRATEGIES_PATH):
    """The whole library as {name: rules}. Missing file = empty library."""
    if not os.path.exists(path):
        return {}
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def delete_strategy(name, path=STRATEGIES_PATH):
    library = list_strategies(path)
    if name in library:
        del library[name]
        _write_json(path, library)
        return True
    return False


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------- presets

# A dozen one-click templates. Each only pre-fills the same schema fields;
# everything stays editable in the form.
PRESETS = {
    "momentum": {
        "label": "Momentum",
        "blurb": "Ride short-term strength with a 2R target.",
        "rules": {"direction": "long_only", "stop_pct": 0.5, "target_pct": 1.0,
                  "session_start": "09:30", "session_end": "15:30",
                  "max_size": 500, "exit_style": "fixed", "slippage_bps": 2},
    },
    "mean_reversion": {
        "label": "Mean reversion",
        "blurb": "Fade stretched moves back to the average; quick exits.",
        "rules": {"direction": "both", "stop_pct": 0.6, "target_pct": 0.6,
                  "session_start": "10:00", "session_end": "15:30",
                  "max_size": 400, "exit_style": "fixed", "slippage_bps": 2},
    },
    "breakout": {
        "label": "Breakout",
        "blurb": "Buy new highs, let winners run with a trailing stop.",
        "rules": {"direction": "long_only", "stop_pct": 0.7, "target_pct": None,
                  "session_start": "09:30", "session_end": "15:45",
                  "max_size": 400, "exit_style": "trailing", "slippage_bps": 3},
    },
    "trend": {
        "label": "Trend following",
        "blurb": "Enter with the intraday trend, trail until it bends.",
        "rules": {"direction": "both", "stop_pct": 0.8, "target_pct": None,
                  "session_start": "09:45", "session_end": "15:45",
                  "max_size": 300, "exit_style": "trailing", "slippage_bps": 2},
    },
    "scalp": {
        "label": "Scalp",
        "blurb": "Tiny, fast trades; tight stop, tight target.",
        "rules": {"direction": "both", "stop_pct": 0.15, "target_pct": 0.2,
                  "session_start": "09:30", "session_end": "16:00",
                  "max_size": 1000, "exit_style": "fixed", "slippage_bps": 4},
    },
    "pullback": {
        "label": "Pullback",
        "blurb": "Buy the dip inside an uptrend; move stop to breakeven fast.",
        "rules": {"direction": "long_only", "stop_pct": 0.5, "target_pct": 1.2,
                  "session_start": "09:45", "session_end": "15:30",
                  "max_size": 500, "exit_style": "breakeven", "slippage_bps": 2},
    },
    "fade": {
        "label": "Fade",
        "blurb": "Counter-trend: short the euphoric spike, buy the panic flush.",
        "rules": {"direction": "both", "stop_pct": 0.7, "target_pct": 1.0,
                  "session_start": "09:30", "session_end": "11:30",
                  "max_size": 300, "exit_style": "fixed", "slippage_bps": 3},
    },
    "opening_range": {
        "label": "Opening range breakout",
        "blurb": "Trade the break of the first 30 minutes' range.",
        "rules": {"direction": "both", "stop_pct": 0.4, "target_pct": 0.9,
                  "session_start": "10:00", "session_end": "12:00",
                  "max_size": 500, "exit_style": "fixed", "slippage_bps": 2},
    },
    "vwap_reversion": {
        "label": "VWAP reversion",
        "blurb": "Fade extensions away from VWAP back to the line.",
        "rules": {"direction": "both", "stop_pct": 0.5, "target_pct": 0.5,
                  "session_start": "10:00", "session_end": "15:30",
                  "max_size": 500, "exit_style": "fixed", "slippage_bps": 2},
    },
    "gap_and_go": {
        "label": "Gap and go",
        "blurb": "Trade the morning gap continuation; done by lunch.",
        "rules": {"direction": "long_only", "stop_pct": 0.8, "target_pct": 1.6,
                  "session_start": "09:30", "session_end": "11:00",
                  "max_size": 400, "exit_style": "fixed", "slippage_bps": 4},
    },
    "range": {
        "label": "Range trading",
        "blurb": "Buy support, sell resistance while the tape chops sideways.",
        "rules": {"direction": "both", "stop_pct": 0.35, "target_pct": 0.7,
                  "session_start": "11:00", "session_end": "15:00",
                  "max_size": 500, "exit_style": "fixed", "slippage_bps": 2},
    },
    "power_hour": {
        "label": "Power hour",
        "blurb": "Trade the late-day push into the close; time-based exit.",
        "rules": {"direction": "both", "stop_pct": 0.4, "target_pct": None,
                  "session_start": "15:00", "session_end": "16:00",
                  "max_size": 400, "exit_style": "time", "slippage_bps": 3},
    },
}


def get_preset(name):
    """A preset's full rules dict (validated), ready to pre-fill the form."""
    if name not in PRESETS:
        raise KeyError(f"No preset named '{name}' (have: {', '.join(PRESETS)})")
    clean, errors = normalize_rules(PRESETS[name]["rules"])
    if errors:  # presets are code; this should never happen
        raise ValueError(f"Preset '{name}' is invalid: " + "; ".join(errors))
    return clean


# ------------------------------------------------------------ live readout


def readout(rules):
    """The live numbers under the form, in plain english.

    Returns {reward_to_risk, breakeven_win_rate, text}. The first two are
    None when there is no fixed target. Recompute on every keystroke —
    it's just arithmetic.
    """
    clean, errors = normalize_rules(rules)
    if errors:
        return {"reward_to_risk": None, "breakeven_win_rate": None,
                "text": "Fix the highlighted fields to see reward-to-risk."}

    stop, target = clean["stop_pct"], clean["target_pct"]
    unit = "%" if clean["price_unit"] == "percent" else " points"
    if target is None:
        style = clean["exit_style"]
        how = {
            "trailing": "a trailing stop follows the move",
            "breakeven": "the stop moves to breakeven and the trade runs",
            "time": "the trade exits after its time window",
            "fixed": "exits are discretionary",
        }[style]
        return {
            "reward_to_risk": None,
            "breakeven_win_rate": None,
            "text": (f"No fixed target — {how}. You risk {stop:g}{unit} per trade; "
                     "reward depends on how far each winner runs, so there is "
                     "no fixed breakeven win rate."),
        }

    r = target / stop
    breakeven = 1.0 / (1.0 + r)
    return {
        "reward_to_risk": round(r, 2),
        "breakeven_win_rate": round(breakeven, 4),
        "text": (f"Reward-to-risk: {r:.1f}R — you risk {stop:g}{unit} to make {target:g}{unit}. "
                 f"Breakeven win rate: {breakeven:.0%} — win more often than that "
                 "(before costs) and this profile makes money."),
    }
