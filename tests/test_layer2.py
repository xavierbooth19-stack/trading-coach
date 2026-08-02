"""Offline checks for Layer 2 (rules, presets, readout, AI path).

No network and no API key needed: the AI client is stubbed, and the
keyless path is tested explicitly (deterministic features must work
without any key).

Run:  python tests/test_layer2.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai_client
import ai_config
import rules as rules_mod
import strategy_ai

PASSED = 0


def ok(condition, label):
    global PASSED
    assert condition, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


def clear_keys():
    for var in ai_client.KEY_ENV_VARS + ("AI_MODEL",):
        os.environ.pop(var, None)
    ai_config.API_KEY = ""
    ai_config.AI_MODEL = ""


def test_schema_validation():
    print("rules: schema validation")
    clean, errors = rules_mod.normalize_rules({})
    ok(errors == [] and clean == rules_mod.DEFAULT_RULES, "empty input -> defaults, no errors")

    clean, errors = rules_mod.normalize_rules({
        "direction": "Long_only", "stop_pct": "0.4", "target_pct": "null",
        "session_start": "9:45", "session_end": "15:30",
        "max_size": 250.0, "exit_style": "TRAILING", "slippage_bps": 3,
    })
    ok(errors == [], "coerces strings, case, and null target")
    ok(clean["direction"] == "long_only" and clean["exit_style"] == "trailing",
       "direction/exit_style normalized")
    ok(clean["target_pct"] is None and clean["max_size"] == 250, "null target + int size")
    ok(clean["session_start"] == "09:45", "times normalized to HH:MM")

    _, errors = rules_mod.normalize_rules({
        "direction": "sideways", "stop_pct": -1, "session_start": "16:00",
        "session_end": "09:30", "max_size": 0, "exit_style": "vibes",
        "slippage_bps": "many",
    })
    ok(len(errors) == 6, f"bad fields each get an error ({len(errors)})")
    ok(all(isinstance(e, str) and e for e in errors), "errors are plain english lines")


def test_persistence_and_library(tmpdir):
    print("rules: persistence + named-strategy library")
    rules_path = os.path.join(tmpdir, "rules.json")
    lib_path = os.path.join(tmpdir, "strategies.json")

    ok(rules_mod.load_rules(rules_path) is None, "no rules.json yet -> None")
    saved = rules_mod.save_rules({"stop_pct": 0.3, "target_pct": 0.9}, path=rules_path)
    ok(rules_mod.load_rules(rules_path) == saved, "active rules round-trip")

    try:
        rules_mod.save_rules({"stop_pct": -5}, path=rules_path)
        ok(False, "saving invalid rules should raise")
    except ValueError as e:
        ok("stop_pct" in str(e), "invalid save -> clear error")

    rules_mod.save_strategy("My scalps", {"stop_pct": 0.2, "target_pct": 0.25}, path=lib_path)
    rules_mod.save_strategy("Trend days", {"exit_style": "trailing", "target_pct": None}, path=lib_path)
    ok(sorted(rules_mod.list_strategies(lib_path)) == ["My scalps", "Trend days"],
       "library lists saved strategies")
    reloaded = rules_mod.load_strategy("My scalps", path=lib_path)
    ok(reloaded["stop_pct"] == 0.2, "one-click reload returns the saved rules")
    ok(rules_mod.delete_strategy("My scalps", path=lib_path)
       and "My scalps" not in rules_mod.list_strategies(lib_path), "delete works")


def test_presets():
    print("rules: presets")
    ok(len(rules_mod.PRESETS) >= 12, f"a dozen presets ({len(rules_mod.PRESETS)})")
    for name in rules_mod.PRESETS:
        preset = rules_mod.get_preset(name)
        ok(set(preset) == set(rules_mod.DEFAULT_RULES), f"preset '{name}' fills the exact schema")
    ok(any(rules_mod.get_preset(n)["target_pct"] is None for n in rules_mod.PRESETS),
       "some presets have no fixed target")


def test_readout():
    print("rules: live readout")
    r = rules_mod.readout({"stop_pct": 0.5, "target_pct": 1.0})
    ok(r["reward_to_risk"] == 2.0, "R = target/stop")
    ok(abs(r["breakeven_win_rate"] - 1 / 3) < 1e-3, "breakeven win rate = 1/(1+R)")
    ok("2.0R" in r["text"] and "33%" in r["text"], "plain english mentions both numbers")

    r = rules_mod.readout({"target_pct": None, "exit_style": "trailing"})
    ok(r["reward_to_risk"] is None and "trailing" in r["text"],
       "no fixed target -> explained, no fake numbers")

    r = rules_mod.readout({"stop_pct": -2})
    ok(r["reward_to_risk"] is None and "Fix" in r["text"], "invalid input -> gentle nudge")


def test_provider_routing():
    print("ai_client: key discovery + provider routing")
    clear_keys()
    ok(ai_client.get_api_key() is None and not ai_client.is_configured(),
       "no key anywhere -> not configured")

    ok(ai_client.provider_for("sk-ant-abc123") == "anthropic", "sk-ant-... -> Anthropic")
    ok(ai_client.provider_for("sk-abc123") == "openai", "plain sk-... -> OpenAI")
    try:
        ai_client.provider_for("hunter2")
        ok(False, "bad key format should raise")
    except ai_client.AIClientError:
        ok(True, "unrecognized key format -> clear error")

    os.environ["AI_API_KEY"] = "sk-ant-envwins"
    ai_config.API_KEY = "sk-configloses"
    ok(ai_client.get_api_key() == "sk-ant-envwins", "environment beats ai_config.py")
    del os.environ["AI_API_KEY"]
    ok(ai_client.get_api_key() == "sk-configloses", "ai_config.py is the fallback")

    ok(ai_client.get_model("anthropic") == ai_client.DEFAULT_ANTHROPIC_MODEL
       and ai_client.get_model("openai") == ai_client.DEFAULT_OPENAI_MODEL,
       "per-provider default models")
    os.environ["AI_MODEL"] = "my-model-override"
    ok(ai_client.get_model("anthropic") == "my-model-override"
       and ai_client.get_model("openai") == "my-model-override",
       "AI_MODEL overrides either way")
    clear_keys()


def test_keyless_path():
    print("strategy_ai: keyless behavior")
    clear_keys()
    ok(not strategy_ai.is_available(), "AI path unavailable without a key")
    ok("Presets and manual entry work" in strategy_ai.unavailable_note(),
       "short note for the greyed-out UI")
    try:
        strategy_ai.draft_rules("Momentum", "I buy strength")
        ok(False, "draft without a key should raise")
    except strategy_ai.StrategyAIError as e:
        ok("no api key" in str(e).lower() or "no ai key" in str(e).lower(),
           "keyless draft -> clear error")


def test_ai_draft_and_approve(tmpdir):
    print("strategy_ai: draft -> review -> approve (stubbed model)")
    ok(len(strategy_ai.FAMILIES) >= 15, f"~15 trading families ({len(strategy_ai.FAMILIES)})")

    reply = json.dumps({
        "direction": "long_only", "stop_pct": 0.4, "target_pct": 1.2,
        "session_start": "09:30", "session_end": "11:00",
        "max_size": 300, "exit_style": "breakeven", "slippage_bps": 2,
    })
    original = ai_client.complete
    captured = {}

    def fake_complete(system, user, max_tokens=4096):
        captured["system"], captured["user"] = system, user
        return f"```json\n{reply}\n```"  # model misbehaves with fences; we cope

    os.environ["AI_API_KEY"] = "sk-ant-test"
    ai_client.complete = fake_complete
    try:
        rules, warnings = strategy_ai.draft_rules("Momentum", "I buy the first pullback after a strong open.")
    finally:
        ai_client.complete = original
        clear_keys()

    ok("STRICT" in captured["system"] or "EXACTLY one JSON" in captured["system"],
       "system prompt demands strict JSON")
    ok("Momentum" in captured["user"] and "pullback" in captured["user"],
       "family + playbook sent to the model")
    ok(warnings == [], "clean AI reply -> no warnings")
    ok(rules["exit_style"] == "breakeven" and rules["max_size"] == 300,
       "AI fields land in the editable form")

    # the user edits a field in the review form, then approves
    rules["max_size"] = 200
    rules_path = os.path.join(tmpdir, "rules.json")
    playbook_path = os.path.join(tmpdir, "playbook.json")
    saved = strategy_ai.approve(rules, "Momentum", "I buy the first pullback after a strong open.",
                                rules_path=rules_path, playbook_path=playbook_path)
    ok(saved["max_size"] == 200, "approve saves the edited rules")
    ok(rules_mod.load_rules(rules_path) == saved, "approve persists rules.json")
    playbook = strategy_ai.load_playbook(playbook_path)
    ok(playbook["locked"] and playbook["family"] == "Momentum"
       and "pullback" in playbook["playbook"], "playbook stored for the coach")

    # garbage AI replies fail with one clear line
    ai_client.complete = lambda s, u, max_tokens=4096: "Sure! Here are your rules: nice trades"
    os.environ["AI_API_KEY"] = "sk-ant-test"
    try:
        strategy_ai.draft_rules("Momentum", "whatever")
        ok(False, "non-JSON reply should raise")
    except strategy_ai.StrategyAIError as e:
        ok("not valid JSON" in str(e), "non-JSON reply -> clear error")
    finally:
        ai_client.complete = original
        clear_keys()

    # a reply wrapped in prose still parses (first balanced object wins)
    parsed = strategy_ai.parse_strict_json('Here you go: {"a": {"b": "}"}, "c": 1} thanks')
    ok(parsed == {"a": {"b": "}"}, "c": 1}, "JSON extracted from prose, braces in strings ok")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_schema_validation()
        test_persistence_and_library(tmpdir)
        test_presets()
        test_readout()
        test_provider_routing()
        test_keyless_path()
        test_ai_draft_and_approve(tmpdir)
    print(f"\nall {PASSED} checks passed")


if __name__ == "__main__":
    main()
