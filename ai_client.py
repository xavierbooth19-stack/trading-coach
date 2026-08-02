"""Minimal AI provider client.

Reads the key from the environment first, ai_config.py as fallback, and
routes by prefix: "sk-ant-..." goes to the Anthropic Messages API, plain
"sk-..." goes to the OpenAI chat completions API. AI_MODEL (env, then
ai_config) overrides the default model either way.

If there is NO key at all, is_configured() is False and the UI greys out
the plain-english path — presets and manual entry keep working, and
everything deterministic downstream runs fine keyless.
"""

from __future__ import annotations

import os

import requests

import ai_config

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

TIMEOUT_SECONDS = 90  # Claude models may think for a while before answering
KEY_ENV_VARS = ("AI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


class AIClientError(RuntimeError):
    """One clear line about what went wrong talking to the provider."""


def get_api_key():
    """Environment first, ai_config.py as fallback. None if nothing is set."""
    for var in KEY_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    value = (ai_config.API_KEY or "").strip()
    return value or None


def provider_for(key):
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    raise AIClientError(
        "Unrecognized API key format: expected 'sk-ant-...' (Anthropic) or 'sk-...' (OpenAI)"
    )


def get_model(provider):
    """AI_MODEL (env, then ai_config) overrides the default either way."""
    override = os.environ.get("AI_MODEL", "").strip() or (ai_config.AI_MODEL or "").strip()
    if override:
        return override
    return DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else DEFAULT_OPENAI_MODEL


def is_configured():
    """True when a usable key exists somewhere."""
    key = get_api_key()
    if not key:
        return False
    try:
        provider_for(key)
        return True
    except AIClientError:
        return False


def complete(system, user, max_tokens=4096):
    """One completion: system + user prompt in, response text out.

    Raises AIClientError with a single clear line on any failure
    (no key, timeout, HTTP error, refusal, malformed response).
    """
    key = get_api_key()
    if not key:
        raise AIClientError(
            "No AI key configured — set AI_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY), "
            "or fill in ai_config.py"
        )
    provider = provider_for(key)
    model = get_model(provider)
    if provider == "anthropic":
        return _complete_anthropic(key, model, system, user, max_tokens)
    return _complete_openai(key, model, system, user, max_tokens)


# -------------------------------------------------------------- providers


def _complete_anthropic(key, model, system, user, max_tokens):
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if model.startswith(("claude-opus-5", "claude-fable-5")):
        # Server-side fallback: if safety classifiers decline the request,
        # the API re-runs it on the recommended fallback model.
        headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
        payload["fallbacks"] = "default"

    data = _post(ANTHROPIC_URL, headers, payload, "Anthropic")

    if data.get("stop_reason") == "refusal":
        raise AIClientError("Anthropic declined this request (safety refusal) — rephrase and retry")
    for block in data.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            return block["text"]
    raise AIClientError("Anthropic returned no text content")


def _complete_openai(key, model, system, user, max_tokens):
    headers = {
        "Authorization": f"Bearer {key}",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = _post(OPENAI_URL, headers, payload, "OpenAI")

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AIClientError("OpenAI returned an unexpected response shape") from None
    if not text:
        raise AIClientError("OpenAI returned no text content")
    return text


def _post(url, headers, payload, provider_name):
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.Timeout:
        raise AIClientError(f"{provider_name} request timed out after {TIMEOUT_SECONDS}s") from None
    except requests.RequestException as exc:
        raise AIClientError(f"Could not reach {provider_name}: {exc}") from None

    if response.status_code != 200:
        raise AIClientError(f"{provider_name} error {response.status_code}: {_short_error(response)}")
    try:
        return response.json()
    except ValueError:
        raise AIClientError(f"{provider_name} returned a non-JSON response") from None


def _short_error(response):
    try:
        err = response.json().get("error", {})
        return str(err.get("message") or err or response.text)[:300]
    except ValueError:
        return response.text[:300]
