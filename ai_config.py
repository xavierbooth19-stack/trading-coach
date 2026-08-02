"""Fallback AI configuration.

The environment is checked first (AI_API_KEY, ANTHROPIC_API_KEY,
OPENAI_API_KEY, AI_MODEL); values here are used only when the
environment has nothing. Fine to leave empty — everything except the
plain-english strategy path works without a key.
"""

# "sk-ant-..." routes to Anthropic, plain "sk-..." routes to OpenAI.
API_KEY = ""

# Optional model override for either provider, e.g. "claude-opus-5"
# or "gpt-4o-mini". Empty = provider default.
AI_MODEL = ""
