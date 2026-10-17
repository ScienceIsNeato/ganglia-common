"""Which OpenAI models GANGLIA's LLM calls use, and how to call them.

One place to change (or roll back) the model for every caller -- the
conversation dispatcher here, plus ganglia-core's quest classifier and
voice-clone text -- via environment variables:

- ``GANGLIA_LLM_MODEL``: text model (default ``gpt-6-luna``).
- ``GANGLIA_LLM_AUDIO_MODEL``: model for the experimental audio-output mode
  (default ``gpt-audio-mini``).
- ``GANGLIA_LLM_REASONING_EFFORT``: reasoning effort for models that support it
  (default ``none``, OpenAI's recommended starting point when migrating off
  GPT-4.x). Set it empty to omit the parameter.

OpenAI retires gpt-4o / gpt-4o-mini / gpt-4.1 / gpt-4.1-mini on 2027-01-14.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_TEXT_MODEL = "gpt-6-luna"
DEFAULT_AUDIO_MODEL = "gpt-audio-mini"
DEFAULT_REASONING_EFFORT = "none"

# Older chat models reject ``reasoning_effort`` outright; sending it only to
# models that take it keeps GANGLIA_LLM_MODEL=gpt-4o-mini a working rollback.
_NO_REASONING_PREFIXES = ("gpt-3", "gpt-4", "chatgpt-4o", "gpt-audio")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None else value.strip()


def text_model() -> str:
    """The model for normal text replies."""
    return _env("GANGLIA_LLM_MODEL", DEFAULT_TEXT_MODEL) or DEFAULT_TEXT_MODEL


def audio_model() -> str:
    """The model for audio-output replies (text plus spoken audio)."""
    return _env("GANGLIA_LLM_AUDIO_MODEL", DEFAULT_AUDIO_MODEL) or DEFAULT_AUDIO_MODEL


def supports_reasoning(model: str) -> bool:
    """True if ``model`` accepts the ``reasoning_effort`` parameter."""
    return not model.startswith(_NO_REASONING_PREFIXES)


def reasoning_kwargs(model: str) -> dict[str, Any]:
    """Extra keyword arguments for a chat-completions call to ``model``.

    With reasoning effort ``none``, GPT-6-family models also accept
    ``temperature`` values other than the default, which callers like the
    quest classifier (``temperature=0``) rely on.
    """
    effort = _env("GANGLIA_LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    if not effort or not supports_reasoning(model):
        return {}
    return {"reasoning_effort": effort}
