"""Model selection for GANGLIA's LLM calls (env-overridable, rollback-safe)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ganglia_common import llm_settings
from ganglia_common.query_dispatch import ChatGPTQueryDispatcher


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "GANGLIA_LLM_MODEL",
        "GANGLIA_LLM_AUDIO_MODEL",
        "GANGLIA_LLM_REASONING_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_move_off_the_retiring_gpt_4o_family():
    # gpt-4o / 4o-mini / 4.1 / 4.1-mini retire 2027-01-14.
    assert llm_settings.text_model() == "gpt-6-luna"
    assert llm_settings.audio_model() == "gpt-audio-mini"
    assert llm_settings.reasoning_kwargs("gpt-6-luna") == {"reasoning_effort": "none"}


def test_models_and_effort_are_env_overridable(monkeypatch):
    monkeypatch.setenv("GANGLIA_LLM_MODEL", "gpt-6-sol")
    monkeypatch.setenv("GANGLIA_LLM_AUDIO_MODEL", "gpt-audio")
    monkeypatch.setenv("GANGLIA_LLM_REASONING_EFFORT", "low")
    assert llm_settings.text_model() == "gpt-6-sol"
    assert llm_settings.audio_model() == "gpt-audio"
    assert llm_settings.reasoning_kwargs("gpt-6-sol") == {"reasoning_effort": "low"}


def test_rolling_back_to_an_older_model_drops_reasoning_effort(monkeypatch):
    # Older chat models reject reasoning_effort, so a rollback must not send it.
    monkeypatch.setenv("GANGLIA_LLM_MODEL", "gpt-4o-mini")
    assert llm_settings.text_model() == "gpt-4o-mini"
    for model in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-audio-mini"):
        assert llm_settings.reasoning_kwargs(model) == {}


def test_empty_effort_omits_the_parameter(monkeypatch):
    monkeypatch.setenv("GANGLIA_LLM_REASONING_EFFORT", "")
    assert llm_settings.reasoning_kwargs("gpt-6-luna") == {}


def test_blank_model_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("GANGLIA_LLM_MODEL", "   ")
    assert llm_settings.text_model() == "gpt-6-luna"


def _capturing(qd):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Boo."))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    qd.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return seen


def test_dispatcher_sends_the_configured_model_and_effort():
    qd = ChatGPTQueryDispatcher()
    seen = _capturing(qd)
    qd.send_query("hello")
    assert seen["model"] == "gpt-6-luna"
    assert seen["reasoning_effort"] == "none"


def test_dispatcher_rollback_sends_no_effort(monkeypatch):
    monkeypatch.setenv("GANGLIA_LLM_MODEL", "gpt-4o-mini")
    qd = ChatGPTQueryDispatcher()
    seen = _capturing(qd)
    qd.send_query("hello")
    assert seen["model"] == "gpt-4o-mini"
    assert "reasoning_effort" not in seen
