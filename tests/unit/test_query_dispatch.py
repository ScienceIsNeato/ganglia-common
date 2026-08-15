from types import SimpleNamespace

import pytest

from ganglia_common.query_dispatch import ChatGPTQueryDispatcher


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_load_git_repo_into_history():
    dispatcher = ChatGPTQueryDispatcher(pre_prompt="Test pre-prompt")
    token_count = dispatcher.count_tokens()

    assert isinstance(token_count, int)
    assert token_count > 0


def test_query_dispatcher_init():
    """Test that the query dispatcher initializes correctly."""
    dispatcher = ChatGPTQueryDispatcher()
    assert dispatcher.client is not None
    assert dispatcher.messages == []

    # Test with pre_prompt
    pre_prompt = "You are a helpful assistant."
    dispatcher = ChatGPTQueryDispatcher(pre_prompt=pre_prompt)
    assert dispatcher.messages == [{"role": "system", "content": pre_prompt}]


def test_dispatcher_context_history_and_filtering():
    dispatcher = ChatGPTQueryDispatcher()
    dispatcher.add_system_context(["one", "two"])
    assert dispatcher.count_tokens() == 2

    dispatcher.messages = [{"role": "user", "content": "word " * 4100}]
    dispatcher.rotate_session_history()
    assert dispatcher.messages == []

    dispatcher.send_query = lambda _: "rewritten story"
    assert dispatcher.filter_content_for_dalle("story") == (True, "rewritten story")
    assert "Story to rewrite:\nstory" in dispatcher._get_dalle_filter_prompt("story")


def test_dispatcher_filtering_reports_final_failure():
    dispatcher = ChatGPTQueryDispatcher()
    dispatcher.send_query = lambda _: (_ for _ in ()).throw(ValueError("blocked"))
    assert dispatcher.filter_content_for_dalle("story", max_attempts=2) == (False, None)


def test_dispatcher_streams_complete_sentences(tmp_path, monkeypatch):
    monkeypatch.setenv("GANGLIA_TEMP_DIR", str(tmp_path))
    dispatcher = ChatGPTQueryDispatcher()
    dispatcher.client.chat.completions.create = lambda **_: iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello. "))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Goodbye"))]
            ),
        ]
    )

    assert list(dispatcher.send_query_streaming("start")) == ["Hello.", "Goodbye"]
    assert dispatcher.messages[-1] == {
        "role": "assistant",
        "content": "Hello. Goodbye",
    }


def test_prepare_messages_injects_extra_context_without_storing_it():
    dispatcher = ChatGPTQueryDispatcher(pre_prompt="Be spooky.")

    messages_for_call = dispatcher._prepare_messages(
        "hello", extra_system_context="Active quest: trinket hunt"
    )

    # Injected just before the user turn, for this call only
    assert messages_for_call[-2:] == [
        {"role": "system", "content": "Active quest: trinket hunt"},
        {"role": "user", "content": "hello"},
    ]
    # ...and never persisted into session history
    assert dispatcher.messages == [
        {"role": "system", "content": "Be spooky."},
        {"role": "user", "content": "hello"},
    ]


def test_dispatcher_audio_falls_back_when_audio_is_missing():
    dispatcher = ChatGPTQueryDispatcher(audio_output=True)
    dispatcher.client.chat.completions.create = lambda **_: SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="fallback", audio=None))
        ]
    )
    assert dispatcher.send_query("start") == "fallback"


def test_dispatcher_audio_fallback_substitutes_placeholder_for_empty_reply():
    dispatcher = ChatGPTQueryDispatcher(audio_output=True)
    dispatcher.client.chat.completions.create = lambda **_: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, audio=None))]
    )

    assert dispatcher.send_query("start") == "[No response received]"
    assert dispatcher.messages[-1] == {
        "role": "assistant",
        "content": "[No response received]",
    }


def test_dispatcher_audio_writes_wav(tmp_path, monkeypatch):
    monkeypatch.setenv("GANGLIA_TEMP_DIR", str(tmp_path))
    dispatcher = ChatGPTQueryDispatcher(audio_output=True)
    dispatcher.client.chat.completions.create = lambda **_: SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="spoken",
                    audio=SimpleNamespace(data="d2F2", transcript="spoken"),
                )
            )
        ]
    )

    reply, audio_file = dispatcher.send_query("start")

    assert reply == "spoken"
    assert open(audio_file, "rb").read() == b"wav"


# Test removed - send_merged_query method does not exist in the codebase
# The ChatGPTQueryDispatcher maintains conversation history internally via session_history
# and doesn't expose a merged query method
