import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ganglia_common.tts.openai_tts import OpenAITTS
from ganglia_common.tts.types import Voice


def make_openai_tts() -> OpenAITTS:
    tts = object.__new__(OpenAITTS)
    tts.voice = "onyx"
    return tts


def test_streaming_single_sentence_uses_chunking_path():
    tts = make_openai_tts()
    tts._convert_with_chunking = Mock(return_value=(True, "audio.mp3"))

    assert tts.convert_text_to_speech_streaming(["sentence"], voice_id="nova") == (
        True,
        "audio.mp3",
    )
    tts._convert_with_chunking.assert_called_once_with("sentence", voice_id="nova")


def test_oversized_text_without_chunks_fails_cleanly():
    tts = make_openai_tts()
    tts.split_text = Mock(return_value=[])
    tts._concatenate_audio_files = Mock()

    assert tts._convert_with_chunking("x" * 4001) == (False, None)
    tts._concatenate_audio_files.assert_not_called()


def test_openai_tts_initializes_with_supported_fallback_voice():
    with patch("ganglia_common.tts.openai_tts.OpenAI"), patch(
        "ganglia_common.tts.openai_tts.Logger.print_info"
    ):
        assert OpenAITTS("unsupported").voice == "onyx"


def test_convert_impl_rejects_empty_text_and_writes_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("GANGLIA_TEMP_DIR", str(tmp_path))
    tts = make_openai_tts()
    tts.model = "tts-1"
    response = SimpleNamespace(stream_to_file=Mock())
    tts.client = SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(create=Mock(return_value=response))
        )
    )

    assert tts._convert_text_to_speech_impl(" ") == (False, None)
    success, file_path = tts._convert_text_to_speech_impl(
        "hello", voice_id="nova", thread_id="worker"
    )

    assert success is True
    assert file_path.endswith(".mp3")
    response.stream_to_file.assert_called_once_with(file_path)


def test_convert_impl_reports_sdk_errors():
    tts = make_openai_tts()
    tts.model = "tts-1"
    tts.client = SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(create=Mock(side_effect=RuntimeError("offline")))
        )
    )
    assert tts._convert_text_to_speech_impl("hello") == (False, None)


def test_convert_with_chunking_concatenates_audio():
    tts = make_openai_tts()
    tts.split_text = Mock(return_value=["first", "second"])
    tts._convert_text_to_speech_impl = Mock(
        side_effect=[(True, "first.mp3"), (True, "second.mp3")]
    )
    tts._concatenate_audio_files = Mock(return_value="combined.mp3")

    assert tts._convert_with_chunking("x" * 4001) == (True, "combined.mp3")
    tts._concatenate_audio_files.assert_called_once_with(["first.mp3", "second.mp3"])


def test_convert_uses_voice_and_parallel_streaming_handles_failure():
    tts = make_openai_tts()
    tts._convert_with_chunking = Mock(return_value=(True, "audio.mp3"))
    voice = Voice(engine="google", name="voice", id="nova")
    assert tts.convert_text_to_speech("hello", voice, "thread") == (True, "audio.mp3")
    tts._convert_with_chunking.assert_called_once_with(
        "hello", voice_id="nova", thread_id="thread"
    )

    tts._convert_text_to_speech_impl = Mock(
        side_effect=[(True, "first.mp3"), (False, None)]
    )
    assert tts.convert_text_to_speech_streaming(["one", "two"]) == (False, None)
    assert tts.convert_text_to_speech_streaming([]) == (False, None)


def test_concatenate_audio_files_returns_output_or_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("GANGLIA_TEMP_DIR", str(tmp_path))
    (tmp_path / "tts").mkdir()
    first = tmp_path / "first.mp3"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    tts = make_openai_tts()

    with patch("ganglia_common.tts.openai_tts.subprocess.run"):
        output = tts._concatenate_audio_files([str(first), str(second)])
    assert output.endswith(".mp3")
    assert first.exists() is False
    assert second.exists() is False

    with patch(
        "ganglia_common.tts.openai_tts.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    ):
        assert tts._concatenate_audio_files(["fallback.mp3"]) == "fallback.mp3"
