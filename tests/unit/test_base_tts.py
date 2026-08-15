import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ganglia_common.tts import base_tts
from ganglia_common.tts.base_tts import TextToSpeech, stop_active_playback


class DummyTTS(TextToSpeech):
    def convert_text_to_speech(self, text, voice=None, thread_id=None):
        return True, text


def test_split_text_and_local_filepath_detection():
    tts = DummyTTS()
    assert tts.split_text("First sentence. Second sentence!", max_length=10) == [
        "First sent",
        "ence.",
        "Second se",
        "ntence!",
    ]
    assert tts.is_local_filepath("https://example.com/audio.mp3") is True
    assert tts.is_local_filepath("audio.mp3") is False


def test_prepare_playback_and_audio_duration():
    tts = DummyTTS()
    completed = SimpleNamespace(stdout=b"1.5\n")
    with patch("ganglia_common.tts.base_tts.subprocess.run", return_value=completed):
        assert tts.get_audio_duration("audio.mp3") == 1.5
        assert tts.prepare_playback("audio.mp4") == (
            ["ffplay", "-nodisp", "-autoexit", "audio.mp4"],
            1.5,
        )


def test_concatenate_audio_from_text():
    tts = DummyTTS()
    with patch("ganglia_common.tts.base_tts.subprocess.run") as run:
        assert tts.concatenate_audio_from_text("files.txt") == "combined_audio.mp3"
    run.assert_called_once()


def test_play_speech_response_starts_playback(monkeypatch):
    tts = DummyTTS()
    monkeypatch.setenv("PLAYBACK_MEDIA_IN_TESTS", "true")
    process = Mock()  # hashable, like a real Popen — base_tts tracks it in a set
    tts.prepare_playback = Mock(return_value=(["ffplay"], 1.0))

    with patch("ganglia_common.tts.base_tts.subprocess.Popen", return_value=process):
        tts.play_speech_response("audio.mp3", "hello")

    process.wait.assert_called_once()


@pytest.fixture
def clean_playback_registry():
    """Keep the module-level playback registry isolated per test."""
    with base_tts._playback_lock:
        saved = set(base_tts._active_playback_processes)
        base_tts._active_playback_processes.clear()
    yield base_tts._active_playback_processes
    with base_tts._playback_lock:
        base_tts._active_playback_processes.clear()
        base_tts._active_playback_processes.update(saved)


def test_stop_active_playback_terminates_live_processes_and_clears_registry(
    clean_playback_registry,
):
    live = Mock()
    live.poll.return_value = None  # still running
    exited = Mock()
    exited.poll.return_value = 0  # already finished
    clean_playback_registry.update({live, exited})

    stop_active_playback()

    live.terminate.assert_called_once()
    live.wait.assert_called_once_with(timeout=0.3)
    live.kill.assert_not_called()
    exited.terminate.assert_not_called()  # nothing to stop
    assert clean_playback_registry == set()


def test_stop_active_playback_escalates_to_kill_when_wait_times_out(
    clean_playback_registry,
):
    stubborn = Mock()
    stubborn.poll.return_value = None
    stubborn.wait.side_effect = subprocess.TimeoutExpired(cmd="ffplay", timeout=0.3)
    clean_playback_registry.add(stubborn)

    stop_active_playback()

    stubborn.terminate.assert_called_once()
    stubborn.kill.assert_called_once()
    assert clean_playback_registry == set()


def test_stop_active_playback_swallows_terminate_and_kill_errors(
    clean_playback_registry,
):
    hostile = Mock()
    hostile.poll.return_value = None
    hostile.terminate.side_effect = OSError("process vanished")
    hostile.wait.side_effect = subprocess.TimeoutExpired(cmd="ffplay", timeout=0.3)
    hostile.kill.side_effect = OSError("still vanished")
    clean_playback_registry.add(hostile)

    stop_active_playback()  # best-effort: must not raise

    hostile.terminate.assert_called_once()
    hostile.kill.assert_called_once()
    assert clean_playback_registry == set()
