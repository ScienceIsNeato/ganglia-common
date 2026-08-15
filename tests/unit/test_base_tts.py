from types import SimpleNamespace
from unittest.mock import Mock, patch

from ganglia_common.tts.base_tts import TextToSpeech


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
