"""Unit tests for GoogleTTS tunable effects and voice-id handling.

The Google client is patched out everywhere, so these run with no credentials
and no network.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ganglia_common.tts.google_tts import GoogleTTS
from ganglia_common.tts.types import Voice


@pytest.fixture
def google_tts():
    """A GoogleTTS whose gRPC client is a Mock that returns fake MP3 bytes."""
    with patch("ganglia_common.tts.google_tts.tts.TextToSpeechClient") as client_cls:
        client_cls.return_value.synthesize_speech.return_value = SimpleNamespace(
            audio_content=b"fake-mp3-bytes"
        )
        yield GoogleTTS()


def test_init_applies_menacing_defaults(google_tts):
    assert google_tts.apply_effects is True
    assert google_tts.pitch == GoogleTTS.DEFAULT_PITCH
    assert google_tts.speaking_rate == GoogleTTS.DEFAULT_SPEAKING_RATE


def test_set_effects_clamps_to_google_api_ranges(google_tts):
    google_tts.set_effects(pitch=-999, speaking_rate=100)
    assert google_tts.pitch == GoogleTTS.PITCH_RANGE[0]
    assert google_tts.speaking_rate == GoogleTTS.SPEAKING_RATE_RANGE[1]

    google_tts.set_effects(pitch=999, speaking_rate=0.0)
    assert google_tts.pitch == GoogleTTS.PITCH_RANGE[1]
    assert google_tts.speaking_rate == GoogleTTS.SPEAKING_RATE_RANGE[0]

    google_tts.set_effects(pitch=5.5, speaking_rate=1.25)
    assert google_tts.pitch == 5.5
    assert google_tts.speaking_rate == 1.25


def test_set_effects_none_keeps_current_value_and_enables_effects(google_tts):
    google_tts.apply_effects = False
    google_tts.set_effects(pitch=3.0)
    assert google_tts.pitch == 3.0
    assert google_tts.speaking_rate == GoogleTTS.DEFAULT_SPEAKING_RATE  # untouched
    assert google_tts.apply_effects is True  # customizing implicitly enables

    google_tts.set_effects(speaking_rate=2.0)
    assert google_tts.pitch == 3.0  # untouched this time
    assert google_tts.speaking_rate == 2.0


def test_get_effects_reflects_settings_ranges_and_defaults(google_tts):
    google_tts.set_effects(pitch=-4.0, speaking_rate=1.5)
    effects = google_tts.get_effects()
    assert effects["apply_effects"] is True
    assert effects["pitch"] == -4.0
    assert effects["speaking_rate"] == 1.5
    assert effects["pitch_range"] == list(GoogleTTS.PITCH_RANGE)
    assert effects["speaking_rate_range"] == list(GoogleTTS.SPEAKING_RATE_RANGE)
    assert effects["defaults"] == {
        "pitch": GoogleTTS.DEFAULT_PITCH,
        "speaking_rate": GoogleTTS.DEFAULT_SPEAKING_RATE,
    }


def _synthesized_voice_name(google_tts):
    call = google_tts._client.synthesize_speech.call_args
    return call.kwargs["voice"].name


@pytest.mark.parametrize(
    "voice,expected_name",
    [
        (None, "en-US-Wavenet-D"),
        (Voice(engine="google", name="Empty", id=None), "en-US-Wavenet-D"),
        (Voice(engine="google", name="Legacy", id="en-US-Casual-K"), "en-US-Wavenet-D"),
        (Voice(engine="google", name="Real", id="en-US-Neural2-F"), "en-US-Neural2-F"),
    ],
)
def test_voice_id_defaulting(google_tts, voice, expected_name, tmp_path, monkeypatch):
    monkeypatch.setenv("GANGLIA_TEMP_DIR", str(tmp_path))
    success, file_path = google_tts._convert_text_to_speech_impl(
        "Hello (from) the/crypt", voice
    )
    assert success is True
    assert file_path is not None
    assert open(file_path, "rb").read() == b"fake-mp3-bytes"
    assert _synthesized_voice_name(google_tts) == expected_name
    # Filename snippet has slashes/parens sanitized away
    assert "/crypt" not in file_path.rsplit("/", 1)[-1]


def test_convert_text_to_speech_defaults_voice_when_none(
    google_tts, tmp_path, monkeypatch
):
    monkeypatch.setenv("GANGLIA_TEMP_DIR", str(tmp_path))
    success, file_path = google_tts.convert_text_to_speech("Rise from the coffin")
    assert success is True
    assert file_path is not None
    assert _synthesized_voice_name(google_tts) == "en-US-Wavenet-D"


def test_effects_toggle_changes_audio_config(google_tts, tmp_path, monkeypatch):
    monkeypatch.setenv("GANGLIA_TEMP_DIR", str(tmp_path))
    voice = Voice(engine="google", name="Real", id="en-US-Neural2-F")

    google_tts.set_effects(pitch=-10.0, speaking_rate=0.5)
    google_tts._convert_text_to_speech_impl("spooky", voice)
    with_effects = google_tts._client.synthesize_speech.call_args.kwargs["audio_config"]
    assert with_effects.pitch == -10.0
    assert with_effects.speaking_rate == 0.5

    google_tts.apply_effects = False
    google_tts._convert_text_to_speech_impl("plain", voice)
    without_effects = google_tts._client.synthesize_speech.call_args.kwargs[
        "audio_config"
    ]
    assert without_effects.pitch == 0.0  # proto default: no effect applied
    assert without_effects.speaking_rate == 0.0
