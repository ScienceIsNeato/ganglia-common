"""Unit tests for the shared audio conversion helpers (ffmpeg is mocked)."""

import subprocess
from unittest.mock import patch

from ganglia_common.utils.audio_conversion import wav_to_mp3


def test_wav_to_mp3_success_returns_mp3_path():
    with patch("ganglia_common.utils.audio_conversion.subprocess.run") as run:
        result = wav_to_mp3("/tmp/scare.take.1.wav")

    assert result == "/tmp/scare.take.1.mp3"  # only the extension is swapped
    command = run.call_args.args[0]
    assert command[0] == "ffmpeg"
    assert "/tmp/scare.take.1.wav" in command
    assert command[-1] == "/tmp/scare.take.1.mp3"
    assert run.call_args.kwargs["check"] is True


def test_wav_to_mp3_failure_falls_back_to_wav_path():
    with patch(
        "ganglia_common.utils.audio_conversion.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    ):
        assert wav_to_mp3("/tmp/scare.wav") == "/tmp/scare.wav"
