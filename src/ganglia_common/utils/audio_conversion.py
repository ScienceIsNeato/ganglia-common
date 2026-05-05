"""Shared audio conversion helpers."""

from __future__ import annotations

import subprocess

from ganglia_common.logger import Logger


def wav_to_mp3(wav_path: str) -> str:
    """Convert a WAV file to MP3 alongside the original, returning the playable path."""
    mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", mp3_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return mp3_path
    except Exception as exc:
        Logger.print_warning(
            f"WAV to MP3 conversion failed, playing WAV directly: {exc}"
        )
        return wav_path
