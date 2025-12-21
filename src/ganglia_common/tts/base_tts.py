"""Abstract Base Class for Text-to-Speech implementations."""

import re
import os
import subprocess
import sys
import select
from abc import ABC, abstractmethod
from urllib.parse import urlparse

from ganglia_common.logger import Logger
from ganglia_common.utils.performance_profiler import is_timing_enabled
from ganglia_common.tts.types import Voice


class TextToSpeech(ABC):
    """Abstract base class for text-to-speech functionality.

    This class defines the interface for text-to-speech implementations
    and provides common utility methods for audio handling and playback.
    """

    @abstractmethod
    def convert_text_to_speech(self, text: str, voice: Voice,
                             thread_id: str = None):
        """Convert text to speech using the specified voice.

        Args:
            text: The text to convert to speech
            voice: The voice configuration to use
            thread_id: Optional thread ID for logging purposes

        Returns:
            tuple: (success: bool, file_path: str) where file_path is the path
                  to the generated audio file if successful, None otherwise
        """
        pass

    def is_local_filepath(self, file_path: str) -> bool:
        """Check if a file path is a local file path.

        Args:
            file_path: The file path to check

        Returns:
            bool: True if the path is a local file path, False otherwise
        """
        try:
            result = urlparse(file_path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @classmethod
    def split_text(cls, text: str, max_length: int = 250):
        """Split text into chunks of maximum length while preserving sentences.

        Args:
            text: The text to split
            max_length: Maximum length of each chunk (default: 250)

        Returns:
            list: List of text chunks
        """
        sentences = [match.group() for match in re.finditer(r'[^.!?]*[.!?]', text)]
        chunks = []

        for sentence in sentences:
            while len(sentence) > max_length:
                chunk = sentence[:max_length]
                chunks.append(chunk.strip())
                sentence = sentence[max_length:]
            chunks.append(sentence.strip())

        return chunks

    def play_speech_response(self, file_path, raw_response, suppress_text_output=False):
        """Play speech response and handle user interaction.

        Args:
            file_path: Path to the audio file to play
            raw_response: The text response to display
            suppress_text_output: If True, don't print "GANGLIA says..." (for streaming)
        """
        if file_path.endswith('.txt'):
            file_path = self.concatenate_audio_from_text(file_path)

        # Only play audio if explicitly enabled
        if os.getenv('PLAYBACK_MEDIA_IN_TESTS', 'false').lower() == 'true':
            # Prepare the play command and determine the audio duration
            play_command, audio_duration = self.prepare_playback(file_path)

            # Only print header if not suppressed (for streaming playback)
            if not suppress_text_output:
                Logger.print_demon_output(
                    f"\nGANGLIA says... (Audio Duration: {audio_duration:.1f} seconds)"
                )
                Logger.print_demon_output(raw_response)

            # Mark the moment audio playback begins
            if is_timing_enabled():
                Logger.print_perf(f"⏱️  [PLAYBACK] Starting audio playback NOW! (duration: {audio_duration:.1f}s)")

            # Start playback in a non-blocking manner
            playback_process = subprocess.Popen(
                play_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL
            )

            # Wait for playback to finish (no enter key monitoring for streaming)
            playback_process.wait()

    def monitor_enter_keypress(self, playback_process):
        """Monitor for Enter key press to stop playback.

        Args:
            playback_process: The subprocess running the audio playback
        """
        Logger.print_debug("Press Enter to stop playback...")

        while playback_process.poll() is None:  # While playback is running
            # Use select to check if input is available (non-blocking check)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key_press = sys.stdin.read(1)  # Read single character
                if key_press == '\n':  # Check for Enter key
                    Logger.print_debug("Enter key detected. Terminating playback...")
                    playback_process.terminate()
                    break

    def concatenate_audio_from_text(self, text_file_path):
        """Concatenate multiple audio files listed in a text file.

        Args:
            text_file_path: Path to the text file containing audio file paths

        Returns:
            str: Path to the concatenated audio file
        """
        output_file = "combined_audio.mp3"
        concat_command = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", text_file_path, output_file
        ]
        subprocess.run(
            concat_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            check=True
        )
        return output_file

    def prepare_playback(self, file_path):
        """Prepare audio playback command and get duration.

        Args:
            file_path: Path to the audio file

        Returns:
            tuple: (play_command: list, audio_duration: float)
        """
        if file_path.endswith('.mp4'):
            play_command = ["ffplay", "-nodisp", "-autoexit", file_path]
        else:
            play_command = [
                "ffplay", "-nodisp", "-af", "volume=5", "-autoexit", file_path
            ]
        audio_duration = self.get_audio_duration(file_path)
        return play_command, audio_duration

    def get_audio_duration(self, file_path):
        """Get the duration of an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            float: Duration of the audio in seconds
        """
        duration_command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        duration_output = subprocess.run(
            duration_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        ).stdout.decode('utf-8')
        return float(duration_output.strip())
