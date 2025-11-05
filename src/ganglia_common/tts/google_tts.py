"""Text-to-Speech module for GANGLIA.

This module provides text-to-speech functionality using various backends.
Currently supports Google Cloud Text-to-Speech with features including:
- Text chunking for long inputs
- Audio playback with skip functionality
- Error handling and retries
- Local file handling
"""

# Standard library imports
import os
import re
import select
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Third-party imports
from google.cloud import texttospeech_v1 as tts
from google.api_core import exceptions as google_exceptions

# Local imports
from ganglia_common.logger import Logger
from ganglia_common.utils.file_utils import get_tempdir
from ganglia_common.utils.retry_utils import exponential_backoff
from ganglia_common.utils.performance_profiler import is_timing_enabled

class TextToSpeech(ABC):
    """Abstract base class for text-to-speech functionality.

    This class defines the interface for text-to-speech implementations
    and provides common utility methods for audio handling and playback.
    """

    @abstractmethod
    def convert_text_to_speech(self, text: str, voice_id: str = None,
                             thread_id: str = None):
        """Convert text to speech using the specified voice.

        Args:
            text: The text to convert to speech
            voice_id: The ID of the voice to use (default: "en-US-Casual-K")
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





class GoogleTTS(TextToSpeech):
    """Google Cloud Text-to-Speech implementation.

    This class implements text-to-speech functionality using the Google Cloud
    Text-to-Speech API, with support for various voices and audio configurations.
    """

    # Class-level lock for gRPC client creation
    _client_lock = threading.Lock()

    def __init__(self, apply_effects=False):
        """Initialize the Google TTS client.

        Args:
            apply_effects: If True, apply audio effects (pitch shift, reverb, etc.)
        """
        super().__init__()
        self.apply_effects = apply_effects
        Logger.print_info(f"Initializing GoogleTTS{' with audio effects' if apply_effects else ''}...")
        # Create a single shared client instance with thread safety
        with self._client_lock:
            self._client = tts.TextToSpeechClient()

    def _convert_text_to_speech_impl(self, text: str, voice_id="en-US-Casual-K",
                                   thread_id: str = None):
        """Internal implementation of text-to-speech conversion.

        Args:
            text: The text to convert to speech
            voice_id: The ID of the voice to use (default: "en-US-Casual-K")
            thread_id: Optional thread ID for logging purposes

        Returns:
            tuple: (success: bool, file_path: str) where file_path is the path
                  to the generated audio file if successful
        """
        # Set up the text input and voice settings
        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(
            language_code="en-US",
            name=voice_id,
        )

        # Set the audio configuration with optional effects
        if self.apply_effects:
            # Use Google's native audio parameters for deeper, more dramatic voice
            audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3,
                pitch=-20.0,          # Deep pitch for demonic voice (range: -20.0 to 20.0)
                speaking_rate=1,   # Slower for more menacing effect (range: 0.25 to 4.0)
                # effects_profile_id=['headphone-class-device']  # Optional: optimize for headphones
            )
        else:
            audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3
            )

        thread_prefix = f"{thread_id} " if thread_id else ""
        Logger.print_debug(f"{thread_prefix}Converting text to speech ({len(text)} chars)...")

        # Use the shared client instance
        tts_start = time.time()
        response = self._client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        tts_elapsed = time.time() - tts_start

        if is_timing_enabled():
            Logger.print_perf(f"⏱️  [TTS] Audio generated in {tts_elapsed:.2f}s")

        # Create temp directory if it doesn't exist
        temp_dir = get_tempdir()
        os.makedirs(os.path.join(temp_dir, "tts"), exist_ok=True)

        # Sanitize the text for use in filename
        # Take first 3 words and replace problematic characters
        words = text.split()[:3]
        sanitized_words = []
        for word in words:
            # Replace slashes, parentheses, and other problematic characters
            sanitized = re.sub(r'[^\w\s-]', '_', word)
            sanitized_words.append(sanitized)
        snippet = '_'.join(sanitized_words)

        # Save the audio to a file (with microseconds to avoid collisions in parallel generation)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        file_path = os.path.join(
            temp_dir, "tts",
            f"chatgpt_response_{snippet}_{timestamp}.mp3"
        )
        with open(file_path, "wb") as out:
            out.write(response.audio_content)

        return True, file_path

    def convert_text_to_speech(self, text: str, voice_id="en-US-Casual-K",
                             thread_id: str = None):
        """Convert text to speech using the specified voice with retry logic.

        Args:
            text: The text to convert to speech
            voice_id: The ID of the voice to use (default: "en-US-Casual-K")
            thread_id: Optional thread ID for logging purposes

        Returns:
            tuple: (success: bool, file_path: str) where file_path is the path
                  to the generated audio file if successful, None otherwise
        """
        thread_prefix = f"{thread_id} " if thread_id else ""

        try:
            return exponential_backoff(
                lambda: self._convert_text_to_speech_impl(text, voice_id, thread_id),
                max_retries=5,
                initial_delay=1.0,
                thread_id=thread_id
            )
        except (google_exceptions.GoogleAPICallError, IOError) as e:
            Logger.print_error(
                f"{thread_prefix}Error converting text to speech: {e}"
            )
            return False, None

    def convert_text_to_speech_streaming(self, sentences: List[str], voice_id="en-US-Casual-K") -> Tuple[bool, str]:
        """Convert multiple sentences to speech in parallel and concatenate.

        This method generates TTS for multiple sentences concurrently to reduce
        total generation time for multi-sentence responses.

        Args:
            sentences: List of sentences to convert to speech
            voice_id: The ID of the voice to use

        Returns:
            tuple: (success: bool, file_path: str) where file_path is the path
                  to the concatenated audio file if successful, None otherwise
        """
        if not sentences:
            return False, None

        # Single sentence - use regular method
        if len(sentences) == 1:
            return self.convert_text_to_speech(sentences[0], voice_id)

        Logger.print_debug(f"Generating TTS for {len(sentences)} sentences in parallel...")

        # Generate TTS for all sentences in parallel
        with ThreadPoolExecutor(max_workers=min(3, len(sentences))) as executor:
            futures = [
                executor.submit(self.convert_text_to_speech, sentence, voice_id)
                for sentence in sentences
            ]
            results = [future.result() for future in futures]

        # Check if all succeeded
        if not all(success for success, _ in results):
            Logger.print_error("One or more TTS generations failed")
            return False, None

        # Extract file paths
        audio_files = [file_path for success, file_path in results if success]

        if not audio_files:
            return False, None

        # Concatenate audio files
        try:
            temp_dir = get_tempdir()
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            output_file = os.path.join(temp_dir, "tts", f"concatenated_{timestamp}.mp3")

            # Create file list for ffmpeg
            file_list_path = os.path.join(temp_dir, "tts", f"concat_list_{timestamp}.txt")
            with open(file_list_path, "w") as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")

            # Use ffmpeg to concatenate
            subprocess.run(
                ["ffmpeg", "-f", "concat", "-safe", "0", "-i", file_list_path,
                 "-c", "copy", output_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            # Clean up individual files and list
            for audio_file in audio_files:
                try:
                    os.remove(audio_file)
                except Exception:
                    pass
            try:
                os.remove(file_list_path)
            except Exception:
                pass

            Logger.print_debug(f"Concatenated {len(audio_files)} audio files into {output_file}")
            return True, output_file

        except Exception as e:
            Logger.print_error(f"Error concatenating audio files: {e}")
            return False, None
