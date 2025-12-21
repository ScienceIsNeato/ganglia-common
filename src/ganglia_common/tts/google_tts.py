"""Text-to-Speech module for GANGLIA.

This module provides text-to-speech functionality using various backends.
Currently supports Google Cloud Text-to-Speech.
"""

# Standard library imports
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

# Third-party imports
from google.cloud import texttospeech_v1 as tts
from google.api_core import exceptions as google_exceptions

# Local imports
from ganglia_common.logger import Logger
from ganglia_common.utils.file_utils import get_tempdir
from ganglia_common.utils.retry_utils import exponential_backoff
from ganglia_common.utils.performance_profiler import is_timing_enabled
from ganglia_common.tts.base_tts import TextToSpeech
from ganglia_common.tts.types import Voice


class GoogleTTS(TextToSpeech):
    """Google Cloud Text-to-Speech implementation.

    This class implements text-to-speech functionality using the Google Cloud
    Text-to-Speech API.
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

    def _convert_text_to_speech_impl(self, text: str, voice: Voice,
                                   thread_id: str = None):
        """Internal implementation of text-to-speech conversion.

        Args:
            text: The text to convert to speech
            voice: The voice configuration to use
            thread_id: Optional thread ID for logging purposes

        Returns:
            tuple: (success: bool, file_path: str) where file_path is the path
                  to the generated audio file if successful
        """
        # Use provided voice ID or default
        voice_id = voice.id if voice and voice.id else "en-US-Casual-K"

        # Set up the text input and voice settings
        synthesis_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
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
            voice=voice_params,
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

    def convert_text_to_speech(self, text: str, voice: Optional[Voice] = None,
                             thread_id: str = None):
        """Convert text to speech using the specified voice with retry logic.

        Args:
            text: The text to convert to speech
            voice: The voice configuration to use (optional)
            thread_id: Optional thread ID for logging purposes

        Returns:
            tuple: (success: bool, file_path: str) where file_path is the path
                  to the generated audio file if successful, None otherwise
        """
        thread_prefix = f"{thread_id} " if thread_id else ""
        
        # Create default voice if none provided (backward compatibility)
        if voice is None:
            voice = Voice(engine="google", name="Default", id="en-US-Casual-K")

        try:
            return exponential_backoff(
                lambda: self._convert_text_to_speech_impl(text, voice, thread_id),
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
        
        NOTE: This legacy method still takes voice_id directly to avoid breaking
        existing streaming callers, but should eventually migrate to Voice object.

        Args:
            sentences: List of sentences to convert to speech
            voice_id: The ID of the voice to use

        Returns:
            tuple: (success: bool, file_path: str) where file_path is the path
                  to the concatenated audio file if successful, None otherwise
        """
        if not sentences:
            return False, None
            
        # Create temporary voice object for this call
        voice = Voice(engine="google", name="StreamingTemp", id=voice_id)

        # Single sentence - use regular method
        if len(sentences) == 1:
            return self.convert_text_to_speech(sentences[0], voice)

        Logger.print_debug(f"Generating TTS for {len(sentences)} sentences in parallel...")

        # Generate TTS for all sentences in parallel
        with ThreadPoolExecutor(max_workers=min(3, len(sentences))) as executor:
            futures = [
                executor.submit(self.convert_text_to_speech, sentence, voice)
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
