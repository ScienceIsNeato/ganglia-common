"""OpenAI Text-to-Speech implementation for GANGLIA.

This module provides TTS using OpenAI's dedicated TTS API endpoint.
Uses the tts-1 model optimized for low latency.
"""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Tuple, List
from openai import OpenAI

from tts import TextToSpeech
from ganglia_common.logger import Logger
from utils import get_tempdir
from ganglia_common.utils.performance_profiler import is_timing_enabled


class OpenAITTS(TextToSpeech):
    """OpenAI Text-to-Speech implementation using the dedicated TTS API.

    Features:
    - Uses tts-1 model (optimized for low latency)
    - Supports 6 voices: alloy, echo, fable, onyx, nova, shimmer
    - Pricing: $15/1M characters (comparable to Google TTS)
    - Potentially lower latency when paired with OpenAI LLM
    """

    # Available voices
    VOICES = {
        "alloy": "Neutral and balanced",
        "echo": "Slightly deeper voice",
        "fable": "British accent",
        "onyx": "Deep and authoritative",
        "nova": "Warm and friendly",
        "shimmer": "Soft and gentle"
    }

    def __init__(self, voice: str = "onyx"):
        """Initialize OpenAI TTS client.

        Args:
            voice: Voice to use (default: "onyx" - deep voice like GANGLIA)
        """
        self.client = OpenAI()
        self.voice = voice if voice in self.VOICES else "onyx"
        self.model = "tts-1"  # Optimized for speed
        Logger.print_info(f"OpenAI TTS initialized with voice: {self.voice}")

    def _convert_text_to_speech_impl(self, text: str, voice_id: str = None, thread_id: str = None) -> Tuple[bool, str]:
        """Internal implementation of text-to-speech conversion.

        Args:
            text: Text to convert
            voice_id: Voice to use (overrides default)
            thread_id: Optional thread identifier for logging

        Returns:
            Tuple of (success, file_path)
        """
        thread_prefix = f"[Thread {thread_id}] " if thread_id else ""

        if not text or not text.strip():
            Logger.print_error(f"{thread_prefix}Cannot convert empty text to speech")
            return False, None

        try:
            voice = voice_id if voice_id in self.VOICES else self.voice

            if is_timing_enabled():
                Logger.print_perf(f"⏱️  [TTS] Converting text to speech ({len(text)} chars)...")
            else:
                Logger.print_debug(f"{thread_prefix}Converting text to speech with OpenAI TTS...")

            tts_start = time.time()

            # Call OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=text,
                response_format="mp3"
            )

            tts_elapsed = time.time() - tts_start
            if is_timing_enabled():
                Logger.print_perf(f"⏱️  [TTS] Audio generated in {tts_elapsed:.2f}s")

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            temp_dir = get_tempdir()
            os.makedirs(os.path.join(temp_dir, "tts"), exist_ok=True)

            file_path = os.path.join(temp_dir, "tts", f"openai_tts_{timestamp}.mp3")

            # Stream audio to file
            response.stream_to_file(file_path)

            Logger.print_debug(f"{thread_prefix}Audio saved to: {file_path}")
            return True, file_path

        except Exception as e:
            Logger.print_error(f"{thread_prefix}Error generating speech with OpenAI TTS: {e}")
            return False, None

    def convert_text_to_speech(self, text: str, voice_id: str = None, thread_id: str = None) -> Tuple[bool, str]:
        """Convert text to speech using OpenAI TTS API.

        Args:
            text: Text to convert
            voice_id: Voice to use (default: self.voice)
            thread_id: Optional thread identifier for logging

        Returns:
            Tuple of (success, file_path)
        """
        # Check if text is too long (OpenAI has a 4096 character limit)
        MAX_LENGTH = 4000  # Leave some buffer

        if len(text) > MAX_LENGTH:
            Logger.print_warning(f"Text too long ({len(text)} chars), splitting into chunks...")
            # For long text, use the chunking/concatenation approach
            chunks = self.split_text(text, max_length=MAX_LENGTH)
            Logger.print_debug(f"Split into {len(chunks)} chunks")

            audio_files = []
            for i, chunk in enumerate(chunks):
                success, file_path = self._convert_text_to_speech_impl(
                    chunk,
                    voice_id=voice_id,
                    thread_id=f"{thread_id}-{i}" if thread_id else f"chunk-{i}"
                )
                if success:
                    audio_files.append(file_path)
                else:
                    Logger.print_error(f"Failed to convert chunk {i}")
                    return False, None

            # Concatenate audio files
            final_path = self._concatenate_audio_files(audio_files)
            return True, final_path
        else:
            return self._convert_text_to_speech_impl(text, voice_id=voice_id, thread_id=thread_id)

    def convert_text_to_speech_streaming(self, sentences: List[str], voice_id: str = None) -> Tuple[bool, str]:
        """Convert multiple sentences to speech in parallel, then concatenate.

        This method generates audio for multiple sentences concurrently to reduce
        overall latency, then concatenates them into a single audio file.

        Args:
            sentences: List of sentences to convert
            voice_id: Voice to use (default: self.voice)

        Returns:
            Tuple of (success, file_path)
        """
        if not sentences:
            return False, None

        if len(sentences) == 1:
            return self.convert_text_to_speech(sentences[0], voice_id=voice_id)

        Logger.print_debug(f"Generating TTS for {len(sentences)} sentences in parallel...")

        # Generate audio for each sentence in parallel
        audio_files = []

        with ThreadPoolExecutor(max_workers=min(len(sentences), 5)) as executor:
            # Submit all TTS jobs
            futures = []
            for i, sentence in enumerate(sentences):
                future = executor.submit(
                    self._convert_text_to_speech_impl,
                    sentence,
                    voice_id=voice_id,
                    thread_id=f"parallel-{i}"
                )
                futures.append(future)

            # Collect results in order
            for i, future in enumerate(futures):
                success, file_path = future.result()
                if success:
                    audio_files.append(file_path)
                else:
                    Logger.print_error(f"Failed to convert sentence {i}")
                    return False, None

        # Concatenate all audio files
        final_path = self._concatenate_audio_files(audio_files)
        return True, final_path

    def _concatenate_audio_files(self, audio_files: List[str]) -> str:
        """Concatenate multiple audio files into one using ffmpeg.
        
        Args:
            audio_files: List of audio file paths to concatenate
            
        Returns:
            Path to concatenated audio file
        """
        concat_start = time.time()
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        temp_dir = get_tempdir()
        output_path = os.path.join(temp_dir, "tts", f"concatenated_{timestamp}.mp3")

        # Create file list for ffmpeg
        list_file = os.path.join(temp_dir, "tts", f"concat_list_{timestamp}.txt")
        with open(list_file, 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file}'\n")

        # Use ffmpeg to concatenate
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            concat_elapsed = time.time() - concat_start
            
            if is_timing_enabled():
                Logger.print_perf(f"⏱️  [TTS] Concatenation took {concat_elapsed:.2f}s")
            
            Logger.print_debug(f"Concatenated {len(audio_files)} audio files into {output_path}")

            # Clean up individual files and list file
            for audio_file in audio_files:
                try:
                    os.remove(audio_file)
                except Exception:
                    pass
            try:
                os.remove(list_file)
            except Exception:
                pass

            return output_path
        except subprocess.CalledProcessError as e:
            Logger.print_error(f"Failed to concatenate audio: {e}")
            # Return first file as fallback
            return audio_files[0] if audio_files else None
