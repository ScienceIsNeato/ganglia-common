"""Query dispatcher module for handling OpenAI API interactions.

This module provides a ChatGPT query dispatcher that manages conversations with OpenAI's API,
handles session history, and provides content filtering capabilities for DALL-E compatibility.
"""

import base64

# Standard library imports
import os
import typing
from datetime import datetime
from time import time

# Third-party imports
from openai import OpenAI

# Local imports
from ganglia_common.logger import Logger
from ganglia_common.utils.file_utils import get_tempdir
from ganglia_common.utils.performance_profiler import is_timing_enabled


class ChatGPTQueryDispatcher:
    """A dispatcher for managing conversations with OpenAI's ChatGPT.

    This class handles the interaction with OpenAI's API, manages conversation history,
    and provides utilities for content filtering and token management.
    """

    def __init__(
        self,
        pre_prompt: typing.Any = None,
        config_file_path: typing.Any = None,
        audio_output: typing.Any = False,
        audio_voice: typing.Any = "alloy",
    ) -> None:
        """Initialize the ChatGPT query dispatcher.

        Args:
            pre_prompt (str, optional): Initial system prompt to set context
            config_file_path (str, optional): Path to configuration file
            audio_output (bool): If True, use gpt-4o-audio-preview to get audio responses
            audio_voice (str): Voice to use for audio output (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.client: typing.Any = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        default_config = os.path.join(
            os.path.dirname(__file__), "config", "ganglia_config.json"
        )
        self.config_file_path = config_file_path or default_config
        self.messages: list[dict[str, str]] = []
        self.audio_output = audio_output
        self.audio_voice = audio_voice
        self.model = "gpt-4o-audio-preview" if audio_output else "gpt-4o-mini"

        if pre_prompt:
            self.messages.append({"role": "system", "content": pre_prompt})

    def add_system_context(self, context_lines: typing.Any) -> typing.Any:
        """Add system context messages to the conversation.

        Args:
            context_lines (list[str]): Lines of context to add as system messages
        """
        for line in context_lines:
            self.messages.append({"role": "system", "content": line})

    def send_query(self, current_input: typing.Any) -> typing.Any:
        """Send a query to the ChatGPT API and get the response.

        Args:
            current_input (str): The user's input to send to ChatGPT

        Returns:
            str or tuple: If audio_output=False, returns text response.
                         If audio_output=True, returns (text, audio_file_path)
        """
        self.messages.append({"role": "user", "content": current_input})
        start_time = time()

        self.rotate_session_history()  # Ensure history stays under the max length

        if is_timing_enabled():
            Logger.print_perf(
                f"⏱️  [LLM] Sending query to OpenAI API ({self.model})..."
            )
        else:
            Logger.print_debug("Sending query to AI server...")

        if self.audio_output:
            return self._send_audio_query(start_time)
        else:
            # Standard text-only response
            chat = self.client.chat.completions.create(
                model=self.model, messages=self.messages
            )
            reply = chat.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})

            elapsed = time() - start_time
            if is_timing_enabled():
                Logger.print_perf(
                    f"⏱️  [LLM] Response received in {elapsed:.2f}s ({len(reply)} chars)"
                )
            else:
                Logger.print_info(f"AI response received in {elapsed:.1f} seconds.")

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            temp_dir = get_tempdir()

            with open(
                os.path.join(temp_dir, f"chatgpt_output_{timestamp}_raw.txt"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(reply)

            return reply

    def _send_audio_query(self, start_time: typing.Any) -> typing.Any:
        """Send an audio-enabled query and persist the returned WAV."""
        chat = self.client.chat.completions.create(
            model=self.model,
            modalities=["text", "audio"],
            audio={"voice": self.audio_voice, "format": "wav"},
            messages=self.messages,
        )
        reply = chat.choices[0].message.content or ""
        audio_data = chat.choices[0].message.audio
        if not audio_data or not hasattr(audio_data, "data"):
            Logger.print_warning(
                "⚠️  Audio output requested but not received from API. Falling back to TTS."
            )
            reply = reply or "[No response received]"
            self.messages.append({"role": "assistant", "content": reply})
            return reply

        if not reply and hasattr(audio_data, "transcript"):
            reply = audio_data.transcript or "[Audio response - no transcript]"

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        temp_dir = get_tempdir()
        os.makedirs(os.path.join(temp_dir, "tts"), exist_ok=True)
        audio_file = os.path.join(temp_dir, "tts", f"audio_response_{timestamp}.wav")
        with open(audio_file, "wb") as f:
            f.write(base64.b64decode(audio_data.data))
        self.messages.append({"role": "assistant", "content": reply})

        elapsed = time() - start_time
        if is_timing_enabled():
            Logger.print_perf(
                f"⏱️  [LLM+AUDIO] Response received in {elapsed:.2f}s ({len(reply)} chars + audio)"
            )
        else:
            Logger.print_info(
                f"AI response (with audio) received in {elapsed:.1f} seconds."
            )
        with open(
            os.path.join(temp_dir, f"chatgpt_output_{timestamp}_raw.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(reply)
        return reply, audio_file

    def send_query_streaming(self, current_input: typing.Any) -> typing.Any:
        """Send a query to ChatGPT API and stream the response sentence by sentence.

        This enables faster perceived response time by allowing TTS generation to start
        before the full LLM response is complete.

        Args:
            current_input (str): The user's input to send to ChatGPT

        Yields:
            str: Individual sentences from the AI's response as they're completed
        """
        self.messages.append({"role": "user", "content": current_input})
        start_time = time()

        self.rotate_session_history()

        Logger.print_debug("Sending streaming query to AI server...")

        stream = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=self.messages, stream=True
        )

        full_response = ""
        current_sentence = ""
        sentence_endings = (".", "!", "?", "\n")
        first_chunk_received = False

        for chunk in stream:
            if not first_chunk_received and is_timing_enabled():
                Logger.print_perf(
                    f"⏱️  [LLM] First chunk received (TTFB: {time() - start_time:.2f}s)"
                )
                first_chunk_received = True
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                current_sentence += content

                # Check if we've completed a sentence
                if any(
                    current_sentence.rstrip().endswith(end) for end in sentence_endings
                ):
                    sentence = current_sentence.strip()
                    if sentence:  # Only yield non-empty sentences
                        Logger.print_debug(f"Streaming sentence: {sentence[:50]}...")
                        yield sentence
                        current_sentence = ""

        # Yield any remaining text as the final sentence
        if current_sentence.strip():
            Logger.print_debug(f"Streaming final: {current_sentence[:50]}...")
            yield current_sentence.strip()

        # Add the complete response to message history
        self.messages.append({"role": "assistant", "content": full_response})

        elapsed = time() - start_time
        Logger.print_debug(
            f"AI response streamed in {elapsed:.1f} seconds ({len(full_response)} chars)"
        )

        # Save the response to disk
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        temp_dir = get_tempdir()

        with open(
            os.path.join(temp_dir, f"chatgpt_output_{timestamp}_raw.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(full_response)

    def rotate_session_history(self) -> typing.Any:
        """Rotate session history to keep token count under limit."""
        total_tokens = 0
        for message in self.messages:
            total_tokens += len(message["content"].split())

        max_tokens = 4097  # Constant should be uppercase but used locally

        while total_tokens > max_tokens:
            removed_message = self.messages.pop(0)
            removed_length = len(removed_message["content"].split())
            total_tokens -= removed_length
            debug_msg = (
                f"Conversation history getting long - dropping oldest content: "
                f"{removed_message['content']} ({removed_length} tokens)"
            )
            Logger.print_debug(debug_msg)

    def count_tokens(self) -> typing.Any:
        """Count total tokens in the message history."""
        total_tokens = 0
        for message in self.messages:
            total_tokens += len(message["content"].split())
        return total_tokens

    def filter_content_for_dalle(
        self, content: typing.Any, max_attempts: typing.Any = 3
    ) -> typing.Any:
        """Filter content to ensure it passes DALL-E's content filters.

        Args:
            content (str): The content to filter.
            max_attempts (int): Maximum number of filtering attempts.

        Returns:
            tuple: (success, filtered_content) where success is a boolean indicating if
                  filtering was successful, and filtered_content is the filtered text
                  if successful, or None if not.
        """
        prompt = self._get_dalle_filter_prompt(content)

        for attempt in range(max_attempts):
            try:
                attempt_msg = f"Filtering content for DALL-E (attempt {attempt + 1}/{max_attempts})"
                Logger.print_info(attempt_msg)
                filtered_response = self.send_query(prompt)
                filtered_content = filtered_response.strip()
                Logger.print_info(f"Rewritten content:\n{filtered_content}")
                return True, filtered_content
            except (ValueError, IOError, RuntimeError, TimeoutError) as e:
                error_msg = f"Error filtering content (attempt {attempt + 1}): {e}"
                Logger.print_error(error_msg)
                if attempt == max_attempts - 1:  # Last attempt
                    return False, None

        return False, None

    def _get_dalle_filter_prompt(self, content: typing.Any) -> typing.Any:
        """Get the prompt for filtering content for DALL-E.

        Args:
            content (str): The content to filter.

        Returns:
            str: The prompt for filtering content.
        """
        filter_instructions = [
            "Please rewrite this story to pass OpenAI's DALL-E content filters. ",
            "The rewritten version should:",
            "1. Replace all specific names with generic terms (e.g., 'the family', 'the children')",
            "2. Replace specific locations with generic descriptions (e.g., 'a beautiful lake')",
            "3. Remove any potentially sensitive or controversial content",
            "4. Keep the core story and emotional tone\n",
            f"\nStory to rewrite:\n{content}\n",
            "\nReturn only the rewritten story with no additional text or explanation.",
        ]
        return "".join(filter_instructions)
