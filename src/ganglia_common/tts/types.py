"""Common types for Text-to-Speech functionality."""

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Voice:
    """Represents a voice configuration for TTS engines.

    Attributes:
        engine: The TTS engine to use ('google' or 'chatterbox')
        name: Human-readable name for the voice
        id: Engine-specific ID (e.g., 'en-US-Neural2-F') or UUID for clones
        ref_audio: Path to reference audio file (required for cloning)
        created_at: Timestamp when voice was created/registered
        duration_seconds: Duration of reference audio in seconds
        sample_text: Text content of the reference audio sample
    """

    engine: Literal["google", "chatterbox"]
    name: str
    id: str | None = None
    ref_audio: str | None = None

    # Extended Metadata
    created_at: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    sample_text: str | None = None
