# ganglia-common

Shared Python utilities powering [GANGLIA](https://ganglia-ai.com) — a live,
AI-powered Halloween séance you can talk to by voice or text.

This library is the plumbing the GANGLIA services share: LLM query dispatch
with streaming, pluggable text-to-speech engines, a lightweight pub/sub event
bus, themed terminal logging, and assorted utilities (audio conversion, retry
logic, performance profiling, cloud storage).

## Components

| Module | What it does |
|---|---|
| `query_dispatch` | `ChatGPTQueryDispatcher` — OpenAI chat interface with streaming sentence output, per-turn system-context injection, and token-usage reporting for cost telemetry |
| `tts` | Text-to-speech engines behind one `BaseTTS` interface: Google Cloud TTS (runtime-tunable pitch/rate) and OpenAI TTS, plus shared `Voice` types with clone metadata |
| `pubsub` | In-process publish/subscribe event bus with typed events (conversation, quest, and voice-clone lifecycle) |
| `logger` | Centralized, color-themed terminal logging |
| `utils` | Audio format conversion, file/temp-dir helpers, retry with backoff, performance profiling (`ConversationTimer`), Google Cloud Storage upload |

## Installation

```bash
pip install -e .
```

Requires Python 3.9+. TTS engines need the corresponding credentials at
runtime (`OPENAI_API_KEY` for OpenAI; Google Cloud application credentials
for Google TTS) — nothing is required just to import the library.

## Quick start

```python
from ganglia_common.logger import Logger
from ganglia_common.query_dispatch import ChatGPTQueryDispatcher
from ganglia_common.tts.google_tts import GoogleTTS
from ganglia_common.pubsub import get_pubsub

dispatcher = ChatGPTQueryDispatcher(pre_prompt="You are a helpful skeleton.")
for sentence in dispatcher.send_query_streaming("Tell me a short ghost story."):
    Logger.print_demon_output(sentence)
```

## Development

```bash
pip install -e .
pip install -r requirements-dev.txt
pytest
```

## The GANGLIA ecosystem

- **[ganglia-ai.com](https://ganglia-ai.com)** — the live séance. Free to try.
- **ganglia-core** — the FastAPI/WebSocket session host and web client.
- **ganglia-studio** — media generation (voice cloning, imagery), run as a
  separate service.

## License

[MIT](LICENSE)
