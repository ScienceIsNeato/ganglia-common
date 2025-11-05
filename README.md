# ganglia-common

Shared utilities for the GANGLIA ecosystem.

## Components

- **logger**: Centralized logging
- **query_dispatch**: OpenAI API interface
- **tts**: Text-to-speech engines (Google, OpenAI)
- **pubsub**: Event system
- **utils**: File operations, performance profiling, retry logic, cloud storage

## Installation

```bash
pip install -e .
```

## Development

```bash
pip install -e .
pip install -r requirements-dev.txt
pytest
```

## Usage

```python
from ganglia_common.logger import Logger
from ganglia_common.query_dispatch import ChatGPTQueryDispatcher
from ganglia_common.tts.google_tts import GoogleTTS
from ganglia_common.pubsub import get_pubsub
```


