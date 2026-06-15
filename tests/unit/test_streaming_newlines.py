"""send_query_streaming must preserve intentional line/stanza breaks.

The transcript reflects whatever this yields, reassembled by the caller. Verse
(e.g. a recited poem) must keep its line breaks, so a chunk that ends on a
newline carries a trailing "\\n", and a blank line is emitted as its own break.
Previously "\\n" was a declared sentence ending but the rstrip() before the
endswith() test made it dead, so every line break was silently dropped.
"""

import os
from types import SimpleNamespace

# The dispatcher constructs an OpenAI client on init; this test replaces that
# client with a fake stream, so a placeholder key is enough to construct it.
os.environ.setdefault("OPENAI_API_KEY", "test-key-unused")

from ganglia_common.query_dispatch import ChatGPTQueryDispatcher  # noqa: E402
from ganglia_common.utils import get_config_path  # noqa: E402


def _delta(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))], usage=None
    )


def _usage_chunk():
    # The final include_usage chunk carries token counts and empty choices.
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _dispatcher_streaming(deltas):
    qd = ChatGPTQueryDispatcher(config_file_path=get_config_path())
    chunks = [_delta(d) for d in deltas] + [_usage_chunk()]
    qd.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter(chunks))
        )
    )
    return list(qd.send_query_streaming("go"))


def _reassemble(chunks):
    """Mirror the caller's reassembly (conversational_interface), including its
    trailing-space-before-newline tidy-up."""
    import re

    out = ""
    for c in chunks:
        out += c
        if not c.endswith("\n"):
            out += " "
    out = re.sub(r"[ \t]+\n", "\n", out)  # drop a space left before a line break
    return out.strip()


def test_line_breaks_survive_streaming():
    chunks = _dispatcher_streaming(["Line one", "\n", "Line two", "\n", "End."])
    # Each line-terminated chunk carries its newline; the period chunk does not.
    assert chunks[0] == "Line one\n"
    assert chunks[1] == "Line two\n"
    assert chunks[-1] == "End."
    assert _reassemble(chunks) == "Line one\nLine two\nEnd."


def test_blank_line_becomes_a_stanza_break():
    chunks = _dispatcher_streaming(["First.", "\n\n", "Second."])
    assert "\n\n" in chunks  # the blank line is preserved as its own break
    assert _reassemble(chunks) == "First.\n\nSecond."


def test_plain_prose_is_unchanged():
    # No newlines → behaves exactly as before (sentences space-joined).
    chunks = _dispatcher_streaming(["Hello there.", " How are you?"])
    assert all(not c.endswith("\n") for c in chunks)
    assert _reassemble(chunks) == "Hello there. How are you?"
