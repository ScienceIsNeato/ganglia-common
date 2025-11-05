import pytest
import os
from ganglia_common.query_dispatch import ChatGPTQueryDispatcher
from utils import get_config_path


def test_load_git_repo_into_history():
    dispatcher = ChatGPTQueryDispatcher(pre_prompt="Test pre-prompt")
    token_count = dispatcher.count_tokens()

    assert isinstance(token_count, int)
    assert token_count > 0

def test_query_dispatcher_init():
    """Test that the query dispatcher initializes correctly."""
    dispatcher = ChatGPTQueryDispatcher()
    assert dispatcher.client is not None
    assert dispatcher.messages == []

    # Test with pre_prompt
    pre_prompt = "You are a helpful assistant."
    dispatcher = ChatGPTQueryDispatcher(pre_prompt=pre_prompt)
    assert dispatcher.messages == [{"role": "system", "content": pre_prompt}]

# Test removed - send_merged_query method does not exist in the codebase
# The ChatGPTQueryDispatcher maintains conversation history internally via session_history
# and doesn't expose a merged query method
