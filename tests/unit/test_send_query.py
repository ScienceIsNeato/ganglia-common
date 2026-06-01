import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ganglia_common.query_dispatch import ChatGPTQueryDispatcher
from ganglia_common.utils import get_config_path

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def query_dispatcher():
    return ChatGPTQueryDispatcher(config_file_path=get_config_path())


def test_send_query(monkeypatch):
    expected_in_response = "Paris"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    query_dispatcher = ChatGPTQueryDispatcher(config_file_path=get_config_path())
    query_dispatcher.client.chat.completions.create = lambda **_: SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="The capital of France is Paris.")
            )
        ]
    )

    test_prompt = "What is the capital of France?"

    print("Query: ", test_prompt)

    # Call the send_query function without mocking
    response = query_dispatcher.send_query(test_prompt)

    print("response: ", response)
    print("expected_in_response: ", expected_in_response)

    # Assertions based on the expected response
    assert expected_in_response in response
