import sys
from pathlib import Path
import pytest
from ganglia_common.query_dispatch import ChatGPTQueryDispatcher
from utils import get_config_path

sys.path.append(str(Path(__file__).resolve().parent.parent))

@pytest.fixture
def query_dispatcher():
    return ChatGPTQueryDispatcher(config_file_path=get_config_path())


def test_send_query():
    expected_in_response = "Paris"
    query_dispatcher = ChatGPTQueryDispatcher(config_file_path=get_config_path())

    test_prompt = "What is the capital of France?"

    print("Query: ", test_prompt)

    # Call the send_query function without mocking
    response = query_dispatcher.send_query(test_prompt)

    print("response: ", response)
    print("expected_in_response: ", expected_in_response)

    # Assertions based on the expected response
    assert expected_in_response in response
