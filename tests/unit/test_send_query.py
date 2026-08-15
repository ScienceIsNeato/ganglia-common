import sys
from pathlib import Path
import pytest
from ganglia_common.query_dispatch import ChatGPTQueryDispatcher
from ganglia_common.utils import get_config_path

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def query_dispatcher():
    return ChatGPTQueryDispatcher(config_file_path=get_config_path())


def test_send_query():
    expected_in_response = "Paris"
    query_dispatcher = ChatGPTQueryDispatcher(config_file_path=get_config_path())

    test_prompt = "What is the capital of France?"

    print("Query: ", test_prompt)

    # This is a LIVE test — it exercises the real OpenAI API when a valid key
    # is configured, and skips cleanly when one isn't (public contributors and
    # keyless CI shouldn't see a red suite for missing credentials).
    import openai

    try:
        response = query_dispatcher.send_query(test_prompt)
    except openai.AuthenticationError:
        pytest.skip("no valid OPENAI_API_KEY configured; skipping live API test")

    print("response: ", response)
    print("expected_in_response: ", expected_in_response)

    # Assertions based on the expected response
    assert expected_in_response in response
