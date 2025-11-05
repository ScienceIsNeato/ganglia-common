import pytest
from unittest.mock import Mock, patch
from utils import exponential_backoff
from ganglia_common.logger import Logger

def test_exponential_backoff_success():
    """Test that exponential backoff returns immediately on success."""
    mock_func = Mock(return_value="success")
    result = exponential_backoff(mock_func)
    assert result == "success"
    assert mock_func.call_count == 1

def test_exponential_backoff_retry_then_success():
    """Test that exponential backoff retries on failure then succeeds."""
    mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])

    with patch('time.sleep'):  # Don't actually sleep in tests
        result = exponential_backoff(mock_func, max_retries=5, initial_delay=1.0)

    assert result == "success"
    assert mock_func.call_count == 3

def test_exponential_backoff_max_retries():
    """Test that exponential backoff stops after max retries."""
    mock_func = Mock(side_effect=Exception("fail"))

    with patch('time.sleep'):
        with pytest.raises(Exception) as exc_info:
            exponential_backoff(mock_func, max_retries=3, initial_delay=1.0)

    assert str(exc_info.value) == "fail"
    assert mock_func.call_count == 3

def test_exponential_backoff_with_thread_id():
    """Test that exponential backoff logs with thread ID."""
    mock_func = Mock(side_effect=[Exception("fail"), "success"])
    mock_func.__name__ = "test_func"  # Required for logging

    with patch('time.sleep'), \
         patch.object(Logger, 'print_warning') as mock_warning, \
         patch.object(Logger, 'print_info') as mock_info:

        result = exponential_backoff(mock_func, thread_id="test-thread")

    assert result == "success"
    assert mock_func.call_count == 2

    # Verify logging includes thread ID (exponential_backoff uses print_warning and print_info)
    assert any("test-thread" in str(call) for call in mock_warning.call_args_list), \
        f"thread-id not found in warning calls: {mock_warning.call_args_list}"
    assert any("test-thread" in str(call) for call in mock_info.call_args_list), \
        f"thread-id not found in info calls: {mock_info.call_args_list}"
