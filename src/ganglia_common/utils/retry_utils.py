"""Retry and backoff utilities."""

import random
import time
from typing import Callable, Any, Optional
from ganglia_common.logger import Logger

def exponential_backoff(
    func: Callable[..., Any],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    thread_id: Optional[str] = None
) -> Any:
    """Execute a function with exponential backoff retry logic and improved logging.

    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        thread_id: Optional thread ID for logging

    Returns:
        Any: The result of the function if successful

    Raises:
        Exception: The last exception encountered if all retries fail
    """
    thread_prefix = f"{thread_id} " if thread_id else ""
    attempt = 1
    last_exception = None

    while attempt <= max_retries:
        try:
            # Only log on first attempt if it fails, or retries
            return func()
        except Exception as error:
            last_exception = error
            if attempt == max_retries:
                raise

            delay = initial_delay * (2 ** (attempt - 1))
            # Add some jitter to prevent thundering herd
            delay = delay * (0.5 + random.random())

            func_name = getattr(func, '__name__', '<unknown function>')
            Logger.print_warning(
                f"{thread_prefix}Attempt {attempt}/{max_retries} "
                f"calling {func_name} failed: {error}"
            )
            Logger.print_info(f"{thread_prefix}Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            attempt += 1

    raise last_exception
