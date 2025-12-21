"""File and directory utilities."""

import os
import tempfile
from datetime import datetime
from ganglia_common.logger import Logger

# Global variable to store the current TTV directory
_current_ttv_dir = None


def get_tempdir():
    """
    Get the temporary directory in a platform-agnostic way.
    Creates and returns /tmp/GANGLIA for POSIX systems or %TEMP%/GANGLIA for Windows.
    """
    # If the environment variable is set, use the full path directly
    temp_dir = os.getenv("GANGLIA_TEMP_DIR", None)

    # otherwise, use the default temp directory and append GANGLIA
    if temp_dir is None:
        temp_dir = os.path.join(tempfile.gettempdir(), "GANGLIA")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_timestamped_ttv_dir() -> str:
    """Get a timestamped directory path for TTV files.

    Creates a unique directory for each TTV run using the current timestamp.
    Format: /tmp/GANGLIA/ttv/YYYY-MM-DD-HH-MM-SS/

    The directory is created only on the first call and the same path
    is returned for all subsequent calls within the same run.

    Returns:
        str: Path to the timestamped directory
    """
    global _current_ttv_dir  # pylint: disable=global-statement
    if _current_ttv_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        _current_ttv_dir = os.path.join(get_tempdir(), "ttv", timestamp)
        os.makedirs(_current_ttv_dir, exist_ok=True)
        Logger.print_info(f"📁 TTV directory created: {_current_ttv_dir}")
    return _current_ttv_dir


def get_config_path():
    """Get the path to the config directory relative to the project root."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "ganglia_config.json"
    )
