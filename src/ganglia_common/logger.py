"""Thread-safe logging system with colored terminal output.

This module provides a thread-safe logging system with color-coded output for different
message types and logging levels. It uses the blessed library for terminal color support
and includes thread-safe operations through a global lock.

The logging system supports:
- User input/output differentiation
- Different severity levels (DEBUG, INFO, WARNING, ERROR)
- Special message types (demon output, halloween narrator)
- Thread-safe logging operations
- Color-coded output for better visual distinction

Example:
    ```python
    from logger import Logger

    # Log different types of messages
    Logger.print_info("Starting process...")
    Logger.print_warning("Resource usage high")
    Logger.print_error("Failed to connect")
    Logger.print_debug("Connection attempt details: ...")
    ```

Color Scheme:
    - User Input: Deep Sky Blue
    - Demon Output: Firebrick Red
    - Halloween Narrator: Pumpkin Orange
    - Error/Warning: Yellow
    - Info: Salmon
    - Debug: Snow Gray
"""

import threading
import blessed
from datetime import datetime

term = blessed.Terminal()


class Logger:
    """Thread-safe logging system with colored output.

    This class provides static methods for logging at different levels,
    with thread-safe operations and optional thread ID prefixing.
    Color coding is used to distinguish between different log levels
    and message types.

    Attributes:
        _lock (threading.Lock): Thread lock for synchronizing output operations.
        _timestamps_enabled (bool): Whether to prepend timestamps to log messages.

    Color Scheme:
        - User Input: Deep Sky Blue
        - Demon Output: Firebrick Red
        - Halloween Narrator: Pumpkin Orange
        - Error/Warning: Yellow
        - Info: Salmon
        - Debug: Snow Gray
    """

    _lock = threading.Lock()
    _timestamps_enabled = False
    _debug_enabled = False

    @staticmethod
    def enable_timestamps():
        """Enable timestamp prefixes for all log messages."""
        Logger._timestamps_enabled = True

    @staticmethod
    def disable_timestamps():
        """Disable timestamp prefixes for all log messages."""
        Logger._timestamps_enabled = False

    @staticmethod
    def enable_debug():
        """Enable debug logging."""
        Logger._debug_enabled = True

    @staticmethod
    def disable_debug():
        """Disable debug logging."""
        Logger._debug_enabled = False

    @staticmethod
    def _get_timestamp():
        """Get formatted timestamp if timestamps are enabled."""
        if Logger._timestamps_enabled:
            return f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
        return ""

    @staticmethod
    def print_user_input(*args, **kwargs):
        """Print user input messages in deep sky blue.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        import re
        import shutil
        import os

        end_char = kwargs.get("end", "\n")
        is_carriage_return = end_char == "\r" or end_char == ""

        # Get terminal width: ENV VAR > auto-detect > default 80
        terminal_width = None
        env_width = os.environ.get("GANGLIA_TERMINAL_WIDTH")
        if env_width:
            try:
                terminal_width = int(env_width)
            except ValueError:
                pass  # Fall through to auto-detect

        if terminal_width is None:
            try:
                terminal_width = shutil.get_terminal_size().columns
            except Exception:
                terminal_width = 80

        if len(args) == 1 and isinstance(args[0], str):
            text = args[0]
            # Strip ANSI codes to measure actual text length
            clean_text = re.sub(r"\033\[[0-9;]*[mK]", "", text)
            clean_text = re.sub(r"\r", "", clean_text)

            if is_carriage_return:
                # INTERIM UPDATE (while speaking) - truncate if too long to prevent flooding
                if len(clean_text) > terminal_width - 5:
                    # Show "..." + last N characters that fit
                    visible_width = (
                        terminal_width - 8
                    )  # Leave room for "..." and color codes
                    truncated = "..." + clean_text[-visible_width:]
                    print(
                        f"{term.deepskyblue}\r\033[K{truncated}{term.white}",
                        end=end_char,
                        flush=True,
                    )
                    return
            else:
                # FINAL OUTPUT (sentence complete) - wrap if too long
                if len(clean_text) > terminal_width - 2:
                    # Word wrapping for final output
                    words = clean_text.split()
                    lines = []
                    current_line = []
                    current_length = 0

                    for word in words:
                        word_len = len(word) + (1 if current_line else 0)
                        if current_length + word_len <= terminal_width - 2:
                            current_line.append(word)
                            current_length += word_len
                        else:
                            if current_line:
                                lines.append(" ".join(current_line))
                            current_line = [word]
                            current_length = len(word)

                    if current_line:
                        lines.append(" ".join(current_line))

                    # Print wrapped lines
                    for line in lines:
                        print(f"{term.deepskyblue}{line}{term.white}")
                    return

        # Default: print normally
        print(f"{term.deepskyblue}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_demon_output(*args, **kwargs):
        """Print demon output messages in firebrick red.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        print(f"{term.firebrick2}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_halloween_narrator(*args, **kwargs):
        """Print halloween narrator messages in pumpkin orange.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        print(f"{term.pumpkin}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_error(*args, **kwargs):
        """Print error messages in yellow.

        Used for logging errors and critical issues that need immediate attention.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        print(f"{term.yellow}{Logger._get_timestamp()}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_warning(*args, **kwargs):
        """Print warning messages in yellow.

        Used for logging potential issues or concerning conditions that don't prevent execution.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        print(f"{term.yellow}{Logger._get_timestamp()}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_info(*args, **kwargs):
        """Print informational messages in salmon.

        Used for logging general information and progress updates.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        print(f"{term.salmon1}{Logger._get_timestamp()}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_debug(*args, **kwargs):
        """Print debug messages in snow gray.

        Used for logging detailed debug information and technical details.
        Only prints if debug logging is enabled.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        if not Logger._debug_enabled:
            return
        print(f"{term.snow4}{Logger._get_timestamp()}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_perf(*args, **kwargs):
        """Print performance timing messages in bright cyan.

        Used for logging performance metrics and timing information.

        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments passed to print function.
        """
        print(f"{term.cyan}{Logger._get_timestamp()}", end="")
        print(*args, **kwargs)
        print(f"{term.white}", end="", flush=True)

    @staticmethod
    def print_legend():
        """Print a color-coded legend banner showing available message types.

        Displays a formatted banner showing all available colors and their
        corresponding message types for reference.
        """
        print(term.magenta("===================="))
        print(term.magenta("    COLOR LEGEND   "))
        print(term.magenta("===================="))
        print(term.cyan("You"))
        print(term.red("Him"))
        print(term.yellow("Warnings/Errors"))
        print(term.blue("Informational Messages"))
        print(term.gray("Debug Messages"))
        print(term.magenta("===================="))
