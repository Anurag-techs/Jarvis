"""
JARVIS Reusable Logging Utility.

1. Why this module exists:
   Provides a centralized logger factory and logging setup utility adhering to requirement 3.
   Supports console logging, timestamps, module names, default INFO level, and future file logging extension.

2. How it fits into the architecture:
   Part of the Utility layer. Used across all modules for consistent log formatting.

3. Which future modules will interact with it:
   All backend modules, services, tools, and entrypoints.

4. Common mistakes to avoid:
   - Direct `print()` debugging.
   - Instantiating multiple uncoordinated log handlers.

5. Possible future improvements:
   - File logging handlers (RotatingFileHandler / TimedRotatingFileHandler).
"""

import logging
import sys
from typing import Optional


def setup_logger(
    level: str = "INFO",
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configures and returns the root logger for JARVIS.

    Args:
        level: Log level threshold (default "INFO").
        log_format: Custom log format string if provided.

    Returns:
        Configured root logger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if log_format is None:
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"

    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    root_logger = logging.getLogger("jarvis")
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on re-initialization
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    root_logger.propagate = False
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Factory function returning a child logger for a specific module.

    Args:
        name: Module identifier name (e.g. 'jarvis.core.startup').

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
