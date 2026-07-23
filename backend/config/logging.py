"""
JARVIS Centralized Logging Module.

1. Why this module exists:
   Configures standard Python logging across all backend components, avoiding unstructured
   print statements and supporting structured log levels (DEBUG, INFO, WARNING, ERROR).

2. How it fits into the architecture:
   Cross-cutting infrastructure utility invoked during application startup before any
   components or services are instantiated.

3. Which future modules will interact with it:
   - All modules use `logging.getLogger("jarvis.<module_name>")` to output operational logs.
   - Future API endpoints, services, background workers, and tools.

4. Common mistakes to avoid:
   - Using `print()` statements for debugging instead of `logger.debug()` or `logger.info()`.
   - Calling `logging.basicConfig()` multiple times in different modules.

5. Possible future improvements:
   - JSON-formatted structured logs for production ELK / Datadog collectors.
   - Asynchronous log file handlers for high-throughput performance.
"""

import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO", debug: bool = False) -> None:
    """Configures the root logging output for JARVIS.

    Args:
        level: Minimum log level string (e.g. DEBUG, INFO, WARNING, ERROR).
        debug: If True, overrides level to DEBUG and enables detailed formatting.
    """
    log_level = logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO)

    log_format = (
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
        if debug
        else "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    )

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # Silence verbose third-party loggers if any are added in the future
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    root_logger = logging.getLogger("jarvis")
    root_logger.info("Logging initialized for JARVIS (Level: %s, Debug: %s)", level, debug)
