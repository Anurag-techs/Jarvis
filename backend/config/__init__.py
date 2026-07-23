"""
JARVIS Configuration Module.

1. Why this module exists:
   Exposes configuration settings and logging initialization functions to the application.

2. How it fits into the architecture:
   Acts as the central point for importing application settings and setting up structured logging.

3. Which future modules will interact with it:
   All backend modules, services, tools, and main entrypoint.

4. Common mistakes to avoid:
   Importing settings directly without loading environment variables first.

5. Possible future improvements:
   Dynamic configuration re-loading during runtime without restart.
"""

from backend.config.logging import setup_logging
from backend.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings", "setup_logging"]
