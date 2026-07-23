"""
JARVIS Startup Banner Utility.

1. Why this module exists:
   Encapsulates console startup banner formatting separately from initialization logic.

2. How it fits into the architecture:
   Utility layer. Invoked during application startup.

3. Which future modules will interact with it:
   - `backend.core.startup.StartupManager`

4. Common mistakes to avoid:
   - Mixing console formatting ASCII art inside business orchestrators or main.py.

5. Possible future improvements:
   - Colorized ANSI output for supported terminal environments.
"""


def display_banner(version: str = "1.0") -> None:
    """Prints the official JARVIS startup console banner."""
    banner = f"""
====================================================
  JARVIS v{version}
  Production-Quality AI Assistant Foundation
====================================================
"""
    print(banner)
