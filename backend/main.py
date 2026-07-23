"""
JARVIS Application Entrypoint.

1. Why this module exists:
   Serves as the execution entrypoint for running JARVIS Sprint 1 Application Bootstrap.

2. How it fits into the architecture:
   Presentation / Entrypoint layer. Contains NO business logic.
   Delegates bootstrap execution strictly to `backend.core.startup.StartupManager`.

3. Which future modules will interact with it:
   CLI runner (`python -m backend.main`).

4. Common mistakes to avoid:
   - Adding tool registration, setting parsing, or business logic directly inside main.py.

5. Possible future improvements:
   - Command line flag parsing (e.g. `--version`, `--debug`).
"""

import sys
from backend.core.startup import StartupManager
from backend.core.exceptions import JarvisError


def main() -> None:
    """Main application entrypoint executing StartupManager bootstrap."""
    try:
        startup_manager = StartupManager()
        app_container = startup_manager.bootstrap()
    except JarvisError as exc:
        print(f"CRITICAL: Failed to start JARVIS: {exc.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"CRITICAL: Unexpected error during JARVIS startup: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
