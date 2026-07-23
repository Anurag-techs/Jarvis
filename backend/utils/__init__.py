"""
JARVIS Utilities Package Initialization.

1. Why this module exists:
   Exposes cross-cutting system, process execution, and string helper functions.

2. How it fits into the architecture:
   Cross-platform helper utilities used by backend tools and services.

3. Which future modules will interact with it:
   - `backend.tools.*`
   - `backend.services.*`

4. Common mistakes to avoid:
   - Putting domain models or business logic inside general utility packages.

5. Possible future improvements:
   - Platform-agnostic process sandbox and isolated subprocess execution runners.
"""

from backend.utils.system import run_system_command

__all__ = ["run_system_command"]
