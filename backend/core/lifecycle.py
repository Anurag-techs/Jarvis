"""
JARVIS Application Lifecycle Management.

1. Why this module exists:
   Manages application execution lifecycles, enabling seamless delegation from `JarvisApplication.run()`
   to active user interfaces (Console, Web API, Voice loop).

2. How it fits into the architecture:
   Part of Core. Decouples application bootstrapper from specific interface execution modes.

3. Which future modules will interact with it:
   - `backend.core.startup.JarvisApplication`
   - `backend.interfaces.console.ConsoleInterface`
   - `backend.api.server` (Version 2.0 Web interface runner).

4. Common mistakes to avoid:
   - Coupling lifecycle manager directly to a single interface framework.

5. Possible future improvements:
   - Async event loop lifecycle runner with signal handlers.
"""

from abc import ABC, abstractmethod


class BaseLifecycleManager(ABC):
    """Abstract interface for managing interface lifecycle execution."""

    @abstractmethod
    def start(self) -> None:
        """Starts the interface event loop."""

    @abstractmethod
    def stop(self) -> None:
        """Stops the interface event loop cleanly."""
