"""
JARVIS Application Lifecycle Management Skeleton.

1. Why this module exists:
   Skeleton placeholder for managing future full application lifecycle transitions
   (Startup, Main Run Loop, Graceful Shutdown, Signal Handling).

2. How it fits into the architecture:
   Part of Core. Will coordinate lifecycle states across services and server processes in Version 2.0+.

3. Which future modules will interact with it:
   - `backend.main`
   - `backend.api.server`

4. Common mistakes to avoid:
   - Overengineering lifecycle state machines before Version 2.0 API requirements exist.

5. Possible future improvements:
   - Async signal handlers (SIGINT, SIGTERM) and shutdown hook registries.
"""

from abc import ABC, abstractmethod


class BaseLifecycleManager(ABC):
    """Abstract interface skeleton for application lifecycle management."""

    @abstractmethod
    def start(self) -> None:
        """Starts application lifecycle services."""

    @abstractmethod
    def stop(self) -> None:
        """Stops application lifecycle services gracefully."""
