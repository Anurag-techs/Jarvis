"""
JARVIS Core Module Package Initialization.

1. Why this module exists:
   Exposes central domain models, system exceptions, orchestrator, and startup manager.

2. How it fits into the architecture:
   Core domain layer export entrypoint.

3. Which future modules will interact with it:
   Application entrypoint (`main.py`), test suite, and future API endpoints.
"""

from backend.core.exceptions import (
    ConfigurationError,
    JarvisError,
    ProviderError,
    ToolExecutionError,
    VoiceError,
)
from backend.core.lifecycle import BaseLifecycleManager
from backend.core.models import CommandResult, UserIntent
from backend.core.orchestrator import SystemOrchestrator
from backend.core.startup import JarvisApplication, StartupManager

__all__ = [
    "SystemOrchestrator",
    "JarvisApplication",
    "StartupManager",
    "BaseLifecycleManager",
    "JarvisError",
    "ConfigurationError",
    "ProviderError",
    "ToolExecutionError",
    "VoiceError",
    "UserIntent",
    "CommandResult",
]
