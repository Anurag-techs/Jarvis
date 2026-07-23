"""
JARVIS Interfaces Package Initialization.

1. Why this module exists:
   Exposes Console and Voice interfaces.
"""

from backend.interfaces.console import ConsoleInterface
from backend.interfaces.voice import VoiceController

__all__ = [
    "ConsoleInterface",
    "VoiceController",
]
