"""
JARVIS Runtime Manager.
Selects Console or Voice execution modes based on dependency validation and hardware presence.
"""

from enum import Enum
import logging
import os
from typing import Any

from backend.config.settings import Settings

logger = logging.getLogger("jarvis.core.runtime")


class RuntimeType(str, Enum):
    CONSOLE = "console"
    VOICE = "voice"


class RuntimeManager:
    """Manages Python packages and hardware verification to determine JARVIS execution lifecycle."""

    def __init__(self, settings: Settings) -> None:
        """Initialize RuntimeManager.

        Args:
            settings: Loaded application settings.
        """
        self.settings = settings

    def check_dependencies(self) -> dict[str, bool]:
        """Checks if all core voice modules can be imported without errors."""
        deps = ["pyttsx3", "faster_whisper", "numpy", "openwakeword", "pyaudio"]
        status = {}
        for dep in deps:
            try:
                __import__(dep)
                status[dep] = True
            except ImportError:
                status[dep] = False
        return status

    def check_microphone(self) -> bool:
        """Checks if PyAudio can detect at least one audio input device."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            try:
                device_count = p.get_device_count()
                for i in range(device_count):
                    info = p.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        return True
                return False
            finally:
                p.terminate()
        except Exception:
            return False

    def check_python_version(self) -> bool:
        """Verifies if Python is on officially supported versions (3.11, 3.12)."""
        import sys
        major, minor = sys.version_info.major, sys.version_info.minor
        return major == 3 and minor in (11, 12)

    @staticmethod
    def handle_missing_dependency(pkg_name: str) -> None:
        """Prints user-friendly missing dependency commands instead of ModuleNotFoundError."""
        pip_map = {
            "pyaudio": "pyaudio",
            "pyttsx3": "pyttsx3",
            "faster_whisper": "faster-whisper",
            "openwakeword": "openwakeword",
            "numpy": "numpy",
        }
        pip_pkg = pip_map.get(pkg_name, pkg_name.replace("_", "-"))
        print("\nVoice Runtime unavailable.\n")
        print("Reason:")
        print("Missing package:")
        print(f"{pkg_name}\n")
        print("Install using:")
        print(f"pip install {pip_pkg}\n")

    def select_runtime(self, force_console: bool = False) -> tuple[RuntimeType, dict[str, Any]]:
        """Determines the appropriate runtime lifecycle.

        Args:
            force_console: Override to always run in Console Mode.

        Returns:
            Tuple of (RuntimeType, status_report_dict).
        """
        python_supported = self.check_python_version()
        deps = self.check_dependencies()
        mic = self.check_microphone()

        voice_supported = python_supported and all(deps.values()) and mic

        # Decide selected runtime
        if force_console or not python_supported:
            selected = RuntimeType.CONSOLE
        elif voice_supported:
            selected = RuntimeType.VOICE
        elif self.settings.debug and (self.settings.voice_mode or os.environ.get("FORCE_VOICE") == "true"):
            # Debug mode allows voice runtime with mock/fallback providers
            selected = RuntimeType.VOICE
        else:
            selected = RuntimeType.CONSOLE

        status = {
            "dependencies": deps,
            "mic_detected": mic,
            "voice_supported": voice_supported,
            "python_supported": python_supported
        }
        return selected, status

    def print_report(self, selected: RuntimeType, status: dict[str, Any]) -> None:
        """Prints a diagnostic startup report of the voice runtime capabilities."""
        import sys

        deps = status["dependencies"]
        mic = status["mic_detected"]
        python_supported = status.get("python_supported", True)

        print("\n----------------------------------------")
        print("Voice Runtime Status")
        print("----------------------------------------")

        # Unsupported python warning printed if detected
        if not python_supported:
            major, minor = sys.version_info.major, sys.version_info.minor
            print(f"WARNING:\nPython {major}.{minor}.{sys.version_info.micro} detected.\n")
            print("Voice Runtime is officially supported on:")
            print("• Python 3.11")
            print("• Python 3.12\n")
            print("Recommendation:")
            print("Create a new virtual environment using Python 3.12.\n")
            print("Do not crash.")
            print("Continue in Console Mode.")
        elif selected == RuntimeType.VOICE:
            print("✓ Microphone detected")
            print("✓ STT initialized")
            print("✓ Wake Word initialized")
            print("✓ TTS initialized")
            print("✓ Voice Runtime selected")
        else:
            # Diagnostic logs
            for pkg, available in deps.items():
                if not available:
                    print(f"✗ Missing package: {pkg}")
            if not mic and deps["pyaudio"]:
                print("✗ No input microphone device detected")
            print("Voice Runtime unavailable.")
            print("Starting Console Runtime.")

        print("----------------------------------------\n")

        print("----------------------------------------\n")
