"""
JARVIS Voice Dependencies Installer.
Provides automatic installation for pyttsx3, faster-whisper, numpy, openwakeword, and pyaudio.
"""

import subprocess
import sys


def install_voice_dependencies() -> None:
    """Installs required voice packages via pip subprocess calls."""
    packages = ["numpy", "pyttsx3", "faster-whisper", "openwakeword", "pyaudio"]
    print("\n=========================================")
    print("Installing JARVIS Voice Dependencies")
    print("=========================================\n")

    failed = []
    for pkg in packages:
        # Pyaudio package name matches on PyPI, but faster-whisper is with hyphen.
        # Let's clean the name for print.
        print(f"Installing {pkg}...")
        try:
            # sys.executable ensures we install to the active virtual environment python interpreter
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"✗ Failed to install {pkg}")
                print(f"Error details:\n{result.stderr.strip()}\n")
                failed.append((pkg, result.stderr.strip()))
            else:
                print(f"✓ Successfully installed {pkg}")
        except Exception as exc:
            print(f"✗ Unexpected error installing {pkg}: {exc}\n")
            failed.append((pkg, str(exc)))

    print("=========================================")
    if failed:
        print("Dependency installation completed with errors.")
        for pkg, _ in failed:
            print(f"- {pkg} failed")
        print("=========================================\n")
        sys.exit(1)
    else:
        print("All voice dependencies successfully installed!")
        print("=========================================\n")


if __name__ == "__main__":
    install_voice_dependencies()
