"""
JARVIS Environment Verification Script.
Checks Python version, virtual environment status, microphone, speakers, dependencies, and API credentials.
"""

import os
import sys


def check_environment() -> int:
    """Performs validation checks and returns exit code 0 if all core requirements pass, else 1."""
    print("\n=========================================")
    print("JARVIS Environment Checker")
    print("=========================================\n")

    overall_pass = True

    # 1. Python Version
    major, minor = sys.version_info.major, sys.version_info.minor
    python_supported = major == 3 and minor in (11, 12)
    python_ver_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if python_supported:
        print(f"✓ Installed: Python version {python_ver_str} (Supported)")
    else:
        print(f"✗ Missing: Python version {python_ver_str} (Unsupported. Requires 3.11 or 3.12)")
        overall_pass = False

    # 2. Virtual Environment
    venv_active = (sys.prefix != sys.base_prefix) or ("VIRTUAL_ENV" in os.environ)
    if venv_active:
        print("✓ Installed: Virtual Environment (Active)")
    else:
        print("✗ Missing: Virtual Environment (Inactive/Not Detected)")
        overall_pass = False

    # 3. Audio Interfaces (Mic & Speaker)
    mic_ok = False
    speaker_ok = False
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        try:
            device_count = p.get_device_count()
            for i in range(device_count):
                info = p.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    mic_ok = True
                if info.get("maxOutputChannels", 0) > 0:
                    speaker_ok = True
        finally:
            p.terminate()
    except Exception:
        pass

    if mic_ok:
        print("✓ Installed: Microphone device detected")
    else:
        print("✗ Missing: Microphone device not detected")
        overall_pass = False

    if speaker_ok:
        print("✓ Installed: Speaker device detected")
    else:
        print("✗ Missing: Speaker device not detected")
        overall_pass = False

    # 4. Voice Packages
    packages = ["numpy", "pyttsx3", "faster_whisper", "openwakeword", "pyaudio"]
    for pkg in packages:
        import_name = pkg.replace("-", "_")
        try:
            __import__(import_name)
            print(f"✓ Installed: {pkg}")
        except ImportError:
            print(f"✗ Missing: {pkg}")
            overall_pass = False

    # 5. GPU Support (Optional)
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    if gpu_available:
        print("✓ Installed: GPU support (Optional, CUDA available)")
    else:
        print("✗ Missing: GPU support (Optional, CPU-only execution will be used)")

    # 6. Gemini Credentials
    api_key_set = os.environ.get("GEMINI_API_KEY") is not None
    if not api_key_set:
        try:
            from backend.config.settings import get_settings
            api_key_set = get_settings().gemini_api_key is not None
        except Exception:
            pass

    if api_key_set:
        print("✓ Installed: Gemini API Key Configured")
    else:
        print("✗ Missing: Gemini API Key Configuration (Set GEMINI_API_KEY in .env)")
        overall_pass = False

    print("\n=========================================")
    if overall_pass:
        print("Environment Verification: PASS")
        print("=========================================\n")
        return 0
    else:
        print("Environment Verification: FAIL")
        print("=========================================\n")
        return 1


if __name__ == "__main__":
    sys.exit(check_environment())
