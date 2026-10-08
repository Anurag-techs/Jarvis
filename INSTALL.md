# JARVIS Installation and Environment Guide

This document guides you through installing and configuring JARVIS on a Windows machine.

---

## Supported Python Versions

Voice Runtime and speech components are officially supported and tested on:
- **Python 3.11**
- **Python 3.12**

*Note: Running on other versions (such as Python 3.14) will display a warning and fall back to Console Mode rather than crashing.*

---

## Installation Steps (Windows)

For a fresh install on Windows:

1. **Open PowerShell** and navigate to your workspace directory.

2. **Create a virtual environment** using Python 3.12:
   ```powershell
   py -3.12 -m venv .venv
   ```

3. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Run the setup script**:
   ```powershell
   .\setup.ps1
   ```
   This script will automatically upgrade `pip`, install core dependencies (`requirements.txt`), install voice dependencies (`requirements-voice.txt`), and run verification checks.

---

## Execution

### Run JARVIS Hands-Free (Voice Mode)
To launch JARVIS with continuous wake-word listening:
```powershell
py -m backend.main
```

### Run JARVIS Debug Console Mode
To force JARVIS to start in the interactive text-based console mode:
```powershell
py -m backend.main --console
```

---

## How to Update Dependencies

To upgrade or update dependencies later, run:
```powershell
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements-voice.txt
```

---

## Troubleshooting Missing Packages

If the environment checker or startup reports a missing package (e.g., `faster_whisper`):

```
Voice Runtime unavailable.

Reason:
Missing package:
faster_whisper

Install using:
pip install faster-whisper
```

Run the specific install command manually:
```powershell
pip install faster-whisper
```
Or run the automatic dependency installer script:
```powershell
python backend/scripts/install_voice_dependencies.py
```

---

## Expected Startup Output (Voice Mode Success)

```
=====================================
JARVIS Environment Report
=====================================
Python: ✓ Ready (Version 3.12)
Virtual Environment: ✓ Ready
Gemini API: ✓ Ready
Voice: ✓ Ready
TTS: ✓ Ready
STT: ✓ Ready
Wake Word: ✓ Ready
Microphone: ✓ Ready
Speaker: ✓ Ready
Vision: ✓ Ready
Planner: ✓ Ready
Memory: ✓ Ready
Desktop Automation: ✓ Ready
=====================================
```
