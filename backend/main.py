"""
JARVIS Application Entrypoint.

1. Why this module exists:
   Serves as the main operational execution entrypoint for JARVIS.

2. How it fits into the architecture:
   Presentation / Entrypoint layer. Contains NO business logic.
   Delegates bootstrap execution to `StartupManager().bootstrap()` and runs `app.run(voice_mode=...)`.
   Supports `--voice` and `--test-stt` CLI flags.

3. Which future modules will interact with it:
   CLI execution environment (`python -m backend.main`).
"""

import sys
from backend.core.exceptions import JarvisError
from backend.core.startup import StartupManager


def main() -> None:
    """Main application entrypoint bootstrapping and launching JARVIS."""
    args = sys.argv[1:]

    try:
        app = StartupManager().bootstrap()

        # Handle --test-stt verification mode
        if "--test-stt" in args:
            print("\n[STT Verification Mode]")
            print("Listening to microphone for 5 seconds...")
            transcript = app.stt_provider.listen_and_transcribe(duration=5.0)
            print(f"\n[Transcribed Text]: '{transcript}'")
            return

        # Determine voice_mode flag from CLI arguments
        voice_mode = "--voice" in args

        # Default application lifecycle run
        app.run(voice_mode=voice_mode)

    except JarvisError as exc:
        print(f"CRITICAL: Failed to start JARVIS: {exc.message}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"CRITICAL: Unexpected error during JARVIS execution: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
