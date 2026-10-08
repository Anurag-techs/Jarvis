"""
Coding Agent - Editor Controller.
Detects the active editor and inserts generated code via clipboard-paste using DesktopService.
"""

import logging
import re
import time
from typing import Tuple

from backend.agents.coding_agent.interfaces import BaseEditorController
from backend.agents.coding_agent.models import CodeSolution, EditorContext
from backend.agents.computer_agent.models import WorldState
from backend.services.desktop import DesktopService

logger = logging.getLogger("jarvis.agents.coding_agent.editor_controller")

# Maps window title keywords to editor type names
_EDITOR_SIGNATURES = {
    "visual studio code": "vscode",
    "vs code": "vscode",
    "code - ": "vscode",      # VS Code window title pattern: "filename - Code"
    "notepad++": "notepad++",
    "sublime text": "sublime",
    "intellij": "jetbrains",
    "pycharm": "jetbrains",
    "webstorm": "jetbrains",
    "clion": "jetbrains",
    "android studio": "jetbrains",
    "atom": "atom",
    "vim": "vim",
    "neovim": "neovim",
    "emacs": "emacs",
    "notepad": "notepad",
}


class CodingEditorController(BaseEditorController):
    """Detects editor windows and inserts code via clipboard paste."""

    def __init__(self, desktop_service: DesktopService) -> None:
        self._desktop = desktop_service

    def detect_editor(self, world_state: WorldState) -> EditorContext:
        """Identifies the active editor from the current WorldState."""
        title = world_state.active_window
        title_lower = title.lower()

        editor_type = "generic"
        for keyword, etype in _EDITOR_SIGNATURES.items():
            if keyword in title_lower:
                editor_type = etype
                break

        # Try to extract file extension from window title
        file_ext = self._extract_extension(title)

        # Check if editor likely has existing code (heuristic: OCR has code-like tokens)
        has_existing = self._detect_existing_code(world_state.ocr_text)

        context = EditorContext(
            editor_type=editor_type,
            window_title=title,
            has_existing_code=has_existing,
            file_extension=file_ext,
        )

        logger.info(
            "Detected editor: type=%s | title='%s' | extension=%s | has_code=%s",
            context.editor_type, context.window_title,
            context.file_extension, context.has_existing_code,
        )
        return context

    def insert_code(self, solution: CodeSolution, editor_context: EditorContext) -> Tuple[bool, str]:
        """Inserts generated code into the editor using clipboard paste.

        Strategy:
            1. Set clipboard to the generated code
            2. Select all existing content (Ctrl+A)
            3. Paste from clipboard (Ctrl+V) — replaces selected content
        """
        logger.info("Inserting code into %s editor...", editor_context.editor_type)

        try:
            # Step 1: Copy code to clipboard
            ok, msg = self._desktop.set_clipboard(solution.source_code)
            if not ok:
                return False, f"Failed to set clipboard: {msg}"
            logger.debug("Code copied to clipboard (%d chars)", len(solution.source_code))

            # Brief pause to ensure clipboard is ready
            time.sleep(0.3)

            # Step 2: Select all existing content
            ok, msg = self._desktop.hotkey("ctrl", "a")
            if not ok:
                return False, f"Failed to select all: {msg}"
            time.sleep(0.2)

            # Step 3: Paste from clipboard (replaces selection)
            ok, msg = self._desktop.hotkey("ctrl", "v")
            if not ok:
                return False, f"Failed to paste: {msg}"

            logger.info("Code inserted successfully via clipboard paste.")
            return True, "Code inserted via clipboard paste"

        except Exception as e:
            logger.error("Code insertion failed: %s", e)
            return False, f"Code insertion failed: {e}"

    def _extract_extension(self, title: str) -> str | None:
        """Extracts file extension from editor window title."""
        # Common patterns: "filename.py - VS Code", "filename.cpp - Sublime Text"
        match = re.search(r'(\.\w{1,10})\b', title)
        if match:
            return match.group(1)
        return None

    def _detect_existing_code(self, ocr_text: str) -> bool:
        """Heuristic check for existing code in the editor based on OCR text."""
        if not ocr_text:
            return False

        # Code-like indicators
        code_patterns = [
            r'\bdef\s+\w+',      # Python function
            r'\bclass\s+\w+',    # Class definition
            r'\bimport\s+\w+',   # Import statement
            r'\bfor\s+\w+\s+in', # For loop
            r'\bif\s+\w+',       # If statement
            r'[{};]',            # C-style syntax
            r'#include',         # C/C++ include
            r'public\s+class',   # Java class
        ]

        for pattern in code_patterns:
            if re.search(pattern, ocr_text):
                return True
        return False


# Backward compatibility alias
DesktopEditorController = CodingEditorController
