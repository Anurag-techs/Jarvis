"""
JARVIS Desktop Automation Service.
Implements low-level OS tasks (files, command execution, clipboard, and browser control).
"""

import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from typing import Tuple

logger = logging.getLogger("jarvis.services.desktop")


class DesktopService:
    """Service to execute system-level desktop automation actions."""

    def launch_app(self, app_name: str) -> Tuple[bool, str]:
        logger.info("Launching application: %s", app_name)
        from backend.tools.application_tool import _normalize_app_name
        norm_app = _normalize_app_name(app_name)
        os_type = sys.platform

        try:
            if os_type == "win32":
                subprocess.Popen(["cmd", "/c", "start", "", norm_app], shell=True)
            elif os_type == "darwin":
                subprocess.Popen(["open", "-a", norm_app])
            else:
                subprocess.Popen([norm_app])
            return True, f"Application '{app_name}' launched successfully."
        except Exception as exc:
            logger.error("Failed to launch application '%s': %s", app_name, exc)
            return False, f"Could not launch application '{app_name}': {exc}"

    def create_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        logger.info("Creating file: %s", file_path)
        if os.path.exists(file_path):
            return False, f"File already exists at '{file_path}'."
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, f"File created successfully at '{file_path}'."
        except Exception as exc:
            logger.error("Failed to create file '%s': %s", file_path, exc)
            return False, f"Failed to create file: {exc}"

    def delete_file(self, file_path: str) -> Tuple[bool, str]:
        logger.info("Deleting file: %s", file_path)
        if not os.path.exists(file_path):
            return False, f"No file exists at '{file_path}'."
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            return True, f"File at '{file_path}' deleted successfully."
        except Exception as exc:
            logger.error("Failed to delete file '%s': %s", file_path, exc)
            return False, f"Failed to delete file: {exc}"

    def list_files(self, directory_path: str) -> Tuple[bool, str]:
        logger.info("Listing files in directory: %s", directory_path)
        if not os.path.exists(directory_path):
            return False, f"Directory does not exist at '{directory_path}'."
        if not os.path.isdir(directory_path):
            return False, f"Path '{directory_path}' is not a directory."
        try:
            items = os.listdir(directory_path)
            items_str = "\n".join(items) if items else "(empty directory)"
            return True, f"Contents of '{directory_path}':\n{items_str}"
        except Exception as exc:
            logger.error("Failed to list directory '%s': %s", directory_path, exc)
            return False, f"Failed to list directory: {exc}"

    def read_file(self, file_path: str) -> Tuple[bool, str]:
        logger.info("Reading file: %s", file_path)
        if not os.path.exists(file_path):
            return False, f"No file exists at '{file_path}'."
        if os.path.isdir(file_path):
            return False, f"Path '{file_path}' is a directory, not a file."
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return True, content
        except Exception as exc:
            logger.error("Failed to read file '%s': %s", file_path, exc)
            return False, f"Failed to read file: {exc}"

    def write_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        logger.info("Writing file: %s", file_path)
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, f"File written successfully at '{file_path}'."
        except Exception as exc:
            logger.error("Failed to write to file '%s': %s", file_path, exc)
            return False, f"Failed to write file: {exc}"

    def execute_terminal(self, command: str) -> Tuple[bool, str]:
        logger.info("Executing terminal command: %s", command)
        try:
            res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = res.stdout + res.stderr
            success = res.returncode == 0
            return success, output if output.strip() else f"Command exited with code {res.returncode}"
        except subprocess.TimeoutExpired:
            logger.error("Command timed out: %s", command)
            return False, "Command timed out after 30 seconds."
        except Exception as exc:
            logger.error("Failed to execute command: %s", exc)
            return False, f"Failed to execute command: {exc}"

    def set_clipboard(self, text: str) -> Tuple[bool, str]:
        logger.info("Setting clipboard text")
        os_type = sys.platform
        try:
            if os_type == "win32":
                proc = subprocess.Popen(
                    ["powershell.exe", "-NoProfile", "-Command", "Set-Clipboard -Value $Input"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                proc.communicate(input=text)
            elif os_type == "darwin":
                proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE, text=True)
                proc.communicate(input=text)
            else:
                if shutil.which("xclip"):
                    proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, text=True)
                    proc.communicate(input=text)
                elif shutil.which("xsel"):
                    proc = subprocess.Popen(["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE, text=True)
                    proc.communicate(input=text)
                else:
                    return False, "Clipboard utilities (xclip/xsel) not found on Linux."
            return True, "Clipboard content updated."
        except Exception as exc:
            logger.error("Failed to set clipboard: %s", exc)
            return False, f"Failed to set clipboard: {exc}"

    def get_clipboard(self) -> Tuple[bool, str]:
        logger.info("Reading clipboard text")
        os_type = sys.platform
        try:
            if os_type == "win32":
                res = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return True, res.stdout.strip()
            elif os_type == "darwin":
                res = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
                return True, res.stdout
            else:
                if shutil.which("xclip"):
                    res = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True, timeout=5)
                    return True, res.stdout
                elif shutil.which("xsel"):
                    res = subprocess.run(["xsel", "--clipboard", "--output"], capture_output=True, text=True, timeout=5)
                    return True, res.stdout
                else:
                    return False, "Clipboard utilities (xclip/xsel) not found on Linux."
        except Exception as exc:
            logger.error("Failed to read clipboard: %s", exc)
            return False, f"Failed to read clipboard: {exc}"

    def open_browser(self, url: str) -> Tuple[bool, str]:
        logger.info("Opening URL in browser: %s", url)
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        try:
            webbrowser.open(url)
            return True, f"Opened website: {url}"
        except Exception as exc:
            logger.error("Failed to open browser URL '%s': %s", url, exc)
            return False, f"Failed to open URL: {exc}"

    def search_browser(self, query: str) -> Tuple[bool, str]:
        logger.info("Searching browser for: %s", query)
        url = f"https://www.google.com/search?q={query}"
        try:
            webbrowser.open(url)
            return True, f"Opened search query in browser: {query}"
        except Exception as exc:
            logger.error("Failed to perform browser search for '%s': %s", query, exc)
            return False, f"Failed to perform search: {exc}"

    # Low-level Mouse & Keyboard wrappers (pyautogui)
    def move_mouse(self, x: int, y: int) -> Tuple[bool, str]:
        logger.info("Moving mouse to (%d, %d)", x, y)
        try:
            import pyautogui
            pyautogui.moveTo(x, y, duration=0.2)
            return True, f"Moved mouse to ({x}, {y})"
        except Exception as exc:
            logger.error("Failed to move mouse: %s", exc)
            return False, f"Failed to move mouse: {exc}"

    def left_click(self) -> Tuple[bool, str]:
        logger.info("Performing left click")
        try:
            import pyautogui
            pyautogui.click()
            return True, "Left click performed"
        except Exception as exc:
            logger.error("Failed left click: %s", exc)
            return False, f"Failed left click: {exc}"

    def double_click(self) -> Tuple[bool, str]:
        logger.info("Performing double click")
        try:
            import pyautogui
            pyautogui.doubleClick()
            return True, "Double click performed"
        except Exception as exc:
            logger.error("Failed double click: %s", exc)
            return False, f"Failed double click: {exc}"

    def right_click(self) -> Tuple[bool, str]:
        logger.info("Performing right click")
        try:
            import pyautogui
            pyautogui.click(button='right')
            return True, "Right click performed"
        except Exception as exc:
            logger.error("Failed right click: %s", exc)
            return False, f"Failed right click: {exc}"

    def drag(self, x: int, y: int) -> Tuple[bool, str]:
        logger.info("Dragging to (%d, %d)", x, y)
        try:
            import pyautogui
            pyautogui.dragTo(x, y, duration=0.2)
            return True, f"Dragged to ({x}, {y})"
        except Exception as exc:
            logger.error("Failed to drag: %s", exc)
            return False, f"Failed to drag: {exc}"

    def scroll(self, clicks: int) -> Tuple[bool, str]:
        logger.info("Scrolling %d clicks", clicks)
        try:
            import pyautogui
            pyautogui.scroll(clicks)
            return True, f"Scrolled {clicks} clicks"
        except Exception as exc:
            logger.error("Failed to scroll: %s", exc)
            return False, f"Failed to scroll: {exc}"

    def get_mouse_position(self) -> Tuple[bool, str]:
        try:
            import pyautogui
            pos = pyautogui.position()
            return True, f"{pos[0]},{pos[1]}"
        except Exception as exc:
            logger.error("Failed to get mouse position: %s", exc)
            return False, f"Failed to get mouse position: {exc}"

    def type_text(self, text: str) -> Tuple[bool, str]:
        logger.info("Typing text: %s", text)
        try:
            import pyautogui
            pyautogui.write(text, interval=0.01)
            return True, f"Typed text: {text}"
        except Exception as exc:
            logger.error("Failed to type text: %s", exc)
            return False, f"Failed to type text: {exc}"

    def press_key(self, key: str) -> Tuple[bool, str]:
        logger.info("Pressing key: %s", key)
        try:
            import pyautogui
            pyautogui.press(key)
            return True, f"Pressed key: {key}"
        except Exception as exc:
            logger.error("Failed to press key: %s", exc)
            return False, f"Failed to press key: {exc}"

    def hotkey(self, *keys: str) -> Tuple[bool, str]:
        logger.info("Pressing hotkey combo: %s", keys)
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            return True, f"Pressed hotkey: {keys}"
        except Exception as exc:
            logger.error("Failed to press hotkey: %s", exc)
            return False, f"Failed to press hotkey: {exc}"

    def key_down(self, key: str) -> Tuple[bool, str]:
        logger.info("Key down: %s", key)
        try:
            import pyautogui
            pyautogui.keyDown(key)
            return True, f"Key down: {key}"
        except Exception as exc:
            logger.error("Failed key down: %s", exc)
            return False, f"Failed key down: {exc}"

    def key_up(self, key: str) -> Tuple[bool, str]:
        logger.info("Key up: %s", key)
        try:
            import pyautogui
            pyautogui.keyUp(key)
            return True, f"Key up: {key}"
        except Exception as exc:
            logger.error("Failed key up: %s", exc)
            return False, f"Failed key up: {exc}"
