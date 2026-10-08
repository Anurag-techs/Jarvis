"""
JARVIS Transcript Normalizer Service.

1. Why this module exists:
   Speech-to-Text engines (including Faster-Whisper) frequently mis-hear proper nouns,
   brand names, and compound words. This service applies a deterministic dictionary of
   known STT errors → correct forms BEFORE the transcript reaches IntentRouter or the LLM.
   This improves routing accuracy and command recognition without touching the STT engine.

2. How it fits into the architecture:
   Injected into VoiceController. Called in _handle_processing() AFTER transcription and
   BEFORE the confidence gate / intent routing. Pure, stateless, side-effect-free.

3. Which future modules will interact with it:
   - backend.interfaces.voice.VoiceController
   - Future console interface (ConsoleInterface) can optionally inject it too.

4. Common mistakes to avoid:
   - Adding very short substitutions like "code" -> "X" that fire on unrelated sentences.
   - Applying replacements in a random order that causes cascading rewrites.

5. Possible future improvements:
   - Load the dictionary from a user-editable YAML file for zero-code customizations.
   - Phonetic similarity matching (edit distance) for unknown words.
"""

import logging
import re
from typing import Tuple

logger = logging.getLogger("jarvis.services.transcript_normalizer")

# ---------------------------------------------------------------------------
# Master normalization dictionary.
# Keys: all lowercase, may be multi-word phrases.
# Values: the intended, correctly-spelled form.
# Replacements are applied in ORDER — more specific phrases first.
# ---------------------------------------------------------------------------
_NORMALIZATION_MAP: dict[str, str] = {
    # ── Application names ─────────────────────────────────────────────────
    "open crome":              "open chrome",
    "open chorme":             "open chrome",
    "open crom":               "open chrome",
    "google crome":            "google chrome",
    "chrome browser":          "google chrome",
    "open vs code":            "open vscode",
    "open visual studio code": "open vscode",
    "open visual studio":      "open vscode",
    "open vs":                 "open vscode",
    "open note pad":           "open notepad",
    "north air":               "notepad",
    "notebook":                "notepad",
    "open you tube":           "open youtube",
    "you tube":                "youtube",
    "open you-tube":           "open youtube",
    "open calc":               "open calculator",
    "open caliculator":        "open calculator",
    "open calulator":          "open calculator",
    "open spotifiy":           "open spotify",
    "open spotfy":             "open spotify",
    "open disc cord":          "open discord",
    "open disc-cord":          "open discord",
    "open file explorer":      "open explorer",
    "file explorar":           "file explorer",
    "open task manager":       "open taskmgr",
    "open task man":           "open taskmgr",
    "open power shell":        "open powershell",
    "open command prompt":     "open cmd",
    "open terminal":           "open cmd",

    # ── Coding platforms ──────────────────────────────────────────────────
    "leet code":               "leetcode",
    "lead code":               "leetcode",
    "lee code":                "leetcode",
    "lit code":                "leetcode",
    "hacker rank":             "hackerrank",
    "hackar rank":             "hackerrank",
    "code forces":             "codeforces",
    "geeks for geeks":         "geeksforgeeks",
    "geek for geeks":          "geeksforgeeks",

    # ── AI / Web services ─────────────────────────────────────────────────
    "g p t":                   "chatgpt",
    "chat g p t":              "chatgpt",
    "chat gbt":                "chatgpt",
    "chat g b t":              "chatgpt",
    "googel":                  "google",
    "gogle":                   "google",
    "goolge":                  "google",
    "bing ai":                 "bing",

    # ── System commands ───────────────────────────────────────────────────
    "shut down":               "shutdown",
    "shut-down":               "shutdown",
    "shut of":                 "shutdown",
    "power off":               "shutdown",
    "re start":                "restart",
    "re-start":                "restart",
    "re boot":                 "reboot",
    "re-boot":                 "reboot",
    "vol up":                  "volume up",
    "vol down":                "volume down",
    "volume higher":           "volume up",
    "volume lower":            "volume down",
    "make it louder":          "volume up",
    "make it quieter":         "volume down",
    "turn up volume":          "volume up",
    "turn down volume":        "volume down",
    "brighter":                "increase brightness",
    "dimmer":                  "decrease brightness",
    "brightness up":           "increase brightness",
    "brightness down":         "decrease brightness",
    "screen brighter":         "increase brightness",
    "screen dimmer":           "decrease brightness",

    # ── Search / web ──────────────────────────────────────────────────────
    "search for":              "search",
    "look up":                 "search",
    "look for":                "search",
    "find me":                 "search",
    "what is the weather in":  "weather in",
    "whats the weather in":    "weather in",
    "what's the weather":      "weather",

    # ── Filler / preamble removal ─────────────────────────────────────────
    "hey jarvis":              "",
    "okay jarvis":             "",
    "ok jarvis":               "",
    "jarvis":                  "",
}

# Filler words that often appear at the very start of a transcript
_FILLER_PREFIXES: tuple[str, ...] = (
    "um ", "uh ", "ah ", "hmm ", "err ", "like ",
)

# Pre-compile all substitution patterns once (longest key first to avoid partial matches)
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(re.escape(src), re.IGNORECASE), tgt)
    for src, tgt in sorted(_NORMALIZATION_MAP.items(), key=lambda kv: -len(kv[0]))
]


class TranscriptNormalizer:
    """Lightweight, dictionary-based transcript cleaner.

    Applies known STT mis-hearing corrections and strips filler words.
    Designed to be injected into VoiceController and called once per transcript.

    Usage::

        normalizer = TranscriptNormalizer()
        normalized, changed = normalizer.normalize("Open Crome")
        # normalized → "open chrome", changed → True
    """

    def normalize(self, text: str) -> Tuple[str, bool]:
        """Normalizes an STT transcript.

        Args:
            text: Raw transcript string as returned by the STT engine.

        Returns:
            A tuple of (normalized_text, was_changed) where was_changed is True
            if any substitution was applied.
        """
        if not text or not text.strip():
            return text, False

        original = text.strip()
        result = original

        # 1. Strip leading filler prefixes (case-insensitive)
        lower = result.lower()
        for filler in _FILLER_PREFIXES:
            if lower.startswith(filler):
                result = result[len(filler):].strip()
                lower = result.lower()
                break

        # 2. Apply dictionary substitutions
        for pattern, replacement in _COMPILED_PATTERNS:
            result = pattern.sub(replacement, result)

        # 3. Collapse multiple spaces and strip again
        result = " ".join(result.split()).strip()

        changed = result.lower() != original.lower()
        if changed:
            logger.info(
                "TranscriptNormalizer: '%s' → '%s'",
                original, result,
            )
        else:
            logger.debug("TranscriptNormalizer: no changes for '%s'", original[:60])

        return result, changed
