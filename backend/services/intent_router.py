"""
JARVIS Intent Router.

1. Why this module exists:
   Provides a fast, deterministic pre-filter that classifies high-confidence user
   queries BEFORE handing control to the LLM. This eliminates unnecessary AI
   round-trips for intents that can be matched with high precision using keyword
   rules, and avoids false positives by requiring multi-word phrase matches or
   unambiguous single tokens.

2. How it fits into the architecture:
   Injected into SystemOrchestrator. The orchestrator calls `route()` first;
   if a tool name is returned the tool is executed directly, skipping LLM
   tool-selection. If `None` is returned, the LLM handles tool selection as usual.

3. Which future modules will interact with it:
   - backend.core.orchestrator.SystemOrchestrator

4. Common mistakes to avoid:
   - Adding very short, ambiguous keywords (e.g. "code") that will fire on
     unrelated queries like "What is error code 404?".
   - Using can_handle() inside a tool – routing is the sole responsibility of
     this service.

5. Possible future improvements:
   - Load rules from a YAML/JSON config file for runtime reconfiguration.
   - Confidence scoring and multi-label support.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("jarvis.services.intent_router")


# ---------------------------------------------------------------------------
# Rule set: each entry is (phrase_pattern, tool_name).
# Patterns are matched in ORDER – put more-specific phrases first.
# All patterns are compiled case-insensitively.
# Use word-boundary anchors (\b) where needed to prevent substring collisions.
# ---------------------------------------------------------------------------
_ROUTING_RULES: list[tuple[str, str]] = [
    # -- High-precision multi-word coding triggers ------------------------------
    (r"solve\s+this\s+(coding\s+)?(problem|question)", "coding_actions"),
    (r"coding\s+(problem|question|challenge)", "coding_actions"),
    (r"explain\s+(this\s+)?(mcq|multiple[\s-]choice)", "coding_actions"),
    (r"solve\s+(this\s+)?(mcq|multiple[\s-]choice)", "coding_actions"),
    (r"(generate|write|create)\s+(the\s+)?code\s+for\s+this", "coding_actions"),
    (r"insert\s+(the\s+)?code\s+into\s+(the\s+)?editor", "coding_actions"),
    (r"(solve|fix|answer)\s+(the\s+)?(visible|current|on[\s-]screen)\s+(problem|question|challenge)", "coding_actions"),

    # -- Application launches – open_application --------------------------------
    # These MUST NOT call the LLM – pure OS dispatch. Real Windows applications.
    (r"\bopen\s+(chrome|google[\s-]chrome)\b",                     "open_application"),
    (r"\bopen\s+(firefox|mozilla[\s-]firefox)\b",                  "open_application"),
    (r"\bopen\s+(edge|microsoft[\s-]edge)\b",                      "open_application"),
    (r"\bopen\s+(vscode|vs[\s-]code|visual[\s-]studio[\s-]code)\b", "open_application"),
    (r"\bopen\s+(notepad)\b",                                       "open_application"),
    (r"\bopen\s+(calculator|calc)\b",                               "open_application"),
    (r"\bopen\s+(explorer|file[\s-]explorer)\b",                    "open_application"),
    (r"\bopen\s+(terminal|cmd|command[\s-]prompt)\b",               "open_application"),
    (r"\bopen\s+(powershell)\b",                                    "open_application"),
    (r"\bopen\s+(spotify)\b",                                       "open_application"),
    (r"\bopen\s+(discord)\b",                                       "open_application"),
    (r"\bopen\s+(taskmgr|task[\s-]manager)\b",                      "open_application"),
    (r"\bopen\s+(paint|ms[\s-]paint)\b",                            "open_application"),
    (r"\bopen\s+(word|microsoft[\s-]word)\b",                       "open_application"),
    (r"\bopen\s+(excel|microsoft[\s-]excel)\b",                     "open_application"),
    (r"\bopen\s+(settings|windows[\s-]settings)\b",                 "open_application"),
    (r"\blaunch\s+(chrome|google[\s-]chrome|vscode|vs[\s-]code|spotify|discord|firefox|edge)\b", "open_application"),

    # -- Website / Service launches – open_website -----------------------------
    # User's personal & frequent websites: YouTube, LinkedIn, LeetCode, GitHub,
    # ChatGPT, Gmail, Google, NIAT/CCBP, and general online services.
    (
        r"\b(open|launch|go\s+to|visit|browse|navigate\s+to)\s+"
        r"(the\s+)?"
        r"(website\s+)?"
        r"(youtube|linkedin|linked[\s-]in|leetcode|leet[\s-]code|github|git[\s-]hub|chatgpt|chat[\s-]gpt|openai|gmail|google[\s-]mail|google[\s-]maps|google|niat|ccbp|niat[\s-]learning|instagram|insta|facebook|fb|whatsapp|whats[\s-]app|whatsapp[\s-]web|reddit|twitter|\bx\b)"
        r"(\s+website|\s+site|\s+page)?\b",
        "open_website",
    ),
    (r"\b(open|go\s+to)\s+(the\s+)?website\b", "open_website"),
    (r"\b(open|go\s+to|visit)\s+(https?://\S+|www\.\S+|\b\w+\.(com|org|net|io|ai|gov|edu)\b)", "open_website"),

    # -- Coding Agent – platform triggers without open/launch verb --------------
    (r"\bleetcode\b", "coding_actions"),
    (r"\bhackerrank\b", "coding_actions"),
    (r"\bcodeforces\b", "coding_actions"),
    (r"\bgeeksforgeeks\b", "coding_actions"),

    # -- System actions – system_actions / camera_actions -----------------------
    # Camera / Webcam actions
    (r"\b(open|use|start)\s+(the\s+)?(camera|webcam)\b",           "camera_actions"),
    (r"\btake\s+(a\s+)?(photo|picture|snapshot)\b",                "camera_actions"),
    (r"\bcapture\s+(from\s+)?(camera|webcam)\b",                   "camera_actions"),
    (r"\blist\s+(cameras?|webcams?)\b",                            "camera_actions"),
    (r"\banalyze\s+(camera|webcam)\b",                             "camera_actions"),

    (r"\b(shutdown|shut[\s-]down|power[\s-]off)\b",                "system_actions"),
    (r"\b(restart|reboot|re[\s-]start|re[\s-]boot)\b",             "system_actions"),
    (r"\bvolume\s+(up|down|mute|unmute)\b",                        "system_actions"),
    (r"\b(mute|unmute)\s+(volume|audio|sound|mic|microphone)?\b",   "system_actions"),
    (r"\b(increase|decrease|raise|lower)\s+(brightness|volume)\b", "system_actions"),
    (r"\bbrightness\s+(up|down)\b",                                "system_actions"),
    (r"\bscreen\s+(brighter|dimmer)\b",                            "system_actions"),
    (r"\block\s+(screen|computer|pc|windows)\b",                   "system_actions"),
    (r"\b(sleep|hibernate)\s+(mode|computer|pc)?\b",               "system_actions"),
]

# Pre-compile all patterns once at module load time for performance
_COMPILED_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE), tool_name)
    for pattern, tool_name in _ROUTING_RULES
]


class IntentRouter:
    """Deterministic, LLM-free intent classifier for high-confidence user queries.

    Usage::

        router = IntentRouter()
        tool_name = router.route("Solve this LeetCode problem")
        # -> "coding_actions"

        tool_name = router.route("Open YouTube")
        # -> "open_website"

        tool_name = router.route("What is the weather today?")
        # -> None  (falls through to LLM)
    """

    def route(self, query: str) -> Optional[str]:
        """Classifies a query and returns the target tool name, or None.

        Args:
            query: Raw user input (voice-transcribed or typed).

        Returns:
            Tool name string if the query matches a deterministic rule,
            ``None`` otherwise (delegates to LLM tool-selection).
        """
        if not query or not query.strip():
            return None

        for pattern, tool_name in _COMPILED_RULES:
            if pattern.search(query):
                logger.info(
                    "IntentRouter: deterministic route -> '%s' (matched pattern: %s)",
                    tool_name,
                    pattern.pattern,
                )
                return tool_name

        logger.debug("IntentRouter: no deterministic match for query=%r -> delegating to LLM", query[:80])
        return None
