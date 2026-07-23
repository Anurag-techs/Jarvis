"""
JARVIS Memory Pre-Filter.

1. Why this module exists:
   Applies lightweight, rule-based heuristics on user queries to calculate a confidence score
   indicating if the input represents a potential memory event (storing or updating facts/preferences).
   This avoids calling LLM extraction for non-memorable or trivial conversation turns.

2. How it fits into the architecture:
   Used as the entry gate in the MemoryPipeline. If the confidence score is below the configured
   PREFILTER_THRESHOLD, we bypass LLM-based extraction.
"""

import re


class MemoryPreFilter:
    """Lightweight heuristic pre-filter calculating a confidence score for memorable inputs."""

    def __init__(self) -> None:
        # Explicit memory action keywords (remember, remind, forget, recall, memorize)
        self._action_pattern = re.compile(
            r"\b(remember|remind|forget|recall|memorize|dont forget|do not forget|save this)\b", 
            re.IGNORECASE
        )

        # First-person statements/possessives (I, my, me, mine, myself)
        self._first_person_pattern = re.compile(
            r"\b(my|i|me|mine|myself)\b",
            re.IGNORECASE
        )

        # Preference and identity patterns (live in, work at, like to, dislike, hobby, job, favorite, prefer)
        self._preference_pattern = re.compile(
            r"\b(live in|work at|likes? to|dislikes?|hobbies|hobby|occupation|favorite|preference|prefer|name|study|studies|am|likes?)\b",
            re.IGNORECASE
        )

    def calculate_confidence(self, user_input: str) -> float:
        """Calculates a confidence score (0.0 to 1.0) indicating if the input is memorable.

        Args:
            user_input: Raw query text from user.

        Returns:
            float: Confidence score in the range [0.0, 1.0].
        """
        if not user_input:
            return 0.0

        trimmed = user_input.strip()
        words = trimmed.split()

        # Heuristic 1: Minimum sentence length to filter out trivial queries (e.g. "yes", "ok", "hi")
        if len(trimmed) < 10 or len(words) < 3:
            return 0.0

        confidence = 0.0

        # Heuristic 2: Explicit memory action matching (adds 0.5)
        if self._action_pattern.search(trimmed):
            confidence += 0.5

        # Heuristic 3: First-person relationship pattern matching (adds 0.3)
        if self._first_person_pattern.search(trimmed):
            confidence += 0.3

        # Heuristic 4: General preference/identity indicator matching (adds 0.3)
        if self._preference_pattern.search(trimmed):
            confidence += 0.3

        # Cap confidence between 0.0 and 1.0
        return min(max(confidence, 0.0), 1.0)
