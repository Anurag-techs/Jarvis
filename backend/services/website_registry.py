"""
JARVIS Website Alias Registry and Normalization Service.

Centralized repository for common website and online service names, mapping
natural-language service names to canonical URLs, including user's personal
frequent websites.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("jarvis.services.website_registry")

# User's exact configured personal & frequent website entries
PERSONAL_WEBSITES: dict[str, str] = {
    # YouTube
    "youtube": "https://www.youtube.com",
    "you tube": "https://www.youtube.com",
    "yt": "https://www.youtube.com",

    # LinkedIn (exact personal profile URL preserved)
    "linkedin": "https://www.linkedin.com/in/anurag-singh-754a88373/",
    "linked in": "https://www.linkedin.com/in/anurag-singh-754a88373/",

    # LeetCode (exact personal profile URL preserved)
    "leetcode": "https://leetcode.com/u/Anurag_singh_leet/",
    "leet code": "https://leetcode.com/u/Anurag_singh_leet/",

    # GitHub (exact personal profile URL preserved)
    "github": "https://github.com/Anurag-techs",
    "git hub": "https://github.com/Anurag-techs",

    # ChatGPT
    "chatgpt": "https://chatgpt.com",
    "chat gpt": "https://chatgpt.com",
    "openai": "https://chatgpt.com",

    # Gmail
    "gmail": "https://mail.google.com",
    "google mail": "https://mail.google.com",

    # Google
    "google": "https://www.google.com",
    "google search": "https://www.google.com",

    # NIAT / CCBP Learning
    "niat": "https://learning.ccbp.in/",
    "ccbp": "https://learning.ccbp.in/",
    "niat learning": "https://learning.ccbp.in/",
}

# Standard / general website entries
GENERAL_WEBSITES: dict[str, str] = {
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "instagram": "https://www.instagram.com",
    "insta": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "fb": "https://www.facebook.com",
    "whatsapp": "https://web.whatsapp.com",
    "whats app": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "x/twitter": "https://x.com",
    "x twitter": "https://x.com",
}

# Unified registry: personal websites take precedence
WEBSITE_REGISTRY: dict[str, str] = {**GENERAL_WEBSITES, **PERSONAL_WEBSITES}

# Regex to strip action verbs when extracting website targets
_ACTION_VERBS_RE = re.compile(
    r"^\s*(open|launch|go\s+to|visit|browse|navigate\s+to|start|take\s+me\s+to)\s+",
    re.IGNORECASE,
)

# Regex to strip trailing and noise words ("the", "website", "site", punctuation)
_NOISE_WORDS_RE = re.compile(
    r"\b(the|website|web\s+site|site|webpage|page|app)\b",
    re.IGNORECASE,
)
_PUNCTUATION_RE = re.compile(r"[.!?,;:]+\s*$", re.IGNORECASE)


def get_website_url(name: str) -> Optional[str]:
    """Retrieves canonical URL for a website or service name.

    Args:
        name: Service name or alias (e.g., 'YouTube', 'LinkedIn', 'LeetCode', 'NIAT').

    Returns:
        Canonical URL string if matched, None otherwise.
    """
    if not name or not name.strip():
        return None

    raw_trimmed = name.strip()

    # If already a full URL, preserve original casing (URL paths are case-sensitive)
    if raw_trimmed.startswith(("http://", "https://")):
        return raw_trimmed

    cleaned = raw_trimmed.lower()

    # Direct match in registry
    if cleaned in WEBSITE_REGISTRY:
        return WEBSITE_REGISTRY[cleaned]

    # Strip noise words like "the", "website", "site"
    normalized = _NOISE_WORDS_RE.sub("", cleaned)
    normalized = _PUNCTUATION_RE.sub("", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    if normalized in WEBSITE_REGISTRY:
        return WEBSITE_REGISTRY[normalized]

    # Check for standard domain patterns (e.g. "youtube.com", "sub.domain.org")
    if re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$", raw_trimmed):
        return f"https://{raw_trimmed}"

    return None


def is_known_website(name: str) -> bool:
    """Checks whether the given name corresponds to a registered website."""
    return get_website_url(name) is not None


def extract_website_target(query: str) -> Optional[str]:
    """Extracts the target website URL from a natural language query.

    Preserves exact URL casing for profile URLs and paths.

    Args:
        query: Raw user query string.

    Returns:
        Canonical URL string if recognized, None otherwise.
    """
    if not query or not query.strip():
        return None

    trimmed = query.strip()

    # Check if an explicit full URL is present in the query (preserve casing!)
    url_match = re.search(r"https?://\S+", trimmed)
    if url_match:
        return url_match.group(0).rstrip(".!?,;:")

    # Strip leading action verbs
    stripped = _ACTION_VERBS_RE.sub("", trimmed).strip()
    stripped = _PUNCTUATION_RE.sub("", stripped).strip()

    # Attempt to resolve the stripped query
    resolved = get_website_url(stripped)
    if resolved:
        return resolved

    # Fallback: remove internal noise words
    noise_stripped = _NOISE_WORDS_RE.sub("", stripped).strip()
    noise_stripped = re.sub(r"\s+", " ", noise_stripped)
    resolved = get_website_url(noise_stripped)
    if resolved:
        return resolved

    return None
