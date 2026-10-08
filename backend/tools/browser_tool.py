"""
JARVIS Browser Tool.

1. Why this module exists:
   Fulfills Version 1.0 requirement to open requested websites in default web browser.

2. How it fits into the architecture:
   Implements `BaseTool`. Registered in `ToolRegistry`. Uses standard Python `webbrowser` library.

3. Which future modules will interact with it:
   - `backend.tools.registry.ToolRegistry`
   - `backend.core.orchestrator.SystemOrchestrator`

4. Common mistakes to avoid:
   - Failing to prepend standard `https://` protocols to domain queries.

5. Possible future improvements:
   - Browser tab automation via Playwright / Selenium in Version 4.0.
"""

import logging
import webbrowser
from typing import Any

from backend.tools.base import BaseTool, ToolResult

logger = logging.getLogger("jarvis.tools.browser")


class BrowserTool(BaseTool):
    """Tool for opening websites in default system web browser."""

    @property
    def name(self) -> str:
        return "open_website"

    @property
    def description(self) -> str:
        return "Opens websites or URLs in the default web browser."

    def can_handle(self, query: str) -> bool:
        lowered = query.lower()
        return "open website" in lowered or "go to" in lowered or "open url" in lowered or "open http" in lowered

    def execute(self, **kwargs: Any) -> ToolResult:
        raw_target = kwargs.get("url") or kwargs.get("query", "").replace("open website", "").replace("go to", "").strip()

        if not raw_target:
            return ToolResult(
                success=False,
                message="Please provide a valid website URL or domain.",
                error="Missing URL argument",
            )

        # Normalize via centralized website registry if alias or natural language query
        from backend.services.website_registry import extract_website_target, get_website_url
        url = get_website_url(raw_target) or extract_website_target(raw_target) or raw_target

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        logger.info("Opening browser URL: %s", url)
        try:
            webbrowser.open(url)
            return ToolResult(
                success=True,
                message=f"Opening website {url}.",
                data={"url": url},
            )
        except Exception as exc:
            logger.error("Failed to open URL '%s': %s", url, exc)
            return ToolResult(
                success=False,
                message=f"Failed to open website {url}.",
                error=str(exc),
            )
