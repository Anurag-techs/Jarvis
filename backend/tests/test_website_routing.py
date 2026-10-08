"""
Tests for JARVIS Website Routing & Centralized Registry:
- IntentRouter routing for websites vs applications
- Website registry canonical URL resolutions (including user's personal websites)
- Argument extraction in orchestrator
- BrowserTool and ApplicationTool execution
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.core.orchestrator import _extract_tool_arguments, _extract_open_website_args
from backend.services.intent_router import IntentRouter
from backend.services.website_registry import (
    PERSONAL_WEBSITES,
    WEBSITE_REGISTRY,
    extract_website_target,
    get_website_url,
    is_known_website,
)
from backend.tools.application_tool import ApplicationTool
from backend.tools.browser_tool import BrowserTool


class TestPersonalWebsiteRegistry(unittest.TestCase):
    """Verifies user's exact personal frequent website entries are preserved."""

    def test_personal_website_exact_urls(self) -> None:
        personal_expected = {
            "YouTube": "https://www.youtube.com",
            "youtube": "https://www.youtube.com",
            "yt": "https://www.youtube.com",
            "LinkedIn": "https://www.linkedin.com/in/anurag-singh-754a88373/",
            "linkedin": "https://www.linkedin.com/in/anurag-singh-754a88373/",
            "linked in": "https://www.linkedin.com/in/anurag-singh-754a88373/",
            "LeetCode": "https://leetcode.com/u/Anurag_singh_leet/",
            "leetcode": "https://leetcode.com/u/Anurag_singh_leet/",
            "leet code": "https://leetcode.com/u/Anurag_singh_leet/",
            "GitHub": "https://github.com/Anurag-techs",
            "github": "https://github.com/Anurag-techs",
            "git hub": "https://github.com/Anurag-techs",
            "ChatGPT": "https://chatgpt.com",
            "chatgpt": "https://chatgpt.com",
            "chat gpt": "https://chatgpt.com",
            "Gmail": "https://mail.google.com",
            "gmail": "https://mail.google.com",
            "google mail": "https://mail.google.com",
            "Google": "https://www.google.com",
            "google": "https://www.google.com",
            "NIAT": "https://learning.ccbp.in/",
            "niat": "https://learning.ccbp.in/",
            "ccbp": "https://learning.ccbp.in/",
            "niat learning": "https://learning.ccbp.in/",
        }
        for alias, expected_url in personal_expected.items():
            self.assertEqual(
                get_website_url(alias),
                expected_url,
                f"Failed for personal alias: '{alias}'",
            )
            self.assertTrue(is_known_website(alias))

    def test_extract_personal_website_targets(self) -> None:
        commands = [
            ("Open YouTube", "https://www.youtube.com"),
            ("Open LinkedIn", "https://www.linkedin.com/in/anurag-singh-754a88373/"),
            ("Open LeetCode", "https://leetcode.com/u/Anurag_singh_leet/"),
            ("Open GitHub", "https://github.com/Anurag-techs"),
            ("Open ChatGPT", "https://chatgpt.com"),
            ("Open Gmail", "https://mail.google.com"),
            ("Open Google", "https://www.google.com"),
            ("Open NIAT", "https://learning.ccbp.in/"),
            ("Open CCBP", "https://learning.ccbp.in/"),
            ("Open NIAT learning", "https://learning.ccbp.in/"),
            ("Launch LeetCode", "https://leetcode.com/u/Anurag_singh_leet/"),
            ("Go to GitHub", "https://github.com/Anurag-techs"),
        ]
        for cmd, expected_url in commands:
            self.assertEqual(
                extract_website_target(cmd),
                expected_url,
                f"Failed for command: '{cmd}'",
            )


class TestIntentRouterWebsiteRouting(unittest.TestCase):
    """Verifies IntentRouter routes personal websites and real desktop apps correctly."""

    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_open_youtube_routes(self) -> None:
        self.assertEqual(self.router.route("Open YouTube"), "open_website")

    def test_open_linkedin_routes(self) -> None:
        self.assertEqual(self.router.route("Open LinkedIn"), "open_website")

    def test_open_leetcode_routes_to_website(self) -> None:
        self.assertEqual(self.router.route("Open LeetCode"), "open_website")

    def test_open_github_routes(self) -> None:
        self.assertEqual(self.router.route("Open GitHub"), "open_website")

    def test_open_chatgpt_routes(self) -> None:
        self.assertEqual(self.router.route("Open ChatGPT"), "open_website")

    def test_open_gmail_routes(self) -> None:
        self.assertEqual(self.router.route("Open Gmail"), "open_website")

    def test_open_google_routes(self) -> None:
        self.assertEqual(self.router.route("Open Google"), "open_website")

    def test_open_niat_routes(self) -> None:
        self.assertEqual(self.router.route("Open NIAT"), "open_website")

    def test_open_ccbp_routes(self) -> None:
        self.assertEqual(self.router.route("Open CCBP"), "open_website")

    def test_leetcode_coding_still_routes_to_coding_actions(self) -> None:
        self.assertEqual(self.router.route("Solve this LeetCode problem"), "coding_actions")
        self.assertEqual(self.router.route("leetcode"), "coding_actions")

    def test_open_chrome_routes_to_open_application(self) -> None:
        self.assertEqual(self.router.route("Open Chrome"), "open_application")

    def test_open_google_chrome_routes_to_open_application(self) -> None:
        self.assertEqual(self.router.route("Open Google Chrome"), "open_application")

    def test_open_notepad_routes_to_open_application(self) -> None:
        self.assertEqual(self.router.route("Open Notepad"), "open_application")

    def test_open_calculator_routes_to_open_application(self) -> None:
        self.assertEqual(self.router.route("Open Calculator"), "open_application")


class TestOrchestratorWebsiteArgsExtraction(unittest.TestCase):
    """Verifies orchestrator argument extraction for all personal websites."""

    def test_extract_open_website_args_personal(self) -> None:
        cases = [
            ("Open YouTube", "https://www.youtube.com"),
            ("Open LinkedIn", "https://www.linkedin.com/in/anurag-singh-754a88373/"),
            ("Open LeetCode", "https://leetcode.com/u/Anurag_singh_leet/"),
            ("Open GitHub", "https://github.com/Anurag-techs"),
            ("Open ChatGPT", "https://chatgpt.com"),
            ("Open Gmail", "https://mail.google.com"),
            ("Open Google", "https://www.google.com"),
            ("Open NIAT", "https://learning.ccbp.in/"),
        ]
        for cmd, expected_url in cases:
            args = _extract_tool_arguments("open_website", cmd)
            self.assertEqual(args, {"url": expected_url}, f"Failed extraction for: '{cmd}'")

    def test_extract_open_application_args(self) -> None:
        self.assertEqual(
            _extract_tool_arguments("open_application", "Open Chrome"),
            {"app_name": "Chrome"},
        )
        self.assertEqual(
            _extract_tool_arguments("open_application", "Open Notepad"),
            {"app_name": "Notepad"},
        )


class TestBrowserAndApplicationTools(unittest.TestCase):
    """Verifies BrowserTool opens personal URLs directly without LLM roundtrips."""

    def test_browser_tool_personal_urls(self) -> None:
        tool = BrowserTool()
        cases = [
            ("Open LinkedIn", "https://www.linkedin.com/in/anurag-singh-754a88373/"),
            ("Open LeetCode", "https://leetcode.com/u/Anurag_singh_leet/"),
            ("Open GitHub", "https://github.com/Anurag-techs"),
            ("Open NIAT", "https://learning.ccbp.in/"),
        ]
        for query, expected_url in cases:
            with patch("webbrowser.open") as mock_open:
                result = tool.execute(query=query)
                self.assertTrue(result.success)
                self.assertEqual(result.data["url"], expected_url)
                mock_open.assert_called_once_with(expected_url)

    def test_application_tool_redirects_leetcode_to_browser(self) -> None:
        tool = ApplicationTool()
        with patch("webbrowser.open") as mock_open:
            result = tool.execute(app_name="LeetCode")
            self.assertTrue(result.success)
            self.assertEqual(result.data["url"], "https://leetcode.com/u/Anurag_singh_leet/")
            mock_open.assert_called_once_with("https://leetcode.com/u/Anurag_singh_leet/")


if __name__ == "__main__":
    unittest.main()
