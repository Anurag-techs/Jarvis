"""
Unit tests for IntentRouter.
Verifies true-positive coding matches and guards against false positives.
"""

import pytest
from backend.services.intent_router import IntentRouter


@pytest.fixture
def router() -> IntentRouter:
    return IntentRouter()


# ---------------------------------------------------------------------------
# True positives — must route to 'coding_actions'
# ---------------------------------------------------------------------------
class TestCodingActionsTruePositives:
    def test_solve_this_problem(self, router):
        assert router.route("solve this problem") == "coding_actions"

    def test_solve_this_coding_problem(self, router):
        assert router.route("solve this coding problem") == "coding_actions"

    def test_solve_this_question(self, router):
        assert router.route("solve this question") == "coding_actions"

    def test_solve_coding_question(self, router):
        assert router.route("Solve the coding question on screen") == "coding_actions"

    def test_leetcode_bare(self, router):
        assert router.route("leetcode") == "coding_actions"

    def test_leetcode_sentence(self, router):
        assert router.route("Solve this LeetCode problem: Two Sum") == "coding_actions"

    def test_hackerrank(self, router):
        assert router.route("This is a HackerRank challenge") == "coding_actions"

    def test_codeforces(self, router):
        assert router.route("solve the codeforces problem") == "coding_actions"

    def test_geeksforgeeks(self, router):
        assert router.route("answer the geeksforgeeks question") == "coding_actions"

    def test_explain_mcq(self, router):
        assert router.route("explain this MCQ") == "coding_actions"

    def test_explain_multiple_choice(self, router):
        assert router.route("explain this multiple-choice question") == "coding_actions"

    def test_solve_mcq(self, router):
        assert router.route("solve this MCQ") == "coding_actions"

    def test_generate_code_for_this(self, router):
        assert router.route("generate the code for this") == "coding_actions"

    def test_write_code_for_this(self, router):
        assert router.route("write code for this") == "coding_actions"

    def test_create_code_for_this(self, router):
        assert router.route("create code for this problem") == "coding_actions"

    def test_insert_code_into_editor(self, router):
        assert router.route("insert the code into the editor") == "coding_actions"

    def test_solve_visible_problem(self, router):
        assert router.route("solve the visible problem") == "coding_actions"

    def test_answer_current_question(self, router):
        assert router.route("answer the current question") == "coding_actions"

    def test_case_insensitive(self, router):
        assert router.route("SOLVE THIS CODING PROBLEM") == "coding_actions"


# ---------------------------------------------------------------------------
# True negatives — must return None (delegate to LLM)
# ---------------------------------------------------------------------------
class TestFalsePositiveGuards:
    def test_weather_query(self, router):
        assert router.route("What is the weather today?") is None

    def test_generic_open_without_known_app(self, router):
        # 'open notepad' correctly routes to open_application (not a false positive).
        # This test verifies a truly ambiguous 'open' command with no known app
        # does NOT route to any tool.
        assert router.route("open a can of worms") is None

    def test_play_music(self, router):
        assert router.route("play some music") is None

    def test_generic_question(self, router):
        assert router.route("What is the capital of France?") is None

    def test_empty_string(self, router):
        assert router.route("") is None

    def test_whitespace_only(self, router):
        assert router.route("   ") is None

    def test_error_code_query(self, router):
        # "code" alone must NOT match
        assert router.route("What is error code 404?") is None

    def test_source_code_of(self, router):
        # "source code" alone must NOT match
        assert router.route("Show me the source code of the project") is None

    def test_search_query(self, router):
        assert router.route("Search for machine learning tutorials") is None

    def test_screenshot_request(self, router):
        assert router.route("take a screenshot") is None

    def test_news_request(self, router):
        assert router.route("Get me the latest news") is None


# ---------------------------------------------------------------------------
# Website launches – must route to 'open_website'
# ---------------------------------------------------------------------------
class TestWebsiteRouting:
    def test_open_youtube(self, router):
        assert router.route("Open YouTube") == "open_website"

    def test_launch_youtube(self, router):
        assert router.route("Launch YouTube") == "open_website"

    def test_go_to_youtube(self, router):
        assert router.route("Go to YouTube") == "open_website"

    def test_open_the_youtube_website(self, router):
        assert router.route("Open the YouTube website") == "open_website"

    def test_open_gmail(self, router):
        assert router.route("Open Gmail") == "open_website"

    def test_open_github(self, router):
        assert router.route("Open GitHub") == "open_website"

    def test_open_chatgpt(self, router):
        assert router.route("Open ChatGPT") == "open_website"

    def test_open_google_maps(self, router):
        assert router.route("Open Google Maps") == "open_website"

    def test_open_chrome_routes_to_application(self, router):
        assert router.route("Open Chrome") == "open_application"

    def test_open_notepad_routes_to_application(self, router):
        assert router.route("Open Notepad") == "open_application"

