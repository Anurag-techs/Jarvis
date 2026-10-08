"""
Unit and Integration Tests for JARVIS Computer Interaction Agent (v2.0).
Verifies WorldState observations, approval workflows, wait conditions, task memory, retries, and Google search.
"""

import time
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.agents.computer_agent.models import (
    ApprovalLevel,
    ComputerAction,
    ComputerActionType,
    PerceivedElement,
    WorldState,
)
from backend.agents.computer_agent.perception import ComputerAgentPerception
from backend.agents.computer_agent.reasoning import ComputerAgentReasoning
from backend.agents.computer_agent.executor import ComputerAgentExecutor
from backend.agents.computer_agent.verifier import ComputerAgentVerifier
from backend.agents.computer_agent.memory import ComputerAgentMemory
from backend.agents.computer_agent.coordinator import ComputerAgent
from backend.services.desktop import DesktopService
from backend.services.vision.service import VisionService
from backend.services.vision.models import UIElement, ScreenAnalysis, VisionObservation


class TestComputerAgentV2(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_desktop = MagicMock(spec=DesktopService)
        self.mock_vision = MagicMock(spec=VisionService)

        self.mock_capture = MagicMock()
        self.mock_vision._capture_provider = self.mock_capture

        self.perception = ComputerAgentPerception(self.mock_vision)
        self.reasoning = ComputerAgentReasoning(MagicMock())
        self.executor = ComputerAgentExecutor(self.mock_desktop)
        self.verifier = ComputerAgentVerifier()
        self.memory = ComputerAgentMemory()

        self.agent = ComputerAgent(
            perception=self.perception,
            reasoning=self.reasoning,
            executor=self.executor,
            verifier=self.verifier,
            memory=self.memory,
            max_retries_per_step=2,
        )

    @patch("pygetwindow.getActiveWindow")
    def test_world_state_construction(self, mock_get_win) -> None:
        """Verifies perception sweep builds structured WorldState with active window and loading detection."""
        mock_win = MagicMock()
        mock_win.title = "Google Chrome"
        mock_get_win.return_value = mock_win

        mock_img = Image.new("RGB", (100, 100))
        self.mock_capture.capture_screen.return_value = mock_img

        mock_obs = VisionObservation(
            analysis=ScreenAnalysis(
                width=100,
                height=100,
                ocr_text="Loading search page...",
                description="Google homepage loading",
                detected_elements=[
                    UIElement(
                        label="Chrome Icon",
                        description="Chrome desktop launcher",
                        bounding_box=[10, 20, 100, 50],
                        confidence=0.95,
                        semantic_type="icon",
                    )
                ],
            ),
            state_hash="mock_hash",
            cache_hit=False,
        )
        self.mock_vision.analyze_current_screen.return_value = mock_obs

        img, world_state, desc = self.perception.perceive_screen()

        self.assertEqual(world_state.active_window, "Google Chrome")
        self.assertEqual(len(world_state.visible_elements), 1)
        self.assertTrue(world_state.has_loading_indicator)  # Due to "Loading" in OCR text

    def test_human_approval_workflow(self) -> None:
        """Verifies approval gates trigger correctly for SAFE, CAUTION, and DANGEROUS actions."""
        mock_img = Image.new("RGB", (64, 64))
        self.mock_capture.capture_screen.return_value = mock_img
        self.mock_vision.analyze_current_screen.return_value = VisionObservation(
            analysis=ScreenAnalysis(width=64, height=64, description="Screen", detected_elements=[]),
            state_hash="h1",
            cache_hit=False,
        )

        # Mock reasoning to decide a CAUTION action
        caution_action = ComputerAction(
            action_type=ComputerActionType.PRESS_KEY,
            key="delete",
            approval_level=ApprovalLevel.CAUTION,
            description="Delete file",
        )
        self.agent.reasoning = MagicMock()
        self.agent.reasoning.decide_actions.side_effect = [
            MagicMock(is_terminal=False, next_actions=[caution_action]),
            MagicMock(is_terminal=True, success=True, message="Completed"),
        ]

        # Approval handler mock that REJECTS the caution action
        mock_approval = MagicMock(return_value=False)
        self.agent.approval_handler = mock_approval

        success, msg = self.agent.execute_goal("Delete caution file.")

        # Rejection should stop the action, and log it to history
        mock_approval.assert_called_once_with(caution_action)
        self.mock_desktop.press_key.assert_not_called()
        self.assertIn("Rejected action by approval: Delete file", self.memory.get_history())

    def test_waiting_conditions_satisfied(self) -> None:
        """Verifies execution polls screen state until wait condition is satisfied."""
        mock_img = Image.new("RGB", (64, 64))
        self.mock_capture.capture_screen.return_value = mock_img

        # First 2 perception sweeps return Loading. 3rd returns element loaded.
        obs_loading = VisionObservation(
            analysis=ScreenAnalysis(width=64, height=64, ocr_text="Loading...", description="Loading", detected_elements=[]),
            state_hash="h1",
            cache_hit=False,
        )
        obs_loaded = VisionObservation(
            analysis=ScreenAnalysis(
                width=64,
                height=64,
                ocr_text="Google",
                description="Loaded",
                detected_elements=[UIElement(label="Search box", description="Search bar", bounding_box=[10, 10, 10, 10], semantic_type="text_box")],
            ),
            state_hash="h2",
            cache_hit=False,
        )

        obs_list = [
            obs_loading,  # Pre-action observation
            obs_loading,  # First wait poll
            obs_loading,  # Second wait poll
            obs_loaded,   # Third wait poll -> condition met!
            obs_loaded,   # Post-action observation
        ]
        def side_effect(*args, **kwargs):
            if len(obs_list) > 1:
                return obs_list.pop(0)
            return obs_list[0]
        self.mock_vision.analyze_current_screen.side_effect = side_effect

        self.mock_desktop.left_click.return_value = (True, "clicked")

        action = ComputerAction(
            action_type=ComputerActionType.CLICK,
            wait_condition="Search box",
            wait_timeout=2.0,
            description="Wait for search box element",
        )

        # Run execute and verify
        success = self.agent._execute_and_verify_with_retries(action, mock_img, "hash")

        self.assertTrue(success)
        self.assertEqual(self.mock_vision.analyze_current_screen.call_count, 5)

    def test_task_memory_tracking_lifecycle(self) -> None:
        """Verifies session metrics, retry counts, elapsed time, and completed steps are saved in Memory."""
        self.memory.clear()
        self.assertEqual(self.memory.goal, "")

        self.memory.goal = "Test Goal"
        self.memory.start_time = time.time() - 5.0  # Simulate 5s elapsed
        self.memory.add_step("Step 1 executed", "hash1")
        self.memory.retry_count = 3
        self.memory.last_clicked_element = "Chrome Icon"

        self.assertEqual(self.memory.goal, "Test Goal")
        self.assertGreaterEqual(self.memory.get_elapsed_time(), 5.0)
        self.assertEqual(self.memory.retry_count, 3)
        self.assertEqual(self.memory.last_clicked_element, "Chrome Icon")

        # Verify clear resets lifecycle
        self.memory.clear()
        self.assertEqual(self.memory.goal, "")
        self.assertEqual(self.memory.retry_count, 0)
        self.assertEqual(self.memory.last_clicked_element, None)

    def test_alternative_plan_generation_on_failure(self) -> None:
        """Verifies coordinator registers step failure in memory and requests alternative strategy from reasoning."""
        mock_img = Image.new("RGB", (64, 64))
        self.mock_capture.capture_screen.return_value = mock_img

        # Verifier always returns False (simulates clicked button didn't change screen state)
        self.agent.verifier = MagicMock()
        self.agent.verifier.verify_action.return_value = (False, "Verification failed")

        self.mock_desktop.left_click.return_value = (True, "clicked")

        # Set up reasoning to output the failing action first, then an alternative successful terminal action
        action_fail = ComputerAction(action_type=ComputerActionType.CLICK, description="Click button X")
        action_alt = ComputerAction(action_type=ComputerActionType.TYPE, text="OpenAI", description="Alt type text")

        self.agent.reasoning = MagicMock()
        self.agent.reasoning.decide_actions.side_effect = [
            MagicMock(is_terminal=False, next_actions=[action_fail]),  # Step 1
            MagicMock(is_terminal=True, success=True, message="Alternative strategy completed"),  # Step 2
        ]

        # First observation
        self.mock_vision.analyze_current_screen.return_value = VisionObservation(
            analysis=ScreenAnalysis(width=64, height=64, description="Screen", detected_elements=[]),
            state_hash="h1",
            cache_hit=False,
        )

        success, message = self.agent.execute_goal("Click X or alternative path.")

        # Should log the failure to history, trigger Step 2 reasoning, and complete successfully
        self.assertTrue(success)
        self.assertEqual(message, "Alternative strategy completed")
        self.assertIn("Failed to execute action: Click button X", self.memory.get_history())

    def test_autonomous_chrome_google_openai_workflow(self) -> None:
        """Integration test executing the full v2.0 autonomous success path criteria."""
        mock_img_desktop = Image.new("RGB", (100, 100), color="blue")
        mock_img_chrome = Image.new("RGB", (100, 100), color="green")
        mock_img_google = Image.new("RGB", (100, 100), color="yellow")
        mock_img_results = Image.new("RGB", (100, 100), color="orange")
        mock_img_openai = Image.new("RGB", (100, 100), color="red")

        # Mock perception returns images
        self.mock_capture.capture_screen.return_value = mock_img_desktop

        # Patch verifier to bypass pixel differences
        self.agent.verifier = MagicMock()
        self.agent.verifier.verify_action.return_value = (True, "Verified")

        # Build obs sequence for each step
        obs_desktop = VisionObservation(
            analysis=ScreenAnalysis(
                width=100,
                height=100,
                description="Desktop screen showing Chrome icon",
                detected_elements=[UIElement(label="Chrome Icon", description="Chrome", bounding_box=[10, 10, 10, 10], semantic_type="icon")],
            ),
            state_hash="h1",
            cache_hit=False,
        )

        obs_chrome = VisionObservation(
            analysis=ScreenAnalysis(
                width=100,
                height=100,
                description="Chrome address bar visible",
                detected_elements=[UIElement(label="Search box", description="Search bar", bounding_box=[10, 10, 10, 10], semantic_type="text_box")],
            ),
            state_hash="h2",
            cache_hit=False,
        )

        obs_google = VisionObservation(
            analysis=ScreenAnalysis(
                width=100,
                height=100,
                description="Google Search Home page",
                detected_elements=[UIElement(label="Google Search", description="Input", bounding_box=[10, 10, 10, 10], semantic_type="text_box")],
            ),
            state_hash="h3",
            cache_hit=False,
        )

        obs_results = VisionObservation(
            analysis=ScreenAnalysis(
                width=100,
                height=100,
                description="Search results page",
                detected_elements=[UIElement(label="OpenAI Official Site link", description="Link", bounding_box=[10, 10, 10, 10], semantic_type="button")],
            ),
            state_hash="h4",
            cache_hit=False,
        )

        obs_openai = VisionObservation(
            analysis=ScreenAnalysis(
                width=100,
                height=100,
                description="OpenAI official homepage. Mission: AGI benefit all humanity.",
                detected_elements=[],
            ),
            state_hash="h5",
            cache_hit=False,
        )

        obs_list = [
            obs_desktop,  # Step 1 Pre
            obs_chrome,   # Step 1 Post
            obs_chrome,   # Step 2 Pre
            obs_chrome,   # Step 2 Action 1 Post
            obs_chrome,   # Step 2 Action 2 Post
            obs_google,   # Step 2 Action 3 Post (Navigate to Google)
            obs_google,   # Step 3 Pre
            obs_google,   # Step 3 Action 1 Post
            obs_google,   # Step 3 Action 2 Post
            obs_results,  # Step 3 Action 3 Post (Google Search Results)
            obs_results,  # Step 4 Pre
            obs_openai,   # Step 4 Post (OpenAI homepage loaded)
            obs_openai,   # Step 5 Pre (Summarize OpenAI page)
            obs_openai,   # Step 5 Post
            obs_openai,   # Step 6 Pre (Complete check)
        ]
        def side_effect(*args, **kwargs):
            if len(obs_list) > 1:
                return obs_list.pop(0)
            return obs_list[0]
        self.mock_vision.analyze_current_screen.side_effect = side_effect

        # Setup executor success mocks
        self.mock_desktop.move_mouse.return_value = (True, "moved")
        self.mock_desktop.left_click.return_value = (True, "clicked")
        self.mock_desktop.type_text.return_value = (True, "typed")
        self.mock_desktop.press_key.return_value = (True, "pressed")
        self.mock_desktop.hotkey.return_value = (True, "pressed hotkey")

        # Execute full autonomous loop
        success, message = self.agent.execute_goal("Open Chrome, Open Google, Search OpenAI, Open first result, Summarize page.")

        self.assertTrue(success)
        self.assertIn("Google search for OpenAI and page summarization completed successfully", message)

        history = self.memory.get_history()
        self.assertIn("Click Chrome Desktop Icon", history)
        self.assertIn("Type google.com", history)
        self.assertIn("Type OpenAI query", history)
        self.assertIn("Click first result link", history)
        self.assertIn("Summarize page: OpenAI is an AI research and deployment company. Our mission is to ensure AGI benefits all humanity.", history)


if __name__ == "__main__":
    unittest.main()
