"""
Unit and Integration Tests for JARVIS Coding Agent v1.0.
Covers problem parsing, code generation, validation, MCQ explanation,
editor detection/insertion, verification, and full end-to-end workflows.
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image

from backend.agents.coding_agent.models import (
    CodeSolution,
    CodingGoal,
    CodingMode,
    CodingPlatform,
    EditorContext,
    Example,
    MCQOption,
    MCQQuestion,
    ProgrammingLanguage,
    ProgrammingProblem,
    ValidationResult,
)
from backend.agents.coding_agent.parser import CodingProblemParser
from backend.agents.coding_agent.generator import CodingCodeGenerator
from backend.agents.coding_agent.validator import CodingCodeValidator
from backend.agents.coding_agent.editor_controller import CodingEditorController
from backend.agents.coding_agent.verifier import CodingVerifier
from backend.agents.coding_agent.coordinator import CodingAgent
from backend.agents.computer_agent.models import PerceivedElement, WorldState
from backend.services.desktop import DesktopService
from backend.services.vision.service import VisionService
from backend.services.vision.models import ScreenAnalysis, UIElement, VisionObservation


class TestProblemParser(unittest.TestCase):
    """Tests for CodingProblemParser."""

    def setUp(self) -> None:
        self.mock_llm = MagicMock()
        self.parser = CodingProblemParser(self.mock_llm)

    def test_problem_parsing_leetcode(self) -> None:
        """Extracts a LeetCode Two Sum problem from mock screen data."""
        self.mock_llm.generate_raw_completion.return_value = '''{
            "type": "coding",
            "title": "Two Sum",
            "statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            "constraints": ["2 <= nums.length <= 10^4", "-10^9 <= nums[i] <= 10^9"],
            "examples": [
                {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]", "explanation": "Because nums[0] + nums[1] == 9"}
            ],
            "language": "python",
            "difficulty": "Easy",
            "function_signature": "def twoSum(self, nums: List[int], target: int) -> List[int]:"
        }'''

        mock_img = Image.new("RGB", (100, 100))
        ocr_text = "Two Sum leetcode.com Given an array of integers nums"
        description = "LeetCode problem page showing Two Sum"

        problem, mcq = self.parser.parse_screen(mock_img, ocr_text, description)

        self.assertIsNotNone(problem)
        self.assertIsNone(mcq)
        self.assertEqual(problem.title, "Two Sum")
        self.assertEqual(problem.platform, CodingPlatform.LEETCODE)
        self.assertEqual(problem.language, ProgrammingLanguage.PYTHON)
        self.assertEqual(problem.difficulty, "Easy")
        self.assertEqual(len(problem.examples), 1)
        self.assertEqual(len(problem.constraints), 2)
        self.assertIn("twoSum", problem.function_signature)

    def test_problem_parsing_mcq(self) -> None:
        """Detects and parses an MCQ question from mock screen data."""
        self.mock_llm.generate_raw_completion.return_value = '''{
            "type": "mcq",
            "question_text": "What is the time complexity of binary search?",
            "options": [
                {"label": "A", "text": "O(n)"},
                {"label": "B", "text": "O(log n)"},
                {"label": "C", "text": "O(n^2)"},
                {"label": "D", "text": "O(1)"}
            ],
            "topic": "Algorithms"
        }'''

        mock_img = Image.new("RGB", (100, 100))
        ocr_text = "What is the time complexity of binary search? A) O(n) B) O(log n)"
        description = "Multiple choice question about algorithms"

        problem, mcq = self.parser.parse_screen(mock_img, ocr_text, description)

        self.assertIsNone(problem)
        self.assertIsNotNone(mcq)
        self.assertEqual(mcq.question_text, "What is the time complexity of binary search?")
        self.assertEqual(len(mcq.options), 4)
        self.assertEqual(mcq.options[1].label, "B")
        self.assertEqual(mcq.options[1].text, "O(log n)")
        self.assertEqual(mcq.topic, "Algorithms")

    def test_platform_detection(self) -> None:
        """Verifies platform keywords are correctly detected."""
        self.assertEqual(
            self.parser._detect_platform("Visit leetcode.com/problems", "LeetCode page"),
            CodingPlatform.LEETCODE,
        )
        self.assertEqual(
            self.parser._detect_platform("hackerrank challenge", "Problem page"),
            CodingPlatform.HACKERRANK,
        )
        self.assertEqual(
            self.parser._detect_platform("random text", "generic page"),
            CodingPlatform.GENERIC,
        )


class TestCodeGenerator(unittest.TestCase):
    """Tests for CodingCodeGenerator."""

    def setUp(self) -> None:
        self.mock_llm = MagicMock()
        self.generator = CodingCodeGenerator(self.mock_llm)

    def test_code_generation(self) -> None:
        """Generates a Python solution for a parsed problem."""
        self.mock_llm.generate_raw_completion.return_value = '''{
            "source_code": "class Solution:\\n    def twoSum(self, nums, target):\\n        seen = {}\\n        for i, n in enumerate(nums):\\n            comp = target - n\\n            if comp in seen:\\n                return [seen[comp], i]\\n            seen[n] = i\\n        return []",
            "explanation": "Use a hash map to store seen numbers and their indices.",
            "time_complexity": "O(n)",
            "space_complexity": "O(n)",
            "function_name": "twoSum",
            "class_name": "Solution"
        }'''

        problem = ProgrammingProblem(
            title="Two Sum",
            statement="Given an array of integers...",
            language=ProgrammingLanguage.PYTHON,
        )

        solution = self.generator.generate_solution(problem)

        self.assertIn("twoSum", solution.source_code)
        self.assertIn("seen", solution.source_code)
        self.assertEqual(solution.time_complexity, "O(n)")
        self.assertEqual(solution.space_complexity, "O(n)")
        self.assertEqual(solution.function_name, "twoSum")
        self.assertEqual(solution.class_name, "Solution")

    def test_mcq_explanation(self) -> None:
        """Generates per-option explanations for an MCQ."""
        self.mock_llm.generate_raw_completion.return_value = '''{
            "correct_label": "B",
            "correct_text": "O(log n)",
            "overall_reasoning": "Binary search halves the search space each iteration.",
            "options": [
                {"label": "A", "text": "O(n)", "is_correct": false, "reasoning": "O(n) is linear search."},
                {"label": "B", "text": "O(log n)", "is_correct": true, "reasoning": "Binary search halves the space."},
                {"label": "C", "text": "O(n^2)", "is_correct": false, "reasoning": "Quadratic is too slow."},
                {"label": "D", "text": "O(1)", "is_correct": false, "reasoning": "Constant time is not possible for search."}
            ]
        }'''

        mcq = MCQQuestion(
            question_text="What is the time complexity of binary search?",
            options=[
                MCQOption(label="A", text="O(n)"),
                MCQOption(label="B", text="O(log n)"),
                MCQOption(label="C", text="O(n^2)"),
                MCQOption(label="D", text="O(1)"),
            ],
        )

        explanation = self.generator.explain_mcq(mcq)

        self.assertEqual(explanation.correct_label, "B")
        self.assertEqual(explanation.correct_text, "O(log n)")
        self.assertIn("halves", explanation.overall_reasoning)
        self.assertEqual(len(explanation.options), 4)
        self.assertTrue(explanation.options[1].is_correct)
        self.assertFalse(explanation.options[0].is_correct)


class TestCodeValidator(unittest.TestCase):
    """Tests for CodingCodeValidator."""

    def setUp(self) -> None:
        self.validator = CodingCodeValidator()

    def test_valid_python_syntax(self) -> None:
        """Valid Python code passes syntax check."""
        solution = CodeSolution(
            source_code="def add(a, b):\n    return a + b\n",
            language=ProgrammingLanguage.PYTHON,
        )
        problem = ProgrammingProblem(title="Add", statement="Add two numbers")

        result = self.validator.validate(solution, problem)

        self.assertTrue(result.syntax_ok)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_invalid_python_syntax(self) -> None:
        """Invalid Python code fails syntax check."""
        solution = CodeSolution(
            source_code="def broken(\n    return",
            language=ProgrammingLanguage.PYTHON,
        )
        problem = ProgrammingProblem(title="Broken", statement="Broken code")

        result = self.validator.validate(solution, problem)

        self.assertFalse(result.syntax_ok)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("syntax error" in e.lower() for e in result.errors))

    def test_required_function_check(self) -> None:
        """Validates that required function is present in solution."""
        solution = CodeSolution(
            source_code="class Solution:\n    def twoSum(self, nums, target):\n        return []\n",
            language=ProgrammingLanguage.PYTHON,
        )
        problem = ProgrammingProblem(
            title="Two Sum",
            statement="...",
            function_signature="def twoSum(self, nums: List[int], target: int) -> List[int]:",
        )

        result = self.validator.validate(solution, problem)

        self.assertTrue(result.has_required_structure)
        self.assertTrue(result.is_valid)

    def test_missing_required_function(self) -> None:
        """Fails when required function is missing."""
        solution = CodeSolution(
            source_code="class Solution:\n    def wrongName(self):\n        pass\n",
            language=ProgrammingLanguage.PYTHON,
        )
        problem = ProgrammingProblem(
            title="Two Sum",
            statement="...",
            function_signature="def twoSum(self, nums: List[int], target: int) -> List[int]:",
        )

        result = self.validator.validate(solution, problem)

        self.assertFalse(result.has_required_structure)
        self.assertFalse(result.is_valid)

    def test_bracket_balance_valid(self) -> None:
        """Valid C++ code passes bracket check."""
        solution = CodeSolution(
            source_code='int main() {\n    printf("hello");\n    return 0;\n}\n',
            language=ProgrammingLanguage.CPP,
        )
        problem = ProgrammingProblem(title="Hello", statement="Print hello", language=ProgrammingLanguage.CPP)

        result = self.validator.validate(solution, problem)

        self.assertTrue(result.syntax_ok)

    def test_empty_code_rejected(self) -> None:
        """Empty code is rejected by sanity checks."""
        solution = CodeSolution(source_code="", language=ProgrammingLanguage.PYTHON)
        problem = ProgrammingProblem(title="Empty", statement="...")

        result = self.validator.validate(solution, problem)

        self.assertFalse(result.is_valid)

    def test_complexity_estimation(self) -> None:
        """Heuristic complexity estimation counts nested loops."""
        solution = CodeSolution(
            source_code="def solve(arr):\n    for i in arr:\n        for j in arr:\n            print(i, j)\n",
            language=ProgrammingLanguage.PYTHON,
        )
        problem = ProgrammingProblem(title="Nested", statement="...")

        result = self.validator.validate(solution, problem)

        self.assertIsNotNone(result.estimated_time_complexity)
        self.assertIn("n", result.estimated_time_complexity.lower())


class TestEditorController(unittest.TestCase):
    """Tests for CodingEditorController."""

    def setUp(self) -> None:
        self.mock_desktop = MagicMock(spec=DesktopService)
        self.controller = CodingEditorController(self.mock_desktop)

    def test_editor_detection_vscode(self) -> None:
        """Detects VS Code from window title."""
        world_state = WorldState(
            active_window="main.py - Visual Studio Code",
            ocr_text="def hello():\n    print('hello')",
        )

        context = self.controller.detect_editor(world_state)

        self.assertEqual(context.editor_type, "vscode")
        self.assertEqual(context.file_extension, ".py")
        self.assertTrue(context.has_existing_code)

    def test_editor_detection_notepad(self) -> None:
        """Detects Notepad++ from window title."""
        world_state = WorldState(
            active_window="solution.cpp - Notepad++",
            ocr_text="#include <iostream>",
        )

        context = self.controller.detect_editor(world_state)

        self.assertEqual(context.editor_type, "notepad++")
        self.assertEqual(context.file_extension, ".cpp")

    def test_editor_detection_generic(self) -> None:
        """Falls back to generic for unknown editors."""
        world_state = WorldState(active_window="Unknown Editor v3.0")

        context = self.controller.detect_editor(world_state)

        self.assertEqual(context.editor_type, "generic")

    def test_code_insertion_via_clipboard(self) -> None:
        """Verifies clipboard paste workflow calls DesktopService correctly."""
        self.mock_desktop.set_clipboard.return_value = (True, "OK")
        self.mock_desktop.hotkey.return_value = (True, "OK")

        solution = CodeSolution(
            source_code="def twoSum(nums, target):\n    return []\n",
            language=ProgrammingLanguage.PYTHON,
        )
        context = EditorContext(editor_type="vscode", window_title="test.py - VS Code")

        ok, msg = self.controller.insert_code(solution, context)

        self.assertTrue(ok)
        self.mock_desktop.set_clipboard.assert_called_once_with(solution.source_code)
        # Ctrl+A then Ctrl+V
        self.assertEqual(self.mock_desktop.hotkey.call_count, 2)
        calls = self.mock_desktop.hotkey.call_args_list
        self.assertEqual(calls[0][0], ("ctrl", "a"))
        self.assertEqual(calls[1][0], ("ctrl", "v"))

    def test_desktop_editor_controller_alias(self) -> None:
        """Verifies DesktopEditorController alias imports and matches CodingEditorController."""
        from backend.agents.coding_agent.editor_controller import DesktopEditorController
        self.assertIs(DesktopEditorController, CodingEditorController)
        instance = DesktopEditorController(self.mock_desktop)
        self.assertIsInstance(instance, CodingEditorController)


class TestCodingVerifier(unittest.TestCase):
    """Tests for CodingVerifier."""

    def setUp(self) -> None:
        self.verifier = CodingVerifier()

    def test_verification_success(self) -> None:
        """Confirms verifier passes when code tokens are found in OCR."""
        code = "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        return []\n"
        ocr_text = "class Solution def twoSum self nums target seen return"
        mock_img = Image.new("RGB", (100, 100))

        ok, msg = self.verifier.verify_insertion(code, mock_img, ocr_text)

        self.assertTrue(ok)
        self.assertIn("verified", msg.lower())

    def test_verification_failure(self) -> None:
        """Confirms verifier fails when code tokens are missing."""
        code = "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        return []\n"
        ocr_text = "This is a blank editor with no code"
        mock_img = Image.new("RGB", (100, 100))

        ok, msg = self.verifier.verify_insertion(code, mock_img, ocr_text)

        self.assertFalse(ok)
        self.assertIn("failed", msg.lower())

    def test_empty_ocr_fails(self) -> None:
        """Empty OCR text results in verification failure."""
        ok, msg = self.verifier.verify_insertion("def hello(): pass", Image.new("RGB", (10, 10)), "")

        self.assertFalse(ok)


class TestFullSolveWorkflow(unittest.TestCase):
    """Integration test for the complete solve pipeline."""

    def setUp(self) -> None:
        self.mock_perception = MagicMock()
        self.mock_parser = MagicMock()
        self.mock_generator = MagicMock()
        self.mock_validator = MagicMock()
        self.mock_editor = MagicMock()
        self.mock_verifier = MagicMock()

        self.agent = CodingAgent(
            perception=self.mock_perception,
            parser=self.mock_parser,
            generator=self.mock_generator,
            validator=self.mock_validator,
            editor_controller=self.mock_editor,
            verifier=self.mock_verifier,
        )

    def test_full_solve_workflow(self) -> None:
        """End-to-end: screen → parse → generate → validate → insert → verify."""
        mock_img = Image.new("RGB", (100, 100))
        world_state = WorldState(
            active_window="solution.py - Visual Studio Code",
            ocr_text="Two Sum leetcode",
        )

        # Perception returns screen state
        self.mock_perception.perceive_screen.return_value = (mock_img, world_state, "LeetCode Two Sum page")

        # Parser returns a problem
        problem = ProgrammingProblem(
            title="Two Sum",
            statement="Given an array...",
            language=ProgrammingLanguage.PYTHON,
            platform=CodingPlatform.LEETCODE,
        )
        self.mock_parser.parse_screen.return_value = (problem, None)

        # Generator returns a solution
        solution = CodeSolution(
            source_code="class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, n in enumerate(nums):\n            if target - n in seen:\n                return [seen[target-n], i]\n            seen[n] = i\n        return []\n",
            language=ProgrammingLanguage.PYTHON,
            explanation="Hash map approach",
            time_complexity="O(n)",
            space_complexity="O(n)",
            function_name="twoSum",
            class_name="Solution",
        )
        self.mock_generator.generate_solution.return_value = solution

        # Validator approves
        validation = ValidationResult(
            is_valid=True,
            syntax_ok=True,
            has_required_structure=True,
            sample_tests_passed=1,
            sample_tests_total=1,
        )
        self.mock_validator.validate.return_value = validation

        # Editor detected and insertion succeeds
        self.mock_editor.detect_editor.return_value = EditorContext(
            editor_type="vscode",
            window_title="solution.py - Visual Studio Code",
        )
        self.mock_editor.insert_code.return_value = (True, "Code inserted")

        # Verification passes
        self.mock_verifier.verify_insertion.return_value = (True, "Code verified")

        # Execute
        goal = CodingGoal(mode=CodingMode.SOLVE)
        result = self.agent.execute(goal)

        self.assertTrue(result.success)
        self.assertEqual(result.mode, CodingMode.SOLVE)
        self.assertIsNotNone(result.problem)
        self.assertIsNotNone(result.solution)
        self.assertIsNotNone(result.validation)
        self.assertIn("Two Sum", result.message)
        self.assertTrue(len(result.steps_taken) >= 6)

    def test_full_mcq_workflow(self) -> None:
        """End-to-end: screen → parse MCQ → explain → return result."""
        mock_img = Image.new("RGB", (100, 100))
        world_state = WorldState(
            active_window="Quiz - Chrome",
            ocr_text="What is the time complexity of binary search?",
        )

        self.mock_perception.perceive_screen.return_value = (mock_img, world_state, "MCQ quiz page")

        mcq = MCQQuestion(
            question_text="What is the time complexity of binary search?",
            options=[
                MCQOption(label="A", text="O(n)"),
                MCQOption(label="B", text="O(log n)"),
                MCQOption(label="C", text="O(n^2)"),
                MCQOption(label="D", text="O(1)"),
            ],
        )
        self.mock_parser.parse_screen.return_value = (None, mcq)

        from backend.agents.coding_agent.models import MCQExplanation
        explanation = MCQExplanation(
            correct_label="B",
            correct_text="O(log n)",
            overall_reasoning="Binary search halves the search space each iteration.",
            options=[
                MCQOption(label="A", text="O(n)", is_correct=False, reasoning="Linear search"),
                MCQOption(label="B", text="O(log n)", is_correct=True, reasoning="Halves each time"),
                MCQOption(label="C", text="O(n^2)", is_correct=False, reasoning="Quadratic"),
                MCQOption(label="D", text="O(1)", is_correct=False, reasoning="Constant impossible"),
            ],
        )
        self.mock_generator.explain_mcq.return_value = explanation

        goal = CodingGoal(mode=CodingMode.MCQ)
        result = self.agent.execute(goal)

        self.assertTrue(result.success)
        self.assertEqual(result.mode, CodingMode.MCQ)
        self.assertIsNotNone(result.mcq_explanation)
        self.assertEqual(result.mcq_explanation.correct_label, "B")
        self.assertIn("B", result.message)

    def test_validation_rejection_stops_pipeline(self) -> None:
        """When validator rejects code, insertion is never attempted."""
        mock_img = Image.new("RGB", (100, 100))
        world_state = WorldState(active_window="test.py - VS Code")

        self.mock_perception.perceive_screen.return_value = (mock_img, world_state, "Editor")

        problem = ProgrammingProblem(title="Test", statement="...")
        self.mock_parser.parse_screen.return_value = (problem, None)

        solution = CodeSolution(
            source_code="def broken(\n    return",
            language=ProgrammingLanguage.PYTHON,
        )
        self.mock_generator.generate_solution.return_value = solution

        validation = ValidationResult(
            is_valid=False,
            syntax_ok=False,
            errors=["Python syntax error at line 1"],
        )
        self.mock_validator.validate.return_value = validation

        result = self.agent.execute(CodingGoal())

        self.assertFalse(result.success)
        self.assertIn("rejected", result.message.lower())
        # Editor should never have been called
        self.mock_editor.insert_code.assert_not_called()


if __name__ == "__main__":
    unittest.main()
