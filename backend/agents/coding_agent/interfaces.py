"""
Coding Agent - Interfaces.
Abstract base classes defining contracts for ProblemParser, CodeGenerator,
CodeValidator, EditorController, and CodingVerifier.
"""

from abc import ABC, abstractmethod
from PIL import Image
from typing import Tuple

from backend.agents.coding_agent.models import (
    CodeSolution,
    CodingResult,
    EditorContext,
    MCQExplanation,
    MCQQuestion,
    ProgrammingProblem,
    ValidationResult,
)
from backend.agents.computer_agent.models import WorldState


class BaseProblemParser(ABC):
    """Extracts structured problem or MCQ data from screen captures."""

    @abstractmethod
    def parse_screen(
        self,
        image: Image.Image,
        ocr_text: str,
        description: str,
    ) -> Tuple[ProgrammingProblem | None, MCQQuestion | None]:
        """Analyzes screen content and returns either a problem or an MCQ.

        Returns:
            Tuple of (ProgrammingProblem or None, MCQQuestion or None).
            Exactly one should be populated; the other will be None.
        """
        pass


class BaseCodeGenerator(ABC):
    """Generates code solutions and MCQ explanations via LLM."""

    @abstractmethod
    def generate_solution(self, problem: ProgrammingProblem) -> CodeSolution:
        """Generates a complete code solution for the given problem."""
        pass

    @abstractmethod
    def explain_mcq(self, question: MCQQuestion) -> MCQExplanation:
        """Generates a detailed explanation for an MCQ question."""
        pass


class BaseCodeValidator(ABC):
    """Validates generated code before insertion into the editor."""

    @abstractmethod
    def validate(
        self,
        solution: CodeSolution,
        problem: ProgrammingProblem,
    ) -> ValidationResult:
        """Runs syntax checks, sample I/O tests, structure validation, and complexity estimation.

        Returns:
            ValidationResult with detailed pass/fail information.
        """
        pass


class BaseEditorController(ABC):
    """Detects editor context and inserts code via DesktopService."""

    @abstractmethod
    def detect_editor(self, world_state: WorldState) -> EditorContext:
        """Identifies the active editor from the current screen state."""
        pass

    @abstractmethod
    def insert_code(self, solution: CodeSolution, editor_context: EditorContext) -> Tuple[bool, str]:
        """Inserts generated code into the detected editor.

        Returns:
            Tuple of (success, message).
        """
        pass


class BaseCodingVerifier(ABC):
    """Verifies that code was successfully inserted into the editor."""

    @abstractmethod
    def verify_insertion(
        self,
        expected_code: str,
        image: Image.Image,
        ocr_text: str,
    ) -> Tuple[bool, str]:
        """Checks if key tokens from expected_code appear on screen.

        Returns:
            Tuple of (verified, message).
        """
        pass
