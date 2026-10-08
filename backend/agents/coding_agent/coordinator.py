"""
Coding Agent - Coordinator.
Orchestrates the full Coding Agent pipeline:
    Screen Capture → Problem Parsing → Code Generation → Validation → Editor Insertion → Verification.
Also handles the MCQ explanation flow.
"""

import logging
import time
from typing import Tuple

from backend.agents.coding_agent.interfaces import (
    BaseCodeGenerator,
    BaseCodeValidator,
    BaseCodingVerifier,
    BaseEditorController,
    BaseProblemParser,
)
from backend.agents.coding_agent.models import (
    CodingGoal,
    CodingMode,
    CodingResult,
)
from backend.agents.computer_agent.interfaces import BasePerception

logger = logging.getLogger("jarvis.agents.coding_agent.coordinator")


class CodingAgent:
    """Orchestrates the complete Coding Agent pipeline for solving problems and explaining MCQs."""

    def __init__(
        self,
        perception: BasePerception,
        parser: BaseProblemParser,
        generator: BaseCodeGenerator,
        validator: BaseCodeValidator,
        editor_controller: BaseEditorController,
        verifier: BaseCodingVerifier,
    ) -> None:
        self.perception = perception
        self.parser = parser
        self.generator = generator
        self.validator = validator
        self.editor_controller = editor_controller
        self.verifier = verifier

    def execute(self, goal: CodingGoal | None = None) -> CodingResult:
        """Runs the appropriate pipeline based on goal mode."""
        if goal is None:
            goal = CodingGoal()

        start = time.time()
        logger.info("CodingAgent starting execution — mode: %s", goal.mode.value)

        try:
            if goal.mode == CodingMode.MCQ:
                result = self._execute_mcq_flow(goal)
            else:
                result = self._execute_solve_flow(goal)
        except Exception as e:
            logger.error("CodingAgent execution failed: %s", e)
            result = CodingResult(
                success=False,
                mode=goal.mode,
                message=f"Execution error: {e}",
            )

        result.duration_seconds = time.time() - start
        logger.info(
            "CodingAgent finished — success=%s | mode=%s | steps=%d | duration=%.2fs",
            result.success, result.mode.value, len(result.steps_taken), result.duration_seconds,
        )
        return result

    def _execute_solve_flow(self, goal: CodingGoal) -> CodingResult:
        """Full solve pipeline: capture → parse → generate → validate → insert → verify."""
        steps = []

        # Step 1: Capture screen
        steps.append("Capturing screen...")
        image, world_state, description = self.perception.perceive_screen()
        ocr_text = world_state.ocr_text
        steps.append(f"Screen captured. Active window: '{world_state.active_window}'")

        # Step 2: Parse problem
        steps.append("Parsing problem from screen...")
        problem, mcq = self.parser.parse_screen(image, ocr_text, description)

        if mcq is not None:
            # Content was actually an MCQ — switch to MCQ flow
            logger.info("Detected MCQ content instead of coding problem. Switching to MCQ mode.")
            steps.append("Detected MCQ content — switching to MCQ mode")
            return self._handle_mcq(mcq, goal, steps)

        if problem is None:
            steps.append("Failed to parse problem from screen")
            return CodingResult(
                success=False,
                mode=CodingMode.SOLVE,
                steps_taken=steps,
                message="Could not extract a programming problem from the screen.",
            )

        steps.append(f"Problem parsed: '{problem.title}' ({problem.platform.value}, {problem.language.value})")

        # Override language if user specified preference
        if goal.preferred_language != problem.language:
            problem.language = goal.preferred_language
            steps.append(f"Language overridden to: {goal.preferred_language.value}")

        # Step 3: Generate solution
        steps.append("Generating code solution...")
        solution = self.generator.generate_solution(problem)

        if not solution.source_code or "Code generation failed" in solution.source_code:
            steps.append("Code generation failed")
            return CodingResult(
                success=False,
                mode=CodingMode.SOLVE,
                problem=problem,
                solution=solution,
                steps_taken=steps,
                message="Code generation failed.",
            )

        steps.append(
            f"Solution generated: {len(solution.source_code)} chars | "
            f"Time: {solution.time_complexity} | Space: {solution.space_complexity}"
        )

        # Step 4: Validate solution
        steps.append("Validating solution...")
        validation = self.validator.validate(solution, problem)

        if not validation.is_valid:
            steps.append(f"Validation FAILED: {'; '.join(validation.errors)}")
            return CodingResult(
                success=False,
                mode=CodingMode.SOLVE,
                problem=problem,
                solution=solution,
                validation=validation,
                steps_taken=steps,
                message=f"Solution rejected by validator: {'; '.join(validation.errors)}",
            )

        steps.append(
            f"Validation passed: syntax={validation.syntax_ok} | structure={validation.has_required_structure} | "
            f"tests={validation.sample_tests_passed}/{validation.sample_tests_total}"
        )

        # Step 5: Detect editor and insert code
        steps.append("Detecting editor...")
        editor_context = self.editor_controller.detect_editor(world_state)
        steps.append(f"Editor detected: {editor_context.editor_type} ('{editor_context.window_title}')")

        # Insert with retry logic
        for attempt in range(1, goal.max_retries + 1):
            steps.append(f"Inserting code (attempt {attempt}/{goal.max_retries})...")
            insert_ok, insert_msg = self.editor_controller.insert_code(solution, editor_context)

            if not insert_ok:
                steps.append(f"Insertion failed: {insert_msg}")
                continue

            steps.append("Code inserted. Verifying...")

            # Step 6: Verify insertion
            time.sleep(0.5)  # Brief wait for screen to settle
            post_image, post_state, post_desc = self.perception.perceive_screen()
            verify_ok, verify_msg = self.verifier.verify_insertion(
                solution.source_code, post_image, post_state.ocr_text,
            )

            if verify_ok:
                steps.append(f"Verification passed: {verify_msg}")
                return CodingResult(
                    success=True,
                    mode=CodingMode.SOLVE,
                    problem=problem,
                    solution=solution,
                    validation=validation,
                    steps_taken=steps,
                    message=f"Problem '{problem.title}' solved and code inserted successfully.",
                )
            else:
                steps.append(f"Verification failed: {verify_msg}")

        # All retries exhausted
        return CodingResult(
            success=False,
            mode=CodingMode.SOLVE,
            problem=problem,
            solution=solution,
            validation=validation,
            steps_taken=steps,
            message="Code was generated and validated but could not be verified in the editor after retries.",
        )

    def _execute_mcq_flow(self, goal: CodingGoal) -> CodingResult:
        """MCQ flow: capture → parse → explain."""
        steps = []

        # Step 1: Capture screen
        steps.append("Capturing screen for MCQ...")
        image, world_state, description = self.perception.perceive_screen()
        ocr_text = world_state.ocr_text
        steps.append(f"Screen captured. Active window: '{world_state.active_window}'")

        # Step 2: Parse MCQ
        steps.append("Parsing MCQ from screen...")
        problem, mcq = self.parser.parse_screen(image, ocr_text, description)

        if mcq is None:
            steps.append("No MCQ content detected on screen")
            return CodingResult(
                success=False,
                mode=CodingMode.MCQ,
                steps_taken=steps,
                message="Could not detect a multiple-choice question on the screen.",
            )

        return self._handle_mcq(mcq, goal, steps)

    def _handle_mcq(self, mcq, goal, steps) -> CodingResult:
        """Generates and returns an MCQ explanation."""
        steps.append(f"MCQ parsed: '{mcq.question_text[:80]}...'")

        # Step 3: Generate explanation
        steps.append("Generating MCQ explanation...")
        explanation = self.generator.explain_mcq(mcq)
        steps.append(
            f"Explanation generated: correct answer is {explanation.correct_label} "
            f"({explanation.correct_text})"
        )

        return CodingResult(
            success=True,
            mode=CodingMode.MCQ,
            mcq_explanation=explanation,
            steps_taken=steps,
            message=(
                f"Correct answer: {explanation.correct_label}. "
                f"{explanation.overall_reasoning}"
            ),
        )
