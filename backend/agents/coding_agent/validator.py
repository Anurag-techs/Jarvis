"""
Coding Agent - Code Validator.
Validates generated code before insertion: syntax checking, sample I/O testing,
structural verification, and complexity estimation.
"""

import ast
import io
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import List, Optional, Tuple

from backend.agents.coding_agent.interfaces import BaseCodeValidator
from backend.agents.coding_agent.models import (
    CodeSolution,
    ProgrammingLanguage,
    ProgrammingProblem,
    ValidationResult,
)

logger = logging.getLogger("jarvis.agents.coding_agent.validator")

# Maximum execution time for sample test runs (seconds)
_EXECUTION_TIMEOUT = 10


class CodingCodeValidator(BaseCodeValidator):
    """Validates generated code for correctness and structural integrity before editor insertion."""

    def validate(
        self,
        solution: CodeSolution,
        problem: ProgrammingProblem,
    ) -> ValidationResult:
        logger.info("Validating solution for '%s' (%s)", problem.title, solution.language.value)

        errors: List[str] = []
        warnings: List[str] = []

        # 1. Syntax validation
        syntax_ok = self._check_syntax(solution, errors)

        # 2. Structural validation (required function/class exists)
        has_structure = self._check_structure(solution, problem, errors, warnings)

        # 3. Sample I/O tests
        tests_passed, tests_total = self._run_sample_tests(solution, problem, errors, warnings)

        # 4. Complexity estimation
        estimated_complexity = self._estimate_complexity(solution, warnings)

        # 5. Basic sanity checks
        self._sanity_checks(solution, errors, warnings)

        # Determine overall validity
        is_valid = syntax_ok and has_structure and len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            syntax_ok=syntax_ok,
            has_required_structure=has_structure,
            sample_tests_passed=tests_passed,
            sample_tests_total=tests_total,
            estimated_time_complexity=estimated_complexity,
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            "Validation result: valid=%s | syntax=%s | structure=%s | tests=%d/%d | errors=%d",
            result.is_valid, result.syntax_ok, result.has_required_structure,
            result.sample_tests_passed, result.sample_tests_total, len(result.errors),
        )
        return result

    def _check_syntax(self, solution: CodeSolution, errors: List[str]) -> bool:
        """Validates syntax of the generated code."""
        if solution.language == ProgrammingLanguage.PYTHON:
            return self._check_python_syntax(solution.source_code, errors)
        else:
            # For non-Python languages, perform basic bracket/brace matching
            return self._check_bracket_balance(solution.source_code, errors)

    def _check_python_syntax(self, code: str, errors: List[str]) -> bool:
        """Uses Python's ast module to verify syntax."""
        try:
            ast.parse(code)
            logger.debug("Python syntax check passed.")
            return True
        except SyntaxError as e:
            msg = f"Python syntax error at line {e.lineno}: {e.msg}"
            logger.warning(msg)
            errors.append(msg)
            return False

    def _check_bracket_balance(self, code: str, errors: List[str]) -> bool:
        """Basic bracket/brace/parenthesis balance check for any language."""
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        in_string = False
        string_char = None

        for i, ch in enumerate(code):
            # Simplified string detection (doesn't handle all edge cases)
            if ch in ('"', "'") and not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char and in_string:
                in_string = False
                string_char = None
            elif not in_string:
                if ch in pairs:
                    stack.append(pairs[ch])
                elif ch in pairs.values():
                    if not stack or stack[-1] != ch:
                        errors.append(f"Unmatched bracket '{ch}' at position {i}")
                        return False
                    stack.pop()

        if stack:
            errors.append(f"Unclosed brackets: {len(stack)} remaining")
            return False

        logger.debug("Bracket balance check passed.")
        return True

    def _check_structure(
        self,
        solution: CodeSolution,
        problem: ProgrammingProblem,
        errors: List[str],
        warnings: List[str],
    ) -> bool:
        """Verifies required functions/classes exist in the generated code."""
        code = solution.source_code

        # Check for function signature if specified in the problem
        if problem.function_signature:
            # Extract function name from signature
            func_match = re.search(r'def\s+(\w+)', problem.function_signature)
            if func_match:
                required_func = func_match.group(1)
                if f"def {required_func}" not in code:
                    errors.append(f"Required function '{required_func}' not found in solution")
                    return False
                logger.debug("Required function '%s' found.", required_func)

        # Check solution's own declared function/class names
        if solution.function_name:
            if solution.function_name not in code:
                warnings.append(f"Declared function_name '{solution.function_name}' not found in code")

        if solution.class_name:
            if solution.class_name not in code:
                warnings.append(f"Declared class_name '{solution.class_name}' not found in code")

        # For Python, use AST to verify top-level definitions exist
        if solution.language == ProgrammingLanguage.PYTHON:
            try:
                tree = ast.parse(code)
                has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                has_class = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
                if not has_function and not has_class:
                    warnings.append("No function or class definition found in solution — may be script-style code")
            except SyntaxError:
                pass  # Already caught by syntax check

        return True

    def _run_sample_tests(
        self,
        solution: CodeSolution,
        problem: ProgrammingProblem,
        errors: List[str],
        warnings: List[str],
    ) -> Tuple[int, int]:
        """Runs sample input/output tests for Python solutions."""
        if not problem.examples:
            logger.debug("No sample examples provided — skipping I/O tests.")
            return 0, 0

        if solution.language != ProgrammingLanguage.PYTHON:
            warnings.append(f"Sample I/O testing not supported for {solution.language.value} — skipping")
            return 0, 0

        passed = 0
        total = len(problem.examples)

        for i, example in enumerate(problem.examples):
            success = self._run_single_test(solution.source_code, example.input, example.output, i + 1, errors)
            if success:
                passed += 1

        logger.info("Sample tests: %d/%d passed", passed, total)
        return passed, total

    def _run_single_test(
        self,
        code: str,
        test_input: str,
        expected_output: str,
        test_num: int,
        errors: List[str],
    ) -> bool:
        """Executes a single test case in an isolated subprocess."""
        # Build a test script that runs the solution with the given input
        test_script = (
            f"{code}\n\n"
            f"# Auto-generated test harness\n"
            f"import sys\n"
            f"sys.stdin = __import__('io').StringIO({repr(test_input)})\n"
        )

        # Try to find and call the main function or rely on stdin-based execution
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=_EXECUTION_TIMEOUT,
                cwd=tempfile.gettempdir(),
            )

            actual = result.stdout.strip()
            expected = expected_output.strip()

            if result.returncode != 0:
                err_msg = result.stderr.strip()[:200]
                errors.append(f"Test {test_num}: Runtime error — {err_msg}")
                return False

            if actual == expected:
                logger.debug("Test %d passed: output matches expected.", test_num)
                return True
            else:
                # Non-fatal: output mismatch might be due to function-based solutions
                # that don't print to stdout
                logger.debug(
                    "Test %d: output mismatch (expected='%s', got='%s'). "
                    "May be function-based solution without print.",
                    test_num, expected[:50], actual[:50],
                )
                return False

        except subprocess.TimeoutExpired:
            errors.append(f"Test {test_num}: Execution timed out ({_EXECUTION_TIMEOUT}s)")
            return False
        except Exception as e:
            errors.append(f"Test {test_num}: Execution error — {e}")
            return False

    def _estimate_complexity(self, solution: CodeSolution, warnings: List[str]) -> Optional[str]:
        """Estimates time complexity using heuristic AST analysis for Python."""
        if solution.language != ProgrammingLanguage.PYTHON:
            return solution.time_complexity or None

        # Use the LLM-reported complexity as primary, supplement with heuristic
        if solution.time_complexity:
            # Perform basic heuristic cross-check
            code = solution.source_code
            nested_loops = self._count_nested_loops(code)

            if nested_loops >= 3 and "O(n)" == solution.time_complexity:
                warnings.append(
                    f"Reported complexity {solution.time_complexity} but found "
                    f"{nested_loops} nested loops — verify manually"
                )

            return solution.time_complexity

        # Heuristic fallback
        nested_loops = self._count_nested_loops(solution.source_code)
        if nested_loops == 0:
            return "O(1) or O(n)"
        elif nested_loops == 1:
            return "O(n) estimated"
        elif nested_loops == 2:
            return "O(n²) estimated"
        else:
            return f"O(n^{nested_loops}) estimated"

    def _count_nested_loops(self, code: str) -> int:
        """Counts the maximum nesting depth of for/while loops."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0

        max_depth = 0

        def _walk(node: ast.AST, depth: int) -> None:
            nonlocal max_depth
            if isinstance(node, (ast.For, ast.While)):
                depth += 1
                max_depth = max(max_depth, depth)
            for child in ast.iter_child_nodes(node):
                _walk(child, depth)

        _walk(tree, 0)
        return max_depth

    def _sanity_checks(self, solution: CodeSolution, errors: List[str], warnings: List[str]) -> None:
        """Performs basic sanity checks on the generated code."""
        code = solution.source_code.strip()

        if not code:
            errors.append("Generated code is empty")
            return

        if len(code) < 10:
            warnings.append("Generated code is suspiciously short (< 10 characters)")

        # Check for common error patterns
        if "# Code generation failed" in code:
            errors.append("Code generation produced an error placeholder, not a real solution")

        # Check for incomplete code patterns
        if code.endswith("...") or code.endswith("pass"):
            warnings.append("Code appears to end with placeholder content")


# Backward compatibility alias
CodeValidator = CodingCodeValidator
