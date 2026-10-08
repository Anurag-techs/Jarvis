"""
Coding Agent - Code Generator.
Generates production-quality code solutions and MCQ explanations using an LLM provider.
"""

import json
import logging
import re
from typing import Optional

from backend.ai.provider import BaseLLMProvider
from backend.agents.coding_agent.interfaces import BaseCodeGenerator
from backend.agents.coding_agent.models import (
    CodeSolution,
    MCQExplanation,
    MCQOption,
    MCQQuestion,
    ProgrammingLanguage,
    ProgrammingProblem,
)

logger = logging.getLogger("jarvis.agents.coding_agent.generator")

_CODE_SYSTEM_PROMPT = """You are an expert competitive programmer and software engineer.
Generate a complete, production-quality solution for the given programming problem.

Requirements:
- Write clean, well-commented code
- Handle all edge cases mentioned in the constraints
- Use optimal algorithms where possible
- Include the exact function/class signature if one is provided

Respond with ONLY a JSON object in this format:
{
  "source_code": "complete source code here",
  "explanation": "Step-by-step explanation of the approach",
  "time_complexity": "O(...)",
  "space_complexity": "O(...)",
  "function_name": "main entry function name or null",
  "class_name": "class name if applicable or null"
}

Return ONLY the JSON object. No markdown wrapping."""

_MCQ_SYSTEM_PROMPT = """You are an expert educator and subject matter expert.
Analyze the given multiple-choice question and provide a detailed explanation.

For EACH option, explain:
- Whether it is correct or incorrect
- WHY it is correct or incorrect with technical reasoning

This is intended for learning and practice purposes only.

Respond with ONLY a JSON object in this format:
{
  "correct_label": "B",
  "correct_text": "text of the correct option",
  "overall_reasoning": "Detailed explanation of why the correct answer is correct",
  "options": [
    {"label": "A", "text": "option text", "is_correct": false, "reasoning": "Why this is wrong"},
    {"label": "B", "text": "option text", "is_correct": true, "reasoning": "Why this is correct"},
    ...
  ]
}

Return ONLY the JSON object. No markdown wrapping."""


class CodingCodeGenerator(BaseCodeGenerator):
    """Generates code solutions and MCQ explanations via LLM."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm = llm_provider

    def generate_solution(self, problem: ProgrammingProblem) -> CodeSolution:
        logger.info("Generating %s solution for: '%s'", problem.language.value, problem.title)

        # Format examples
        examples_text = ""
        for i, ex in enumerate(problem.examples, 1):
            examples_text += f"\nExample {i}:\n  Input: {ex.input}\n  Output: {ex.output}"
            if ex.explanation:
                examples_text += f"\n  Explanation: {ex.explanation}"

        constraints_text = "\n".join(f"  - {c}" for c in problem.constraints) if problem.constraints else "None specified"

        signature_text = f"\nRequired signature: {problem.function_signature}" if problem.function_signature else ""

        user_prompt = (
            f"PROBLEM: {problem.title}\n\n"
            f"STATEMENT:\n{problem.statement}\n\n"
            f"CONSTRAINTS:\n{constraints_text}\n"
            f"EXAMPLES:{examples_text}\n\n"
            f"LANGUAGE: {problem.language.value}\n"
            f"PLATFORM: {problem.platform.value}\n"
            f"{signature_text}\n\n"
            "Generate the optimal solution."
        )

        try:
            raw = self._llm.generate_raw_completion(
                user_prompt=user_prompt,
                system_prompt=_CODE_SYSTEM_PROMPT,
                response_mime_type="application/json",
            )
            payload = self._parse_json(raw)

            source_code = payload.get("source_code", "")
            # Strip markdown code fences from source code if present
            source_code = self._strip_code_fences(source_code)

            return CodeSolution(
                source_code=source_code,
                language=problem.language,
                explanation=payload.get("explanation", ""),
                time_complexity=payload.get("time_complexity", ""),
                space_complexity=payload.get("space_complexity", ""),
                function_name=payload.get("function_name"),
                class_name=payload.get("class_name"),
            )
        except Exception as e:
            logger.error("Code generation failed: %s", e)
            return CodeSolution(
                source_code=f"# Code generation failed: {e}",
                language=problem.language,
                explanation=f"Error: {e}",
            )

    def explain_mcq(self, question: MCQQuestion) -> MCQExplanation:
        logger.info("Generating MCQ explanation for: '%s'", question.question_text[:80])

        options_text = "\n".join(
            f"  {opt.label}. {opt.text}" for opt in question.options
        )

        user_prompt = (
            f"QUESTION:\n{question.question_text}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
        )
        if question.topic:
            user_prompt += f"TOPIC: {question.topic}\n\n"
        user_prompt += "Analyze all options and identify the correct answer."

        try:
            raw = self._llm.generate_raw_completion(
                user_prompt=user_prompt,
                system_prompt=_MCQ_SYSTEM_PROMPT,
                response_mime_type="application/json",
            )
            payload = self._parse_json(raw)

            options = []
            for opt in payload.get("options", []):
                options.append(MCQOption(
                    label=opt.get("label", ""),
                    text=opt.get("text", ""),
                    is_correct=opt.get("is_correct", False),
                    reasoning=opt.get("reasoning", ""),
                ))

            return MCQExplanation(
                correct_label=payload.get("correct_label", ""),
                correct_text=payload.get("correct_text", ""),
                overall_reasoning=payload.get("overall_reasoning", ""),
                options=options,
            )
        except Exception as e:
            logger.error("MCQ explanation generation failed: %s", e)
            return MCQExplanation(
                correct_label="?",
                correct_text="Unable to determine",
                overall_reasoning=f"Error generating explanation: {e}",
                options=[],
            )

    def _parse_json(self, raw: str) -> dict:
        """Parses JSON from LLM response, stripping markdown fences if needed."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        return json.loads(cleaned)

    def _strip_code_fences(self, code: str) -> str:
        """Removes markdown code fence wrapping from source code."""
        stripped = code.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Remove first line (```python or ```)
            lines = lines[1:]
            # Remove last line if it's closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return stripped


# Backward compatibility alias
LLMCodeGenerator = CodingCodeGenerator
