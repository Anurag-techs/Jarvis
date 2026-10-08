"""
Coding Agent - Data Models.
Pydantic models for programming problems, code solutions, MCQ questions,
editor contexts, validation results, goals, and final outcomes.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CodingPlatform(str, Enum):
    """Known competitive programming / coding platforms."""
    LEETCODE = "leetcode"
    HACKERRANK = "hackerrank"
    CODEFORCES = "codeforces"
    GEEKSFORGEEKS = "geeksforgeeks"
    VSCODE = "vscode"
    GENERIC = "generic"


class CodingMode(str, Enum):
    """Operating mode of the Coding Agent."""
    SOLVE = "solve"
    MCQ = "mcq"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages for code generation."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class Example(BaseModel):
    """A single input/output example for a programming problem."""
    input: str = Field(..., description="Example input")
    output: str = Field(..., description="Expected output")
    explanation: Optional[str] = Field(default=None, description="Optional walkthrough")


class ProgrammingProblem(BaseModel):
    """Structured representation of a programming problem extracted from screen."""
    title: str = Field(..., description="Problem title")
    statement: str = Field(..., description="Full problem statement")
    constraints: List[str] = Field(default_factory=list, description="Constraint lines")
    examples: List[Example] = Field(default_factory=list, description="Sample input/output pairs")
    language: ProgrammingLanguage = Field(default=ProgrammingLanguage.PYTHON, description="Target language")
    platform: CodingPlatform = Field(default=CodingPlatform.GENERIC, description="Detected source platform")
    difficulty: Optional[str] = Field(default=None, description="Difficulty label if available")
    function_signature: Optional[str] = Field(default=None, description="Required function/class signature if specified")


class CodeSolution(BaseModel):
    """Generated code solution with metadata."""
    source_code: str = Field(..., description="Complete source code")
    language: ProgrammingLanguage = Field(default=ProgrammingLanguage.PYTHON)
    explanation: str = Field(default="", description="Step-by-step explanation of the approach")
    time_complexity: str = Field(default="", description="Big-O time complexity")
    space_complexity: str = Field(default="", description="Big-O space complexity")
    function_name: Optional[str] = Field(default=None, description="Entry point function name")
    class_name: Optional[str] = Field(default=None, description="Entry point class name if applicable")


class MCQOption(BaseModel):
    """A single MCQ option with analysis."""
    label: str = Field(..., description="Option label (A, B, C, D)")
    text: str = Field(..., description="Option text content")
    is_correct: bool = Field(default=False)
    reasoning: str = Field(default="", description="Why this option is correct or incorrect")


class MCQQuestion(BaseModel):
    """Structured representation of a multiple-choice question."""
    question_text: str = Field(..., description="The question stem")
    options: List[MCQOption] = Field(default_factory=list, description="Available options")
    topic: Optional[str] = Field(default=None, description="Subject area")
    platform: CodingPlatform = Field(default=CodingPlatform.GENERIC)


class MCQExplanation(BaseModel):
    """Full explanation of an MCQ including per-option reasoning."""
    correct_label: str = Field(..., description="Label of the correct answer (A/B/C/D)")
    correct_text: str = Field(..., description="Text of the correct answer")
    overall_reasoning: str = Field(..., description="Why the correct answer is correct")
    options: List[MCQOption] = Field(default_factory=list, description="Per-option analysis")


class ValidationResult(BaseModel):
    """Result of code validation before insertion."""
    is_valid: bool = Field(default=False, description="Whether the code passed all checks")
    syntax_ok: bool = Field(default=False, description="Syntax check passed")
    has_required_structure: bool = Field(default=False, description="Required function/class exists")
    sample_tests_passed: int = Field(default=0, description="Number of sample I/O tests passed")
    sample_tests_total: int = Field(default=0, description="Total sample I/O tests attempted")
    estimated_time_complexity: Optional[str] = Field(default=None, description="Estimated complexity if determinable")
    errors: List[str] = Field(default_factory=list, description="Error messages from validation")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")


class EditorContext(BaseModel):
    """Describes the currently active text editor state."""
    editor_type: str = Field(default="generic", description="Detected editor name")
    window_title: str = Field(default="", description="Full window title")
    has_existing_code: bool = Field(default=False, description="Whether the editor has existing content")
    file_extension: Optional[str] = Field(default=None, description="File extension from title bar")


class CodingGoal(BaseModel):
    """High-level goal for the Coding Agent."""
    description: str = Field(default="Solve the visible problem", description="Goal description")
    mode: CodingMode = Field(default=CodingMode.SOLVE, description="Operating mode")
    preferred_language: ProgrammingLanguage = Field(default=ProgrammingLanguage.PYTHON)
    max_retries: int = Field(default=2, description="Max retry attempts for insertion")


class CodingResult(BaseModel):
    """Final outcome returned by the Coding Agent."""
    success: bool = Field(default=False)
    mode: CodingMode = Field(default=CodingMode.SOLVE)
    problem: Optional[ProgrammingProblem] = None
    solution: Optional[CodeSolution] = None
    validation: Optional[ValidationResult] = None
    mcq_explanation: Optional[MCQExplanation] = None
    steps_taken: List[str] = Field(default_factory=list, description="Log of steps executed")
    message: str = Field(default="", description="Summary message")
    duration_seconds: float = Field(default=0.0, description="Total execution time")
