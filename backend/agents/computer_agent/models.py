"""
Computer Interaction Agent - Data Models.
Defines models for actions, perception elements, reasoning state, goals, and WorldState (v2.0).
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ApprovalLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"


class ComputerActionType(str, Enum):
    MOVE_TO = "move_to"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE = "type"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    WAIT = "wait"


class ComputerAction(BaseModel):
    """Represents a low-level computer action to be executed."""

    action_type: ComputerActionType = Field(..., description="The type of OS level interaction")
    x: Optional[int] = Field(default=None, description="Target X coordinate")
    y: Optional[int] = Field(default=None, description="Target Y coordinate")
    text: Optional[str] = Field(default=None, description="Text to type if action is TYPE")
    key: Optional[str] = Field(default=None, description="Single key name to press/down/up")
    keys: Optional[List[str]] = Field(default=None, description="List of key names for key combos")
    clicks: Optional[int] = Field(default=None, description="Scroll click count (positive or negative)")
    duration: float = Field(default=0.2, description="Duration in seconds for mouse movements or drag-and-drop")
    description: str = Field(default="", description="Human readable rationale of this step")

    # v2.0 fields
    approval_level: ApprovalLevel = Field(default=ApprovalLevel.SAFE, description="Approval level required for the action")
    wait_condition: Optional[str] = Field(default=None, description="Condition to wait for after execution (e.g. element name, window title)")
    wait_timeout: float = Field(default=10.0, description="Max seconds to wait for wait_condition to become true")


class PerceivedElement(BaseModel):
    """Represents a UI element perceived from screen analysis."""

    label: str
    semantic_type: Optional[str] = None
    bounding_box: Optional[List[int]] = None  # [x, y, w, h]
    confidence: Optional[float] = None
    center_x: Optional[int] = None
    center_y: Optional[int] = None


class WorldState(BaseModel):
    """Structured representation of the current OS and GUI environment state."""

    active_window: str = Field(default="Desktop", description="Title of the currently focused window")
    visible_elements: List[PerceivedElement] = Field(default_factory=list, description="UI elements on screen")
    ocr_text: str = Field(default="", description="Text captured via OCR")
    has_loading_indicator: bool = Field(default=False, description="Whether a loading indicator or spinner is detected")


class AgentGoal(BaseModel):
    """Represents the high-level user goal for the Computer Agent."""

    description: str
    max_steps: int = 15
    verification_criteria: Optional[str] = None


class ReasoningState(BaseModel):
    """Represents the agent's internal thinking state at any step."""

    goal: str
    visual_description: str
    world_state: Optional[WorldState] = Field(default=None, description="The observed world state at this step")
    history: List[str] = Field(default_factory=list)
    next_actions: List[ComputerAction] = Field(default_factory=list)
    is_terminal: bool = False
    success: bool = False
    message: str = ""
