"""
JARVIS Memory Data Models.

1. Why this module exists:
   Defines the structured representations of memories, search modes, and extracted facts.
   Ensures type-safety and structured validation using Pydantic.

2. How it fits into the architecture:
   Part of the Memory layer. Dictates how memories are represented, saved, and searched.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
import uuid
from pydantic import BaseModel, Field


class SearchType(str, Enum):
    """Modes supported for querying the Memory store."""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class ExtractedMemory(BaseModel):
    """Temporary DTO representing an extracted memory fact before persistence validation."""

    content: str = Field(..., description="The main factual context or preference to remember")
    importance: float = Field(
        default=1.0, 
        description="Core priority ranking score (e.g. 1.0 to 5.0) of the memory"
    )
    source: Literal["user", "assistant", "system"] = Field(
        default="user", 
        description="Entity that originated the memory"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata tags, categories or confidence scores"
    )


class MemoryItem(BaseModel):
    """Structured representation of a saved memory stored in the memory manager."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the memory item",
    )
    content: str = Field(..., description="The remembered context or preference statement")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of when the memory was initially saved",
    )
    last_accessed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of when the memory was last retrieved/accessed",
    )
    importance: float = Field(
        default=1.0,
        description="Subjective priority rating of the memory"
    )
    source: Literal["user", "assistant", "system"] = Field(
        default="user",
        description="Preserves where the memory originated (user, assistant, system)"
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiration timestamp for transient memories",
    )
    is_deleted: bool = Field(
        default=False,
        description="Flag indicating if the memory has been soft-deleted",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata storage for future categorization or source tracking",
    )
