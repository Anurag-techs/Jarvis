"""
JARVIS Memory Interfaces Package.

1. Why this module exists:
   Exposes abstract memory store interfaces for future stateful conversation and vector retrieval.

2. How it fits into the architecture:
   Defines clean contracts for memory without locking V1 foundation into specific databases.

3. Which future modules will interact with it:
   - `backend.memory.chroma` / `backend.memory.sqlite` (Version 3.0)
   - `backend.core.orchestrator`

4. Common mistakes to avoid:
   - Implementing memory databases or file writes in Version 1.0.

5. Possible future improvements:
   - Semantic RAG search and episodic session store interfaces.
"""

from backend.memory.base import BaseMemoryStore, PipelineMemoryStore
from backend.memory.models import ExtractedMemory, MemoryItem
from backend.memory.pre_filter import MemoryPreFilter
from backend.memory.extractor import MemoryExtractor
from backend.memory.summarizer import MemorySummarizer
from backend.memory.manager import MemoryManager
from backend.memory.pipeline import MemoryPipeline

__all__ = [
    "BaseMemoryStore",
    "PipelineMemoryStore",
    "ExtractedMemory",
    "MemoryItem",
    "MemoryPreFilter",
    "MemoryExtractor",
    "MemorySummarizer",
    "MemoryManager",
    "MemoryPipeline",
]
