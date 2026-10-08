import re
"""
JARVIS System Orchestrator.

1. Why this module exists:
   Acts as the single control plane orchestrating the lifecycle, receiving user requests,
   coordinating tool decision execution via `ToolExecutor`, delegating conversation context to `ConversationManager`,
   triggering speech output, and returning structured `AssistantResponse`.

2. How it fits into the architecture:
   The core orchestrator lives in `backend.core`. It has ZERO UI logic.
   It relies entirely on Dependency Injection to delegate tasks to AI Providers, Tool Registry, ToolExecutor, ConversationManager, and TTS Service.

3. Which future modules will interact with it:
   - `backend.interfaces.console.ConsoleInterface`
   - `backend.api.router` (FastAPI HTTP / WebSocket endpoints in V2)

4. Common mistakes to avoid:
   - Adding direct console UI formatting or raw subprocess tool execution directly inside orchestrator code.

5. Possible future improvements:
   - Asynchronous event bus emission for real-time UI status notifications.
"""

import logging
import traceback
from typing import Optional

from backend.ai.provider import BaseLLMProvider
from backend.conversation.manager import ConversationManager
from backend.core.exceptions import ProviderError
from backend.core.models import AssistantResponse, ToolCall, ToolResult, UserIntent
from backend.memory.base import BaseMemoryStore
from backend.memory.recall_service import MemoryRecallService
from backend.services.intent_router import IntentRouter
from backend.tools.executor import ToolExecutor
from backend.tools.registry import ToolRegistry
from backend.voice.service import TTSService

logger = logging.getLogger("jarvis.core.orchestrator")

_PROVIDER_FALLBACK_MESSAGE: str = (
    "I'm having trouble reaching the AI service right now. "
    "Please try again in a moment."
)

# ---------------------------------------------------------------------------
# Memory gate — keywords that indicate the user is asking about past context.
# Memory search is SKIPPED for all other queries to reduce latency.
# ---------------------------------------------------------------------------
_MEMORY_KEYWORDS: frozenset[str] = frozenset({
    "remember", "remind", "recall", "last", "previous",
    "what was", "continue", "we were", "working on",
    "earlier", "before", "history", "you told", "i said",
    "last time", "my last", "last project", "last session",
})


def _needs_memory(query: str) -> bool:
    """Returns True when the query likely references past context.

    Only when this returns True will MemoryRecallService be consulted.
    All deterministic commands skip memory entirely for lower latency.
    """
    lowered = query.lower()
    return any(kw in lowered for kw in _MEMORY_KEYWORDS)


# ---------------------------------------------------------------------------
# Argument extractors for deterministic IntentRouter-matched tool calls.
# Each entry maps a tool_name -> extraction function(raw_query) -> dict.
# If no extractor is registered the raw query is passed as {"query": ...}.
# ---------------------------------------------------------------------------
_OPEN_APP_VERB_RE = re.compile(
    r"^\s*(open|launch|start|run)\s+", re.IGNORECASE
)
_OPEN_APP_TRAILER_RE = re.compile(
    r"[.!?,;]+\s*$", re.IGNORECASE
)


def _extract_open_application_args(raw_query: str) -> dict:
    """Extracts app_name from 'Open Chrome', 'Launch Notepad', etc."""
    # Strip leading verbs and trailing punctuation
    app_name = _OPEN_APP_VERB_RE.sub("", raw_query).strip()
    app_name = _OPEN_APP_TRAILER_RE.sub("", app_name).strip()
    return {"app_name": app_name}


def _extract_system_actions_args(raw_query: str) -> dict:
    """Extracts action description for system_actions tool."""
    return {"query": raw_query}


def _extract_camera_args(raw_query: str) -> dict:
    """Extracts action for camera_actions tool."""
    q = raw_query.lower()
    if "list" in q:
        action = "list_cameras"
    elif "analyze" in q or "describe" in q or "what" in q:
        action = "analyze_camera"
    else:
        action = "capture_camera"
    return {"action": action}


_TOOL_ARGUMENT_EXTRACTORS: dict = {
    "open_application": _extract_open_application_args,
    "system_actions": _extract_system_actions_args,
    "camera_actions": _extract_camera_args,
}


def _extract_tool_arguments(tool_name: str, raw_query: str) -> dict:
    """Extracts structured arguments for a deterministically-routed tool call.

    Args:
        tool_name: The name of the target tool returned by IntentRouter.
        raw_query: The raw user input string (already stripped).

    Returns:
        A dict of kwargs suitable for the target tool's execute(**kwargs).
    """
    extractor = _TOOL_ARGUMENT_EXTRACTORS.get(tool_name)
    if extractor:
        return extractor(raw_query)
    # Generic fallback: pass the raw query under a "query" key
    return {"query": raw_query}



class SystemOrchestrator:
    """Core Orchestrator coordinating AI providers, tool registry, tool executor, conversation manager, and speech service."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor | None = None,
        conversation_manager: ConversationManager | None = None,
        tts_service: TTSService | None = None,
        memory_store: BaseMemoryStore | None = None,
        recall_service: MemoryRecallService | None = None,
        intent_router: Optional[IntentRouter] = None,
    ) -> None:
        """Initialize orchestrator via constructor dependency injection.

        Args:
            llm_provider: Abstract AI LLM provider.
            tool_registry: Pluggable tool registry holding executable V1 tools.
            tool_executor: ToolExecutor service managing tool execution pipeline.
            conversation_manager: ConversationManager for multi-turn history.
            tts_service: Optional TTSService handling audio output.
            memory_store: Optional abstract memory store interface.
            recall_service: Optional memory recall service interface.
            intent_router: Optional deterministic pre-classifier. When provided,
                matched queries bypass LLM tool-selection entirely.
        """
        self._llm = llm_provider
        self._tools = tool_registry
        self._tool_executor = tool_executor or ToolExecutor(registry=tool_registry)
        self._conversation_manager = conversation_manager or ConversationManager(llm_provider=llm_provider)
        self._tts = tts_service
        self._memory = memory_store
        self._recall_service = recall_service
        self._intent_router: Optional[IntentRouter] = intent_router

        logger.info(
            "SystemOrchestrator initialized with LLM: %s, Registered Tools: %d",
            self._llm.provider_name,
            len(self._tools),
        )

    def process(self, user_input: str) -> AssistantResponse:
        """Orchestrates processing of raw user text or transcribed speech input.

        This method is guaranteed never to raise. All provider failures are caught,
        logged with full stack traces, and returned as a friendly AssistantResponse
        so the conversation loop stays alive.

        Args:
            user_input: Raw text command from user interface.

        Returns:
            AssistantResponse containing text, tool_calls, should_speak flag, and execution status.
        """
        logger.info("Processing command input: '%s'", user_input)
        clean_input = user_input.strip()

        if not clean_input:
            return AssistantResponse(
                text="I did not receive any input.",
                should_speak=False,
                success=False,
                error="Empty input provided",
            )

        try:
            # 1. Deterministic intent routing — bypass LLM for high-confidence queries
            if self._intent_router:
                routed_tool = self._intent_router.route(clean_input)
                if routed_tool:
                    logger.info(
                        "IntentRouter matched query to '%s' — skipping LLM tool-selection",
                        routed_tool,
                    )
                    direct_call = ToolCall(tool=routed_tool, arguments=_extract_tool_arguments(routed_tool, clean_input))
                    tool_results: list[ToolResult] = self._tool_executor.execute_tool_calls([direct_call])
                    result = tool_results[0] if tool_results else ToolResult(
                        success=False, message="Tool execution returned no results."
                    )
                    response_text = result.message
                    direct_response = AssistantResponse(
                        text=response_text,
                        should_speak=True,
                        success=result.success,
                        tool_calls=[direct_call],
                        error=result.error,
                    )
                    if self._tts and direct_response.should_speak:
                        self._tts.speak(direct_response.text)
                    if self._memory and direct_response.success:
                        try:
                            self._memory.store_interaction(clean_input, direct_response.text)
                        except Exception as exc:
                            logger.error("Failed to store interaction in memory: %s", exc, exc_info=True)
                    return direct_response

            # 2. Fetch available tool metadata schemas for AI provider
            available_tool_schemas = self._tools.get_available_tools()

            # 3. Build context using MemoryRecallService (only for conversational queries)
            injected_prompt = clean_input
            if self._recall_service and _needs_memory(clean_input):
                injected_prompt = self._recall_service.build_context(clean_input)
            elif self._recall_service:
                logger.debug("Memory search skipped for query=%r (no memory keywords detected)", clean_input[:60])

            # 4. Query AI provider & ConversationManager for intent & tool call decisions
            initial_response: AssistantResponse = self._conversation_manager.generate_response(
                user_prompt=clean_input,
                available_tools=available_tool_schemas,
                injected_prompt=injected_prompt,
            )

            # 5. Check if response contains tool execution decisions
            if initial_response.tool_calls:
                logger.info("Orchestrator detected %d tool call requests", len(initial_response.tool_calls))

                # Execute tool calls via isolated ToolExecutor service
                llm_tool_results: list[ToolResult] = self._tool_executor.execute_tool_calls(initial_response.tool_calls)

                # Send tool execution results back to AI provider to generate final natural language summary
                final_response = self._conversation_manager.generate_final_response(llm_tool_results)
                final_response.tool_calls = initial_response.tool_calls

                if self._tts and final_response.should_speak:
                    self._tts.speak(final_response.text)

                # Save interaction to memory store if present and response succeeded
                if self._memory and final_response.success:
                    try:
                        self._memory.store_interaction(clean_input, final_response.text)
                    except Exception as exc:
                        logger.error("Failed to store interaction in memory: %s", exc, exc_info=True)

                return final_response

            # 6. Standard conversational response without tool execution
            if self._tts and initial_response.should_speak:
                self._tts.speak(initial_response.text)

            # Save interaction to memory store if present and response succeeded
            if self._memory and initial_response.success:
                try:
                    self._memory.store_interaction(clean_input, initial_response.text)
                except Exception as exc:
                    logger.error("Failed to store interaction in memory: %s", exc, exc_info=True)

            return initial_response

        except ProviderError as exc:
            # Safety net: GeminiProvider should already return a friendly AssistantResponse
            # on exhausted retries, but this catches any provider errors that still escape
            # (e.g. from ConversationManager internals or tool synthesis calls).
            logger.error(
                "ProviderError escaped into orchestrator for input=%r — "
                "returning friendly fallback. Error: %s\n%s",
                clean_input[:80],
                exc.message,
                traceback.format_exc(),
            )
            return AssistantResponse(
                text=_PROVIDER_FALLBACK_MESSAGE,
                tool_calls=[],
                should_speak=True,
                success=False,
                error=exc.message,
            )
        except Exception as exc:
            # Catch-all for unexpected non-provider failures (e.g. tool execution bugs)
            logger.error(
                "Unexpected orchestrator error for input=%r: %s\n%s",
                clean_input[:80],
                exc,
                traceback.format_exc(),
            )
            return AssistantResponse(
                text="Something unexpected happened. Please try again.",
                tool_calls=[],
                should_speak=True,
                success=False,
                error=str(exc),
            )

    def shutdown(self) -> None:
        """Cleanly shuts down orchestrator components."""
        logger.info("Shutting down SystemOrchestrator...")
        if self._memory and hasattr(self._memory, "shutdown"):
            try:
                self._memory.shutdown()
            except Exception as e:
                logger.warning("Error during memory store shutdown: %s", e)


