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

from backend.ai.provider import BaseLLMProvider
from backend.conversation.manager import ConversationManager
from backend.core.models import AssistantResponse, ToolResult, UserIntent
from backend.memory.base import BaseMemoryStore
from backend.tools.executor import ToolExecutor
from backend.tools.registry import ToolRegistry
from backend.voice.service import TTSService

logger = logging.getLogger("jarvis.core.orchestrator")


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
    ) -> None:
        """Initialize orchestrator via constructor dependency injection.

        Args:
            llm_provider: Abstract AI LLM provider.
            tool_registry: Pluggable tool registry holding executable V1 tools.
            tool_executor: ToolExecutor service managing tool execution pipeline.
            conversation_manager: ConversationManager for multi-turn history.
            tts_service: Optional TTSService handling audio output.
            memory_store: Optional abstract memory store interface.
        """
        self._llm = llm_provider
        self._tools = tool_registry
        self._tool_executor = tool_executor or ToolExecutor(registry=tool_registry)
        self._conversation_manager = conversation_manager or ConversationManager(llm_provider=llm_provider)
        self._tts = tts_service
        self._memory = memory_store

        logger.info(
            "SystemOrchestrator initialized with LLM: %s, Registered Tools: %d",
            self._llm.provider_name,
            len(self._tools),
        )

    def process(self, user_input: str) -> AssistantResponse:
        """Orchestrates processing of raw user text or transcribed speech input.

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

        # 1. Fetch available tool metadata schemas for AI provider
        available_tool_schemas = self._tools.get_available_tools()

        # 2. Query AI provider & ConversationManager for intent & tool call decisions
        initial_response: AssistantResponse = self._conversation_manager.generate_response(
            user_prompt=clean_input,
            available_tools=available_tool_schemas,
        )

        # 3. Check if response contains tool execution decisions
        if initial_response.tool_calls:
            logger.info("Orchestrator detected %d tool call requests", len(initial_response.tool_calls))
            
            # Execute tool calls via isolated ToolExecutor service
            tool_results: list[ToolResult] = self._tool_executor.execute_tool_calls(initial_response.tool_calls)
            
            # Send tool execution results back to AI provider to generate final natural language summary
            final_response = self._conversation_manager.generate_final_response(tool_results)
            final_response.tool_calls = initial_response.tool_calls

            if self._tts and final_response.should_speak:
                self._tts.speak(final_response.text)

            return final_response

        # 4. Standard conversational response without tool execution
        if self._tts and initial_response.should_speak:
            self._tts.speak(initial_response.text)

        return initial_response
