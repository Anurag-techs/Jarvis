"""
JARVIS System Orchestrator.

1. Why this module exists:
   Acts as the single control plane orchestrating the lifecycle, receiving user requests,
   coordinating intent resolution via AI/Tools, and returning final user feedback.

2. How it fits into the architecture:
   The core orchestrator lives in backend.core. It has ZERO direct business logic.
   It relies entirely on Dependency Injection to delegate tasks to AI Services, Voice Managers,
   and the Tool Registry.

3. Which future modules will interact with it:
   - backend.main (CLI runner)
   - backend.api.router (FastAPI HTTP / WebSocket endpoints in V2)
   - Future background job event listeners

4. Common mistakes to avoid:
   - Adding direct tool execution logic or string manipulation inside this file.
   - Instantiating concrete services or third-party APIs directly inside `__init__`.

5. Possible future improvements:
   - Asynchronous event bus emission for real-time UI status notifications.
"""

import logging

from backend.ai.provider import BaseLLMProvider
from backend.core.models import CommandResult, UserIntent
from backend.memory.base import BaseMemoryStore
from backend.tools.registry import ToolRegistry
from backend.voice.manager import VoiceManager

logger = logging.getLogger("jarvis.core.orchestrator")


class SystemOrchestrator:
    """Core Orchestrator coordinating AI providers, tool registry, and voice manager."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        tool_registry: ToolRegistry,
        voice_manager: VoiceManager | None = None,
        memory_store: BaseMemoryStore | None = None,
    ) -> None:
        """Initialize orchestrator via constructor dependency injection.

        Args:
            llm_provider: Abstract AI LLM provider.
            tool_registry: Pluggable tool registry holding executable V1 tools.
            voice_manager: Optional voice manager handling STT/TTS lifecycle.
            memory_store: Optional abstract memory store interface (V1 placeholder).
        """
        self._llm = llm_provider
        self._tools = tool_registry
        self._voice = voice_manager
        self._memory = memory_store

        logger.info(
            "SystemOrchestrator initialized with LLM: %s, Registered Tools: %d",
            self._llm.provider_name,
            len(self._tools),
        )

    def process_command(self, user_input: str) -> CommandResult:
        """Orchestrates processing of raw user text or transcribed speech input.

        Args:
            user_input: Raw text command from user or speech-to-text.

        Returns:
            CommandResult containing execution success state and text response.
        """
        logger.info("Processing command input: '%s'", user_input)
        clean_input = user_input.strip()

        if not clean_input:
            return CommandResult(
                success=False,
                response_text="I did not receive any input.",
                error_message="Empty input provided",
            )

        # Step 1: Parse intent via AI provider (or tool keyword matching strategy)
        intent: UserIntent = self._parse_intent(clean_input)

        # Step 2: Route according to intent type
        if intent.intent_type == "tool_call" and intent.tool_name:
            tool_result = self._tools.execute_tool(intent.tool_name, intent.parameters)
            response_text = tool_result.message

            # Speak output if voice manager is active
            if self._voice:
                self._voice.speak(response_text)

            return CommandResult(
                success=tool_result.success,
                response_text=response_text,
                data=tool_result.data,
                error_message=tool_result.error,
            )

        # Step 3: Default conversational response via LLM
        llm_response = self._llm.generate_response(clean_input)

        if self._voice:
            self._voice.speak(llm_response)

        return CommandResult(
            success=True,
            response_text=llm_response,
            data={"intent": "conversation"},
        )

    def _parse_intent(self, text: str) -> UserIntent:
        """Internal helper mapping user query to structured intent.

        In V1.0, checks tool registry keyword triggers before falling back to LLM conversation.
        """
        lowered = text.lower()

        # Simple deterministic intent routing for V1 foundation tools
        for tool_name in self._tools.list_tools():
            tool = self._tools.get_tool(tool_name)
            if tool and tool.can_handle(lowered):
                return UserIntent(
                    raw_text=text,
                    intent_type="tool_call",
                    tool_name=tool_name,
                    parameters={"query": text},
                )

        return UserIntent(
            raw_text=text,
            intent_type="conversation",
        )
