"""
JARVIS Google Gemini AI Provider.

1. Why this module exists:
   Integrates Google Gemini models (`gemini-2.5-flash`) via the official `google-genai` SDK.
   Parses structured JSON responses directly in the provider layer and returns `AssistantResponse`.

2. How it fits into the architecture:
   Concrete subclass of `BaseLLMProvider`. Injected into `SystemOrchestrator` via `StartupManager`.

3. Which future modules will interact with it:
   - `backend.core.startup.StartupManager`
   - `backend.conversation.manager.ConversationManager`

4. Common mistakes to avoid:
   - Returning raw JSON strings to orchestrators instead of parsing into `AssistantResponse`.
   - Exposing SDK-specific errors directly to callers.

5. Possible future improvements:
   - TODO: Add exponential backoff and retry logic for transient 429/503 network errors.
   - Native Gemini FunctionDeclaration tool calling bindings.
"""

import json
import logging
from typing import Any

from backend.ai.provider import BaseLLMProvider
from backend.config.settings import AIConfig
from backend.core.exceptions import ProviderError
from backend.core.models import AssistantResponse, ToolCall

logger = logging.getLogger("jarvis.ai.providers.gemini")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini AI Provider utilizing the official google-genai SDK with structured response parsing."""

    def __init__(self, config: AIConfig) -> None:
        """Initializes the Gemini client instance once using provided AIConfig.

        Args:
            config: Encapsulated AI provider configuration object.

        Raises:
            ProviderError: If API key is missing or SDK client initialization fails.
        """
        self._config = config
        self._model_name = config.model_name

        if not config.api_key:
            raise ProviderError(
                message="Google Gemini API key is unconfigured. Set GEMINI_API_KEY in .env file.",
                details={"provider": "gemini", "model": self._model_name},
            )

        try:
            import google.genai as genai  # Deferred import for clean SDK loading

            # Single client initialization reused across all requests
            self._client = genai.Client(api_key=config.api_key)
        except Exception as exc:
            raise ProviderError(
                message=f"Failed to initialize Google GenAI SDK client: {exc}",
                details={"provider": "gemini", "model": self._model_name},
            ) from exc

    @property
    def provider_name(self) -> str:
        return f"GeminiProvider({self._model_name})"

    def generate_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> AssistantResponse:
        """Generates structured completion response directly parsing Gemini JSON output into AssistantResponse.

        Args:
            user_prompt: Target query string from user.
            system_prompt: Optional instructions guiding AI persona and system behavior.
            history: Optional list of past conversation turns.
            available_tools: Optional tool metadata schemas.

        Returns:
            AssistantResponse DTO populated with text message and parsed tool_calls.

        Raises:
            ProviderError: Wrapped exception if Gemini API call fails.
        """
        if not user_prompt or not user_prompt.strip():
            return AssistantResponse(
                text="I did not receive a prompt to generate.",
                should_speak=False,
                success=False,
            )

        # TODO: Implement exponential backoff and retry logic for transient API network errors (e.g. Rate Limit / 429)

        try:
            import google.genai.types as types

            # Build system prompt with tool schemas and JSON formatting instructions
            full_system_instruction = self._build_system_instruction(system_prompt, available_tools)

            request_config = types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
                response_mime_type="application/json",
            )

            # Single client reused for API request
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=user_prompt,
                config=request_config,
            )

            if not response or not response.text:
                raise ProviderError(
                    message="Gemini API returned an empty completion response.",
                    details={"model": self._model_name},
                )

            return self._parse_json_response(response.text.strip())

        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                message=f"Gemini AI generation failed: {exc}",
                details={"provider": "gemini", "model": self._model_name, "prompt": user_prompt},
            ) from exc

    def _build_system_instruction(
        self, base_system_prompt: str | None, available_tools: list[dict[str, Any]] | None
    ) -> str:
        """Formats the system prompt with tool metadata schemas and JSON schema instructions."""
        prompt_parts = [base_system_prompt or "You are JARVIS, a production-quality AI assistant."]

        if available_tools:
            tools_json = json.dumps(available_tools, indent=2)
            prompt_parts.append(
                f"\nAVAILABLE TOOLS:\n{tools_json}\n\n"
                "INSTRUCTIONS:\n"
                "If the user query requires executing a tool, include the tool name and arguments in 'tool_calls'.\n"
                "Otherwise, set 'tool_calls' to an empty list [].\n\n"
                "REQUIRED JSON OUTPUT SCHEMA:\n"
                "{\n"
                '  "message": "Human readable text summary",\n'
                '  "tool_calls": [\n'
                '    {"tool": "tool_name", "arguments": {"arg": "value"}}\n'
                "  ]\n"
                "}"
            )
        else:
            prompt_parts.append(
                "\nREQUIRED JSON OUTPUT SCHEMA:\n"
                '{\n  "message": "Response text",\n  "tool_calls": []\n}'
            )

        return "\n\n".join(prompt_parts)

    def _parse_json_response(self, raw_text: str) -> AssistantResponse:
        """Parses raw JSON output into AssistantResponse model."""
        try:
            payload = json.loads(raw_text)
            message = payload.get("message", raw_text)
            tool_calls_data = payload.get("tool_calls", [])

            parsed_tool_calls: list[ToolCall] = []
            for call_item in tool_calls_data:
                tool_name = call_item.get("tool") or call_item.get("name")
                args = call_item.get("arguments") or call_item.get("args") or {}
                if tool_name:
                    parsed_tool_calls.append(ToolCall(tool=tool_name, arguments=args))

            return AssistantResponse(
                text=message,
                tool_calls=parsed_tool_calls,
                should_speak=True,
                success=True,
            )
        except Exception as exc:
            logger.warning("Failed to parse Gemini JSON output (%s); returning text directly", exc)
            return AssistantResponse(text=raw_text, tool_calls=[], should_speak=True)
