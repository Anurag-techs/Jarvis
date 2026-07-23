"""
JARVIS Application Startup Manager & Application Container.

1. Why this module exists:
   Encapsulates application bootstrapping logic. Responsible for loading settings, initializing
   logging, registering tools, initializing TTS/STT/WakeWord services, selecting configured AI providers,
   instantiating ToolExecutor, ConversationManager, and VoiceController, constructing orchestrator via DI,
   and measuring startup latency.

2. How it fits into the architecture:
   Part of the Core domain layer. Invoked directly by `backend.main`.

3. Which future modules will interact with it:
   - `backend.main`
   - `backend.api` (For server startup bootstrapping in Version 2.0).

4. Common mistakes to avoid:
   - Putting startup sequence logic inside `main.py`.
   - Triggering audio output during application startup (startup must remain completely silent).

5. Possible future improvements:
   - Dynamic plugin scanner integration during tool registration.
"""

from dataclasses import dataclass
import logging
import time

from backend.ai.provider import BaseLLMProvider, MockLLMProvider
from backend.ai.providers.gemini_provider import GeminiProvider
from backend.config.settings import Settings, get_settings
from backend.conversation.manager import ConversationManager
from backend.conversation.system_prompt import DefaultSystemPromptProvider
from backend.core.exceptions import ConfigurationError, JarvisError
from backend.core.orchestrator import SystemOrchestrator
from backend.interfaces.console import ConsoleInterface
from backend.interfaces.voice import VoiceController
from backend.tools.application_tool import ApplicationTool
from backend.tools.browser_tool import BrowserTool
from backend.tools.executor import ToolExecutor
from backend.tools.news_tool import NewsTool
from backend.tools.registry import ToolRegistry
from backend.tools.screenshot_tool import ScreenshotTool
from backend.tools.search_tool import SearchTool
from backend.tools.system_tool import SystemTool
from backend.tools.weather_tool import WeatherTool
from backend.utils.banner import display_banner
from backend.utils.logger import setup_logger
from backend.voice.base import BaseWakeWordDetector, SpeechToTextProvider
from backend.voice.providers import (
    FasterWhisperProvider,
    MockSTTProvider,
    MockTTSProvider,
    Pyttsx3Provider,
)
from backend.voice.service import TTSService
from backend.voice.wake_word import MockWakeWordDetector, OpenWakeWordDetector

logger = logging.getLogger("jarvis.core.startup")


@dataclass
class JarvisApplication:
    """Holds fully initialized application components and lifecycle runner."""

    settings: Settings
    logger: logging.Logger
    tool_registry: ToolRegistry
    orchestrator: SystemOrchestrator
    tts_service: TTSService
    stt_provider: SpeechToTextProvider
    wake_word_detector: BaseWakeWordDetector
    conversation_manager: ConversationManager
    tool_executor: ToolExecutor
    voice_controller: VoiceController

    def run(self, voice_mode: bool = False) -> None:
        """Delegates execution to the active lifecycle interface runner (Console or Voice)."""
        if voice_mode or self.settings.voice_mode:
            logger.info("Launching JarvisApplication in Voice Interface lifecycle...")
            self.voice_controller.start()
        else:
            logger.info("Launching JarvisApplication in Console Interface lifecycle...")
            console_interface = ConsoleInterface(orchestrator=self.orchestrator)
            console_interface.start()


class StartupManager:
    """Manager responsible for application bootstrapping and dependency graph construction."""

    def bootstrap(self) -> JarvisApplication:
        """Executes the application bootstrap sequence.

        Returns:
            Fully initialized JarvisApplication container.

        Raises:
            JarvisError: If initialization fails.
        """
        start_time = time.perf_counter()
        display_banner(version="1.0")

        try:
            # Step 1: Load Configuration
            logger.info("Loading configuration...")
            settings = get_settings()
            logger.info("[OK] Configuration loaded (Environment: %s)", settings.env)

            # Step 2: Initialize Logger
            logger.info("Initializing logger...")
            app_logger = setup_logger(level=settings.log_level, log_format=None)
            logger.info("[OK] Logger ready")

            # Step 3: Initialize TTS/STT/WakeWord Voice Services (Keep Startup Completely Silent)
            logger.info("Initializing TTS, STT & WakeWord voice services...")
            tts_provider = self._create_tts_provider(settings.voice_provider)
            tts_service = TTSService(provider=tts_provider)
            stt_provider = self._create_stt_provider(settings)
            wake_word_detector = self._create_wake_word_detector(settings)
            logger.info(
                "[OK] TTS ready (%s), STT ready (%s), WakeWord ready (%s)",
                type(tts_provider).__name__,
                type(stt_provider).__name__,
                type(wake_word_detector).__name__,
            )

            # Step 4: Register Tools & Instantiate ToolExecutor
            logger.info("Registering tools & ToolExecutor...")
            registry = ToolRegistry()
            registry.register(ApplicationTool())
            registry.register(BrowserTool())
            registry.register(NewsTool())
            registry.register(ScreenshotTool())
            registry.register(SearchTool())
            registry.register(SystemTool())
            registry.register(WeatherTool())
            tool_executor = ToolExecutor(registry=registry)
            logger.info("[OK] %d tools registered & ToolExecutor ready", len(registry))

            # Step 5: Initialize AI Provider & ConversationManager
            logger.info("Initializing AI Provider & ConversationManager...")
            llm_provider = self._create_llm_provider(settings)
            system_prompt_provider = DefaultSystemPromptProvider(assistant_name=settings.assistant_name)
            conversation_manager = ConversationManager(
                llm_provider=llm_provider,
                history_limit=settings.conversation_history_limit,
                system_prompt_provider=system_prompt_provider,
            )
            logger.info("[OK] ConversationManager ready (Limit: %d turns)", settings.conversation_history_limit)

            # Step 6: Initialize Orchestrator & VoiceController via Dependency Injection
            logger.info("Initializing orchestrator & VoiceController...")
            orchestrator = SystemOrchestrator(
                llm_provider=llm_provider,
                tool_registry=registry,
                tool_executor=tool_executor,
                conversation_manager=conversation_manager,
                tts_service=tts_service,
            )
            voice_controller = VoiceController(
                orchestrator=orchestrator,
                stt_provider=stt_provider,
                wake_word_detector=wake_word_detector,
                tts_service=tts_service,
                listen_timeout=settings.listen_timeout,
                post_response_timeout=settings.post_response_timeout,
                cooldown_seconds=settings.voice_cooldown_seconds,
            )
            logger.info("[OK] Ready (AI Provider: %s)", llm_provider.provider_name)

            # Measure & Log Startup Duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info("JARVIS startup completed in %.2f ms", duration_ms)
            logger.info("JARVIS is online. Waiting for commands...")

            return JarvisApplication(
                settings=settings,
                logger=app_logger,
                tool_registry=registry,
                orchestrator=orchestrator,
                tts_service=tts_service,
                stt_provider=stt_provider,
                wake_word_detector=wake_word_detector,
                conversation_manager=conversation_manager,
                tool_executor=tool_executor,
                voice_controller=voice_controller,
            )

        except JarvisError as err:
            logger.error("Initialization failed due to domain error: %s", err.message)
            raise
        except Exception as exc:
            logger.error("Unexpected initialization failure: %s", exc)
            raise ConfigurationError(
                message=f"Failed to bootstrap JARVIS application: {exc}"
            ) from exc

    def _create_wake_word_detector(self, settings: Settings) -> BaseWakeWordDetector:
        """Creates target WakeWordDetector instance with clean fallback."""
        if settings.wakeword_provider.lower() == "openwakeword":
            try:
                return OpenWakeWordDetector(
                    wake_word=settings.wake_word,
                    sensitivity=settings.wakeword_sensitivity,
                )
            except JarvisError as err:
                logger.warning(
                    "OpenWakeWordDetector initialization failed (%s); falling back to MockWakeWordDetector",
                    err.message,
                )
                return MockWakeWordDetector()
        return MockWakeWordDetector()

    def _create_stt_provider(self, settings: Settings) -> SpeechToTextProvider:
        """Creates target STT provider instance with clean fallback."""
        if settings.stt_provider.lower() == "faster_whisper":
            try:
                return FasterWhisperProvider(
                    model_size=settings.whisper_model,
                    device=settings.stt_device,
                    compute_type=settings.stt_compute_type,
                )
            except JarvisError as err:
                logger.warning(
                    "FasterWhisperProvider initialization failed (%s); falling back to MockSTTProvider",
                    err.message,
                )
                return MockSTTProvider()
        return MockSTTProvider()

    def _create_llm_provider(self, settings: Settings) -> BaseLLMProvider:
        """Creates target AI LLM provider using AIConfig."""
        ai_config = settings.get_ai_config()
        if ai_config.provider.lower() == "gemini":
            try:
                return GeminiProvider(config=ai_config)
            except JarvisError as err:
                logger.warning(
                    "GeminiProvider initialization failed (%s); falling back to MockLLMProvider",
                    err.message,
                )
                return MockLLMProvider(model_name=ai_config.model_name)
        return MockLLMProvider(model_name=ai_config.model_name)

    def _create_tts_provider(self, provider_name: str):
        """Creates target TTS provider instance with clean fallback."""
        if provider_name.lower() == "pyttsx3":
            try:
                return Pyttsx3Provider()
            except JarvisError as err:
                logger.warning(
                    "Pyttsx3Provider initialization failed (%s); falling back to MockTTSProvider",
                    err.message,
                )
                return MockTTSProvider()
        return MockTTSProvider()
