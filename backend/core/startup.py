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
from typing import Any
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
from backend.tools.camera_tool import CameraTool
from backend.tools.browser_tool import BrowserTool
from backend.tools.executor import ToolExecutor
from backend.tools.news_tool import NewsTool
from backend.tools.registry import ToolRegistry
from backend.tools.screenshot_tool import ScreenshotTool
from backend.tools.search_tool import SearchTool
from backend.tools.system_tool import SystemTool
from backend.services.desktop import DesktopService
from backend.services.approval import TerminalApprovalService
from backend.services.intent_router import IntentRouter
from backend.services.transcript_normalizer import TranscriptNormalizer
from backend.tools.desktop_automation_tool import DesktopAutomationTool

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

    def run(self, console_mode: bool = False) -> None:
        """Determines the runtime dynamically using RuntimeManager and launches it."""
        from backend.core.runtime import RuntimeManager, RuntimeType

        runtime_manager = RuntimeManager(self.settings)
        selected_runtime, status = runtime_manager.select_runtime(force_console=console_mode)
        runtime_manager.print_report(selected_runtime, status)

        if selected_runtime == RuntimeType.VOICE:
            logger.info("Launching JarvisApplication in Voice Interface lifecycle...")
            self.voice_controller.start()
        else:
            logger.info("Launching JarvisApplication in Console Interface lifecycle...")
            from backend.interfaces.console import ConsoleInterface
            console_interface = ConsoleInterface(orchestrator=self.orchestrator)
            console_interface.start()

    def shutdown(self) -> None:
        """Cleanly shuts down all application components and releases resources."""
        logger.info("Shutting down JarvisApplication...")
        if hasattr(self.orchestrator, "shutdown"):
            try:
                self.orchestrator.shutdown()
            except Exception as e:
                logger.warning("Error during orchestrator shutdown: %s", e)
        if hasattr(self.tts_service, "stop"):
            try:
                self.tts_service.stop()
            except Exception as e:
                logger.warning("Error stopping TTS service: %s", e)
        logger.info("JarvisApplication shutdown complete.")



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

            # Check Python compatibility & dependencies
            from backend.core.runtime import RuntimeManager
            runtime_mgr = RuntimeManager(settings)
            _, runtime_status = runtime_mgr.select_runtime()

            # Trigger auto-installer if requested and needed
            if not runtime_status["voice_supported"] and runtime_status["python_supported"]:
                if settings.auto_install_dependencies:
                    logger.info("Missing dependencies detected. settings.auto_install_dependencies=True: Installing automatically...")
                    try:
                        from backend.scripts.install_voice_dependencies import install_voice_dependencies
                        install_voice_dependencies()
                        # Re-evaluate runtime status
                        _, runtime_status = runtime_mgr.select_runtime()
                    except Exception as e:
                        logger.error("Auto dependency installation failed: %s", e)

            # Print warning and advice if missing packages exist
            if not runtime_status["voice_supported"] and runtime_status["python_supported"]:
                for pkg, available in runtime_status["dependencies"].items():
                    if not available:
                        RuntimeManager.handle_missing_dependency(pkg)

            # Step 2: Initialize Logger
            logger.info("Initializing logger...")
            app_logger = setup_logger(level=settings.log_level, log_format=None)
            logger.info("[OK] Logger ready")

            # Step 3: Initialize TTS/STT/WakeWord Voice Services (Keep Startup Completely Silent)
            logger.info("Initializing TTS, STT & WakeWord voice services...")
            tts_provider = self._create_tts_provider(settings.voice_provider, settings)
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
            registry.register(CameraTool())
            registry.register(BrowserTool())
            registry.register(NewsTool())
            registry.register(ScreenshotTool())
            registry.register(SearchTool())
            registry.register(SystemTool())
            registry.register(WeatherTool())
            
            # Desktop Automation v1.0
            desktop_service = DesktopService()
            approval_service = TerminalApprovalService()
            registry.register(
                DesktopAutomationTool(
                    desktop_service=desktop_service,
                    approval_service=approval_service,
                )
            )

            tool_executor = ToolExecutor(registry=registry)
            logger.info("[OK] %d tools registered & ToolExecutor ready", len(registry))


            # Step 5: Initialize AI Provider, Memory & ConversationManager
            logger.info("Initializing AI Provider, Memory & ConversationManager...")
            llm_provider = self._create_llm_provider(settings)

            # Initialize Vision Service & Vision Tool
            logger.info("Initializing Vision Service & Vision Tool...")
            vision_service = self._create_vision_service(settings, llm_provider)
            from backend.tools.vision_tool import VisionTool
            registry.register(VisionTool(vision_service=vision_service))

            # Step 4b: Build and register CodingAgentTool
            logger.info("Initializing CodingAgent & CodingAgentTool...")
            coding_agent_tool = self._create_coding_agent_tool(
                desktop_service=desktop_service,
                llm_provider=llm_provider,
                vision_service=vision_service,
            )
            if coding_agent_tool:
                registry.register(coding_agent_tool)
                logger.info("[OK] CodingAgent tool registered as 'coding_actions'")
            else:
                logger.warning("CodingAgent could not be initialized; 'coding_actions' will not be available")

            # Initialize Planning Coordinator & Planner Tool
            logger.info("Initializing Planning Coordinator & Planner Tool...")
            planner_coordinator = self._create_planner_coordinator(
                settings=settings,
                llm_provider=llm_provider,
                tool_executor=tool_executor,
                tool_registry=registry,
                vision_service=vision_service
            )
            from backend.tools.planner_tool import PlannerTool
            registry.register(PlannerTool(coordinator=planner_coordinator))

            # Initialize Memory components using Dependency Injection
            from backend.memory import MemoryPipeline, PipelineMemoryStore, MemoryRecallService
            memory_pipeline = MemoryPipeline(llm_provider=llm_provider)
            memory_store = PipelineMemoryStore(pipeline=memory_pipeline)
            recall_service = MemoryRecallService(memory_store=memory_store)

            system_prompt_provider = DefaultSystemPromptProvider(assistant_name=settings.assistant_name)
            conversation_manager = ConversationManager(
                llm_provider=llm_provider,
                history_limit=settings.conversation_history_limit,
                system_prompt_provider=system_prompt_provider,
            )
            logger.info("[OK] ConversationManager ready (Limit: %d turns)", settings.conversation_history_limit)

            # Step 6: Initialize Orchestrator & VoiceController via Dependency Injection
            logger.info("Initializing orchestrator & VoiceController...")
            intent_router = IntentRouter()
            orchestrator = SystemOrchestrator(
                llm_provider=llm_provider,
                tool_registry=registry,
                tool_executor=tool_executor,
                conversation_manager=conversation_manager,
                tts_service=tts_service,
                memory_store=memory_store,
                recall_service=recall_service,
                intent_router=intent_router,
            )
            transcript_normalizer = TranscriptNormalizer()
            voice_controller = VoiceController(
                wake_acknowledgement=settings.wake_acknowledgement,
                orchestrator=orchestrator,
                stt_provider=stt_provider,
                wake_word_detector=wake_word_detector,
                tts_service=tts_service,
                cooldown_seconds=settings.voice_cooldown_seconds,
                speech_start_timeout=settings.voice_speech_start_timeout,
                silence_timeout=settings.voice_silence_timeout,
                silence_threshold=settings.voice_silence_threshold,
                beep_enabled=settings.voice_beep_enabled,
                beep_frequency=settings.voice_beep_frequency,
                beep_duration=settings.voice_beep_duration,
                follow_up_timeout=settings.voice_follow_up_timeout,
                barge_in_enabled=settings.voice_barge_in_enabled,
                pre_roll_buffer_ms=settings.voice_pre_roll_buffer_ms,
                vad_threshold=settings.voice_vad_threshold,
                summary_threshold=settings.voice_summary_threshold,
                confidence_high=settings.stt_confidence_high,
                confidence_low=settings.stt_confidence_low,
                transcript_normalizer=transcript_normalizer,
            )

            logger.info("[OK] Ready (AI Provider: %s)", llm_provider.provider_name)

            # Measure & Log Startup Duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info("JARVIS startup completed in %.2f ms", duration_ms)
            logger.info("JARVIS is online. Waiting for commands...")

            app = JarvisApplication(
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

            self._print_startup_diagnostics(app)
            return app

        except JarvisError as err:
            logger.error("Initialization failed due to domain error: %s", err.message)
            raise
        except Exception as exc:
            logger.error("Unexpected initialization failure: %s", exc)
            raise ConfigurationError(
                message=f"Failed to bootstrap JARVIS application: {exc}"
            ) from exc

    def _print_startup_diagnostics(self, app: JarvisApplication) -> None:
        """Evaluates all system components and prints a comprehensive startup report."""
        print("\n=====================================")
        print("JARVIS Environment Report")
        print("=====================================")

        # 1. Python
        import sys
        major, minor = sys.version_info.major, sys.version_info.minor
        python_ok = major == 3 and minor in (11, 12)
        if python_ok:
            print(f"Python: [OK] Ready (Version {major}.{minor})")
        else:
            print(f"Python: [FAILED] Failed (Unsupported version {major}.{minor}, voice requires 3.11 or 3.12)")

        # 2. Virtual Environment
        import os
        venv_active = (sys.prefix != sys.base_prefix) or ("VIRTUAL_ENV" in os.environ)
        if venv_active:
            print("Virtual Environment: [OK] Ready")
        else:
            print("Virtual Environment: [FAILED] Failed (Not running in a virtual environment)")

        # 3. Gemini API
        ai_config = app.settings.get_ai_config()
        if ai_config.provider.lower() == "gemini" and ai_config.api_key:
            print("Gemini API: [OK] Ready")
        else:
            reason = "LLM_PROVIDER is not gemini" if ai_config.provider.lower() != "gemini" else "GEMINI_API_KEY is missing"
            print(f"Gemini API: [FAILED] Failed ({reason})")

        # 4. Voice, TTS, STT, Wake Word
        from backend.voice.providers import MockTTSProvider, MockSTTProvider
        from backend.voice.wake_word import MockWakeWordDetector

        tts_real = not isinstance(app.tts_service._provider, MockTTSProvider) if app.tts_service else False
        stt_real = not isinstance(app.stt_provider, MockSTTProvider)
        ww_real = not isinstance(app.wake_word_detector, MockWakeWordDetector)

        voice_ready = tts_real and stt_real and ww_real
        print(f"Voice: {'[OK] Ready' if voice_ready else '[FAILED] Failed (One or more voice services are running in Mock mode)'}")
        print(f"TTS: {'[OK] Ready' if tts_real else '[FAILED] Failed (Pyttsx3Provider is unavailable or not selected)'}")
        print(f"STT: {'[OK] Ready' if stt_real else '[FAILED] Failed (FasterWhisperProvider is unavailable or not selected)'}")
        print(f"Wake Word: {'[OK] Ready' if ww_real else '[FAILED] Failed (OpenWakeWordDetector is unavailable or not selected)'}")

        # 5. Microphone & Speaker
        mic_ok = False
        speaker_ok = False
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            try:
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        mic_ok = True
                    if info.get("maxOutputChannels", 0) > 0:
                        speaker_ok = True
            finally:
                p.terminate()
        except Exception:
            pass
        print(f"Microphone: {'[OK] Ready' if mic_ok else '[FAILED] Failed (No microphone detected)'}")
        print(f"Speaker: {'[OK] Ready' if speaker_ok else '[FAILED] Failed (No speaker detected)'}")

        # 6. Vision
        vision_tool = app.tool_registry.get_tool("vision_actions")
        if vision_tool:
            print("Vision: [OK] Ready")
        else:
            print("Vision: [FAILED] Failed (Vision tool not registered)")

        # 7. Planner
        planner_tool = app.tool_registry.get_tool("planner_actions")
        if planner_tool:
            print("Planner: [OK] Ready")
        else:
            print("Planner: [FAILED] Failed (Planner tool not registered)")

        # 8. Memory
        import sqlite3
        try:
            conn = sqlite3.connect(app.settings.memory_db_path)
            conn.close()
            print("Memory: [OK] Ready")
        except Exception as e:
            print(f"Memory: [FAILED] Failed ({e})")

        # 9. Desktop Automation
        da_tool = app.tool_registry.get_tool("desktop_automation")
        if da_tool:
            print("Desktop Automation: [OK] Ready")
        else:
            print("Desktop Automation: [FAILED] Failed (Desktop Automation tool not registered)")

        # 10. Coding Agent
        coding_tool = app.tool_registry.get_tool("coding_actions")
        if coding_tool:
            print("Coding Agent: [OK] Ready")
        else:
            print("Coding Agent: [FAILED] Failed (coding_actions tool not registered)")

        print("=====================================\n")

    def _create_wake_word_detector(self, settings: Settings) -> BaseWakeWordDetector:
        """Creates target WakeWordDetector instance with clean fallback."""
        if settings.wakeword_provider.lower() == "openwakeword":
            try:
                return OpenWakeWordDetector(
                    wake_word=settings.wake_word,
                    sensitivity=settings.wakeword_sensitivity,
                    rolling_window_size=settings.wakeword_rolling_window_size,
                    min_consecutive_detections=settings.wakeword_min_consecutive_detections,
                    cooldown_duration=settings.wakeword_cooldown_duration,
                    lower_threshold=settings.wakeword_lower_threshold,
                )
            except Exception as err:
                if not settings.debug:
                    raise ConfigurationError(
                        message=f"Failed to initialize OpenWakeWordDetector in production: {err}"
                    ) from err
                logger.warning(
                    "OpenWakeWordDetector initialization failed (%s); falling back to MockWakeWordDetector",
                    err,
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
            except Exception as err:
                if not settings.debug:
                    raise ConfigurationError(
                        message=f"Failed to initialize FasterWhisperProvider in production: {err}"
                    ) from err
                logger.warning(
                    "FasterWhisperProvider initialization failed (%s); falling back to MockSTTProvider",
                    err,
                )
                return MockSTTProvider()
        return MockSTTProvider()

    def _create_llm_provider(self, settings: Settings) -> BaseLLMProvider:
        """Creates target AI LLM provider using AIConfig with transparent reason logging.

        Selection priority:
          1. If ``llm_provider`` is "gemini" AND ``gemini_api_key`` is present → GeminiProvider.
          2. If ``llm_provider`` is "gemini" BUT key is missing → warn and fall back to Mock.
          3. If ``llm_provider`` is "mock" (or anything else) → Mock (explicit or default).
          4. If GeminiProvider raises during init (SDK error, bad key) → warn and fall back to Mock.
        """
        ai_config = settings.get_ai_config()

        # Always log the resolved configuration so startup is self-diagnosing
        logger.info(
            "Provider config: LLM_PROVIDER=%s | GEMINI_API_KEY set=%s | model=%s",
            ai_config.provider,
            bool(ai_config.api_key),
            ai_config.model_name,
        )

        if ai_config.provider.lower() != "gemini":
            logger.info(
                "AI Provider: MockLLMProvider selected "
                "(reason: LLM_PROVIDER is '%s', not 'gemini'). "
                "Set LLM_PROVIDER=gemini in .env to use the real AI.",
                ai_config.provider,
            )
            return MockLLMProvider(model_name=ai_config.model_name)

        if not ai_config.api_key:
            logger.warning(
                "AI Provider: falling back to MockLLMProvider "
                "(reason: LLM_PROVIDER=gemini but GEMINI_API_KEY is not set in .env). "
                "Add your Gemini API key to enable real AI responses."
            )
            return MockLLMProvider(model_name=ai_config.model_name)

        try:
            provider = GeminiProvider(config=ai_config)
            logger.info(
                "AI Provider: GeminiProvider loaded successfully (model: %s).",
                ai_config.model_name,
            )
            return provider
        except JarvisError as err:
            logger.warning(
                "AI Provider: GeminiProvider initialization failed (%s); "
                "falling back to MockLLMProvider.",
                err.message,
                exc_info=True,
            )
            return MockLLMProvider(model_name=ai_config.model_name)



    def _create_tts_provider(self, provider_name: str, settings: Settings):
        """Creates target TTS provider instance with clean fallback."""
        if provider_name.lower() == "pyttsx3":
            try:
                return Pyttsx3Provider()
            except Exception as err:
                if not settings.debug:
                    raise ConfigurationError(
                        message=f"Failed to initialize Pyttsx3Provider in production: {err}"
                    ) from err
                logger.warning(
                    "Pyttsx3Provider initialization failed (%s); falling back to MockTTSProvider",
                    err,
                )
                return MockTTSProvider()
        return MockTTSProvider()

    def _create_vision_service(self, settings: Settings, llm_provider: BaseLLMProvider) -> Any:
        """Helper to create and configure the VisionService and its swappable providers."""
        from backend.services.vision import (
            MssScreenCaptureProvider,
            MockScreenCaptureProvider,
            EasyOCROCRProvider,
            MockOCRProvider,
            LLMVisionProvider,
            MockVisionProvider,
            VisionService
        )

        # 1. Screen Capture Provider (Fast native screenshots via mss)
        try:
            capture_provider = MssScreenCaptureProvider()
        except Exception as e:
            logger.warning("Failed to initialize MssScreenCaptureProvider (%s); falling back to MockScreenCaptureProvider", e)
            capture_provider = MockScreenCaptureProvider()

        # 2. OCR Provider (Optional)
        ocr_provider = None
        if settings.ocr_provider.lower() == "easyocr":
            try:
                ocr_provider = EasyOCROCRProvider()
            except Exception as e:
                logger.warning("Failed to initialize EasyOCROCRProvider (%s); falling back to MockOCRProvider", e)
                ocr_provider = MockOCRProvider()
        else:
            ocr_provider = MockOCRProvider()

        # 3. Vision Provider (Optional/Mockable)
        vision_provider = None
        if settings.vision_provider.lower() == "gemini":
            try:
                ai_config = settings.get_ai_config()
                vision_provider = LLMVisionProvider(config=ai_config)
            except Exception as e:
                logger.warning("Failed to initialize LLMVisionProvider (%s); falling back to MockVisionProvider", e)
                vision_provider = MockVisionProvider()
        else:
            vision_provider = MockVisionProvider()

        logger.info(
            "VisionService initialized. Capture: %s | OCR: %s | Vision: %s",
            type(capture_provider).__name__,
            type(ocr_provider).__name__ if ocr_provider else "None",
            type(vision_provider).__name__ if vision_provider else "None"
        )
        return VisionService(
            capture_provider=capture_provider,
            ocr_provider=ocr_provider,
            vision_provider=vision_provider
        )

    def _create_planner_coordinator(
        self,
        settings: Settings,
        llm_provider: BaseLLMProvider,
        tool_executor: ToolExecutor,
        tool_registry: ToolRegistry,
        vision_service: Any
    ) -> Any:
        """Helper to bootstrap the PlanningCoordinator and pluggable VerificationManager."""
        from backend.services.planner import (
            PlanRepository,
            LLMPlanner,
            MockPlanner,
            VerificationManager,
            VisionVerifier,
            MockVerifier,
            Replanner,
            PlanExecutor,
            PlanningCoordinator
        )

        # 1. Repository (Persistence storage inside instance/plans)
        storage_dir = settings.memory_db_path.replace("jarvis_memory.db", "plans")
        repository = PlanRepository(storage_dir=storage_dir)

        # 2. Select planner implementation
        from backend.ai.provider import MockLLMProvider
        if isinstance(llm_provider, MockLLMProvider):
            planner = MockPlanner()
        else:
            planner = LLMPlanner(llm_provider=llm_provider)

        # 3. Pluggable Verifier
        verifier_manager = VerificationManager()
        verifier_manager.add_verifier(MockVerifier(default_success=True))
        if vision_service:
            try:
                vision_verifier = VisionVerifier(vision_service=vision_service, llm_provider=llm_provider)
                verifier_manager.add_verifier(vision_verifier)
            except Exception as e:
                logger.warning("Failed to initialize VisionVerifier (%s)", e)

        # 4. Replanner
        replanner = Replanner(llm_provider=llm_provider)

        # 5. PlanExecutor
        executor = PlanExecutor(
            tool_executor=tool_executor,
            verifier_manager=verifier_manager,
            replanner=replanner
        )

        # 6. Coordinator control plane
        coordinator = PlanningCoordinator(
            planner=planner,
            executor=executor,
            repository=repository,
            tool_registry=tool_registry
        )

        logger.info(
            "PlanningCoordinator initialized with repository storage: %s",
            storage_dir
        )
        return coordinator

    def _create_coding_agent_tool(
        self,
        desktop_service: "DesktopService",
        llm_provider: Any = None,
        vision_service: Any = None,
    ) -> "CodingAgentTool | None":
        """Build and return a fully wired CodingAgentTool, or None on failure.

        Creates the complete CodingAgent dependency graph:
          VisionPerception -> CodingProblemParser -> CodingCodeGenerator -> CodingCodeValidator
          -> CodingEditorController -> CodingVerifier -> CodingAgent -> CodingAgentTool

        Returns:
            CodingAgentTool ready for registration, or None if any dependency fails.
        """
        try:
            from backend.agents.coding_agent.coordinator import CodingAgent
            from backend.agents.coding_agent.editor_controller import CodingEditorController, DesktopEditorController
            from backend.agents.coding_agent.generator import CodingCodeGenerator, LLMCodeGenerator
            from backend.agents.coding_agent.parser import CodingProblemParser, VisionProblemParser
            from backend.agents.coding_agent.validator import CodingCodeValidator, CodeValidator
            from backend.agents.coding_agent.verifier import CodingVerifier, OCRCodingVerifier
            from backend.tools.coding_agent_tool import CodingAgentTool
            from backend.agents.computer_agent.perception import ComputerAgentPerception, VisionPerception
            from backend.config.settings import get_settings

            settings = get_settings()

            if llm_provider is None:
                llm_provider = self._create_llm_provider(settings)

            if vision_service is None:
                vision_service = self._create_vision_service(settings, llm_provider)

            perception = ComputerAgentPerception(vision_service=vision_service)
            parser = CodingProblemParser(llm_provider=llm_provider)
            generator = CodingCodeGenerator(llm_provider=llm_provider)
            validator = CodingCodeValidator()
            editor_controller = CodingEditorController(desktop_service=desktop_service)
            verifier = CodingVerifier()

            agent = CodingAgent(
                perception=perception,
                parser=parser,
                generator=generator,
                validator=validator,
                editor_controller=editor_controller,
                verifier=verifier,
            )
            return CodingAgentTool(coding_agent=agent)

        except Exception as exc:
            logger.warning("Failed to build CodingAgentTool: %s", exc, exc_info=True)
            return None