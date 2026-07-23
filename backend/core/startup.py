"""
JARVIS Application Startup Manager & Application Container.

1. Why this module exists:
   Encapsulates application bootstrapping logic. Responsible for loading settings, initializing
   logging, registering tools, constructing the orchestrator via Dependency Injection,
   and measuring startup latency.

2. How it fits into the architecture:
   Part of the Core domain layer. Invoked directly by `backend.main`.

3. Which future modules will interact with it:
   - `backend.main`
   - `backend.api` (For server startup bootstrapping in Version 2.0).

4. Common mistakes to avoid:
   - Putting startup sequence logic inside `main.py`.
   - Using `print()` instead of the central logger for operational logs.

5. Possible future improvements:
   - Dynamic plugin scanner integration during tool registration.
"""

from dataclasses import dataclass
import logging
import time

from backend.ai.provider import MockLLMProvider
from backend.config.settings import Settings, get_settings
from backend.core.exceptions import ConfigurationError, JarvisError
from backend.core.orchestrator import SystemOrchestrator
from backend.tools.application_tool import ApplicationTool
from backend.tools.browser_tool import BrowserTool
from backend.tools.news_tool import NewsTool
from backend.tools.registry import ToolRegistry
from backend.tools.screenshot_tool import ScreenshotTool
from backend.tools.search_tool import SearchTool
from backend.tools.system_tool import SystemTool
from backend.tools.weather_tool import WeatherTool
from backend.utils.banner import display_banner
from backend.utils.logger import setup_logger

logger = logging.getLogger("jarvis.core.startup")


@dataclass
class JarvisApplication:
    """Holds fully initialized application components and dependencies."""

    settings: Settings
    logger: logging.Logger
    tool_registry: ToolRegistry
    orchestrator: SystemOrchestrator


class StartupManager:
    """Manager responsible for application bootstrapping and dependency graph construction."""

    def bootstrap(self) -> JarvisApplication:
        """Executes the Sprint 1 application bootstrap sequence.

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

            # Step 3: Register Tools
            logger.info("Registering tools...")
            registry = ToolRegistry()
            registry.register(ApplicationTool())
            registry.register(BrowserTool())
            registry.register(NewsTool())
            registry.register(ScreenshotTool())
            registry.register(SearchTool())
            registry.register(SystemTool())
            registry.register(WeatherTool())
            logger.info("[OK] %d tools registered", len(registry))

            # Step 4: Initialize Orchestrator via Dependency Injection
            logger.info("Initializing orchestrator...")
            llm_provider = MockLLMProvider(model_name=settings.llm_model)
            orchestrator = SystemOrchestrator(
                llm_provider=llm_provider,
                tool_registry=registry,
            )
            logger.info("[OK] Ready")

            # Measure & Log Startup Duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info("JARVIS startup completed in %.2f ms", duration_ms)
            logger.info("JARVIS is online. Waiting for commands...")

            return JarvisApplication(
                settings=settings,
                logger=app_logger,
                tool_registry=registry,
                orchestrator=orchestrator,
            )

        except JarvisError as err:
            logger.error("Initialization failed due to domain error: %s", err.message)
            raise
        except Exception as exc:
            logger.error("Unexpected initialization failure: %s", exc)
            raise ConfigurationError(
                message=f"Failed to bootstrap JARVIS application: {exc}"
            ) from exc
