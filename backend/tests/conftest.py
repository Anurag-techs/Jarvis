"""
Pytest fixtures for JARVIS test suite.

1. Why this module exists:
   Provides shared test fixtures for Pydantic Settings, ToolRegistry, Mock AI Providers, and SystemOrchestrator.

2. How it fits into the architecture:
   Part of the Testing framework. Isolates tests from environmental side effects.
"""

import pytest

from backend.ai.provider import MockLLMProvider
from backend.config.settings import Settings
from backend.core.orchestrator import SystemOrchestrator
from backend.tools.application_tool import ApplicationTool
from backend.tools.browser_tool import BrowserTool
from backend.tools.news_tool import NewsTool
from backend.tools.registry import ToolRegistry
from backend.tools.search_tool import SearchTool
from backend.tools.system_tool import SystemTool
from backend.tools.weather_tool import WeatherTool


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing clean Settings instance."""
    return Settings(
        env="development",
        debug=True,
        assistant_name="JARVIS-Test",
        wake_word="jarvis",
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Fixture providing mock LLM provider."""
    return MockLLMProvider(model_name="test-v1")


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Fixture providing a populated ToolRegistry with all V1 foundation tools."""
    registry = ToolRegistry()
    registry.register(ApplicationTool())
    registry.register(BrowserTool())
    registry.register(NewsTool())
    registry.register(SearchTool())
    registry.register(SystemTool())
    registry.register(WeatherTool())
    return registry


@pytest.fixture
def orchestrator(mock_llm: MockLLMProvider, tool_registry: ToolRegistry) -> SystemOrchestrator:
    """Fixture providing SystemOrchestrator with mock dependencies."""
    return SystemOrchestrator(
        llm_provider=mock_llm,
        tool_registry=tool_registry,
    )
