# JARVIS Architecture Guide

## Overview

JARVIS is built using **Clean Architecture** and **SOLID Design Principles** to ensure long-term maintainability, strict separation of concerns, and modular evolution across multiple releases.

## Core Architectural Layers

```
+-----------------------------------------------------------------------+
|                            PRESENTATION / API                         |
|                 (CLI Entrypoint / Future FastAPI / Frontend)          |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                            CORE ORCHESTRATION                         |
|                     (backend.core.orchestrator)                      |
+-----------------------------------------------------------------------+
                    /               |               \
                   v                v                v
+-----------------------+ +-------------------+ +-----------------------+
|       SERVICES        | |   TOOL REGISTRY   | |    VOICE LIFECYCLE    |
| (LLM, Speech, System, | | (Search, Weather, | | (STT, TTS, WakeWord)  |
|    News, Weather)     | | Applications, etc)| |                     |
+-----------------------+ +-------------------+ +-----------------------+
          |                         |                        |
          v                         v                        v
+-----------------------------------------------------------------------+
|                         ABSTRACT INTERFACES                           |
|        (BaseLLMProvider, BaseTool, BaseVoiceEngine, BaseMemoryStore)  |
+-----------------------------------------------------------------------+
```

## Package Responsibilities

### 1. `backend/config/`
- **Responsibility**: Environment management using `pydantic-settings` and centralized Python logging configuration.
- **Rules**: No hardcoded API keys or magic constants. All configuration options are read from `.env` or system environment variables.

### 2. `backend/core/`
- **Responsibility**: System orchestration (`SystemOrchestrator`), domain models (`Intent`, `CommandResult`), and custom domain exception hierarchy.
- **Rules**: Core contains zero business logic or external API coupling. It relies strictly on dependency injection to coordinate services, voice lifecycle, and tools.

### 3. `backend/ai/`
- **Responsibility**: AI provider abstractions (`BaseLLMProvider`) and concrete provider integrations.
- **Rules**: Never couple core code to a specific LLM vendor (e.g. OpenAI, Anthropic, Ollama). Use strategy patterns to swap providers seamlessly.

### 4. `backend/services/`
- **Responsibility**: Application services bridging core orchestration with business operations (e.g., `LLMService`, `SpeechService`, `SystemService`, `WeatherService`, `NewsService`).
- **Rules**: Services implement business capabilities and orchestrate low-level providers or external APIs.

### 5. `backend/tools/`
- **Responsibility**: Actionable tools callable by the assistant (System operations, Web browsing, Weather, News, Screenshots, Search).
- **Rules**: All tools implement `BaseTool` and register dynamically with `ToolRegistry`.

### 6. `backend/voice/`
- **Responsibility**: Voice interaction lifecycles, speech-to-text (STT), text-to-speech (TTS), and wake-word detection interfaces.
- **Rules**: Voice code is strictly isolated from core reasoning. Voice produces text inputs for core and consumes text outputs.

### 7. `backend/memory/`
- **Responsibility**: Abstractions for conversational and long-term memory (`BaseMemoryStore`).
- **Rules**: Only interfaces exist in V1.0. Storage implementations (vector DBs, graph memory) will be introduced in V2+.

### 8. `backend/api/`
- **Responsibility**: REST / WebSocket API gateway placeholder for V2+.
- **Rules**: Empty directory placeholder in V1.0 per architectural roadmap.

### 9. `backend/utils/`
- **Responsibility**: Cross-platform OS helpers, subprocess runners, and string sanitizers.
