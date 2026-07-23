# JARVIS Product Evolution Roadmap (Version 1.0 - Version 6.0)

## Version 1.0 — Core Foundation & System Control (Current Scope)
- **Goal**: Establish Clean Architecture, SOLID interfaces, Pydantic settings, logging, dependency injection, and core system tools.
- **Capabilities**:
  - Wake word & Voice interfaces (STT/TTS stubs)
  - AI conversation abstraction
  - System control tools: Open applications, Open websites, Weather, News, Time, Search, Screenshot, Shutdown, Restart, Lock PC.
- **Dependencies**: `pydantic`, `pydantic-settings`, `python-dotenv`, `pytest`.

## Version 2.0 — Web API & Desktop GUI Integration
- **Goal**: Expose backend capabilities over REST/WebSocket and launch desktop UI.
- **Capabilities**:
  - FastAPI server in `backend/api/` with real-time WebSocket streaming.
  - Desktop frontend (Electron / PySide / React) connected via `backend/api/`.
  - Production STT (Faster-Whisper) & TTS (Piper/Kokoro) engines.

## Version 3.0 — Personal Memory & Vision Capabilities
- **Goal**: Introduce stateful contextual awareness and computer vision.
- **Capabilities**:
  - Vector database integration in `backend/memory/` (Chroma / Qdrant) for RAG & context recall.
  - Screen capture analysis & camera vision using Multimodal LLMs.

## Version 4.0 — Autonomous Coding & Deep Research
- **Goal**: Multi-step agentic execution for code generation and web research.
- **Capabilities**:
  - Code execution sandbox and terminal tools.
  - Autonomous deep web research agents with document parsing.

## Version 5.0 — Multi-Agent Architecture & Local LLM Offloading
- **Goal**: Distributed intelligence and offline privacy-first execution.
- **Capabilities**:
  - Specialist agents (Coder, System Admin, Researcher) managed by core orchestrator.
  - Local LLM inference integration (Ollama / vLLM / llama.cpp).

## Version 6.0 — IoT & Hardware Control (ESP32 & Mobile)
- **Goal**: Ubiquitous ambient intelligence across home devices and mobile.
- **Capabilities**:
  - ESP32 micro-controller integration via MQTT/Home Assistant.
  - Mobile client application (iOS / Android) with push notifications.
