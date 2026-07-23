# JARVIS AI Assistant (Version 1.0 Foundation)

JARVIS is a production-quality, modular AI Assistant designed using **Clean Architecture**, **SOLID principles**, and **Dependency Injection** in **Python 3.12+**.

## 🌟 Architectural Features
- **Clean Architecture Separation**: Pure core domain orchestration decoupled from infrastructure, tools, AI providers, and voice hardware.
- **Dependency Injection**: Core components depend strictly on abstract interfaces (`BaseLLMProvider`, `BaseTool`, `BaseVoiceEngine`, `BaseMemoryStore`).
- **Pluggable Tool Registry**: Easily add new tools by subclassing `BaseTool` without altering core orchestrator logic.
- **Strict V1 Dependencies**: Core foundation relies strictly on `pydantic`, `pydantic-settings`, `python-dotenv`, and `pytest`.

---

## 📁 Directory Structure

```
jarvis/
├── backend/
│   ├── ai/          # Abstract BaseLLMProvider interface & Mock Provider
│   ├── api/         # Created API module directory (FastAPI routes reserved for V2.0)
│   ├── config/      # Pydantic Settings & centralized logging
│   ├── core/        # Orchestrator, domain exceptions, and data models
│   ├── services/    # Business services (LLM, Speech, System, Weather, News)
│   ├── tools/       # Pluggable Tool Registry & V1.0 tools (Apps, Browser, Weather, News, System, Search, Screenshot)
│   ├── voice/       # Voice lifecycle manager & STT/TTS abstract interfaces
│   ├── memory/      # Abstract BaseMemoryStore interface ONLY (no DB code in V1)
│   ├── utils/       # Cross-platform OS helpers & process runners
│   ├── tests/       # Pytest unit suite verifying config, registry, and orchestrator
│   └── main.py      # Entrypoint & CLI execution loop
│
├── docs/            # Architecture documentation, ADR decisions, and multi-version roadmap
├── frontend/        # Directory placeholder for future UI clients
├── .env.example     # Environment variable template
├── .gitignore       # Standard Python gitignore rules
├── README.md        # Project guide
└── requirements.txt # Version 1.0 production dependencies
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
Ensure Python 3.12+ is installed on your system.

```bash
# Copy example environment file
cp .env.example .env

# Install Version 1.0 production dependencies
pip install -r requirements.txt
```

### 2. Running Unit Tests
Execute the pytest suite to verify settings loading, tool registration, and orchestrator routing:

```bash
pytest backend/tests
```

### 3. Launching JARVIS
Run the interactive CLI application:

```bash
python -m backend.main
```

---

## 📚 Documentation & Roadmap
- **Architecture Guide**: [docs/architecture.md](docs/architecture.md)
- **ADR Decisions**: [docs/decisions.md](docs/decisions.md)
- **Version Evolution Roadmap**: [docs/roadmap.md](docs/roadmap.md)
