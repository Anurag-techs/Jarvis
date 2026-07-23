# Architecture Decision Records (ADR)

## Decision 001: Adoption of Clean Architecture
* **Status**: Accepted
* **Context**: JARVIS is planned as a multi-year project expanding from simple system commands to complex multi-agent execution, voice pipelines, vision, and memory systems.
* **Decision**: Enforce Clean Architecture boundaries separating Core Orchestration from Infrastructure (Services, Tools, Voice, AI Providers).
* **Consequences**:
  - Core domain logic has zero external dependencies.
  - Third-party APIs (OpenAI, Weather APIs, STT engines) can be swapped without touching core orchestrator logic.
  - Slightly higher upfront boilerplating for interfaces, but drastically higher long-term testability and maintainability.

---

## Decision 002: Strict Dependency Injection in System Orchestrator
* **Status**: Accepted
* **Context**: Concrete implementations of voice engines, LLMs, and memory stores change rapidly in the AI ecosystem.
* **Decision**: Pass abstract interfaces (`BaseLLMProvider`, `BaseVoiceEngine`, `BaseMemoryStore`, `ToolRegistry`) into `SystemOrchestrator` via constructor injection.
* **Consequences**:
  - Eliminates tight coupling to specific vendor libraries.
  - Allows instant unit testing with lightweight mock objects.
  - Enables dynamic runtime provider switching via configuration.

---

## Decision 003: Dynamic Tool Registry Pattern
* **Status**: Accepted
* **Context**: JARVIS requires expandable tool capabilities (System, Browser, Apps, Weather, Search) that will grow to hundreds of tools in future releases.
* **Decision**: Implement a pluggable `ToolRegistry` managing instances of `BaseTool`.
* **Consequences**:
  - New tools can be added simply by subclassing `BaseTool` and calling `registry.register()`.
  - Core orchestrator interacts uniformly with all tools using standard input/output models (`ToolResult`).
  - Prevents monolithic `if/elif/else` command routing trees in core code.
