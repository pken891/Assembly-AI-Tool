# Portable Hybrid AI Gateway

This software is a lightweight, portable AI routing engine designed to run entirely off a USB drive. It acts as a dynamic middleware layer that intercepts user prompts, calculates token size and estimated execution costs, and routes the workload to the most appropriate model. It operates as a hybrid system: when offline, it directs prompts to local, quantized `llamafile` models (like Qwen 7B or 30B) based on available system RAM; when online, it unlocks web search capabilities and routes complex tasks to cloud APIs (Gemini, Claude, GPT) using securely stored keys.

## Development Roadmap

- [ ] **Phase 1: Local Execution & Routing**
  - **Feature:** Implement `gateway.py` to route prompts to existing local endpoints configured confi.json to detect and respond to users inputted models on their local machine, as well as connect there online models
  - **Significance:** Establishes the foundational offline pipeline, ensuring the core chat and prompt execution loop works entirely locally without external dependencies.

- [ ] **Phase 2: Token Evaluation & Cost Approximation**
  - **Feature:** Integrate `tiktoken` to analyze input prompt size prior to execution. For local models, it displays token counts to prevent context window overflow; for cloud models, it maps input tokens to a pricing dictionary to display estimated costs.
  - **Significance:** Provides exact resource visibility before a task executes, protecting hardware from crashing due to oversized prompts and preventing unexpected API billing charges.

- [ ] **Phase 3: Cloud Fallback & Hybrid Connectivity**
  - **Feature:** Build a network connectivity toggle that detects internet access. When online, the system loads `.env` variables via `python-dotenv` and enables `litellm` routing to external APIs, while also unlocking the existing DuckDuckGo search agent for live web context.
  - **Significance:** Transforms the gateway from a strictly local tool into a highly capable hybrid router that can fall back to frontier cloud models and pull real-time data when a connection is present.

- [ ] **Phase 4: Unified Streamlit Dashboard**
  - **Feature:** Wrap the backend Python logic into a local browser-based Streamlit web application. The UI will bundle the model selection dropdowns, real-time token/cost metrics, and API key management into a single dashboard launched directly from the master batch script.
  - **Significance:** Eliminates the need for terminal-based navigation, providing a clean, accessible graphical interface that unifies all diagnostic data and chat interactions in one window.
- [ ] **Phase 5: Wrap all into one application that can be downloaded/used on local machine instead of USB**
  - **Feature:** 
  - **Significance:**
