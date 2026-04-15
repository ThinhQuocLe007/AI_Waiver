# System Architecture: Old vs. New Pipeline

This document contains Mermaid diagrams for the AI Waiter architecture. You can copy the code below directly into [Mermaid Live Editor](https://mermaid.live/) or any Markdown viewer that supports Mermaid (like VS Code or GitHub) for your slides.

---

## 1. The "Old" Pipeline (Sequential & Rigid)

The old system followed a brittle, linear path. If any stage failed or the user asked something "outside the plan," the whole chain broke.

```mermaid
graph LR
    User([User Voice]) --> STT[STT: Phowhisper]
    STT --> Intent{Intent Classifier<br/>'if/else' logic}
    
    subgraph "Hardcoded Logic"
    Intent -- "Ordering" --> Order[Order Logic]
    Intent -- "Menu info" --> Search[Menu Lookup]
    Intent -- "Payment" --> Pay[Payment Logic]
    end
    
    Order --> TTS[TTS: Engine]
    Search --> TTS
    Pay --> TTS
    
    TTS --> Voice([Robot Voice])
    
    style User fill:#f9f,stroke:#333
    style Voice fill:#f9f,stroke:#333
    style Intent fill:#fff4dd,stroke:#d4a017
```

### Why it was "Stupid":
- **No Memory**: Every interaction was a fresh start.
- **Rigid flow**: Couldn't handle multi-step requests (e.g., "Check the price AND THEN order").
- **Fragile**: Adding a new feature required rewriting the central `if/else` logic.

---

## 2. The "New" Pipeline (LangGraph Agentic)

The new system is **cyclic** and **stateful**. The LLM acts as the "Brain" that decides which "Hands" (Tools) to use based on the current context stored in a persistent checkpoint.

```mermaid
graph TD
    %% Define Nodes
    START((START)) --> Agent[<b>Agent Node</b><br/>Brain: LLM llama3.1]
    
    %% Main Loop
    Agent -->|Decides| Tools[<b>Tools Node</b><br/>Hands: Python Actions]
    Tools -->|Result| Agent
    
    %% Tool specific paths
    subgraph Tools_Registry [Core Tools]
        direction LR
        S[search_menu]
        O[place_order]
        P[request_payment]
    end
    
    %% Persistence Layer
    Persistence[(SQLite Persistence<br/>MemorySaver)] -.->|Update State| Agent
    
    %% Exit
    Agent -->|Final Answer| END((END))
    
    %% Styling
    style Agent fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Tools fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Persistence fill:#f3e5f5,stroke:#4a148c
    style START fill:#c8e6c9,stroke:#2e7d32
    style END fill:#ffcdd2,stroke:#c62828
```

### Why it is "Smart":
- **Cyclic Reasoning**: The agent can call multiple tools in one "turn" (e.g., Search -> Calculate -> Confirm).
- **Persistent State**: SQLite acts as the robot's long-term memory. It remembers you across sessions.
- **Modular Tools**: Adding a new feature is as simple as adding a new tool function; the LLM automatically learns how to use it.

---

## 3. Comparative Summary

| Feature | Old Pipeline | LangGraph Agent |
| :--- | :--- | :--- |
| **Logic** | Hardcoded `if/else` | LLM Reasoning |
| **Flow** | Linear (One-way) | Cyclic (Looping) |
| **Memory** | None (Stateless) | SQLite (Persistent) |
| **Context** | Single-shot response | Multi-turn conversation |
| **Complexity** | Simple but fragile | Sophisticated & robust |
