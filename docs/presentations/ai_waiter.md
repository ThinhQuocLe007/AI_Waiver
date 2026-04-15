# AI Waiter — Pipeline Evolution: From Monolithic Chain to Agentic Graph

> **Presenter:** Le Quoc Thinh — 22134013
> **Topic:** Evolution of an AI Waiter System

---

## 1. The "True" Old Architecture (Project Legacy)

This is the system as originally built—a robust but complex 4-layer monolithic pipeline.

### 1.1 Old Architecture Diagram

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'background': '#191919', 'mainBkg': '#1e1e1e', 'lineColor': '#6b7280', 'edgeLabelBackground': '#191919', 'clusterBkg': '#1a1a1a', 'clusterBorder': '#374151', 'titleColor': '#f9fafb', 'nodeTextColor': '#f9fafb', 'primaryTextColor': '#f9fafb'}}}%%
flowchart LR
    %% Styles
    classDef input fill:#0d47a1,stroke:#64b5f6,stroke-width:2px,color:#fff
    classDef process fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff
    classDef db fill:#006064,stroke:#4dd0e1,stroke-width:2px,color:#fff
    classDef tool fill:#bf360c,stroke:#ffcc80,stroke-width:2px,stroke-dasharray: 5 5,color:#fff
    classDef output fill:#1b5e20,stroke:#81c784,stroke-width:2px,color:#fff
    classDef ui fill:#4a148c,stroke:#ce93d8,stroke-width:2px,color:#fff

    %% 1. Input & Perception Layer
    subgraph Layer1 ["1. Perception Layer (Input)"]
        In(["Microphone"]) --> VAD["Silero VAD"]
        VAD -->|"Voice Detected"| STT["Pho-Whisper STT"]
        Touch(["iPad Touch Input"])
    end

    %% 2. Orchestration Layer
    subgraph Layer2 ["2. Orchestration Layer (Brain)"]
        STT -->|"Vietnamese Text"| Orchestrator{"LLM Orchestrator<br>+ System Prompt"}
        Touch -->|"UI Events / Item Selection"| Orchestrator
        Orchestrator <-->|"Loads/Saves via table_id"| ChatHist[("Shared Chat History DB")]
    end

    %% 3. Action & Tool Layer
    subgraph Layer3 ["3. Action Layer (Tools)"]
        Orchestrator -->|"JSON Tool Call"| Dispatcher{"Tool Dispatcher"}
        
        Dispatcher --> T_Search["Tool: search"]
        Dispatcher --> T_Order["Tool: place_order"]
        Dispatcher --> T_QR["Tool: qr_payment"]
        
        T_Search <-->|"Query"| RAG[("Vector DB")]
        T_Search -->|"Context Return"| Orchestrator
    end

    %% 4. Output & UI Layer
    subgraph Layer4 ["4. Output Layer (UI & Audio)"]
        Orchestrator -->|"Text Response"| TTS["TTS Engine"]
        TTS --> Out(["Robot Speaker"])
        
        Orchestrator -->|"State & Subtitles"| WS(("WebSocket Server"))
        T_Order -.->|"Order Status Updates"| WS
        T_QR -.->|"QR Code Payload"| WS
        
        WS <--> Monitor["iPad Monitor Display"]
    end

    %% Database & Hardware Exits
    T_Order == "DB Connection" ==> DB[("Restaurant DB")]
    Orchestrator -.->|"Semantic Intent"| Nav["ROS 2 Task Control"]

    %% Assign Classes
    class In,VAD,Touch input;
    class STT,Orchestrator,TTS process;
    class RAG,ChatHist,DB db;
    class Dispatcher,T_Search,T_Order,T_QR tool;
    class Out,Nav output;
    class WS,Monitor ui;
```

---

## 2. Advanced Analysis: Why This "Advanced" System Failed

Even though this system looks professional, it suffered from **"Monolithic Thinking."**

### 2.1 The "Manual Context Hot-Potato" 🥔
In your old system, you had to manually `"Load/Save via table_id"` from a `Shared Chat History DB`.
- **The Stupidity**: The Orchestrator had to wait for a database query to even *remember* who it was talking to before processing.
- **LangGraph Fix**: `Checkpointers` are built-in. History is not a separate DB call you manage; it's the **State** of the graph itself, automatically managed by `thread_id`.

### 2.2 The "Rigid Dispatcher" ⚙️
You had a separate `Tool Dispatcher` node that interpreted JSON calls.
- **The Stupidity**: If the LLM made a small typo in the JSON, the Dispatcher would crash the whole action layer. The LLM couldn't "see" the crash to fix it.
- **LangGraph Fix**: The **Tools Node** is part of the graph loop. If a tool fails, the error goes **back into the Agent's message history**. The Agent reads the error and retries or corrects its JSON automatically.

### 2.3 Subgraph Synchronization Nightmares 😵‍💫
Look at `Layer 4`. You had `T_Order` and `T_QR` sending updates directly to a `WebSocket Server` while the `Orchestrator` sent subtitles.
- **The Stupidity**: **De-synchronization**. The iPad might show "Order Complete" (via tool) before the Robot speaks "Your order is ready" (via orchestrator). The UI and Audio could easily get out of sync.
- **LangGraph Fix**: The **State is the Single Source of Truth**. Everything (Audio, UI state, Order status) is written to the `AgentState` first, and then emitted as a single, synchronized update.

### 2.4 Semantic vs. Task Conflict 🧠
You had one `LLM Orchestrator` trying to handle:
1. Low-level Hardware Tasks (ROS 2 Nav)
2. High-level Social Chat (TTS)
3. UI State (WS Server)
- **The Stupidity**: **Instruction Dilution**. Testing one part (e.g., social chat) might accidentally break how the robot navigates because they share the same massive System Prompt.
- **LangGraph Fix**: **Modular Nodes**. You can have a `navigation_node` and a `chat_node` with separate, focused instructions, while still sharing the same state.

---

## 3. Comparison of the "Brain" Logic

| Dimension | Old Orchestrator (Monolith) | LangGraph (Agentic) |
| :--- | :--- | :--- |
| **History** | Manual fetch from DB via `table_id` | Automatic state recovery via `thread_id` |
| **Tool Errors** | Dispatcher crashes or returns code | Error goes to Agent's brain for "Self-Healing" |
| **Side Effects** | Multi-path (WS, ROS 2, TTS concurrently) | Unified state updates (Linear logic) |
| **Scaling** | Complex global state management | Isolated, parallel graph instances |
| **Hardware** | Hardcoded ROS 2 calls | Decoupled Navigation Tools |

---

### Key Presentation Takeaway:
> "My old system was a **Multi-Layer Monolith**. It looked organized on paper, but in production, the *side effects* and *manual session management* made it fragile. LangGraph allowed me to stop being a **State Manager** and start being a **Workflow Designer**."
