# Open-Source LLMs & llama.cpp
> A guide to the open-source LLM ecosystem — models, tools, and how they all fit together.

---

## 1. The Big Picture

```
Open-Source Model (weights)     →  Inference Engine        →  Your App
─────────────────────────────────────────────────────────────────────────
Llama 3 (Meta)                  →  llama.cpp               →  CLI / C++ app
Mistral (Mistral AI)            →  Ollama (wraps llama.cpp) →  REST API / CLI
Gemma (Google)                  →  vLLM                    →  High-throughput server
Qwen (Alibaba)                  →  text-generation-webui   →  Browser UI
Phi (Microsoft)                 →  LM Studio               →  Desktop GUI
```

**Key insight:** The **model** (weights) and the **engine** (software that runs it) are separate things.

---

## 2. What is llama.cpp?

[llama.cpp](https://github.com/ggerganov/llama.cpp) is a **C/C++ inference engine** for running LLMs. Written by Georgi Gerganov.

### Why it matters
- **Pure C/C++** — no Python dependencies, no PyTorch
- **Runs on CPU** — works even without a GPU
- **GPU acceleration** — CUDA (NVIDIA), ROCm (AMD), Metal (Apple)
- **Quantization** — converts huge models into small, fast versions (GGUF format)
- **Foundation** — Ollama, LM Studio, and many other tools are built on top of it

### llama.cpp vs Ollama

| Feature | llama.cpp | Ollama |
|---|---|---|
| What is it | Low-level C++ inference engine | User-friendly wrapper around llama.cpp |
| Interface | CLI + C API | CLI + REST API |
| Setup difficulty | Compile from source | One-line install |
| Model format | GGUF files | Auto-downloads from registry |
| Customization | Full control over everything | Simplified via Modelfile |
| Use case | Power users, custom builds, research | Developers, quick prototyping |

> **For your AI Waiter project:** Use **Ollama** (easy). You only need llama.cpp directly if you want maximum performance tuning or want to embed an LLM into a C/C++ application.

---

## 3. The GGUF Format

GGUF (GPT-Generated Unified Format) is the **standard file format** for quantized LLM weights used by llama.cpp.

```
Original Model (PyTorch)  →  Convert  →  model.gguf  →  llama.cpp / Ollama
   ~16 GB (fp16)                           ~4 GB (q4)
```

### Where to find GGUF models
- **Hugging Face** — search for "GGUF" → [huggingface.co/models?search=gguf](https://huggingface.co/models?search=gguf)
- **TheBloke** — popular quantizer: [huggingface.co/TheBloke](https://huggingface.co/TheBloke)
- **bartowski** — another well-known quantizer

### Quantization levels (same as in Ollama)
| Quant | Size (7B model) | Quality | Speed |
|---|---|---|---|
| Q2_K | ~2.5 GB | Low | Fastest |
| Q4_0 | ~3.5 GB | Good | Fast |
| Q4_K_M | ~4.0 GB | Better | Fast |
| Q5_K_M | ~4.5 GB | Great | Moderate |
| Q8_0 | ~7.0 GB | Near perfect | Slower |
| F16 | ~14 GB | Perfect | Slowest |

> **Sweet spot:** `Q4_K_M` — best balance of quality and speed for most use cases.

---

## 4. Open-Source Models — The Landscape

### Tier 1: Frontier Models (Best Quality)

| Model | Creator | Sizes | Strengths |
|---|---|---|---|
| **Llama 3 / 3.1** | Meta | 8B, 70B, 405B | Best overall open model, strong reasoning |
| **Mistral / Mixtral** | Mistral AI | 7B, 8x7B, 8x22B | Fast, great for code and chat |
| **Qwen 2.5** | Alibaba | 0.5B–72B | Excellent multilingual, **best for Vietnamese/Asian** |
| **Gemma 2** | Google | 2B, 9B, 27B | Efficient, good for edge deployment |
| **DeepSeek V3** | DeepSeek | 671B (MoE) | Open MoE, very strong reasoning |
| **Command R+** | Cohere | 104B | Built for RAG and tool use |

### Tier 2: Small & Efficient (Good for Edge/IoT)

| Model | Creator | Sizes | Strengths |
|---|---|---|---|
| **Phi-3 / Phi-4** | Microsoft | 3.8B, 14B | Surprisingly good for their size |
| **SmolLM 2** | Hugging Face | 135M–1.7B | Tiny models for embedded use |
| **TinyLlama** | Community | 1.1B | Chat-capable at 1B params |

### Understanding Model Names
```
llama-3.1-8b-instruct-q4_K_M.gguf
│         │  │         │
│         │  │         └─ Quantization level
│         │  └─ Fine-tuned for chat/instructions
│         └─ 8 billion parameters
└─ Model family and version
```

- **Base** model = raw pretrained, completes text
- **Instruct** model = fine-tuned to follow instructions (what you want for AI Waiter)
- **Chat** model = fine-tuned for multi-turn conversation

---

## 5. Key Concepts

### Parameters (7B, 13B, 70B...)
- More parameters = smarter but **slower and larger**
- 7B–8B models are the practical sweet spot for local use with consumer GPUs

### Context Window (num_ctx)
- How many tokens the model can "see" at once
- Llama 3: 8,192 tokens (extendable to 128K with RoPE scaling)
- More context = more VRAM needed

### Fine-Tuning vs RAG vs Prompt Engineering

| Approach | What it does | When to use |
|---|---|---|
| **Prompt Engineering** | Craft better instructions | Always start here |
| **RAG** | Inject external data into prompt | When LLM needs your data (menu, docs) |
| **Fine-Tuning (LoRA)** | Retrain the model on your data | When behavior/style needs fundamental change |

> **For AI Waiter:** Use **RAG** for menu search + **Prompt Engineering** for conversation style. Fine-tuning is overkill for now.

### Mixture of Experts (MoE)
- Models like Mixtral 8x7B have multiple "expert" sub-networks
- Only a subset activates per token → fast despite large total size
- Example: Mixtral 8x7B has 46B total params but runs like a 13B model

---

## 6. Inference Engines — All the Tools

| Tool | Type | Best For |
|---|---|---|
| **llama.cpp** | C++ engine | Maximum control, custom builds |
| **Ollama** | CLI + API wrapper | Easy local dev (what you use) |
| **LM Studio** | Desktop GUI | Non-technical users, visual interface |
| **vLLM** | Python server | High-throughput production serving |
| **text-generation-inference (TGI)** | Docker server | Hugging Face ecosystem, production |
| **text-generation-webui** | Browser UI | Experimenting with many models |
| **LocalAI** | API server | OpenAI-compatible drop-in replacement |
| **Jan** | Desktop app | Privacy-focused alternative to LM Studio |

### When to use which

```
Just learning / prototyping  →  Ollama ← (you are here)
Need a GUI                   →  LM Studio
Production API server        →  vLLM or TGI
Embedded / C++ integration   →  llama.cpp directly
Research / experimentation   →  text-generation-webui
```

---

## 7. Where to Find Models

| Source | URL | What's there |
|---|---|---|
| **Hugging Face** | huggingface.co | The GitHub of ML models — everything is here |
| **Ollama Library** | ollama.com/library | Pre-packaged models ready to `ollama pull` |
| **LMSYS Chatbot Arena** | chat.lmsys.org | Compare models head-to-head with blind voting |
| **Open LLM Leaderboard** | huggingface.co/spaces/open-llm-leaderboard | Benchmark rankings |

---

## 8. The Open-Source LLM Stack

Here's how all the pieces fit together for a project like AI Waiter:

```
┌─────────────────────────────────────────────────────────┐
│  YOUR APPLICATION (AI Waiter)                           │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ LangChain    │  │ RAG Pipeline │  │ Tool Calling │  │
│  │ (framework)  │  │ (Chroma DB)  │  │ (Order API)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └────────────┬─────┘────────────────┘           │
│                      ▼                                  │
│         ┌──────────────────────┐                        │
│         │  Ollama (localhost)  │                        │
│         │  REST API :11434    │                        │
│         └──────────┬───────────┘                        │
│                    ▼                                    │
│         ┌──────────────────────┐                        │
│         │  llama.cpp (engine) │                        │
│         └──────────┬───────────┘                        │
│                    ▼                                    │
│         ┌──────────────────────┐                        │
│         │  Llama 3 8B (GGUF) │                        │
│         │  GPU: NVIDIA CUDA  │                        │
│         └──────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Recommended Learning Path

```
1. ✅  Ollama basics (you're here)
2. →   LangChain + Ollama integration
3. →   RAG with ChromaDB
4. →   Tool Calling / Agents
5. →   LangGraph for routing
6. →   (Optional) llama.cpp for advanced tuning
7. →   (Optional) Fine-tuning with LoRA
```

---

## 10. Recommended Models for AI Waiter

| Model | Size | Why |
|---|---|---|
| **llama3:8b** | 4.7 GB | Best overall reasoning, strong tool calling |
| **qwen2.5:7b** | 4.7 GB | Better Vietnamese/multilingual understanding |
| **nomic-embed-text** | 274 MB | Embeddings for RAG menu search |
| **phi3:mini** | 2.3 GB | Lightweight alternative if GPU VRAM is limited |

Start with `llama3:8b`. If Vietnamese responses need improvement, try `qwen2.5:7b`.
