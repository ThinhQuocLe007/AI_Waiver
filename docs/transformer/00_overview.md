# Transformer — Module 00: Overview & Motivation

> **Paper:** "Attention Is All You Need" — Vaswani et al., 2017
> **Link:** https://arxiv.org/abs/1706.03762
>
> This document covers the **"Why"** — the problems the Transformer was built to solve,
> the big picture architecture, and how data flows from input to output.

---

## 1. The Problem: What Was Wrong Before?

Before the Transformer (2017), the best models for language tasks were **RNNs** (Recurrent Neural Networks) and their improved variant **LSTMs** (Long Short-Term Memory).

### How RNNs Work (and Why They Fail)

An RNN processes a sentence **one word at a time, left to right**. Each word updates a **hidden state** that carries memory forward.

```mermaid
graph LR
    W1["Word 1 'The'"] --> H1["Hidden State h₁"]
    H1 --> H2["Hidden State h₂"]
    W2["Word 2 'cat'"] --> H2
    H2 --> H3["Hidden State h₃"]
    W3["Word 3 'sat'"] --> H3
    H3 --> H4["Hidden State h₄"]
    W4["Word 4 'on'"] --> H4
    H4 --> OUT["Output"]

    style H1 fill:#2d3748,stroke:#4299e1,color:#e2e8f0
    style H2 fill:#2d3748,stroke:#4299e1,color:#e2e8f0
    style H3 fill:#2d3748,stroke:#4299e1,color:#e2e8f0
    style H4 fill:#2d3748,stroke:#4299e1,color:#e2e8f0
```

**Critical Problems with this approach:**

| Problem | Explanation |
| :--- | :--- |
| **Sequential bottleneck** | Word 4 cannot be processed until Word 3 is done. Cannot parallelize on GPU. Slow training. |
| **Vanishing gradients** | In long sentences (100+ words), gradients shrink to nearly zero as they backpropagate. The model "forgets" early words. |
| **Long-range dependency** | "The cat that sat on the mat **is** happy" — the verb "is" refers to "cat", but they are far apart. RNNs lose this connection. |
| **Fixed-size bottleneck** | The entire sentence is compressed into a single hidden state vector. This is lossy — information is destroyed. |

### The Seq2Seq Bottleneck (Translation Example)

For machine translation, RNNs used an **Encoder-Decoder** design with a single "context vector":

```mermaid
graph LR
    subgraph Encoder
        E1["'Xin'"] --> EH1["h₁"]
        E2["'chào'"] --> EH2["h₂"]
        E3["'thế'"] --> EH3["h₃"]
        E4["'giới'"] --> EH4["h₄"]
    end

    EH4 -->|"Single context vector (bottleneck!)"| C["Context Vector"]

    subgraph Decoder
        C --> DH1["h₁"]
        DH1 --> O1["'Hello'"]
        DH1 --> DH2["h₂"]
        DH2 --> O2["'world'"]
    end

    style C fill:#c53030,stroke:#fc8181,color:#fff
```

**The bottleneck problem:** The entire input sentence is compressed into ONE vector. For long sentences, this single vector cannot remember everything. Quality collapses.

> This was solved by **Attention Mechanism** (Bahdanau et al., 2015), which let the decoder look back at **all** encoder hidden states. The Transformer takes this idea and makes attention **the only mechanism** — removing the RNN entirely.

---

## 2. The Core Idea: "Attention Is All You Need"

The Transformer's core insight:

> **"You don't need recurrence or convolution. Just attention, applied repeatedly."**

Instead of processing left-to-right and passing a hidden state, the Transformer lets **every word look at every other word directly** — all at once, in parallel.

```mermaid
graph TB
    subgraph Old["❌ Old Way: RNN Sequential"]
        direction LR
        r1["Word 1"] --> r2["Word 2"] --> r3["Word 3"] --> r4["Word 4"]
    end

    subgraph New["✅ New Way: Transformer Parallel"]
        direction TB
        t1["Word 1"] & t2["Word 2"] & t3["Word 3"] & t4["Word 4"]
        t1 <-->|attention| t2
        t1 <-->|attention| t3
        t1 <-->|attention| t4
        t2 <-->|attention| t3
        t2 <-->|attention| t4
        t3 <-->|attention| t4
    end

    style Old fill:#2d1b1b,stroke:#fc8181,color:#e2e8f0
    style New fill:#1b2d1b,stroke:#68d391,color:#e2e8f0
```

**Key benefits:**
- ✅ **Fully parallelizable** — all words processed simultaneously → 10–100x faster training on GPU
- ✅ **Constant path length** — Word 1 and Word 100 are always directly connected (O(1) vs O(N) in RNN)
- ✅ **No information bottleneck** — every decoder step can look at every encoder position directly

---

## 3. The Big Picture: Transformer Architecture

The Transformer has two halves — an **Encoder** and a **Decoder**, each made of stacked layers.

```mermaid
graph TB
    subgraph INPUT["📥 Input Side"]
        SRC["Source Sentence: 'Xin chào thế giới'"]
        SRC --> IE["Input Embedding + Positional Encoding"]
    end

    subgraph ENC["🔵 ENCODER — N=6 identical layers"]
        direction TB
        IE --> EL1["Encoder Layer 1: Self-Attention → FFN"]
        EL1 --> EL2["Encoder Layer 2: Self-Attention → FFN"]
        EL2 --> ELN["Encoder Layer N: Self-Attention → FFN"]
        ELN --> MEM["Encoder Memory — rich representation of source"]
    end

    subgraph OUTPUT["📤 Output Side"]
        TGT["Target Sentence shifted right: 'Hello world'"]
        TGT --> OE["Output Embedding + Positional Encoding"]
    end

    subgraph DEC["🟢 DECODER — N=6 identical layers"]
        direction TB
        OE --> DL1["Decoder Layer 1: Masked Self-Attn → Cross-Attn → FFN"]
        MEM -->|"cross-attention"| DL1
        DL1 --> DL2["Decoder Layer 2: Masked Self-Attn → Cross-Attn → FFN"]
        MEM --> DL2
        DL2 --> DLN["Decoder Layer N: Masked Self-Attn → Cross-Attn → FFN"]
        MEM --> DLN
    end

    subgraph PRED["🎯 Prediction"]
        DLN --> LIN["Linear Layer"]
        LIN --> SMX["Softmax"]
        SMX --> PROB["Probability over vocabulary → 'Hello', 'world', ..."]
    end

    style INPUT fill:#1a202c,stroke:#4299e1,color:#e2e8f0
    style ENC fill:#1a2744,stroke:#4299e1,color:#e2e8f0
    style OUTPUT fill:#1a202c,stroke:#68d391,color:#e2e8f0
    style DEC fill:#1a2d1a,stroke:#68d391,color:#e2e8f0
    style PRED fill:#2d2a1a,stroke:#f6ad55,color:#e2e8f0
```

---

## 4. The Building Blocks (Module Map)

Every Encoder and Decoder layer is built from the same set of sub-modules:

```mermaid
graph LR
    M1["📦 Module 01 — Input Embedding + Positional Encoding"]
    M2["📦 Module 02 — Scaled Dot-Product Attention"]
    M3["📦 Module 03 — Multi-Head Attention"]
    M4["📦 Module 04 — Feed-Forward + Add & Norm"]
    M5["📦 Module 05 — Full Encoder"]
    M6["📦 Module 06 — Full Decoder"]
    M7["📦 Module 07 — Full Transformer + Example"]

    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> M5
    M4 --> M6
    M5 --> M7
    M6 --> M7

    style M1 fill:#2c3e50,stroke:#3498db,color:#ecf0f1
    style M2 fill:#2c3e50,stroke:#e74c3c,color:#ecf0f1
    style M3 fill:#2c3e50,stroke:#e74c3c,color:#ecf0f1
    style M4 fill:#2c3e50,stroke:#2ecc71,color:#ecf0f1
    style M5 fill:#2c3e50,stroke:#3498db,color:#ecf0f1
    style M6 fill:#2c3e50,stroke:#2ecc71,color:#ecf0f1
    style M7 fill:#2c3e50,stroke:#f39c12,color:#ecf0f1
```

---

## 5. How Data Flows — End to End

Let's trace the sentence **"Xin chào"** → **"Hello"** through the full model:

```mermaid
sequenceDiagram
    participant SRC as Source Tokens
    participant EMB as Embedding + PosEnc
    participant ENC as Encoder x6
    participant MEM as Encoder Memory
    participant TGT as Target Tokens
    participant DEC as Decoder x6
    participant OUT as Output Prediction

    SRC->>EMB: ["Xin", "chào"] → token IDs
    EMB->>ENC: vectors (seq_len × d_model=512)
    Note over ENC: Each layer: Self-Attention → FFN
    ENC->>MEM: rich contextual representations

    TGT->>DEC: [start] → token IDs + PosEnc
    MEM->>DEC: encoder output (cross-attention)
    Note over DEC: Layer 1: Masked Self-Attn → Cross-Attn → FFN
    DEC->>OUT: logits → softmax → "Hello" (argmax)

    OUT->>DEC: feed "Hello" back as next input
    DEC->>OUT: logits → softmax → "world"
    OUT->>DEC: feed "world" back
    DEC->>OUT: logits → softmax → end token → stop
```

---

## 6. Key Hyperparameters from the Paper

The paper defines a "base" and "big" model. Here are the base model settings you will see throughout the code:

| Hyperparameter | Symbol | Base Model Value | What It Means |
| :--- | :--- | :--- | :--- |
| Model dimension | `d_model` | **512** | Size of every vector throughout the model |
| Number of heads | `h` | **8** | How many parallel attention perspectives |
| Head dimension | `d_k = d_v` | **64** | `d_model / h = 512 / 8` |
| FFN inner dim | `d_ff` | **2048** | Size of the hidden layer in feed-forward |
| Encoder layers | `N` | **6** | How many encoder layers stacked |
| Decoder layers | `N` | **6** | How many decoder layers stacked |
| Dropout | `p_drop` | **0.1** | Regularization rate |

> [!NOTE]
> These numbers are used consistently in all subsequent module documents. When you see `d_model=512` in code, this is where it comes from.

---

## 7. Why the Transformer Dominated Everything

The paper's results (2017) were dramatic:

- Achieved **28.4 BLEU** on English→German translation — better than all previous models
- Trained in **3.5 days on 8 GPUs** — much cheaper than LSTM alternatives
- **Generalized to other tasks** (parsing, summarization) with no architectural changes

More importantly, the architecture became the foundation for everything that followed:

| Model | Year | Based On |
| :--- | :--- | :--- |
| BERT | 2018 | Transformer **Encoder** only |
| GPT-1/2/3/4 | 2018–2023 | Transformer **Decoder** only |
| T5, BART | 2019–2020 | Full Encoder-Decoder Transformer |
| LLaMA, Mistral | 2023+ | Decoder-only with improvements |

---

## 8. What's Next

You now have the full picture. Each subsequent document dives into **one building block** at a time — concept first, then code:

| Next Document | Topic |
| :--- | :--- |
| `01_input_embedding.md` | Token Embedding + Positional Encoding |
| `02_attention.md` | Scaled Dot-Product Attention |
| `03_multihead.md` | Multi-Head Attention |
| `04_ffn_norm.md` | Feed-Forward + Residual + LayerNorm |
| `05_encoder.md` | Full Encoder Block |
| `06_decoder.md` | Full Decoder Block |
| `07_full.md` | Complete Transformer + Working Example |
