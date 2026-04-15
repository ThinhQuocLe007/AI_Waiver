# RAG Research: From Naive to Agentic RAG

> Based on analysis of `ai_waiter_core` pipeline — April 2026

---

## Part 1: The RAG Landscape (Big Picture)

RAG (Retrieval Augmented Generation) is a technique that gives an LLM access to external knowledge before generating an answer. Instead of relying only on its training data, the LLM retrieves relevant documents and uses them as context.

There are three main generations of RAG:

```
Naive RAG → Advanced RAG → Agentic RAG
  (1-pass)   (multi-step     (LLM steers
              pipeline)       the search)
```

---

## Part 2: Naive RAG

### How it works

```
User Query
    │
    ▼
[Retrieve Top-K Docs]
    │
    ▼
[Stuff into Prompt]
    │
    ▼
[LLM Generates Answer]
```

### Core Components
1. **Chunking**: Split raw data into smaller pieces.
2. **Embedding**: Convert chunks to vectors.
3. **Vector Store**: Store and retrieve vectors (FAISS, ChromaDB).
4. **LLM**: Generate answer from query + chunks.

### Limitations
- Single embedding model bias (misses exact keyword matches).
- No reranking — top-K may have low-relevance results.
- Fixed chunking (no awareness of document structure).
- No feedback loop (no way to know if results were good).

---

## Part 3: Analysis of Your Current Pipeline

You have already moved **well beyond Naive RAG**. Let's map your system:

```
search_menu(query)                  ← Tool exposed to LangGraph agent
      │
      ▼
RetrieverManager.hybrid_search()
      │
      ├──► BM25Index.search()       ← Keyword search (rank_bm25, BM25Okapi)
      │         │
      │    bm25_raw scores
      │
      ├──► VectorStore.search()     ← Semantic search (FAISS + Vietnamese_Embedding)
      │         │
      │    vector_raw distances
      │         │
      │    normalize_vector_score() ← 1 / (1 + distance)
      │
      └──► _merge_scores()          ← Union by hash(page_content)
                │
           _rank_and_format()
                │
           normalize_bm25_batch()   ← Sigmoid normalization (mean-centered)
                │
           calculate_hybrid_score() ← 0.6 * BM25 + 0.4 * Vector
                │
           Filter by threshold      ← 0.3 default
                │
                ▼
           List[SearchResult]       ← Sorted by hybrid score
```

### ✅ What You Are Doing Well

| Feature | Status | Notes |
| :--- | :--- | :--- |
| Hybrid Search (BM25 + Vector) | ✅ Done | Good architectural decision |
| Vietnamese Embedding Model | ✅ Done | Domain-specific — crucial for Vietnamese food data |
| Score Normalization (Sigmoid) | ✅ Done | Statistically sound approach |
| Persistent Index (Disk) | ✅ Done | Fast startup, no rebuild needed |
| Rich Metadata | ✅ Done | `category`, `diet_type`, `price` stored |
| LangGraph Tool Integration | ✅ Done | Agent can call `search_menu` autonomously |

### ⚠️ Current Weaknesses to Fix

| Weakness | Where | Impact | Priority |
| :--- | :--- | :--- | :--- |
| **No Re-ranking** | After `hybrid_search()` | Top-K results may not be the most relevant | 🔴 High |
| **BM25 tokenizer is too simple** | `bm25.py:28` — just `.lower().split()` | Misses Vietnamese morphology, punctuation | 🔴 High |
| **Fixed hybrid weight (0.6/0.4)** | `scoring.py:33` | Not adaptive to query type (keyword vs semantic) | 🟠 Medium |
| **No query expansion/rewriting** | Agent `call_model` | "spicy noodle" won't match "mì cay" | 🟠 Medium |
| **Chunking is per-item** | `document_loader.py` | No sub-document awareness, no overlap | 🟡 Low |
| **No feedback / evaluation** | Entire pipeline | You don't know when search quality is bad | 🟡 Low |

---

## Part 4: Advanced RAG Improvements

### 4.1 Add a Re-Ranker (Highest Impact)

After BM25 + Vector retrieval, a **Cross-Encoder** re-ranks the top-K results by "deeply" reading the query and document together — much more accurate than score fusion.

```
Query + Top-10 Hybrid Results
          │
          ▼
    [Cross-Encoder]           ← e.g., ms-marco-MiniLM-L-6-v2
    Scores each (query, doc) pair
          │
          ▼
    Top-3 Re-ranked Results   ← Only use these in the prompt
```

**Why it helps your case:** Vietnamese food queries are short and ambiguous. A re-ranker can handle "Cho tôi món ngon" better by understanding full context.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [(query, doc.page_content) for doc in results]
scores = reranker.predict(pairs)
reranked = sorted(zip(docs, scores), key=lambda x: -x[1])
```

### 4.2 Query Rewriting / Expansion

Before searching, let the LLM **reformulate** the user's query to be clearer or generate multiple sub-queries.

```
User: "Cho tôi xem món chay ăn được không?"
          │
          ▼
   [LLM Query Rewriter]
          │
          ▼
Generated: "vegetarian dishes", "diet_type: chay", "no meat menu"
          │
          ▼
   [Search with all 3 queries] → Merge results → Re-rank
```

This is especially powerful because your documents are in Vietnamese but a user might mix English/Vietnamese.

### 4.3 Better BM25 Tokenization

Replace `.lower().split()` with proper tokenization for Vietnamese:

```python
# Option 1: underthesea (Vietnamese NLP library)
from underthesea import word_tokenize
tokens = word_tokenize(doc.page_content.lower(), format="text").split()

# Option 2: Keep your tokenizer but add synonym expansion
# "bít tết" → ["bít tết", "steak", "thịt bò"]
```

### 4.4 Metadata Filtering Before Search

Use the rich metadata you already store to **pre-filter** before embedding search:

```python
# Example: User says "I want vegetarian food"
# → Filter by diet_type == "chay" first
# → Then run hybrid search only on those docs

filtered_docs = [d for d in self._documents if d.metadata.get("diet_type") == "chay"]
# Then build a temporary BM25/FAISS only on filtered_docs
```

---

## Part 5: Agentic RAG

In Agentic RAG, the LLM is **in control of the retrieval strategy**. Instead of calling search once, the agent decides when, how, and how many times to search.

### 5.1 Corrective RAG (CRAG)

After retrieval, the agent evaluates whether the results are good enough. If not, it tries again with a different strategy.

```
Query → Retrieve → [Relevance Evaluator LLM]
                          │
                   Is it good enough?
                   /              \
                 YES               NO
                  │                │
           Generate Answer    [Retry with
                               rewritten query
                               or web search]
```

**Integration point in your code:** Add a new LangGraph node `evaluate_results` between `tools` and `agent`.

### 5.2 Self-RAG

The LLM decides **whether it even needs retrieval** for a given question.

```
User: "What time do you close?"
          │
          ▼
   [LLM: "Do I need to search?"] → YES → search_menu()
          │
   User: "How are you?"
          ▼
   [LLM: "Do I need to search?"] → NO → Answer directly
```

This is actually **partially implemented** in your system already through `tools_condition` in LangGraph!

### 5.3 Multi-Step Agentic RAG (Most Advanced)

The agent can break a complex query into multiple search steps:

```
User: "Tôi bị dị ứng đậu phộng, có món chay nào dưới 100k không?"

Step 1: search_menu("vegetarian dishes under 100k")
Step 2: search_menu("dishes with peanuts") → Blacklist
Step 3: Filter Step 1 results that are NOT in blacklist
Step 4: Answer user
```

This is native to LangGraph — the agent loops until it has enough confidence.

---

## Part 6: Latency Analysis — What Actually Slows You Down?

For a restaurant robot waiter, **latency is a UX requirement**, not an afterthought. A customer should get an answer in under 2 seconds.

### Your Current Baseline

| Step | Component | Estimated Time | Notes |
| :--- | :--- | :--- | :--- |
| Embed query | `Vietnamese_Embedding` (HuggingFace) | ~80–200ms | Runs on CPU, first call is slower |
| BM25 search | `BM25Index.search()` | ~1–5ms | Pure in-memory, very fast |
| FAISS search | `VectorStore.search()` | ~5–20ms | In-memory, scales well |
| Score fusion | `_merge_scores` + `_rank_and_format` | ~1ms | Python arithmetic |
| **Total retrieval** | **`hybrid_search()`** | **~90–230ms** | Dominated by embedding |
| LLM (Ollama local) | `call_model` via ChatOllama | ~1,000–4,000ms | The biggest bottleneck |

> [!IMPORTANT]
> Your LLM call is already 10–20x slower than your entire retrieval pipeline.
> Adding moderate improvements to retrieval (e.g., re-ranker) adds maybe +200ms on top of a 3s LLM response.
> **The real enemy is the LLM, not the retriever.**

---

### Latency Impact of Each Improvement

| Improvement | Added Latency | Notes | Verdict |
| :--- | :--- | :--- | :--- |
| **Vietnamese Tokenizer (`underthesea`)** | +5–15ms | Word segmentation is fast | ✅ No concern |
| **Metadata Pre-Filter** | **−30 to −80ms** | Reduces search space → actually FASTER | ✅ Free win |
| **Adaptive Weights** | ~0ms | Arithmetic only | ✅ Free |
| **Cross-Encoder Re-Ranker (CPU)** | +300–800ms | Runs N (query, doc) pairs through a model | ⚠️ Manageable |
| **Cross-Encoder Re-Ranker (GPU)** | +30–80ms | If you have a GPU available | ✅ Acceptable |
| **Query Rewriting (extra LLM call)** | +1,000–3,000ms | Full LLM roundtrip before search | 🔴 Costly |
| **CRAG (retry path)** | +2,000–5,000ms | Only on failure, but doubles your worst case | 🔴 Use carefully |
| **Multi-Step Agent (3 searches)** | +300–600ms | 3x retrieval, but retrieval is cheap | ⚠️ Acceptable |

---

### Latency Mitigation Strategies

#### Strategy 1: Parallelize BM25 + Vector + Re-ranker (Best for Re-Ranker)

Instead of: `BM25 → Vector → Re-rank` (sequential), run:

```python
import asyncio

async def hybrid_search_fast(query):
    # BM25 and Vector run in parallel
    bm25_task = asyncio.to_thread(bm25_engine.search, query, k=10)
    vector_task = asyncio.to_thread(vector_engine.search, query, k=10)
    bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

    merged = merge_scores(bm25_results, vector_results)

    # Re-rank only the top-5 (not all 20)
    reranked = reranker.predict([(query, doc.page_content) for doc in merged[:5]])
    return reranked
```

**Result:** Re-ranking 5 docs on CPU takes ~150ms, not 800ms.

#### Strategy 2: Cache Search Results

If two customers ask "có phở không?" within 60 seconds, don't search twice:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_search(query: str) -> tuple:
    return tuple(retriever.hybrid_search(query, k=3))
```

#### Strategy 3: Skip Query Rewriting for Simple Queries

Only rewrite complex or ambiguous queries — detect with a fast heuristic:

```python
def needs_rewrite(query: str) -> bool:
    # Simple heuristic: short + no Vietnamese food keywords
    if len(query.split()) > 6:
        return True
    return False

# Skip the extra LLM call for "Cho tôi phở" but rewrite "tôi muốn ăn gì đó ngon"
```

#### Strategy 4: Use a Smaller, Faster Re-Ranker

| Model | Speed (CPU, 5 docs) | Quality |
| :--- | :--- | :--- |
| `ms-marco-MiniLM-L-2-v2` | ~80ms | Good |
| `ms-marco-MiniLM-L-6-v2` | ~180ms | Better |
| `ms-marco-MiniLM-L-12-v2` | ~350ms | Best |

For a restaurant menu of ~50–100 items, `L-2-v2` is likely sufficient.

---

### Recommended Latency-Aware Pipeline

```
User Query
    │
    ▼ (~5ms)
[Metadata Pre-Filter]   ← Fast: filter by category/diet_type from query keywords
    │
    ▼ (~90–200ms, parallel)
[BM25] ─────────────────┐
[Vector Embed + Search] ─┴── [Merge Scores]
    │
    ▼ (~80–150ms, only top-5)
[Cross-Encoder Re-Ranker]
    │
    ▼
[Top-3 Results → LLM Context]
    │
    ▼ (~1–4s)
[LLM Response]

Total retrieval: ~200–400ms  (was ~100–200ms, +200ms for re-ranker)
Total e2e:       ~1.5–4.5s   (dominated by LLM, retrieval is rounding error)
```

> [!TIP]
> **The re-ranker is worth it!** It adds ~200ms on top of a 2–4 second LLM call. But it significantly improves the quality of context the LLM receives, which means the LLM actually generates a better answer.

---

## Part 7: Your RAG Evolution Roadmap

```
Current State                 Next Steps                 Future
──────────────                ──────────                 ──────
Hybrid Search                 Add Re-Ranker              Multi-Step Search
(BM25 + FAISS)          →     Better Tokenizer    →      Self-RAG Evaluation
Vietnamese Embedding          Query Rewriting             CRAG with fallback
LangGraph Tool                Metadata Filtering          Streaming results
```

### Recommended Order of Implementation (with Latency Risk)

| Step | Improvement | Latency Added | Quality Impact | Worth It? |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Vietnamese Tokenizer** (`underthesea`) | +10ms | 🔴 High | ✅ Yes |
| 2 | **Metadata Pre-Filter** | −80ms (faster!) | 🟠 Medium | ✅ Yes |
| 3 | **Cross-Encoder Re-Ranker (L-2)** | +80–150ms | 🔴 High | ✅ Yes |
| 4 | **Async Parallel Search** | −50ms | 🟡 Low | ✅ Free win |
| 5 | **Result Cache** | −200ms (cache hit) | 🟡 Low | ✅ Yes |
| 6 | **Query Rewriting** (conditional) | +500–1000ms | 🟠 Medium | ⚠️ Only for complex queries |
| 7 | **CRAG Evaluator Node** | +2s (retry path) | 🟠 Medium | ⚠️ Only for critical accuracy |

---

## Part 7: Resources

### 📄 Papers (Must Read)

| Paper | What it covers |
| :--- | :--- |
| **[RAG Survey (2024)](https://arxiv.org/abs/2312.10997)** | Full overview of all RAG techniques — start here |
| **[Self-RAG (2023)](https://arxiv.org/abs/2310.11511)** | LLM decides when to retrieve and critiques itself |
| **[CRAG (2024)](https://arxiv.org/abs/2401.15884)** | Corrective RAG — retry with web search if results are bad |
| **[HyDE (2022)](https://arxiv.org/abs/2212.10496)** | Hypothetical Document Embedding — generate a fake answer to search for real docs |
| **[ColBERT (2020)](https://arxiv.org/abs/2004.12832)** | Token-level late interaction — more powerful than bi-encoder |

### 📖 Docs & Guides

| Resource | Link | Notes |
| :--- | :--- | :--- |
| **LangChain RAG Docs** | https://python.langchain.com/docs/concepts/rag/ | Official guide |
| **LangGraph Multi-Agent** | https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-network/ | Advanced agentic patterns |
| **Pinecone RAG Guide** | https://www.pinecone.io/learn/retrieval-augmented-generation/ | Visual explanations |
| **Sentence Transformers (Reranker)** | https://www.sbert.net/docs/cross_encoder/pretrained_models.html | Cross-encoder models |
| **underthesea (Vietnamese NLP)** | https://github.com/undertheseanlp/underthesea | Vietnamese tokenizer |
| **RAGAS (Evaluation)** | https://github.com/explodinggradients/ragas | Framework to evaluate your RAG quality |
| **Advanced RAG 12 techniques** | https://towardsdatascience.com/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6 | Illustrated overview |

### 🎥 Videos

| Resource | Notes |
| :--- | :--- |
| **[LangChain RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)** | Series of short, practical videos |
| **[Agentic RAG with LangGraph](https://www.youtube.com/watch?v=rl4VS2fS0-g)** | Direct application to LangGraph |
