# 🔍 RAG (Retrieval Augmented Generation) — Complete Tutorial

> A stage-by-stage guide to understanding and building a production-grade RAG search system.
> From first principles to Agentic RAG.

---

## How to Use This Tutorial

Each **Stage** has three sections:
1. **📚 Concept** — What is it, why does it exist, how does it work (no code yet).
2. **💻 Code** — A clean, minimal implementation to see the concept in action.
3. **📖 Resources** — Papers, docs, and videos to go deeper.

Read the stages in order. Each one solves a problem introduced by the previous one.

---

## Stage 1: Naive RAG

### 📚 Concept

**The core problem RAG solves:**
LLMs have a knowledge cutoff and cannot access private data (like your restaurant menu). RAG solves this by fetching relevant information at query time and giving it to the LLM as context.

**How does it work?**

The pipeline has two phases:

**Indexing Phase** (done once, offline):
```
Raw Documents
      │
  [Chunking]         Split a large PDF into small pieces (~200-500 tokens)
      │
  [Embedding]        Convert each chunk into a vector (list of numbers)
      │
  [Vector Store]     Save all vectors to a searchable database (FAISS, ChromaDB)
```

**Query Phase** (done at runtime, every request):
```
User Question
      │
  [Embed Question]   Convert question to the same vector space
      │
  [Similarity Search] Find the K most similar document chunks
      │
  [Build Prompt]     "Here are some facts: {docs}. Answer this: {question}"
      │
  [LLM]             Generates the final answer
```

**Why "naive"?**
It works, but it has a critical flaw: it only searches by **semantic similarity**. If the user asks for "Phở Bò", a semantic search might return any noodle dish that's conceptually similar (e.g., ramen, udon). It cannot prioritize **exact term matches**.

---

### 💻 Code

```python
# Minimal Naive RAG from scratch

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

# 1. Your knowledge base
documents = [
    Document(page_content="Phở Bò is a Vietnamese beef noodle soup. Price: 55,000 VND"),
    Document(page_content="Bánh Mì is a Vietnamese baguette sandwich. Price: 35,000 VND"),
    Document(page_content="Gỏi Cuốn are fresh spring rolls with shrimp. Price: 45,000 VND"),
]

# 2. Embed and store
embedding = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")
vector_store = FAISS.from_documents(documents, embedding)

# 3. Retrieve
def retrieve(query: str, k: int = 2):
    return vector_store.similarity_search(query, k=k)

# 4. Generate
llm = ChatOllama(model="llama3.1")

def rag_query(question: str) -> str:
    docs = retrieve(question)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm.invoke(prompt).content

# Test
print(rag_query("What noodle dishes do you have?"))
```

**What to observe:**
- Change the query to "Phở" and see what comes back. Does FAISS return the right doc?
- Add 20 more menu items. Does quality hold up?

---

### 📖 Resources

| Resource | Link | Why |
| :--- | :--- | :--- |
| RAG Survey (2024) | https://arxiv.org/abs/2312.10997 | Section 2: Naive RAG — read this first |
| LangChain RAG Quickstart | https://python.langchain.com/docs/concepts/rag/ | Official walkthrough |
| RAG From Scratch (Video) | https://youtu.be/sVcwVQRHIc8 | Visual walk of the pipeline |

---

## Stage 2: BM25 — Keyword-First Search

### 📚 Concept

**The problem with pure vector search:**
Embeddings capture *semantic meaning*, but they are weak at exact term matching.

|  | Vector Search | BM25 |
|:---|:---|:---|
| Query: "Phở Bò" | Returns "noodle soups in general" | Returns docs containing "Phở Bò" first |
| Query: "something warm and filling" | ✅ Returns soups, stews | ❌ No match (no exact words) |

BM25 is the gold standard **keyword search** algorithm. It is used inside Elasticsearch, Solr, and Lucene.

**How BM25 scores a document:**

```
Score(doc, query) = Σ IDF(t) × f(t, doc) × (k1 + 1) / (f(t,doc) + k1 × (1-b+b×|doc|/avgdl))
```

In plain English:
- **IDF** (Inverse Document Frequency): Rare words score higher than common words.
  - "the", "is", "a" → very low IDF (everyone has them)
  - "Phở" → high IDF (only menu docs have it)
- **TF saturation** (`k1`): The 10th occurrence of a word adds much less score than the 1st.
- **Length normalization** (`b`): A short doc mentioning "Phở" once scores higher than a long doc mentioning "Phở" once.

**Tokenization is critical:**
BM25 works on **tokens** (words). The quality of your tokenizer directly affects quality.
- English: `.split()` is often fine.
- Vietnamese: `.split()` is **wrong**. "bít tết" is ONE compound word, not "bít" + "tết".

---

### 💻 Code

```python
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# --- With simple tokenizer (English-style, wrong for Vietnamese) ---
documents = [
    Document(page_content="Phở Bò is a Vietnamese beef noodle soup"),
    Document(page_content="Bánh Mì is a crispy baguette sandwich"),
    Document(page_content="Bún Bò Huế is a spicy beef vermicelli soup"),
]

# Tokenize
tokenized_docs = [doc.page_content.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Search
query = "beef noodle"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)

for doc, score in zip(documents, scores):
    print(f"Score: {score:.3f} | {doc.page_content[:50]}")

# Expected: "Phở Bò" and "Bún Bò Huế" should score higher (both have "beef")
```

```python
# --- With Vietnamese tokenizer (correct approach) ---
from underthesea import word_tokenize

def vn_tokenize(text: str) -> list[str]:
    # word_tokenize joins compound words with underscore: "bít_tết"
    return word_tokenize(text.lower(), format="text").split()

tokenized_docs = [vn_tokenize(doc.page_content) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
```

**What to observe:**
- Search "bít tết" with the naive tokenizer. Does it miss "Bít Tết Wagyu"?
- Install `underthesea` and repeat. Does it improve?

---

### 📖 Resources

| Resource | Link | Why |
| :--- | :--- | :--- |
| BM25 Algorithm Deep-Dive | https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables | Best explanation with graphs |
| `rank_bm25` library | https://github.com/dorianbrown/rank_bm25 | The library used in production code |
| underthesea (Vietnamese NLP) | https://github.com/undertheseanlp/underthesea | Vietnamese tokenizer |
| Robertson BM25 Paper (1994) | https://dl.acm.org/doi/10.5555/188490.188561 | The original paper (optional) |

---

## Stage 3: Hybrid Search & Score Fusion

### 📚 Concept

**The idea:** Run BM25 and Vector search in parallel, then combine their scores to get the best of both worlds.

```
User Query
    │
    ├──► [BM25 Search]      → [(doc1, 12.4), (doc3, 8.1), (doc5, 3.2)]
    │
    └──► [Vector Search]    → [(doc2, 0.91), (doc1, 0.85), (doc4, 0.72)]
                │
           [Score Fusion]   → Merge, normalize, and combine
                │
           [(doc1, 0.88), (doc2, 0.76), (doc3, 0.61)]  ← Final ranked list
```

**The normalization problem:**
You cannot directly add BM25 scores and vector scores — they live in completely different scales.
- BM25 score: can be 0, 5, 50, 500 (unbounded)
- FAISS distance: 0.01, 0.5, 2.0 (lower = closer)

You must normalize both to the same scale `[0, 1]` first.

**Common normalization methods:**

| Method | Formula | Pros | Cons |
| :--- | :--- | :--- | :--- |
| Min-Max | `(x - min) / (max - min)` | Simple | Destroyed by outliers |
| Sigmoid | `1 / (1 + e^(-x))` | Smooth, bounded | Needs tuning of center/scale |
| `1/(1+x)` | For distances | Intuitive | Non-linear, compresses far distances |

**Fusion strategies:**

| Strategy | Formula | Notes |
| :--- | :--- | :--- |
| Weighted Sum | `w1 * bm25_n + w2 * vec_n` | Simple, need to tune weights |
| Reciprocal Rank Fusion (RRF) | `Σ 1 / (k + rank_i)` | No tuning needed, rank-based |

**RRF** is often preferred because it doesn't require knowing the score scale — it only uses the **rank** (position) of each document in each result list.

---

### 💻 Code

```python
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="Phở Bò: Vietnamese beef noodle soup, price 55k"),
    Document(page_content="Bánh Mì: crispy baguette sandwich, price 35k"),
    Document(page_content="Gỏi Cuốn: fresh spring rolls with shrimp, price 45k"),
    Document(page_content="Bún Bò Huế: spicy beef vermicelli, price 60k"),
]

embedding = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")

# Build both indexes
bm25 = BM25Okapi([doc.page_content.lower().split() for doc in documents])
faiss_store = FAISS.from_documents(documents, embedding)

def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """Combine two ranked lists using RRF. k=60 is the standard default."""
    scores = {}

    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = hash(doc.page_content)
        scores[doc_id] = scores.get(doc_id, {"doc": doc, "score": 0})
        scores[doc_id]["score"] += 1 / (k + rank + 1)

    for rank, (doc, _) in enumerate(vector_results):
        doc_id = hash(doc.page_content)
        scores[doc_id] = scores.get(doc_id, {"doc": doc, "score": 0})
        scores[doc_id]["score"] += 1 / (k + rank + 1)

    return sorted(scores.values(), key=lambda x: -x["score"])

def hybrid_search(query: str, k: int = 3):
    # BM25 results
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_results = sorted(zip(documents, bm25_scores), key=lambda x: -x[1])

    # Vector results
    vector_results = faiss_store.similarity_search_with_score(query, k=len(documents))

    # Fuse
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    return fused[:k]

# Test
results = hybrid_search("spicy beef noodle")
for r in results:
    print(f"Score: {r['score']:.4f} | {r['doc'].page_content[:60]}")
```

**What to observe:**
- Search "beef" — does BM25 or vector perform better alone?
- Search "something warm and filling" — which engine helps more?
- Compare RRF vs weighted sum. Do results differ?

---

### 📖 Resources

| Resource | Link | Why |
| :--- | :--- | :--- |
| Hybrid Search Explained | https://www.pinecone.io/learn/hybrid-search-intro/ | Best visual explanation |
| RRF Paper (2009) | https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf | Original RRF algorithm |
| Weaviate: BM25 vs Vector vs Hybrid | https://weaviate.io/blog/hybrid-search-explained | Real benchmarks |

---

## Stage 4: Re-Ranking with Cross-Encoders

### 📚 Concept

**The fundamental limitation of all previous stages:**
BM25 scores each document independently. Vector search compares pre-computed vectors. Both approaches compute relevance **without ever looking at the query and document together**.

This is called a **Bi-Encoder** architecture:
```
[Query]    → [Model] → query_vector
[Document] → [Model] → doc_vector
                              ↓
               cosine_similarity(query_vector, doc_vector) = score
```

**The Cross-Encoder difference:**
A Cross-Encoder feeds **query + document together** into a single model pass.
The model can attend to both simultaneously — much deeper understanding.

```
[Query + Document] → [Transformer] → relevance_score  (0 to 1)
```

**The tradeoff:**

| | Bi-Encoder | Cross-Encoder |
| :--- | :--- | :--- |
| Speed | ✅ Fast (pre-compute docs) | ❌ Slow (run at query time) |
| Accuracy | ⚠️ Approximate | ✅ High |
| Scalable to millions of docs? | ✅ Yes | ❌ No |
| Best use | Stage 1: Get candidates | Stage 2: Re-rank top-K |

**Two-stage retrieval (the solution):**
```
Step 1: Hybrid Search → Top 20 candidates   [cheap, fast]
Step 2: Cross-Encoder → Re-rank those 20    [expensive, accurate]
Step 3: Take Top 3 → Feed to LLM
```
You only run the expensive model on 20 docs, not the entire database.

---

### 💻 Code

```python
from sentence_transformers import CrossEncoder

# Load once at startup (model downloads first time, ~100MB)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
# L-2: fastest (~80ms for 5 docs on CPU)
# L-6: balanced (~180ms) 
# L-12: most accurate (~350ms)

def rerank(query: str, documents: list, top_k: int = 3) -> list:
    """
    Re-rank a list of documents using a Cross-Encoder.
    
    Args:
        query:     The user's search query
        documents: List of Document objects (your hybrid_search results)
        top_k:     How many to return after re-ranking
    
    Returns:
        Top-K documents, re-ordered by cross-encoder score
    """
    # Create (query, doc_text) pairs
    pairs = [(query, doc.page_content) for doc in documents]
    
    # Score all pairs at once (batch is faster than one-by-one)
    scores = reranker.predict(pairs)
    
    # Sort by cross-encoder score (descending)
    doc_score_pairs = sorted(zip(documents, scores), key=lambda x: -x[1])
    
    # Return top-K documents only
    return [doc for doc, score in doc_score_pairs[:top_k]]


# Example usage — plug into your existing pipeline
def full_pipeline(query: str) -> list:
    # Stage 1: Hybrid retrieval (fast, gets 20 candidates)
    candidates = hybrid_search(query, k=20)
    candidate_docs = [r["doc"] for r in candidates]

    # Stage 2: Re-rank (accurate, runs on those 20 only)
    final_results = rerank(query, candidate_docs, top_k=3)
    
    return final_results
```

**What to observe:**
- Take the output of `hybrid_search` before and after `rerank`. Does the order change?
- Try an ambiguous query like "good food". Does the re-ranker make better choices?

---

### 📖 Resources

| Resource | Link | Why |
| :--- | :--- | :--- |
| Bi-Encoder vs Cross-Encoder | https://www.sbert.net/examples/applications/retrieve_rerank/README.html | Best visual comparison |
| Cohere Re-rank Guide | https://docs.cohere.com/docs/reranking | Production perspective |
| Cross-Encoder Pretrained Models | https://www.sbert.net/docs/cross_encoder/pretrained_models.html | Model comparison table |
| **[Paper] ColBERT (2020)** | https://arxiv.org/abs/2004.12832 | Faster alternative (late interaction) |
| **[Paper] MonoT5 (2020)** | https://arxiv.org/abs/2003.06713 | T5-based re-ranker |

---

## Stage 5: Query Transformation

### 📚 Concept

**The problem:** What the user types is often not the best search query.

| User Query | The Real Intent | Problem |
| :--- | :--- | :--- |
| "I'm hungry" | Find food | Too vague for any engine |
| "I don't want meat" | Find vegetarian dishes | Negation is hard for embedding |
| "What's popular here?" | Find bestsellers/recommendations | No keyword to match |

**Three query transformation techniques:**

**1. Query Rewriting** — Use LLM to make the query more searchable:
```
Input:  "I don't feel like meat today"
Output: "vegetarian dishes, plant-based food, no meat menu options"
```

**2. Multi-Query** — Generate multiple search queries from one question:
```
Input:  "something light for lunch"
Output: ["light lunch dishes", "low calorie food", "salads and soups", "healthy options"]
→ Run all 4 searches, deduplicate, merge results
```

**3. HyDE** (Hypothetical Document Embeddings) — Generate a fake "perfect answer" and search for that:
```
Input:  "What's romantic for a date night?"
[LLM generates]: "For a romantic dinner, we recommend our candlelit pasta, 
                  premium wagyu steak, and wine-pairing seafood..."
[Search with that hypothetical text instead of the raw question]
→ Returns: premium, romantic menu items
```

**Why HyDE works:**
A hypothetical document lives in the same vector space as real documents.
It's often a better "bridge" to real content than the short user question.

---

### 💻 Code

```python
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.1", temperature=0)

# --- Technique 1: Query Rewriting ---
def rewrite_query(query: str) -> str:
    prompt = f"""You are a search query optimizer for a Vietnamese restaurant.
Rewrite the following user input into a better search query.
Return ONLY the improved query, nothing else.

User input: {query}
Improved query:"""
    return llm.invoke(prompt).content.strip()

# --- Technique 2: Multi-Query ---
def generate_multi_queries(query: str, n: int = 3) -> list[str]:
    prompt = f"""Generate {n} different search queries for a restaurant menu search.
Each query should approach the topic from a different angle.
Return only the queries, one per line.

Original question: {query}"""
    response = llm.invoke(prompt).content
    return [line.strip() for line in response.split("\n") if line.strip()]

# --- Technique 3: HyDE ---
def hyde_query(query: str) -> str:
    """Generate a hypothetical menu item description to search with."""
    prompt = f"""Write a short restaurant menu item description that would 
perfectly answer this question. Be specific about ingredients and style.

Question: {query}
Hypothetical menu item description:"""
    return llm.invoke(prompt).content.strip()

# Usage
original = "I want something healthy and not too heavy"
print("Rewritten:", rewrite_query(original))
print("Multi:", generate_multi_queries(original))
print("HyDE doc:", hyde_query(original))
```

**What to observe:**
- How does HyDE change what gets retrieved for vague queries?
- Multi-query adds latency (N×LLM calls). Is the quality gain worth it?

---

### 📖 Resources

| Resource | Link | Why |
| :--- | :--- | :--- |
| **[Paper] HyDE (2022)** | https://arxiv.org/abs/2212.10496 | Hypothetical Document Embeddings |
| LangChain Multi-Query Retriever | https://python.langchain.com/docs/how_to/MultiQueryRetriever/ | Ready-to-use implementation |
| RAG From Scratch: Query Translation | https://youtu.be/h0OPWlEOank | Video walkthrough of all techniques |

---

## Stage 6: Agentic RAG

### 📚 Concept

**The problem with all previous stages:**
Every approach is a **fixed pipeline** — one query goes in, results come out.
None of them can handle multi-step reasoning:

> "I'm allergic to peanuts and shellfish. I want a vegetarian dish under 80,000 VND.
>  What do you recommend, and can you also tell me if the kitchen can make it less spicy?"

This requires multiple searches, filtering, and conditional logic. A fixed pipeline cannot do this.

**Agentic RAG** gives the LLM **control** over the retrieval strategy.

**Three core patterns:**

**1. Self-RAG** — LLM decides if it even needs to search:
```
Query: "Hello, how are you?"  → LLM: "I don't need to search" → Direct answer
Query: "Do you have pho?"    → LLM: "I need to search"        → search() → answer
```

**2. Corrective RAG (CRAG)** — LLM evaluates its own search results:
```
Query → Search → [Evaluator LLM]: "Are these results relevant?"
                              │
                    YES ──────┤──────── NO
                              │            │
                         Answer          Retry with
                                         rewritten query
```

**3. Multi-Step Agent** — LLM chains multiple searches autonomously:
```
Complex Query
     │
Step 1: search("vegetarian dishes")       → [salad, tofu stir-fry, spring rolls]
Step 2: search("dishes containing nuts")  → [peanut sauce, walnut cake]
Step 3: Filter Step 1 minus Step 2        → [salad, spring rolls]
Step 4: Answer: "Here are safe options..."
```

---

### 💻 Code

```python
# Agentic RAG with LangGraph
from typing import Annotated, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define state
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# 2. Define tools (your retriever as a LangChain tool)
@tool
def search_menu(query: str) -> str:
    """Search the restaurant menu for dishes, prices, and information."""
    results = full_pipeline(query)  # Uses your Stage 4 pipeline
    if not results:
        return "No results found."
    return "\n".join([doc.page_content for doc in results])

# 3. Bind tools to LLM
llm = ChatOllama(model="llama3.1", temperature=0.1)
llm_with_tools = llm.bind_tools([search_menu])

# 4. Agent node
def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 5. Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode([search_menu]))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # Self-RAG: decide to search or not
graph.add_edge("tools", "agent")                       # Loop back after search

app = graph.compile()

# 6. Run
response = app.invoke({
    "messages": [{"role": "user", "content": "I'm allergic to peanuts. What vegetarian dishes do you recommend?"}]
})
print(response["messages"][-1].content)
```

**What to observe:**
- Ask a simple question ("hello"). Does the agent call `search_menu`?
- Ask a complex question. How many times does the agent loop?
- Add a second tool (e.g., `check_order_status`). Does the agent use the right tool?

---

### 📖 Resources

| Resource | Link | Why |
| :--- | :--- | :--- |
| **[Paper] Self-RAG (2023)** | https://arxiv.org/abs/2310.11511 | LLM critiques its own retrieval |
| **[Paper] CRAG (2024)** | https://arxiv.org/abs/2401.15884 | Corrective RAG with web fallback |
| LangGraph Agentic RAG Tutorial | https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/ | Full LangGraph implementation |
| Agentic RAG (Video) | https://www.youtube.com/watch?v=rl4VS2fS0-g | High-level overview |

---

## Stage 7: Evaluation — How Do You Know It's Working?

### 📚 Concept

**The problem:** How do you know if adding a re-ranker actually made results better?
Without evaluation, your improvements are guesswork.

**RAGAS** is the standard framework for evaluating RAG pipelines.
It measures four dimensions:

| Metric | Question it answers | Score |
| :--- | :--- | :--- |
| **Faithfulness** | Does the LLM answer only from the retrieved context? | 0–1 |
| **Answer Relevancy** | Is the answer actually relevant to the question? | 0–1 |
| **Context Recall** | Did we retrieve ALL the docs needed to answer? | 0–1 |
| **Context Precision** | Were the retrieved docs actually useful (no noise)? | 0–1 |

**The eval dataset:**
You need gold-standard Q&A pairs:
```json
{
  "question": "How much does Phở Bò cost?",
  "ground_truth": "Phở Bò costs 55,000 VND",
  "contexts": ["Phở Bò is a Vietnamese beef noodle soup. Price: 55,000 VND"]
}
```

---

### 💻 Code

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# Your test cases
eval_data = {
    "question": [
        "How much does Phở Bò cost?",
        "Do you have vegetarian options?",
    ],
    "answer": [
        "Phở Bò costs 55,000 VND.",   # Your RAG system's answer
        "Yes, we have Gỏi Cuốn for 45,000 VND.",
    ],
    "contexts": [
        ["Phở Bò is a Vietnamese beef noodle soup. Price: 55,000 VND"],
        ["Gỏi Cuốn: fresh spring rolls with shrimp. Price: 45,000 VND"],
    ],
    "ground_truth": [
        "55,000 VND",
        "Yes, Gỏi Cuốn is vegetarian.",
    ]
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)

print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, 'context_recall': 1.0, 'context_precision': 0.83}
```

**How to use this:**
1. Run baseline evaluation (Naive RAG): write down scores.
2. Add re-ranker. Run evaluation again.
3. Did all 4 metrics improve? If yes, keep the change.

---

### 📖 Resources

| Resource | Link | Why |
| :--- | :--- | :--- |
| RAGAS GitHub | https://github.com/explodinggradients/ragas | Installation and quickstart |
| RAGAS Docs | https://docs.ragas.io/ | All metrics explained |
| **[Paper] RAGAS (2023)** | https://arxiv.org/abs/2309.15217 | The original paper |

---

## Complete Resource Index

### Papers (in reading order)

| # | Paper | Stage | Link |
| :--- | :--- | :--- | :--- |
| 1 | RAG Survey (2024) | All stages | https://arxiv.org/abs/2312.10997 |
| 2 | HyDE (2022) | Query Transform | https://arxiv.org/abs/2212.10496 |
| 3 | ColBERT (2020) | Re-ranking | https://arxiv.org/abs/2004.12832 |
| 4 | Self-RAG (2023) | Agentic | https://arxiv.org/abs/2310.11511 |
| 5 | CRAG (2024) | Agentic | https://arxiv.org/abs/2401.15884 |
| 6 | RAGAS (2023) | Evaluation | https://arxiv.org/abs/2309.15217 |

### Videos (in order)

| # | Title | Link |
| :--- | :--- | :--- |
| 1 | RAG From Scratch (Full Series) | https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x |
| 2 | Agentic RAG with LangGraph | https://www.youtube.com/watch?v=rl4VS2fS0-g |

### Libraries

| Library | Purpose | Install |
| :--- | :--- | :--- |
| `rank_bm25` | BM25 search | `pip install rank_bm25` |
| `underthesea` | Vietnamese NLP | `pip install underthesea` |
| `sentence_transformers` | Cross-encoder re-ranker | `pip install sentence_transformers` |
| `faiss-cpu` | Vector store | `pip install faiss-cpu` |
| `ragas` | RAG evaluation | `pip install ragas` |
| `langgraph` | Agentic RAG | `pip install langgraph` |
