# Native RAG — Deep Dive Tutorial

> A comprehensive tutorial to fully understand Naive RAG: the problem it solves,
> every component in depth, how to build it, and where it breaks down.
>
> **Papers referenced:**
> - Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401))
> - Gao et al. (2023) — "RAG for LLMs: A Survey" ([arXiv:2312.10997](https://arxiv.org/abs/2312.10997))

---

## Part 1: Why Does RAG Exist?

### The Problem with Pure LLMs

A Large Language Model (LLM) is trained on a massive dataset (the internet, books, Wikipedia, etc.) crawled up to a **cutoff date**. Its knowledge is frozen in its **model weights** — billions of parameters that encode statistical patterns from training data.

This creates fundamental limitations:

| Problem | Explanation | Example |
| :--- | :--- | :--- |
| **Knowledge Cutoff** | LLM doesn't know about events after training | "What is on our menu today?" |
| **Hallucination** | LLM confidently generates plausible but false info | "The Phở Bò costs 30k" (wrong price) |
| **Private Data Blind Spot** | LLM never saw your internal documents | "What are our restaurant opening hours?" |
| **No Source Traceability** | You can't verify where the LLM got an answer | Medical, legal, financial applications |

### The Two Types of LLM "Memory"

The original RAG paper (Lewis et al., 2020) introduced a key distinction:

> *"We introduce RAG models which combine **parametric memory** (the pre-trained LLM weights) and **non-parametric memory** (a dense vector index of documents)."*

```
Parametric Memory              Non-Parametric Memory
─────────────────              ─────────────────────
Stored in LLM weights          Stored in external database
Fixed after training           Updated anytime
Knows general world knowledge  Knows your specific data
Cannot be changed cheaply      Changed by editing documents
```

**RAG's core idea:** At query time, retrieve relevant documents from the non-parametric memory, and give them to the LLM as context. This grounds the LLM's answer in real, verifiable information.

---

## Part 2: The Naive RAG Pipeline (Full Detail)

*"Naive RAG" is the term used by Gao et al. (2023) to describe the original, simplest formulation of RAG.*

The pipeline has two completely separate phases:

```
═══════════════════════════════════════════
 PHASE 1: INDEXING  (offline, done once)
═══════════════════════════════════════════
 
 Raw Documents
      │
      ▼
 ┌─────────────┐
 │  Chunking   │  Break large docs into smaller pieces
 └─────────────┘
      │
      ▼
 ┌─────────────┐
 │  Embedding  │  Convert each chunk → vector of numbers
 └─────────────┘
      │
      ▼
 ┌──────────────┐
 │ Vector Store │  Save and index all vectors for fast search
 └──────────────┘


═══════════════════════════════════════════
 PHASE 2: RETRIEVAL + GENERATION  (online, every request)
═══════════════════════════════════════════

 User Query: "How much does Phở Bò cost?"
      │
      ▼
 ┌─────────────┐
 │  Embedding  │  Embed the query using the SAME model
 └─────────────┘
      │
      ▼
 ┌──────────────────┐
 │ Similarity Search│  Find top-K most similar chunks
 └──────────────────┘
      │
      ▼
 ┌──────────────┐
 │ Prompt Build │  "Here are some facts: {chunks}. Answer: {query}"
 └──────────────┘
      │
      ▼
 ┌─────┐
 │ LLM │  Generate the final answer
 └─────┘
      │
      ▼
 "Phở Bò costs 55,000 VND."
```

---

## Part 3: Component 1 — Chunking (Breaking Documents Apart)

### Why Chunking Exists

Embedding models have a **context window limit** — a maximum number of tokens (roughly words) they can process into a single vector. For example:
- `text-embedding-ada-002` (OpenAI): 8,191 tokens
- `AITeamVN/Vietnamese_Embedding`: ~512 tokens
- `sentence-transformers/all-MiniLM-L6-v2`: 256 tokens

A restaurant PDF with 50 menu items and policy text is far larger than 512 tokens. You cannot embed the entire document as a single vector. You must split it first.

But there is a deeper reason beyond the size limit:

> *"If our chunks are too small or too large, it may lead to imprecise search results or missed opportunities to surface relevant content."* — Pinecone Chunking Guide

This means chunking directly determines **retrieval precision**:
- **Too large**: The chunk contains too many topics. The embedding is "diluted" — it represents an average of many ideas.
- **Too small**: The chunk loses context. A sentence like "Yes, we can do that" is meaningless without surrounding conversation.

The **Goldilocks rule**: A chunk is good if a human can understand it *without reading what comes before or after it*.

---

### The Four Main Chunking Strategies

#### Strategy 1: Fixed-Size Chunking

Split every N tokens (or characters), regardless of content structure.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,      # Max characters per chunk
    chunk_overlap=50,    # Overlap to avoid cutting context at boundaries
    separator="\n"       # Prefer splitting at newlines
)

chunks = splitter.split_text(raw_text)
```

**Pros:** Simple, predictable, fast.  
**Cons:** Splits mid-sentence, mid-paragraph, mid-idea.

**When to use:** Your documents have no meaningful structure (e.g., plain text logs). Good as a starting point.

---

#### Strategy 2: Recursive Character Chunking (Most Common)

LangChain's `RecursiveCharacterTextSplitter` tries a hierarchy of separators: `["\n\n", "\n", " ", ""]`. It first tries to split on double newlines (paragraphs), then single newlines, then spaces.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_text(raw_text)
```

**Pros:** Respects natural text structure (paragraphs first, then sentences, then words).  
**Cons:** Still size-based, not meaning-based.

**When to use:** General purpose. This is the default for most RAG applications.

---

#### Strategy 3: Structure-Aware Chunking

When your documents have known structure (Markdown headers, JSON keys, HTML tags), use that structure to define chunk boundaries.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

# Each chunk now has metadata: {"Header 1": "Menu", "Header 2": "Main Courses"}
```

**Pros:** Chunks respect document semantics. Metadata is automatically rich.  
**Cons:** Requires structured documents.

**When to use:** Markdown docs, web pages, technical documentation.

---

#### Strategy 4: Semantic Chunking (Advanced Naive)

Split based on **meaning shifts** rather than size. Embed each sentence, then find boundaries where consecutive sentences become semantically dissimilar.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")
splitter = SemanticChunker(embedding, breakpoint_threshold_type="percentile")
chunks = splitter.split_text(raw_text)
```

**Pros:** Chunks are semantically coherent — topic-based, not size-based.  
**Cons:** Expensive (requires embedding every sentence) and slower.

**When to use:** Long documents with many different topics (books, research papers).

---

### Chunking Decision Guide

```
Is your document structured (Markdown, HTML, JSON)?
  YES → Strategy 3 (Structure-Aware)
  NO  ↓
Is topic consistency per-chunk critical?
  YES → Strategy 4 (Semantic Chunking)
  NO  ↓
Is this your first version?
  YES → Strategy 2 (Recursive, start with chunk_size=500, overlap=50)
  NO  → Benchmark different sizes for your data
```

---

### The Overlap Parameter

When you split text, the last words of chunk N and the first words of chunk N+1 overlap:

```
Chunk 1: "Phở Bò is a traditional Vietnamese soup. It is made with beef broth"
Chunk 2: "It is made with beef broth, rice noodles, and fresh herbs."
                   │████████████████████████│
                   These 8 words appear in BOTH chunks (overlap)
```

**Why?** Without overlap, a sentence about "beef broth being the secret to flavor" might be split, with "beef broth" in chunk 1 and "being the secret to flavor" in chunk 2. Neither chunk alone answers "What makes Phở Bò special?"

**Typical values:** 10–15% of chunk size. For chunk_size=500, use overlap=50.

---

## Part 4: Component 2 — Embeddings (Converting Text to Vectors)

### What is an Embedding?

An **embedding** is a vector (an ordered list of floating-point numbers) that represents the *meaning* of a piece of text. The key property:

> Texts with **similar meaning** produce **similar vectors** (close together in vector space).
> Texts with **different meaning** produce **dissimilar vectors** (far apart).

```
"Phở Bò is a beef noodle soup" → [0.23, -0.14, 0.87, 0.05, ..., 0.61]  (768 numbers)
"Bún Bò Huế is a spicy beef soup"  → [0.21, -0.12, 0.84, 0.08, ..., 0.59]  (close!)
"Stock market crashes 10%"          → [-0.45, 0.71, -0.23, 0.88, ..., -0.12]  (far!)
```

### How Embeddings are Created: The Bi-Encoder Architecture

Modern embedding models are **Transformer** neural networks (like BERT) fine-tuned with contrastive learning:

1. **Input:** A sentence or paragraph.
2. **Tokenization:** Split into subword tokens. "Phở Bò" → `["Ph", "##ở", "B", "##ò"]`
3. **Transformer encoding:** Each token attends to all other tokens. Rich contextual representations are built.
4. **Pooling:** The token representations are compressed into a single vector (usually mean pooling of all token vectors, or using the `[CLS]` token).

```
Input text: "Phở Bò tastes great"
     │
[Tokenizer]  → ["Phở", "Bò", "tastes", "great"]
     │
[Transformer]
  Token 1 attends to Tokens 1,2,3,4
  Token 2 attends to Tokens 1,2,3,4
  ...all tokens see all other tokens...
     │
[Mean Pooling]  → Average of all token vectors
     │
Output Vector: [0.23, -0.14, ..., 0.61]  (size = embedding_dim, e.g., 768)
```

### Choosing the Right Embedding Model

> [!IMPORTANT]
> **Domain matters enormously.** An English-trained model will produce poor vectors for Vietnamese text because it has never seen Vietnamese patterns during training. Always use a model matched to your language and domain.

| Model | Language | Dimensions | Use Case |
| :--- | :--- | :--- | :--- |
| `text-embedding-ada-002` (OpenAI) | English | 1536 | General English |
| `text-embedding-3-small` (OpenAI) | Multilingual | 512–1536 | Multilingual, API |
| `sentence-transformers/all-MiniLM-L6-v2` | English | 384 | Lightweight English |
| `AITeamVN/Vietnamese_Embedding` | Vietnamese | 768 | Vietnamese domain ✅ |
| `bge-m3` (BAAI) | Multilingual | 1024 | Strong multilingual |

### Cosine Similarity vs Dot Product vs Euclidean Distance

Once you have vectors for both query and documents, how do you measure similarity?

**Cosine Similarity** (most common):
```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: -1 to +1 (in practice, usually 0 to 1 for normalized vectors)
- Measures the **angle** between vectors — ignores magnitude (length)
- Best for comparing meaning regardless of text length

**Euclidean Distance** (used by FAISS L2):
```
distance(A, B) = √(Σ (Aᵢ - Bᵢ)²)
```
- Range: 0 to ∞ (lower = more similar)
- **Note:** FAISS returns distances, not similarities. That's why you need to convert: `similarity = 1 / (1 + distance)`

**Inner Product / Dot Product:**
```
dot(A, B) = Σ (Aᵢ × Bᵢ)
```
- When vectors are normalized (unit length), dot product = cosine similarity.

---

### Code: Understanding Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

embedding_model = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")

# Embed some texts
texts = [
    "Phở Bò là món ăn truyền thống của Việt Nam",     # Pho Bo is traditional Vietnamese
    "Bún Bò Huế là món bún cay của miền Trung",        # Another noodle soup
    "Thị trường chứng khoán giảm mạnh hôm nay",       # Stock market crashed today
]

vectors = embedding_model.embed_documents(texts)
vectors = np.array(vectors)

print(f"Vector shape: {vectors.shape}")  # (3, 768)

# Calculate cosine similarity between all pairs
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Phở Bò ↔ Bún Bò Huế: {cosine_sim(vectors[0], vectors[1]):.4f}")   # HIGH: both food
print(f"Phở Bò ↔ Stock market: {cosine_sim(vectors[0], vectors[2]):.4f}") # LOW: unrelated

# Embed a query
query_vector = embedding_model.embed_query("Phở bò giá bao nhiêu?")  # How much is pho?
print(f"Query ↔ Phở Bò: {cosine_sim(query_vector, vectors[0]):.4f}")  # HIGH: same topic
```

**What to observe:**
- Print the actual vector values — they are abstract but the numbers encode meaning.
- Try translating "Phở Bò" to English and re-embed. Does similarity drop with a Vietnamese model?

---

## Part 5: Component 3 — Vector Store (Indexing for Fast Search)

### The Scalability Problem

Imagine you have 10,000 menu items and restaurant documents. A user asks a question.

**Naive approach:** Compute cosine similarity between the query vector and ALL 10,000 document vectors.  
**Time complexity:** O(N × D) where N=documents, D=dimensions.  
For 10,000 documents × 768 dimensions = 7,680,000 multiplications **per query**. Still fast on modern hardware, but it becomes a real problem at 1M, 10M documents.

**Vector stores** solve this with **Approximate Nearest Neighbor (ANN)** search — trading a tiny bit of accuracy for 100x speed improvements.

### FAISS: How it Works Under the Hood

**FAISS** (Facebook AI Similarity Search) is the most popular in-memory vector library.

**The core index types:**

| Index Type | Speed | Accuracy | Memory | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `IndexFlatL2` | ⚠️ Slow (exact) | ✅ Perfect | High | Small collections (<100k) |
| `IndexIVFFlat` | ✅ Fast | ⚠️ Approx | Medium | Medium collections |
| `IndexHNSW` | ✅ Very fast | ✅ High | High | Production systems |

**IVF (Inverted File) — the intuition:**
1. During indexing: Cluster all vectors into K groups (like K-Means). Each vector belongs to its nearest cluster centroid.
2. During search: Find the closest cluster centroids to the query. Only search within those clusters (not the entire index).

```
10,000 documents grouped into 100 clusters
→ Only search ~100 documents per cluster × top-5 clusters = 500 comparisons
→ vs. 10,000 comparisons for exhaustive search
→ 20x faster, very slight accuracy loss
```

**HNSW (Hierarchical Navigable Small World) — the intuition:**
Builds a graph where nearby vectors are connected. Search traverses the graph like a GPS — always moving towards the target, stopping when no closer neighbor exists.

### Code: Working with FAISS

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")

# Your chunks (from Part 3)
documents = [
    Document(
        page_content="Phở Bò: Vietnamese beef noodle soup. Price: 55,000 VND. Rich broth, rice noodles.",
        metadata={"source": "menu.json", "category": "soup", "price": 55000}
    ),
    Document(
        page_content="Bánh Mì: Crispy French baguette with pork filling. Price: 35,000 VND.",
        metadata={"source": "menu.json", "category": "sandwich", "price": 35000}
    ),
    Document(
        page_content="Gỏi Cuốn: Fresh spring rolls with shrimp and herbs. Price: 45,000 VND.",
        metadata={"source": "menu.json", "category": "appetizer", "price": 45000}
    ),
]

# --- BUILD the index ---
vector_store = FAISS.from_documents(documents, embedding)

# --- SAVE to disk (so you don't rebuild every startup) ---
vector_store.save_local("./data/vector_db")

# --- LOAD from disk ---
vector_store = FAISS.load_local(
    "./data/vector_db",
    embedding,
    allow_dangerous_deserialization=True  # Required for local FAISS
)

# --- SEARCH: returns (Document, distance) pairs ---
results_with_scores = vector_store.similarity_search_with_score("beef noodle soup", k=2)

for doc, distance in results_with_scores:
    similarity = 1 / (1 + distance)  # Convert FAISS L2 distance to 0-1 similarity
    print(f"Similarity: {similarity:.3f} | {doc.page_content[:60]}")
    print(f"Metadata: {doc.metadata}\n")
```

---

## Part 6: Component 4 — Retrieval (Bringing It Together)

### The Retrieval Process in Detail

When a user submits a query:

1. **Embed the query** using the **same** embedding model used during indexing.

   > [!IMPORTANT]
   > You MUST use the same model for indexing and querying. If you index with `Vietnamese_Embedding` and query with `all-MiniLM-L6-v2`, your vectors live in different spaces — similarity scores become meaningless.

2. **Search the vector store** for the K nearest neighbours.

3. **Return documents** with their similarity scores.

### The K Parameter (How Many Documents to Retrieve?)

**Too few (K=1):** You might get the only relevant answer, but if the documents don't perfectly cover the topic, the LLM has nothing to work with.

**Too many (K=10+):** The LLM's context window fills up with marginally relevant documents. This leads to the **"lost-in-the-middle" problem**: information in the middle of a long context is ignored by the LLM.

**Research finding (Liu et al., 2023 — "Lost in the Middle"):** LLMs pay most attention to information at the **beginning and end** of the context. Documents placed in the middle of a long context tend to be ignored.

```
Context: [Doc5, Doc3, Doc1, Doc7, Doc2]
LLM pays most attention to: [Doc5          ...         Doc2]
LLM tends to ignore: [            Doc1, Doc7             ]
```

**Practical default:** K=3 to K=5. Start here and increase only if answers are incomplete.

---

## Part 7: Component 5 — Prompt Construction

### The Prompt is the Bridge

The retrieved documents are useless unless the LLM is told **how to use them**. This is the job of the prompt template.

### Three Prompt Patterns

**Pattern 1: Stuff (Most Common)**

Put all documents into a single prompt. Simple, but breaks if context is too long.

```python
def build_stuff_prompt(query: str, documents: list) -> str:
    context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    return f"""You are a helpful restaurant assistant. 
Use ONLY the following information to answer the question.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""
```

**Pattern 2: Map-Reduce (For Long Documents)**

Step 1 (Map): Ask the LLM to answer from each document separately.
Step 2 (Reduce): Combine the individual answers into a final answer.

```python
# Step 1: Map - summarize each document independently
individual_answers = []
for doc in documents:
    prompt = f"Based on this excerpt, answer: '{query}'\n\nExcerpt: {doc.page_content}\n\nAnswer:"
    answer = llm.invoke(prompt).content
    individual_answers.append(answer)

# Step 2: Reduce - combine all answers
combined = "\n".join(individual_answers)
final_prompt = f"Combine these answers into one final answer for: '{query}'\n\n{combined}"
final_answer = llm.invoke(final_prompt).content
```

**Pattern 3: Refine**

Iteratively refine the answer as you process each document:

```python
answer = "No information found."
for doc in documents:
    prompt = f"""Current answer: {answer}

New information from document:
{doc.page_content}

Refine the answer to the question '{query}' using any new information above.
If the new information is irrelevant, return the current answer unchanged."""
    answer = llm.invoke(prompt).content
```

---

### The Full Naive RAG Pipeline — Production Code

```python
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── 1. CONFIGURATION ──
VECTOR_DB_PATH = "./data/vector_db"
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
LLM_MODEL = "llama3.1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# ── 2. INITIALIZE MODELS ──
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ── 3. INDEXING PHASE (run once) ──
def build_index(raw_texts: list[str], metadatas: list[dict] = None):
    """
    Chunk, embed, and index a list of raw text strings.
    Saves the index to disk.
    """
    # Chunk each raw text
    all_chunks = []
    for i, text in enumerate(raw_texts):
        chunks = splitter.split_text(text)
        meta = metadatas[i] if metadatas else {}
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=meta))

    print(f"Created {len(all_chunks)} chunks from {len(raw_texts)} documents")

    # Embed and store
    vector_store = FAISS.from_documents(all_chunks, embedding)
    vector_store.save_local(VECTOR_DB_PATH)

    print(f"Index saved to {VECTOR_DB_PATH}")
    return vector_store


# ── 4. RETRIEVAL PHASE ──
def load_index():
    """Load the pre-built index from disk."""
    return FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)


def retrieve(vector_store, query: str, k: int = TOP_K) -> list[Document]:
    """Retrieve the top-K most relevant documents for a query."""
    results = vector_store.similarity_search_with_score(query, k=k)

    # Log scores for transparency
    for doc, distance in results:
        sim = 1 / (1 + distance)
        print(f"  [score={sim:.3f}] {doc.page_content[:60]}...")

    return [doc for doc, _ in results]


# ── 5. GENERATION PHASE ──
def build_prompt(query: str, documents: list[Document]) -> str:
    """Build the context-aware prompt."""
    context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    return f"""You are a helpful Vietnamese restaurant assistant.
Answer the question using ONLY the information provided below.
If the answer is not in the context, say: "Sorry, I don't have that information."

Context:
{context}

Question: {query}
Answer:"""


def generate(prompt: str) -> str:
    """Call the LLM with the built prompt."""
    return llm.invoke(prompt).content


# ── 6. FULL RAG QUERY ──
def rag_query(vector_store, question: str) -> str:
    """Full Naive RAG pipeline: retrieve → prompt → generate."""
    print(f"\nQuery: {question}")
    print("Retrieving documents...")
    docs = retrieve(vector_store, question)
    
    prompt = build_prompt(question, docs)
    
    print("Generating answer...")
    answer = generate(prompt)
    
    return answer


# ── 7. EXAMPLE USAGE ──
if __name__ == "__main__":
    # Build index from menu data (run once)
    menu_data = [
        "Phở Bò: Vietnamese beef noodle soup. Rich bone broth simmered for 12 hours. Price: 55,000 VND. Category: soup.",
        "Bánh Mì: Crispy French baguette with grilled pork, pickled vegetables, and chili. Price: 35,000 VND. Category: sandwich.",
        "Gỏi Cuốn: Fresh spring rolls with shrimp, pork, vermicelli, and herbs in rice paper. Price: 45,000 VND. Category: appetizer.",
        "Restaurant hours: Open daily 7:00 AM - 10:00 PM. Located at 123 Nguyen Hue Street, District 1, Ho Chi Minh City.",
    ]
    
    if not os.path.exists(VECTOR_DB_PATH):
        vs = build_index(menu_data)
    else:
        vs = load_index()
    
    # Test queries
    print(rag_query(vs, "How much does Phở Bò cost?"))
    print(rag_query(vs, "What time does the restaurant close?"))
    print(rag_query(vs, "What is the capital of France?"))  # Should say "I don't have that info"
```

---

## Part 8: The Failure Modes of Naive RAG

*This section is drawn from Gao et al. (2023) Section 2.1: "Limitations of Naive RAG".*

The 2023 survey paper formally identified three categories of failure in Naive RAG:

### Failure 1: Retrieval Quality

> *"The challenges in the retrieval phase include the difficulty of retrieving precise information when queries are vague or complex."* — Gao et al.

| Problem | Example | Effect |
| :--- | :--- | :--- |
| Low precision | Retrieve 5 docs but only 2 are relevant | LLM gets noise, produces confused answer |
| Low recall | The right document exists but isn't retrieved | LLM has no context, hallucinates |
| Semantic gap | Query and document use different words for same concept | Miss retrieval entirely |

**Example semantic gap:**
- User asks: "Do you have anything for vegetarians?"
- Your menu says: "Gỏi Cuốn: *chay* (plant-based), no meat." ← "Chay" ≠ "vegetarian" to the embedding model if not seen enough times together.

### Failure 2: Generation Quality

Even with perfect retrieval, the LLM can still fail:

- **Hallucination**: LLM invents details not in the retrieved context.
- **Irrelevance**: LLM ignores the provided context and answers from its training data.
- **Toxicity / Bias**: If retrieved docs contain biased information, LLM may amplify it.

### Failure 3: Augmentation Problems

- **Context window overflow**: Too many retrieved documents don't fit in the prompt.
- **"Lost in the middle"**: Key information buried in the middle of the context is ignored.
- **Redundancy**: Multiple retrieved chunks say the same thing — wastes context tokens.

---

## Part 9: Benchmarking Naive RAG Performance

Before moving to Advanced RAG, measure your baseline. Here are the key metrics:

### Metric 1: Retrieval Recall@K

*"Of all the documents that could answer this question, how many did we retrieve in top-K?"*

```python
# You need a labelled test set: question → list of relevant doc IDs
def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_top_k & relevant) / len(relevant)

# Example
retrieved = ["doc_1", "doc_5", "doc_3"]   # What we retrieved
relevant  = ["doc_1", "doc_3"]             # What was actually correct
print(recall_at_k(retrieved, relevant, k=3))  # → 2/2 = 1.0 (perfect)
```

### Metric 2: Retrieval Precision@K

*"Of the K documents we retrieved, how many were actually relevant?"*

```python
def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_top_k & relevant) / k

print(precision_at_k(retrieved, relevant, k=3))  # → 2/3 = 0.67
```

### Metric 3: Answer Faithfulness (with RAGAS)

*"Does the LLM's answer only contain information from the retrieved context?"*

```python
from ragas.metrics import faithfulness
# Score: 1.0 = every claim in the answer is grounded in context
# Score: 0.0 = the answer is completely hallucinated
```

---

## Part 10: When to Move Beyond Naive RAG

Ask yourself these questions about your current system:

| Question | If YES → consider... |
| :--- | :--- |
| Does search miss exact product/person names? | → Add BM25 (Stage 2) |
| Are retrieved docs ranked in the wrong order? | → Add Re-Ranker (Stage 4) |
| Are queries too vague for good retrieval? | → Add Query Rewriting (Stage 5) |
| Does the LLM need to "look up" multiple things? | → Add Agentic RAG (Stage 6) |
| Do you not know if performance is good? | → Add RAGAS evaluation (Stage 7) |

---

## Summary

```
NAIVE RAG
─────────
Problem:   LLMs don't know your private/recent data
Solution:  Retrieve relevant documents at query time and inject into prompt

Pipeline:
Chunk → Embed → Index     (Offline, once)
Query → Embed → Retrieve → Prompt → LLM     (Online, every request)

Key Decisions:
  Chunking:    Size, overlap, strategy (fixed vs semantic vs structure-aware)
  Embedding:   Must match your language domain
  Vector Store: FAISS (exact) or IVF/HNSW (approximate) for scale
  K:            3-5 is usually optimal (lost-in-the-middle at K>5)
  Prompt:      Stuffing, MapReduce, or Refine patterns

Key Failures:
  1. Retrieval: Semantic gap, low precision/recall
  2. Generation: Hallucination despite good context
  3. Augmentation: Context overflow, lost-in-the-middle
```

---

## Resources

| Type | Title | Link | What to Read |
| :--- | :--- | :--- | :--- |
| 📄 **Paper** | RAG (Lewis et al., 2020) | https://arxiv.org/abs/2005.11401 | The original — Sections 1, 2, 3 |
| 📄 **Paper** | RAG Survey (Gao et al., 2023) | https://arxiv.org/abs/2312.10997 | Section 2 (Naive RAG) and Section 3 (Retrieval) |
| 📄 **Paper** | Lost in the Middle (Liu et al., 2023) | https://arxiv.org/abs/2307.03172 | Why K matters and position bias |
| 📄 **Paper** | DPR (Karpukhin et al., 2020) | https://arxiv.org/abs/2004.04906 | How dense retrieval (bi-encoder) was developed |
| 📖 **Guide** | Chunking Strategies (Pinecone) | https://www.pinecone.io/learn/chunking-strategies/ | Full chunking comparison |
| 📖 **Guide** | What are Embeddings? (OpenAI) | https://platform.openai.com/docs/guides/embeddings | Embedding fundamentals |
| 📖 **Docs** | FAISS Documentation | https://github.com/facebookresearch/faiss/wiki | FAISS index types |
| 📖 **Docs** | LangChain Text Splitters | https://python.langchain.com/docs/how_to/#text-splitters | All splitting options |
| 🎥 **Video** | RAG From Scratch (LangChain) | https://youtu.be/sVcwVQRHIc8 | Visual walk of the pipeline |
| 🎥 **Video** | Chunking for RAG (Greg Kamradt) | https://youtu.be/8OJC21T2SL4 | 5 levels of text splitting |
