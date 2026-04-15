# Advanced Routing — Fast Alternatives to LLM Routing
> Your current pipeline calls the LLM twice (classify → respond). Here's how to make routing instant.

---

## 1. The Problem — Why LLM Routing is Slow

Your current AI Waiter pipeline:

```
User Input → [LLM Call #1: classify intent] → route → [LLM Call #2: generate response]
              ~1-3 seconds                              ~2-5 seconds
              
Total: 3-8 seconds just to respond
```

**LLM Call #1 is wasted time.** You're burning a full LLM inference just to get one word back ("ORDER" / "MENU_QUERY" / "CHAT"). There are much faster ways to classify intent.

---

## 2. Solution Overview

| Method | Latency | Accuracy | Complexity |
|---|---|---|---|
| LLM Router (current) | 1-3s | ⭐⭐⭐⭐⭐ | Easy |
| **Embedding classifier** | ~5ms | ⭐⭐⭐⭐ | Easy |
| **Keyword / Regex** | <1ms | ⭐⭐⭐ | Easy |
| **Small classifier model** | ~10ms | ⭐⭐⭐⭐⭐ | Medium |
| **Semantic Router** | ~5ms | ⭐⭐⭐⭐ | Easy |
| **Hybrid (fast first, LLM fallback)** | ~5ms avg | ⭐⭐⭐⭐⭐ | Medium |

---

## 3. Method 1: Embedding-Based Classifier (Recommended)

**Idea:** Pre-compute embeddings for example phrases per category. At runtime, embed the user input and find the nearest category. No LLM call needed.

```python
import numpy as np
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Define example phrases for each intent
INTENT_EXAMPLES = {
    "ORDER": [
        "Cho tôi 2 tô phở bò",
        "Gọi 1 cơm sườn",
        "Bàn 3 muốn thêm 2 ly trà đá",
        "Tôi muốn gọi món",
        "Thêm 1 phần cơm chiên cho bàn 5",
        "Gọi cho tôi 3 ly nước cam",
        "Đặt 2 tô bún bò Huế",
    ],
    "MENU_QUERY": [
        "Có món chay không?",
        "Phở bò bao nhiêu tiền?",
        "Menu có những gì?",
        "Cho tôi xem thực đơn",
        "Có nước ép trái cây không?",
        "Món nào ngon nhất?",
        "Giá cơm sườn là bao nhiêu?",
    ],
    "GENERAL_CHAT": [
        "Xin chào",
        "Nhà hàng mở mấy giờ?",
        "Cảm ơn bạn",
        "Wifi password là gì?",
        "Nhà vệ sinh ở đâu?",
        "Hôm nay thời tiết đẹp quá",
        "Tạm biệt",
    ],
}

# Pre-compute intent embeddings (do this ONCE at startup)
intent_vectors = {}
for intent, examples in INTENT_EXAMPLES.items():
    vecs = embeddings.embed_documents(examples)
    intent_vectors[intent] = np.mean(vecs, axis=0)  # centroid

def classify_fast(user_input: str) -> str:
    """Classify intent using embedding similarity. ~5ms, no LLM call."""
    input_vec = np.array(embeddings.embed_query(user_input))
    
    best_intent = None
    best_score = -1
    
    for intent, centroid in intent_vectors.items():
        # Cosine similarity
        score = np.dot(input_vec, centroid) / (
            np.linalg.norm(input_vec) * np.linalg.norm(centroid)
        )
        if score > best_score:
            best_score = score
            best_intent = intent
    
    return best_intent

# Test
print(classify_fast("Gọi 1 tô phở"))        # → ORDER
print(classify_fast("Có bún chả không?"))    # → MENU_QUERY
print(classify_fast("Chào bạn"))             # → GENERAL_CHAT
```

### Pros & Cons
- ✅ **~5ms** instead of 1-3 seconds
- ✅ No LLM call needed
- ✅ Easy to add new intents (just add examples)
- ⚠️ Needs an embedding model running (`nomic-embed-text` is tiny: 274MB)
- ⚠️ Slightly less accurate than LLM for ambiguous inputs

---

## 4. Method 2: Semantic Router (Library)

[Semantic Router](https://github.com/aurelio-labs/semantic-router) is a purpose-built library for this exact problem.

```bash
pip install semantic-router
```

```python
from semantic_router import Route, RouteLayer
from semantic_router.encoders import FastEmbedEncoder

# Define routes with example utterances
order_route = Route(
    name="ORDER",
    utterances=[
        "Cho tôi 2 tô phở bò",
        "Gọi 1 cơm sườn",
        "Tôi muốn gọi món",
        "Thêm 1 phần cho bàn 5",
        "Đặt 2 tô bún bò Huế",
    ],
)

menu_route = Route(
    name="MENU_QUERY",
    utterances=[
        "Có món chay không?",
        "Phở bò bao nhiêu tiền?",
        "Cho tôi xem thực đơn",
        "Món nào ngon nhất?",
    ],
)

chat_route = Route(
    name="GENERAL_CHAT",
    utterances=[
        "Xin chào",
        "Nhà hàng mở mấy giờ?",
        "Cảm ơn bạn",
        "Wifi password là gì?",
    ],
)

# Create router (uses local embedding model, no API key needed)
encoder = FastEmbedEncoder()
router = RouteLayer(encoder=encoder, routes=[order_route, menu_route, chat_route])

# Classify — returns in milliseconds
result = router("Gọi cho tôi 1 ly cà phê")
print(result.name)  # → ORDER
```

### Why Semantic Router is great
- ✅ Built specifically for intent routing
- ✅ ~5ms classification
- ✅ Uses local embeddings (no API calls)
- ✅ Clean API, easy to maintain
- ✅ Supports dynamic route updates

---

## 5. Method 3: Keyword / Regex Router (Fastest, Simplest)

For when you want **sub-millisecond** routing with zero ML:

```python
import re

ORDER_PATTERNS = [
    r"cho\s+(tôi|bàn)",
    r"gọi\s+\d",
    r"đặt\s+\d",
    r"thêm\s+\d",
    r"muốn\s+(gọi|đặt|thêm)",
    r"\d+\s+(tô|phần|ly|dĩa|chai)",
]

MENU_PATTERNS = [
    r"bao\s+nhiêu\s+tiền",
    r"giá\s+.*là",
    r"thực\s*đơn",
    r"menu",
    r"có\s+.*không",
    r"món\s+(nào|gì)",
]

def classify_regex(text: str) -> str:
    text_lower = text.lower()
    
    for pattern in ORDER_PATTERNS:
        if re.search(pattern, text_lower):
            return "ORDER"
    
    for pattern in MENU_PATTERNS:
        if re.search(pattern, text_lower):
            return "MENU_QUERY"
    
    return "GENERAL_CHAT"

print(classify_regex("Cho tôi 2 tô phở"))     # → ORDER
print(classify_regex("Phở bao nhiêu tiền?"))  # → MENU_QUERY
print(classify_regex("Xin chào"))              # → GENERAL_CHAT
```

### Pros & Cons
- ✅ **<1ms** — literally instant
- ✅ Zero dependencies, zero models
- ✅ 100% predictable — no randomness
- ❌ Brittle — misses creative phrasing
- ❌ Hard to maintain as intents grow
- ❌ Language-specific patterns need expertise

---

## 6. Method 4: Small Classifier Model (Best Accuracy)

Train a tiny text classifier (not a full LLM) specifically for intent detection.

### Option A: scikit-learn (simplest ML)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Training data
texts = [
    "Cho tôi 2 tô phở bò", "Gọi 1 cơm sườn", "Đặt 3 ly trà đá",
    "Có món chay không?", "Phở bao nhiêu?", "Menu có gì?",
    "Xin chào", "Cảm ơn", "Nhà hàng mở mấy giờ?",
]
labels = [
    "ORDER", "ORDER", "ORDER",
    "MENU_QUERY", "MENU_QUERY", "MENU_QUERY",
    "GENERAL_CHAT", "GENERAL_CHAT", "GENERAL_CHAT",
]

# Train
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
clf = LogisticRegression()
clf.fit(X, labels)

# Save
pickle.dump((vectorizer, clf), open("intent_model.pkl", "wb"))

# Predict (~1ms)
def classify_ml(text: str) -> str:
    X_new = vectorizer.transform([text])
    return clf.predict(X_new)[0]

print(classify_ml("Gọi 2 phần cơm chiên"))  # → ORDER
```

### Option B: Small transformer (best accuracy)
```bash
pip install sentence-transformers
```

Fine-tune a small model like `phobert-base` (Vietnamese-specific) on your intent data. This gives the best accuracy but requires collecting training data.

---

## 7. Method 5: Hybrid Router (⭐ Best Overall)

**Use fast methods first, fall back to LLM only when uncertain.**

```python
import numpy as np

CONFIDENCE_THRESHOLD = 0.75

def hybrid_router(user_input: str) -> str:
    """
    1. Try regex first (instant)
    2. Try embedding similarity (5ms)  
    3. Fall back to LLM only if uncertain (1-3s)
    """
    
    # Layer 1: Regex (instant, high-confidence patterns)
    regex_result = classify_regex(user_input)
    if regex_result != "GENERAL_CHAT":  # regex matched a specific pattern
        return regex_result
    
    # Layer 2: Embedding similarity (fast, good accuracy)
    input_vec = np.array(embeddings.embed_query(user_input))
    best_intent = None
    best_score = -1
    
    for intent, centroid in intent_vectors.items():
        score = np.dot(input_vec, centroid) / (
            np.linalg.norm(input_vec) * np.linalg.norm(centroid)
        )
        if score > best_score:
            best_score = score
            best_intent = intent
    
    if best_score >= CONFIDENCE_THRESHOLD:
        return best_intent
    
    # Layer 3: LLM fallback (slow but accurate, only for edge cases)
    return router_chain.invoke({"input": user_input}).strip().upper()

# Most inputs resolve at Layer 1 or 2 (~5ms)
# Only ambiguous inputs hit Layer 3 (~2s)
```

### Performance result
```
Layer 1 (Regex):      <1ms    handles ~40% of inputs
Layer 2 (Embedding):  ~5ms    handles ~50% of inputs
Layer 3 (LLM):        ~2s     handles ~10% of inputs (edge cases)

Average latency: ~10ms (instead of 1-3s for every request)
```

---

## 8. Optimizing Your Full Pipeline

### Current (slow)
```
Input → LLM classify (1-3s) → tool/RAG (0.5s) → LLM respond (2-5s)
Total: 3.5 - 8.5 seconds
```

### Optimized
```
Input → Hybrid router (~5ms) → tool/RAG (0.5s) → LLM respond (2-5s)
Total: 2.5 - 5.5 seconds  (saved 1-3 seconds!)
```

### More speed tricks

| Trick | Saves | How |
|---|---|---|
| **Hybrid router** | 1-3s | Replace LLM classifier with embedding/regex |
| **Streaming** | Perceived 2-3s | Start showing response before it finishes |
| **Smaller model for routing** | 0.5-1s | Use `phi3:mini` (2.3GB) for classification only |
| **Cache frequent queries** | 1-5s | Cache "xin chào" → response, skip LLM entirely |
| **Parallel RAG + LLM** | 0.5-1s | Start RAG search while LLM is still generating |
| **Quantized model** | 0.5-1s | Use `q4_0` instead of `q8_0` |

---

## 9. Caching Common Responses

For a restaurant, many queries repeat (greetings, "what's on the menu?"):

```python
from functools import lru_cache
import hashlib

# Simple cache
response_cache = {}

def get_cached_response(user_input: str) -> str | None:
    """Check if we've seen this (or very similar) input before."""
    # Normalize
    key = user_input.lower().strip()
    return response_cache.get(key)

def cache_response(user_input: str, response: str):
    key = user_input.lower().strip()
    response_cache[key] = response

# Usage
cached = get_cached_response(user_input)
if cached:
    return cached  # instant!
else:
    result = full_pipeline(user_input)  # slow path
    cache_response(user_input, result)
    return result
```

### Smarter: Semantic cache (cache similar meanings)
```python
# If "xin chào" was cached, "chào bạn" should also hit the cache
# Use embedding similarity to match similar inputs to cached responses

def semantic_cache_lookup(user_input: str, threshold=0.95):
    input_vec = embeddings.embed_query(user_input)
    for cached_input, (cached_vec, cached_response) in cache.items():
        similarity = cosine_similarity(input_vec, cached_vec)
        if similarity >= threshold:
            return cached_response
    return None
```

---

## 10. Decision Guide

```
Q: How many intents do you have?
├── 2-5 intents → Embedding classifier or Semantic Router
├── 5-20 intents → Small ML classifier (sklearn)
└── 20+ intents → Fine-tuned transformer

Q: How important is latency?
├── Must be <50ms → Regex + Embedding (no LLM)
├── <500ms is fine → Small model classifier
└── Don't care → LLM router is simplest

Q: How varied is user language?
├── Predictable patterns → Regex works great
├── Somewhat varied → Embedding classifier
└── Very creative/varied → LLM or fine-tuned model

★ For AI Waiter: Start with HYBRID (Method 5)
  - Regex catches obvious orders ("gọi 2 tô phở")
  - Embeddings handle the rest
  - LLM fallback for edge cases
```

---

## 11. Summary — Recommended Setup for AI Waiter

```python
# startup.py — initialize once
router = HybridRouter(
    regex_patterns=ORDER_PATTERNS + MENU_PATTERNS,
    embedding_model="nomic-embed-text",
    llm_fallback="llama3",
    confidence_threshold=0.75,
)

# runtime — per request
intent = router.classify(user_input)  # ~5ms average

if intent == "ORDER":
    response = agent_executor.invoke(user_input)      # tool calling
elif intent == "MENU_QUERY":
    response = rag_chain.invoke(user_input)            # vector search
else:
    response = chat_chain.invoke(user_input)            # direct LLM
```

**Result: 1-3 seconds saved per request.** Over thousands of customer interactions, that's a much better experience.
