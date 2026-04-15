# Reciprocal Rank Fusion (RRF) — Deep Dive Tutorial

> A comprehensive tutorial on Reciprocal Rank Fusion, the standard algorithm
> for combining multiple search engines without falling into the Normalization Trap.
>
> **Papers referenced:**
> - Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009) — "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *SIGIR 2009*, pp. 758–759.

---

## Part 1: The Problem — The Normalization Trap

When you build a **Hybrid Search Pipeline** (Stage 3 of the RAG Roadmap), you are combining two completely different mathematical universes:

1. **Semantic Search (Vector/FAISS):** Outputs distances (e.g., `0.5`, `1.2`). These easily convert to a bounded similarity score between `0.0` and `1.0`.
2. **Keyword Search (BM25):** Outputs unbounded probabilistic scores. A document might score `4.2`, `85.7`, or `402.1` depending on the query length and term rarity.

### Why Weighted Sums Fail

If you try `Final_Score = (BM25 * 0.5) + (Vector * 0.5)`, the BM25 score (e.g., `85.0`) completely obliterates the Vector score (e.g., `0.9`). 

To fix this, developers often try to "squash" the BM25 score into a `0.0 - 1.0` range using **Sigmoid Normalization** based on the batch average:

```python
# The brittle approach:
mean_score = sum(bm25_scores) / len(bm25_scores)
normalized = 1 / (1 + math.exp(-(raw_score - mean_score)))
```

**Why this breaks in production:**
This approach relies on the *batch average*. If you search for "Phở", you might get documents scoring `[20, 18, 16]`. The mean is `18`. The score `20` becomes roughly `0.88`.
If you add one extremely relevant document to your database that scores `100`, the new results are `[100, 20, 18, 16]`. The new mean is `38.5`. 
Because the mean shifted drastically, your document that previously scored `20` now normalizes to `0.000000009`. 

The same document, for the same query, got completely different scores simply because of what *other* documents were in the database. This makes debugging impossible.

---

## Part 2: Reciprocal Rank Fusion (RRF) Explained

In 2009, researchers at the University of Waterloo (Cormack et al.) proposed a radically simple solution: **Throw away the raw scores completely.**

If an engine says a document is the best match, we don't care *how* it got that score (whether it was an 85.0 or a 0.9). We only care that it ranked **1st**.

### The RRF Formula

$$\text{RRF Score} = \sum_{r \in R} \frac{1}{k + \text{rank}(r)}$$

Where:
- $R$ is the set of rank lists from the different search engines.
- $\text{rank}(r)$ is the position of the document in that list (1st, 2nd, 3rd...).
- $k$ is a constant smoothing parameter (the 2009 paper strongly recommends `k = 60`).

### Step-by-step Example

Imagine searching for `"Wagyu steak"`. 
- **BM25** finds exact matches.
- **Vector Store** finds semantic concepts.

**BM25 Results:**
1. Document C (Raw score: 140.2) 
2. Document A (Raw score: 18.5)
3. Document B (Raw score: 12.1)

**Vector Results:**
1. Document A (Raw score: 0.94)
2. Document D (Raw score: 0.88)
3. Document C (Raw score: 0.72)

Let's calculate RRF with `k = 60`:

| Document | BM25 Rank | Vector Rank | RRF Calculation | Final Score |
|:---|:---|:---|:---|:---|
| **Doc A** | 2nd | 1st | `(1 / (60+2)) + (1 / (60+1))` = `0.0161 + 0.0163` | **0.0324** 🏆 (Winner) |
| **Doc C** | 1st | 3rd | `(1 / (60+1)) + (1 / (60+3))` = `0.0163 + 0.0158` | 0.0321 |
| **Doc B** | 3rd | (unranked) | `(1 / (60+3)) + 0` = `0.0158 + 0` | 0.0158 |
| **Doc D** | (unranked) | 2nd | `0 + (1 / (60+2))` = `0 + 0.0161` | 0.0161 |

**Notice what happened:**
- Document A won overall because it ranked highly in *both* engines (2nd and 1st). 
- Document C ranked 1st in BM25 with a massively skewed raw score (`140.2`), but because RRF ignores raw score magnitude, it didn't unfairly dominate Document A.

---

## Part 3: Why k = 60?

The parameter `k = 60` was determined empirically by the original authors after testing across massive TREC datasets. 

Why not `k = 0`?
If `k = 0`, the formula is just `1 / rank`.
- Rank 1 gets `1 / 1 = 1.0`
- Rank 2 gets `1 / 2 = 0.5`
- Rank 3 gets `1 / 3 = 0.33`

This decay is way too fast. A document that ranks #1 in BM25 and gets ignored by Vector (`1.0 + 0 = 1.0`) would easily beat a document that ranks #2 in *both* engines (`0.5 + 0.5 = 1.0`). We want to effectively reward consensus (documents that appear in both lists).

By setting `k = 60`:
- Rank 1 gets `1 / 61 ≈ 0.01639`
- Rank 2 gets `1 / 62 ≈ 0.01612`
- Rank 3 gets `1 / 63 ≈ 0.01587`

The differences between ranks are small but consistent. A document appearing in the top 10 of *both* lists will always beat a document appearing at #1 in only one list.

---

## Part 4: Implementation Code

Here is a clean, dependency-free implementation of RRF for your AI Waiter:

```python
from typing import List, Tuple, Dict
from langchain_core.documents import Document

def rrf_hybrid_search(
    bm25_results: List[Tuple[Document, float]], 
    vector_results: List[Tuple[Document, float]], 
    k: int = 60,
    top_n: int = 5
) -> List[Tuple[Document, float]]:
    """
    Combine BM25 and Vector results using Reciprocal Rank Fusion.
    
    Args:
        bm25_results: List of (Document, score) sorted by score descending.
        vector_results: List of (Document, score) sorted by score descending.
        k: Smoothing parameter (default 60).
        top_n: How many final results to return.
        
    Returns:
        List of (Document, RRF_score) sorted by RRF_score descending.
    """
    
    rrf_scores: Dict[str, dict] = {}
    
    # 1. Process BM25 rankings
    for rank, (doc, raw_score) in enumerate(bm25_results):
        # Use document content (or a unique ID) as the dictionary key
        doc_key = doc.page_content 
        
        if doc_key not in rrf_scores:
            rrf_scores[doc_key] = {"doc": doc, "score": 0.0}
            
        # rank is 0-indexed, so we add 1
        rrf_scores[doc_key]["score"] += 1.0 / (k + rank + 1)

    # 2. Process Vector rankings
    for rank, (doc, raw_score) in enumerate(vector_results):
        doc_key = doc.page_content
        
        if doc_key not in rrf_scores:
            rrf_scores[doc_key] = {"doc": doc, "score": 0.0}
            
        rrf_scores[doc_key]["score"] += 1.0 / (k + rank + 1)
        
    # 3. Sort by final RRF score
    sorted_fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # 4. Format and return top_n
    final_results = []
    for item in sorted_fused[:top_n]:
        final_results.append((item["doc"], item["score"]))
        
    return final_results
```

---

## Part 5: The Equal Weighting Problem (Weighted RRF)

If you look closely at the standard RRF formula, you might notice a theoretical problem: **It assumes all search engines are equally trustworthy.** 

If Document A ranks #1 in BM25, and Document B ranks #1 in Vector Search, standard RRF gives them the exact same base score (`1/61`). 
But what if, for your specific application (like searching a restaurant menu), BM25 exact matches are almost *always* exactly what the user wants, and Vector Search is just a backup for loose semantic queries? In this case, treating them equally dilutes the power of BM25.

### The Solution: Weighted RRF (wRRF)

To solve this, you can simply multiply the RRF score by a weight specific to that engine:

$$\text{wRRF Score} = \sum_{e \in Engines} \left( W_e \times \frac{1}{k + \text{rank}_e} \right)$$

For example, if you trust BM25 roughly twice as much as Vector Search, you could set:
- $W_{BM25} = 0.7$
- $W_{Vector} = 0.3$

### Code update for wRRF

```python
def weighted_rrf_search(bm25_results, vector_results, k=60, w_bm25=0.7, w_vector=0.3):
    rrf_scores = {}
    
    for rank, (doc, _) in enumerate(bm25_results):
        doc_key = doc.page_content 
        rrf_scores[doc_key] = rrf_scores.get(doc_key, {"doc": doc, "score": 0.0})
        # Apply the BM25 weight
        rrf_scores[doc_key]["score"] += w_bm25 * (1.0 / (k + rank + 1))

    for rank, (doc, _) in enumerate(vector_results):
        doc_key = doc.page_content
        rrf_scores[doc_key] = rrf_scores.get(doc_key, {"doc": doc, "score": 0.0})
        # Apply the Vector weight
        rrf_scores[doc_key]["score"] += w_vector * (1.0 / (k + rank + 1))
        
    return sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
```

**When should you use Weighted RRF?**
You should use wRRF if you are *not* using a Cross-Encoder Re-ranker in the next stage, and you know definitively that one engine outperforms the other on your specific dataset. If you *are* using a Cross-Encoder next, standard RRF is usually fine because the Re-ranker will fix any minor ordering mistakes as long as the right documents make it into the top-K pool.

---

## Part 6: When to use RRF vs Re-Ranker

RRF is a powerful, zero-cost technique for **Stage 1 (Retrieval)**. It guarantees high diverse recall by elegantly merging exact keyword matches and semantic matches.

However, RRF is fundamentally "blind" to the actual text of the query. It trusts that the upstream engines did their job correctly.

If you are implementing a **Cross-Encoder Re-Ranker** (Stage 4 of the roadmap), here is how your architecture should look:

```
[User Query]
    │
    ├─► BM25 (gets top 30) ──┐
    │                        │
    └─► Vector (gets top 30) ┴─► [ RRF Fusion ]
                                        │
                         (Yields 30 diverse candidates)
                                        │
                              [ Cross-Encoder ]  ← (Re-evaluates every pair line-by-line)
                                        │
                                (Yields final Top 3)
```

In this architecture, RRF is the perfect bridge. It doesn't waste time trying to perfectly calibrate arbitrary math weights (since the Cross-Encoder will overwrite the scores anyway). It focuses entirely on feeding a rich, high-recall candidate pool to the final precision stage.
