# BM25 — Deep Dive Tutorial

> A comprehensive tutorial to fully understand BM25 (Best Match 25):
> the theory behind it, how it actually scores documents, the critical
> tokenization problem for Vietnamese, and where it breaks down.
>
> **Papers referenced:**
> - Spärck Jones, K. (1972) — "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." *Journal of Documentation*, 28(1), 11–21.
> - Robertson, S. E., & Walker, S. (1994) — "Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted IR." *SIGIR 1994*, pp. 232–241.
> - Lv, Y., & Zhai, C. (2011) — "Lower-Bounding Term Frequency Normalization." *CIKM 2011*, pp. 7–16. (BM25+ / BM25L)
> - Thakur, N. et al. (2021) — "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models." *NeurIPS 2021*. ([arXiv:2104.08663](https://arxiv.org/abs/2104.08663))
> - Gao et al. (2023) — "RAG for LLMs: A Survey." ([arXiv:2312.10997](https://arxiv.org/abs/2312.10997))

---

## Part 1: Why BM25 Exists — The Problem with Naive RAG Search

### Where We Left Off

You finished the Naive RAG tutorial knowing that it works, but has a fundamental flaw in its retrieval engine:

> **Naive RAG only searches by semantic similarity.**

Let's make this concrete with your AI Waiter menu:

```
User query: "Phở Bò"

[Embedding model encodes "Phở Bò" → a vector]
[FAISS finds vectors CLOSE to that vector]

Possible matches returned:
  ✅ "Phở Bò: Vietnamese beef noodle soup, 55,000 VND"    ← correct
  ⚠️  "Ramen: Japanese noodle soup with pork broth"        ← semantically similar, but WRONG
  ⚠️  "Bún Bò Huế: spicy beef vermicelli"                 ← also beef noodles, ranked above Phở Bò
```

The embedding model knows "Phở Bò", "Ramen", and "Bún Bò Huế" are all noodle soups. It cannot
distinguish **which one the user literally asked for by name**.

This is the ***vocabulary gap problem*** — the user typed an exact product name, and the semantic
search engine turned it into an abstract concept.

### The Two Types of Queries

| Query Type | Example | Best Engine |
|:---|:---|:---|
| **Exact / named entity** | "Phở Bò", "Bít Tết Wagyu", table #3 | Keyword search (BM25) |
| **Semantic / intent** | "something warm and filling", "light lunch" | Vector search |

No single engine handles both well. BM25 is the answer to the first type.

---

## Part 2: The Intellectual History of BM25

Understanding *where* BM25 came from helps you understand *why* it works the way it does.

### 1972: Spärck Jones Invents IDF

Karen Spärck Jones published a paper in 1972 that proposed a radical idea:
**the rarer a term is in the entire collection, the more valuable a match on that term is.**

> *"Term specificity should be interpreted as a statistical property of term use rather
> than a semantic one."* — Spärck Jones (1972)

She called this **Inverse Document Frequency (IDF)**. The intuition:
- "is", "the", "a" appear in 99% of all documents → matching on them tells you nothing.
- "Bít Tết Wagyu" appears in 0.1% of documents → a match is highly informative.

```
IDF(term) = log( N / df(term) )

Where:
  N      = Total number of documents in the collection
  df(t)  = Number of documents containing term t
```

For a restaurant menu with 50 items:
```
IDF("is")       = log(50 / 50) = log(1)  = 0.0       ← no signal
IDF("bò")       = log(50 / 12) = log(4.2) ≈ 1.43     ← medium signal
IDF("wagyu")    = log(50 / 1)  = log(50)  ≈ 3.91     ← strong signal
```

This was the first key insight that became the foundation of BM25.

### 1994: Robertson & Walker Formalize BM25

Stephen Robertson and Steve Walker combined Spärck Jones' IDF with **probabilistic
term frequency (TF) modelling** at SIGIR 1994 to produce Okapi BM25.

The critical contributions beyond raw IDF were:

1. **TF Saturation** — the 10th occurrence of "bò" in a document adds far less information
   than the 1st occurrence. Raw TF is unbounded; BM25 saturates it.

2. **Document Length Normalization** — a long menu with "beef" 20 times should not automatically
   rank above a short, precise description of "Bít Tết Wagyu" with "beef" 3 times.

The resulting formula became one of the most cited algorithms in all of computer science.

---

## Part 3: The BM25 Formula — Complete Dissection

### The Full Formula

$$\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t,D) \cdot (k_1 + 1)}{f(t,D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- `Q` = set of query terms
- `t` = each term in the query
- `D` = the document being scored
- `f(t, D)` = frequency of term `t` in document `D` (raw count)
- `|D|` = length of document `D` (in tokens)
- `avgdl` = average document length across the collection
- `k₁` ∈ [1.2, 2.0] = TF saturation parameter
- `b` ∈ [0, 1] = length normalization parameter

### Dissecting Each Component

#### Component 1: IDF (the rarity score)

```
IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

The `+0.5` smoothing prevents division-by-zero when a term appears in every document.

**Intuition:** Rare terms → high IDF → matched document scores higher.

```
Collection: 50 menu documents

term "Wagyu"     → df=1  → IDF ≈ 3.89  (very informative match)
term "bò"        → df=12 → IDF ≈ 1.38  (moderately informative)
term "của"       → df=50 → IDF ≈ 0.01  (nearly useless)
```

---

#### Component 2: Saturated TF (the frequency score)

```
TF_saturated = f(t, D) × (k₁ + 1) / (f(t, D) + k₁)
```

Without the denominator, this is just `f(t, D) × (k₁ + 1)` — linear growth.
With saturation, the function grows fast at first, then flattens:

```
f(t,D)  | k₁=1.5 | Raw TF
--------|---------|-------
1       | 2.0     | 1       ← the 1st occurrence matters most
2       | 2.4     | 2
5       | 2.73    | 5
10      | 2.87    | 10
100     | 2.99    | 100     ← the 100th adds almost nothing
```

**Intuition:** A document mentioning "Bít Tết" 10 times should not automatically
rank 10x higher than one that mentions it once. BM25 prevents this.

---

#### Component 3: Length Normalization (the fairness factor)

```
length_factor = 1 - b + b × (|D| / avgdl)
```

| Parameters | Effect |
|:---|:---|
| `b = 0` | No length normalization. Long documents dominate. |
| `b = 1` | Full length normalization. Every document treated as same length. |
| `b = 0.75` | **Default.** Partial normalization — the standard balance. |

The length factor is put in the denominator, so:
- A long document (high `|D|`) → high denominator → **lower** TF score.
- A short document (low `|D|`) → low denominator → **higher** TF score.

**Why this matters for your AI Waiter:**

```
Document A: "Phở Bò. Price 55,000 VND."
  → |D| = 7 tokens. Very short. Dense mention of "Phở Bò".

Document B: "Phở Bò is a rich, slow-cooked Vietnamese noodle soup served with fresh
  herbs, bean sprouts, lime wedges, hoisin sauce, and your choice of beef cuts including
  brisket, tendon, and tripe. Our broth takes 12 hours to prepare. Price: 55,000 VND."
  → |D| = 52 tokens. Long. Also mentions "Phở Bò" once.

Without length normalization: Document B scores as high as A (same "Phở Bò" count).
With b=0.75: Document A scores higher — it's more *about* "Phở Bò".
```

### Putting It All Together — A Manual Calculation

Let's score a query `"bò nướng"` (grilled beef) against two documents by hand:

```
Documents in collection (N=4):
  D1: "bò nướng sả ớt thơm ngon"        → 6 tokens, contains "bò"×1, "nướng"×1
  D2: "bò kho tiêu đen đậm đà"          → 6 tokens, contains "bò"×1
  D3: "gà nướng mật ong"                → 4 tokens, contains "nướng"×1
  D4: "cơm trắng dẻo thơm"              → 4 tokens, contains neither

avgdl = (6+6+4+4) / 4 = 5.0
k₁ = 1.5, b = 0.75

IDF("bò")    → df=2, N=4 → log((4-2+0.5)/(2+0.5)+1) ≈ 0.98
IDF("nướng") → df=2, N=4 → same ≈ 0.98

Score(D1, "bò nướng"):
  [for "bò"]:    0.98 × (1×2.5) / (1 + 1.5×(1 - 0.75 + 0.75×6/5)) = 0.98 × 0.96 ≈ 0.94
  [for "nướng"]: 0.98 × (1×2.5) / (1 + 1.5×(1 - 0.75 + 0.75×6/5)) = 0.94
  TOTAL: 1.88  ✅ Highest — contains both query terms

Score(D2, "bò nướng"):
  [for "bò"]:    0.94
  [for "nướng"]: 0.0  (term not in document)
  TOTAL: 0.94

Score(D3, "bò nướng"):
  [for "bò"]:    0.0
  [for "nướng"]: same ≈ 0.94
  TOTAL: 0.94

Score(D4, "bò nướng"):
  TOTAL: 0.0   ← no matches, zero score
```

D1 wins. Correct. BM25 naturally elevates the document that matches the most query terms.

---

## Part 4: The Parameters — What k₁ and b Actually Do

### Tuning k₁ (TF Saturation)

```python
import numpy as np
import matplotlib.pyplot as plt

# TF saturation curve for different k1 values
tf_values = np.arange(0, 20, 0.1)

for k1 in [0, 0.5, 1.2, 1.5, 2.0]:
    saturated = tf_values * (k1 + 1) / (tf_values + k1 + 1e-9)
    print(f"k1={k1}: max achievable TF factor = {k1 + 1:.1f}")

# k1=0:   Binary — term either present (score=1) or absent (score=0)
# k1=1.2: Recommended for short queries (web search)
# k1=1.5: Recommended for long documents (default in rank_bm25)
# k1=2.0: Allows more TF influence — good for long documents with rich vocabulary
```

**When to increase k₁:**
- Your corpus has long, detailed descriptions (e.g., medical records, legal documents)
- Repeated terms genuinely increase relevance (e.g., a recipe mentioning "beef" 5 times
  truly is more "beef-focused" than one mentioning it once)

**When to decrease k₁:**
- Short documents (tweets, menu item names)
- You want presence/absence to matter more than frequency

---

### Tuning b (Length Normalization)

```python
# Effect of b parameter on a long vs short document
# Query: "beef noodle"
# short_doc = 5 tokens, long_doc = 50 tokens, avgdl = 20 tokens

def length_factor(doc_len, avgdl, b):
    return 1 - b + b * (doc_len / avgdl)

for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
    lf_short = length_factor(5, 20, b)
    lf_long  = length_factor(50, 20, b)
    print(f"b={b}: short_doc factor={lf_short:.2f}, long_doc factor={lf_long:.2f}")

# b=0.0: short=1.00, long=1.00 → no normalization, equal treatment
# b=0.75: short=0.44, long=2.13 → short doc benefits, long doc penalized
# b=1.0: short=0.25, long=2.50 → maximum effect
```

**Default values used in practice:**
- `k₁ = 1.5`, `b = 0.75` (used by `rank_bm25` and Elasticsearch)
- If your corpus is very uniform in document length: lower `b` (it doesn't matter as much)
- For very long documents (e.g., PDFs): consider `b = 0.5` to reduce penalization

---

## Part 5: Tokenization — The Hidden Critical Decision

### Why Tokenization is the Most Important Component

The BM25 formula is fixed math. What you put *into* it is entirely up to you.

BM25 operates on **tokens**. Every aspect of retrieval quality depends on whether your tokenizer
correctly identifies meaningful units in the text.

```
Bad tokenizer output for "Phở bò viên":    ["Phở", "bò", "viên"]    ← 3 tokens ✅ (accidental success)
Bad tokenizer output for "bít tết wagyu":  ["bít", "tết", "wagyu"]  ← 3 tokens ❌

"bít tết" is a SINGLE compound word meaning "steak".
"bít" alone means nothing meaningful.
"tết" alone means "Lunar New Year" — completely wrong!
```

### The Vietnamese Compound Word Problem

Vietnamese is a **monosyllabic, isolating language** where:
1. Every syllable is pronounced separately and written with spaces.
2. Most meaningful words are **multi-syllabic compounds** written with spaces between syllables.
3. A naive `.split()` on spaces treats each syllable as an independent word.

```
English: "beef steak"  → split → ["beef", "steak"]  ← mostly OK
Vietnamese: "bít tết"  → split → ["bít", "tết"]    ← completely wrong
```

**The BM25 inverted index breaks silently:**

```
With naive split(): Index built with tokens: ["bít", "tết"]
User query "bít tết":  also splits to ["bít", "tết"]
Result: BM25 finds matches ← appears to work

BUT:
User query "steak" (the concept, not exact):  No match at all
User query "bít" alone:  Matches "Tết Holiday Menu" (wrong document!)
User query "tết" alone:  Retrieves New Year dishes instead of steak!
```

The system appears functional for exact-name lookups but is deeply broken for anything else.

### Fixing It: Vietnamese Tokenizers Compared

| Tool | Method | Accuracy | Speed | Install |
|:---|:---|:---|:---|:---|
| `.split()` | Naïve whitespace | ❌ Treats syllables as words | ✅ Instant | Built-in |
| `underthesea` | CRF + Dictionary | ✅ State of the art | ⚠️ ~5ms/sentence | `pip install underthesea` |
| `pyvi` | Pattern-based | ✅ Good | ✅ Fast | `pip install pyvi` |
| `vncorenlp` | Deep learning | ✅ Excellent | ❌ Requires Java | Complex setup |

### Code: Seeing the Tokenization Difference

```python
from underthesea import word_tokenize

# ----- Raw test phrases -----
phrases = [
    "bít tết bò wagyu",       # Wagyu beef steak
    "nước chanh dây",          # Passion fruit juice
    "phở bò tái chín",         # Half-done / well-done pho
    "cơm chiên dương châu",    # Yangzhou fried rice
    "đồ uống không cồn",       # Non-alcoholic beverages
]

print("=" * 60)
print(f"{'Phrase':<25} | {'Naive split':<20} | Underthesea")
print("=" * 60)

for phrase in phrases:
    naive   = phrase.lower().split()
    correct = word_tokenize(phrase.lower(), format="text").split()
    print(f"{phrase:<25} | {str(naive):<20} | {correct}")

# Expected output:
# bít tết bò wagyu    | ['bít', 'tết', 'bò', 'wagyu'] | ['bít_tết', 'bò', 'wagyu']
# nước chanh dây      | ['nước', 'chanh', 'dây']       | ['nước', 'chanh_dây']
# phở bò tái chín    | ['phở', 'bò', 'tái', 'chín']  | ['phở_bò', 'tái', 'chín']
```

**Key insight:** `underthesea` joins compound words with underscores.
`"chanh_dây"` (passion fruit) is now a single token — it won't be confused with
`"chanh"` (lemon) or `"dây"` (wire/string).

---

### Code: Building a Tokenization-Aware BM25

```python
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize
from langchain_core.documents import Document

# ── 1. Your restaurant menu (realistic Vietnamese data) ──
documents = [
    Document(page_content="Bít tết bò Wagyu nướng bơ tỏi. Giá: 250,000 VND. Hạng mục: Bò"),
    Document(page_content="Phở bò tái chín nước dùng hầm 12 tiếng. Giá: 65,000 VND. Hạng mục: Phở"),
    Document(page_content="Gà nướng mật ong chanh dây. Giá: 120,000 VND. Hạng mục: Gà"),
    Document(page_content="Nước chanh dây tươi mát. Giá: 35,000 VND. Hạng mục: Đồ uống"),
    Document(page_content="Cơm chiên dương châu trứng và rau. Giá: 55,000 VND. Hạng mục: Cơm"),
    Document(page_content="Bún bò Huế cay nồng đặc trưng xứ Huế. Giá: 60,000 VND. Hạng mục: Bún"),
    Document(page_content="Gỏi cuốn tôm thịt rau sống. Giá: 45,000 VND. Hạng mục: Khai vị"),
]


# ── 2. Tokenizer functions ──
def naive_tokenize(text: str) -> list[str]:
    """Wrong for Vietnamese."""
    return text.lower().split()


def vn_tokenize(text: str) -> list[str]:
    """Correct: uses underthesea CRF model."""
    return word_tokenize(text.lower(), format="text").split()


# ── 3. Build BM25 with BOTH tokenizers for comparison ──
naive_bm25   = BM25Okapi([naive_tokenize(d.page_content) for d in documents])
correct_bm25 = BM25Okapi([vn_tokenize(d.page_content)   for d in documents])


def search_bm25(query: str, bm25_index, tokenizer_fn, k: int = 3) -> list[dict]:
    """Search and return top-k results with scores."""
    tokens = tokenizer_fn(query)
    scores = bm25_index.get_scores(tokens)

    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: -x[1])

    return [
        {"score": score, "content": doc.page_content, "tokens": tokens}
        for doc, score in doc_scores[:k]
        if score > 0
    ]


# ── 4. Test queries that reveal the difference ──
test_queries = [
    "bít tết",         # Should find Wagyu steak
    "chanh dây",       # Should find passion fruit juice (not regular lemon!)
    "phở bò",          # Should find pho
    "tết",             # Test word meaning confusion: "tết" = New Year, not "bít tết"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")

    naive_results   = search_bm25(query, naive_bm25, naive_tokenize)
    correct_results = search_bm25(query, correct_bm25, vn_tokenize)

    print(f"\n[Naive .split()]:")
    for r in naive_results:
        print(f"  Score {r['score']:.3f} | Tokens used: {r['tokens']} | {r['content'][:55]}...")

    print(f"\n[Underthesea correct tokenizer]:")
    for r in correct_results:
        print(f"  Score {r['score']:.3f} | Tokens used: {r['tokens']} | {r['content'][:55]}...")
```

**What to observe:**
- Query `"tết"` with naive tokenizer: returns "Bít Tết Wagyu" with a score (matching token "tết").
  But "tết" alone means Lunar New Year — this is a false positive.
- With `underthesea`: `"tết"` alone returns nothing because "bít_tết" is one compound token.
  This is correct — "tết" alone should not match a steak dish.
- Query `"chanh dây"`: naive gets two separate tokens `["chanh", "dây"]`, which could match
  unrelated documents containing either "chanh" (lemon) or "dây" (wire). The correct tokenizer
  produces `["chanh_dây"]` — one token for passion fruit.

---

## Part 6: The Complete Production BM25 Class

This is a production-ready BM25 search engine for your AI Waiter, with:
- Configurable tokenizer (naive vs Vietnamese)
- Persistent index (no re-building every restart)
- Score transparency for debugging
- Top-k with minimum score threshold

```python
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


@dataclass
class BM25Result:
    document: Document
    score: float
    rank: int


class VietBM25:
    """
    Production-ready BM25 search engine for Vietnamese restaurant data.

    Design decisions:
    - Tokenizer is injected (swappable: naive for English, underthesea for Vietnamese)
    - Index is built once and optionally cached to disk
    - Scores are always returned for transparency and downstream fusion
    """

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        # Default to Vietnamese tokenizer if underthesea is available, else naive
        if tokenizer is None:
            try:
                from underthesea import word_tokenize
                self.tokenizer = lambda t: word_tokenize(t.lower(), format="text").split()
                print("INFO: Using underthesea Vietnamese tokenizer.")
            except ImportError:
                self.tokenizer = lambda t: t.lower().split()
                print("WARNING: underthesea not found. Using naive tokenizer. Install with: pip install underthesea")
        else:
            self.tokenizer = tokenizer

        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []

    # ── INDEXING ────────────────────────────────────────────────────────
    def build_index(self, documents: list[Document]) -> None:
        """Tokenize all documents and build the BM25 inverted index."""
        self.documents = documents
        self._tokenized_corpus = [
            self.tokenizer(doc.page_content) for doc in documents
        ]
        self.bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        print(f"INFO: BM25 index built with {len(documents)} documents.")

    def save_index(self, path: str) -> None:
        """Cache index to disk. Avoids re-tokenizing on every startup."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "tokenized_corpus": self._tokenized_corpus,
                "k1": self.k1,
                "b": self.b,
            }, f)
        print(f"INFO: BM25 index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load a cached index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents          = data["documents"]
        self._tokenized_corpus  = data["tokenized_corpus"]
        self.k1                 = data["k1"]
        self.b                  = data["b"]
        self.bm25 = BM25Okapi(
            self._tokenized_corpus, k1=self.k1, b=self.b
        )
        print(f"INFO: BM25 index loaded from {path}. {len(self.documents)} documents.")

    # ── RETRIEVAL ────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[BM25Result]:
        """
        Search the index and return top-k results.

        Args:
            query:      The user's search query
            k:          Number of results to return
            min_score:  Filter out results below this score threshold

        Returns:
            List of BM25Result, sorted by score descending
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        query_tokens = self.tokenizer(query)
        scores       = self.bm25.get_scores(query_tokens)

        # Pair documents with scores and sort
        doc_score_pairs = sorted(
            zip(self.documents, scores),
            key=lambda x: -x[1]
        )

        # Apply k limit and min_score threshold
        results = []
        for rank, (doc, score) in enumerate(doc_score_pairs[:k]):
            if score >= min_score:
                results.append(BM25Result(document=doc, score=score, rank=rank + 1))

        return results

    def explain(self, query: str, document: Document) -> dict:
        """
        Debug tool: Show per-term scores for a query-document pair.
        Useful for understanding why a document was ranked as it was.
        """
        query_tokens = self.tokenizer(query)
        doc_tokens   = self.tokenizer(document.page_content)

        explanation = {
            "query_tokens": query_tokens,
            "doc_tokens": doc_tokens,
            "per_term_scores": {},
        }

        for token in query_tokens:
            # Build a single-token query for isolated scoring
            single_bm25 = BM25Okapi([doc_tokens], k1=self.k1, b=self.b)
            score = single_bm25.get_scores([token])[0]
            explanation["per_term_scores"][token] = round(score, 4)

        return explanation


# ── EXAMPLE USAGE ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INDEX_PATH = "./data/bm25_index.pkl"

    # Build the engine
    engine = VietBM25(k1=1.5, b=0.75)

    documents = [
        Document(
            page_content="Bít tết bò Wagyu nướng bơ tỏi thơm lừng. Giá: 250,000 VND.",
            metadata={"id": "item_001", "category": "bò", "price": 250000}
        ),
        Document(
            page_content="Phở bò tái chín nước dùng hầm 12 tiếng. Giá: 65,000 VND.",
            metadata={"id": "item_002", "category": "phở", "price": 65000}
        ),
        Document(
            page_content="Gà nướng mật ong và chanh dây. Giá: 120,000 VND.",
            metadata={"id": "item_003", "category": "gà", "price": 120000}
        ),
        Document(
            page_content="Nước chanh dây tươi ép. Giá: 35,000 VND.",
            metadata={"id": "item_004", "category": "đồ uống", "price": 35000}
        ),
        Document(
            page_content="Bún bò Huế sả ớt đặc trưng. Giá: 60,000 VND.",
            metadata={"id": "item_005", "category": "bún", "price": 60000}
        ),
    ]

    # Build and optionally cache
    if os.path.exists(INDEX_PATH):
        engine.load_index(INDEX_PATH)
    else:
        engine.build_index(documents)
        engine.save_index(INDEX_PATH)

    # Search
    results = engine.search("phở bò", k=3, min_score=0.1)
    for r in results:
        print(f"Rank {r.rank} | Score {r.score:.4f} | {r.document.page_content[:60]}")

    # Debug: why did this document score the way it did?
    print("\n--- Term-by-term explanation ---")
    explanation = engine.explain("phở bò", documents[1])
    for term, score in explanation["per_term_scores"].items():
        print(f"  '{term}' contributes: {score}")
```

---

## Part 7: BM25 Variants — What Was Fixed After 1994

The 1994 paper was not the last word. Researchers identified bugs in BM25 and proposed fixes.

### BM25+ (Lv & Zhai, CIKM 2011)

**The problem they found:**

In standard BM25, as a document gets very long, the length normalization factor can cause
the TF component to approach **zero** — even if the query term appears many times.

```
Extreme case:
  Document: 10,000 tokens (very long), contains "Wagyu" 5 times
  avgdl = 50 tokens

  TF_component ≈ 5 × 2.5 / (5 + 1.5 × (0.25 + 0.75 × 10000/50))
               ≈ 12.5 / (5 + 1.5 × 150.25)
               ≈ 12.5 / 230.4
               ≈ 0.054  ← extremely small!

  Result: A very long, relevant document about Wagyu steak gets a near-zero TF score.
  Anything else beats it.
```

**BM25+ fix:** Add a lower-bound constant `δ` (delta) to ensure every term occurrence
always contributes *something* positive to the score:

```
TF_bm25plus = f(t,D) × (k₁ + 1) / (f(t,D) + k₁ × length_factor) + δ

Default: δ = 1.0
```

```python
# BM25+ is available in rank_bm25
from rank_bm25 import BM25Plus

bm25plus = BM25Plus(tokenized_corpus, k1=1.5, b=0.75, delta=1.0)
```

### BM25L (Lv & Zhai, CIKM 2011)

A complementary fix that smooths the length normalization curve itself, rather than
adding `δ` as an afterthought. Empirically, BM25L sometimes outperforms BM25+.

```python
from rank_bm25 import BM25L

bm25l = BM25L(tokenized_corpus, k1=1.5, b=0.75, delta=0.5)
```

### When to use BM25+ or BM25L for your AI Waiter?

| Your corpus | Recommendation |
|:---|:---|
| Short menu items (1–2 sentences) | Standard BM25Okapi is fine |
| Long dish descriptions + dietary info + reviews | Use BM25+ or BM25L |
| Mix of short and long documents | BM25+ is the safe default |

---

## Part 8: The Failure Modes of BM25

*This section is the mirror of Part 8 in the Naive RAG tutorial, but focused on BM25's specific failure modes, informed by both practical experience and the BEIR benchmark (Thakur et al., 2021).*

### Failure 1: The Semantic Gap (The Fundamental Weakness)

BM25 is a **bag-of-words model**. It cannot understand meaning — only exact token matches.

> *"BM25 has no innate capability to understand query intent, synonyms, or semantic context.
>  It treats semantically related terms as entirely independent."* — BEIR Analysis (2021)

| User asks | Menu says | BM25 result |
|:---|:---|:---|
| "something filling for lunch" | "Cơm Sườn: pork ribs on steamed rice, filling and hearty" | ❌ 0 score — no word overlap |
| "I want vegetarian food" | "Gỏi Cuốn: chay, no meat" | ❌ 0 score — "vegetarian" ≠ "chay" |
| "sirloin steak" | "Bít Tết Thăn ngoại bò Wagyu" | ❌ 0 score — English ≠ Vietnamese |
| "món ăn ngon" (tasty food) | "Phở Bò: thơm ngon, đậm đà" | ⚠️ partial — "ngon" matches |

The first three cases are complete failures. The user's intent is clear, the answer exists
in the menu, but BM25 returns nothing because no tokens match.

**When does this matter most for your AI Waiter?**
- Tourists unfamiliar with Vietnamese food names
- Users describing dietary needs in general terms
- Any semantic or intent-based query ("romantic food", "good for sharing")

---

### Failure 2: Word Form Blindness (No Morphological Awareness)

BM25 matches tokens **exactly** — it knows nothing about word morphology.

```
Example:
  Menu: "bò nướng bơ tỏi"   (grilled beef with garlic butter)

  Query "bò nướng" → Matches ✅
  Query "nướng bò" → Matches ✅ (bag of words is unordered)
  Query "bò nướng?" (with punctuation) → MAY FAIL if punctuation not stripped
  Query "Bò Nướng" (title case) → FAILS without lowercasing 
  Query "grilled beef"  → ❌ FAILS — Vietnamese ≠ English
```

**Fix:** Always normalize: lowercase, strip punctuation, strip accents if needed.

```python
import unicodedata
import re

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

### Failure 3: The Long Document Penalization Bug (Pre-BM25+)

Standard BM25 can **penalize** long, highly relevant documents.

*Formally proven by Lv & Zhai (2011), CIKM.*

```
Scenario: Your menu has a detailed item description page

Document A (short): "Bít tết Wagyu. 250,000 VND."
  → 5 tokens, mentions "Wagyu" 1 time

Document B (long, detailed): "Bít tết bò Wagyu chuẩn A5 Nhật Bản. Thịt bò Wagyu 
  nổi tiếng với vân mỡ đặc biệt (marbling). Giá 250,000 VND. Phục vụ cùng khoai tây
  chiên, salad rau xanh, và nước chấm đặc biệt của nhà hàng."
  → 40 tokens, mentions "Wagyu" 2 times

With standard BM25 (b=0.75):
  Document A gets higher score! (shorter → lower denominator → higher TF score)
  Document B, despite having MORE information about Wagyu, is penalized.

→ Use BM25+ (rank_bm25.BM25Plus) with delta=1.0 to fix this.
```

---

### Failure 4: No Positional Information (Bag of Words)

BM25 treats a document as an **unordered set of tokens**. Word order is invisible.

```
Document 1: "Bò không ngon, gà ngon hơn"  (beef is not tasty, chicken is better)
Document 2: "Bò ngon, gà không ngon hơn"  (beef is tasty, chicken is not)

Query: "bò ngon"

BM25 Score(Document 1) = BM25 Score(Document 2)  ← IDENTICAL!
```

BM25 cannot detect negation, cannot understand phrase proximity, and cannot
distinguish "A is not B" from "A is B".

**Impact on your AI Waiter:**
- "Gà không cay" (chicken NOT spicy) vs "Gà cay" (spicy chicken) → same BM25 score
- "Món nào không có đậu phộng" (which dishes have NO peanuts) → BM25 matches any doc with "đậu phộng"

---

### Failure 5: Hyperparameter Brittleness

BM25 with wrong `k₁` and `b` can underperform a naive TF-IDF:

```
Corpus: All menu items are one sentence (very uniform length)
Parameter: b=0.75 (designed for varied lengths)
Effect: Length normalization divides uniformly — no benefit, possible harm

Correct: b=0.0 or b=0.25 for a uniform-length corpus
```

> *"BM25 performance is highly dependent on k₁ and b, which require manual tuning
>  based on the specific characteristics of the corpus."* — Multiple BEIR follow-up analyses

**Practical advice:** Run the BEIR-style comparison: measure Recall@5 with 3 settings of `k₁`
(1.0, 1.5, 2.0) × 3 settings of `b` (0.25, 0.50, 0.75) = 9 combinations.
Pick the best on a held-out validation set.

---

### Failure 6: The BEIR Finding — BM25 Still Loses on Semantic Tasks

The BEIR benchmark (Thakur et al., NeurIPS 2021) evaluated BM25 across 18 diverse IR datasets:

| Dataset Type | BM25 NDCG@10 | Dense BiEncoder NDCG@10 |
|:---|:---|:---|
| Exact fact-lookup (TREC-COVID) | 0.656 | 0.481 ← BM25 wins! |
| Argument retrieval (ArguAna) | 0.315 | 0.415 ← Dense wins |
| Sci-fi domain (SCIDOCS) | 0.158 | 0.122 ← BM25 wins (barely) |
| General QA (Natural Questions) | 0.329 | 0.524 ← Dense wins by far |

**Key finding:** BM25 wins when queries use the same vocabulary as documents.
Dense retrieval wins when there is a vocabulary gap or semantic reasoning is needed.

*This is the central motivation for Hybrid Search (Stage 3).*

---

### Failure Mode Summary Table

| Failure | Example in AI Waiter context | Fix |
|:---|:---|:---|
| Semantic gap | "filling food" → misses all results | → Add vector search (Stage 3) |
| Vocabulary gap | "steak" → misses "bít tết" | → Vietnamese tokenizer + metadata tagging |
| Word form blindness | "Bò" vs "bò" | → Lowercase normalization |
| Long doc penalization | Detailed description loses to summary | → Use BM25+ |
| No negation handling | "không cay" same as "cay" | → Add metadata filters |
| Parameter sensitivity | Wrong b/k₁ for your corpus | → Grid search on held-out set |

---

## Part 9: The Normalization Trap (When BM25 Meets Vector Search)

If you decide to combine BM25 and Vector Search (Native RAG) into a **Hybrid Search**, you will immediately hit a mathematical wall known as the Normalization Trap.

### The Problem: Incompatible Scales
- **Vector Search (FAISS L2 distance):** Outputs distances mapping natively to `0.0` (perfectly identical) to `2.0` (opposite). You can easily convert this to a bounded `0.0 - 1.0` similarity score using `1 / (1 + distance)`.
- **BM25:** Outputs a score from `0.0` to `Infinity`. A score could be `5.4`, `85.2`, or `402.1` depending on the document length, query length, and term rarity.

If you try to simply add them together (`Final_Score = BM25_Score + Vector_Score`), **BM25 will completely crush the Vector score.** If BM25 outputs 85.0 and Vector outputs 0.8, the vector score is mathematically irrelevant.

### The Naive Solution: Batch Sigmoid Normalization
Many developers (and common codebases) try to solve this by forcing BM25 into a `0.0 - 1.0` scale using a **Sigmoid curve**. 

```python
# A common, but brittle approach (The "DEO HIEU" problem)
def normalize_bm25_batch(scores: list[float]) -> list[float]:
    mean_score = sum(scores) / len(scores)  # Calculate average of the batch
    
    normalized = []
    for score in scores:
        # Push through a sigmoid curve centered at the mean
        exponent = -1.0 * (score - mean_score)
        sigmoid = 1.0 / (1.0 + math.exp(exponent))
        normalized.append(sigmoid)
        
    return normalized
```

**Why this is a trap (The `#TODO: DEO HIEU` problem):**
1. **Batch Dependency:** The normalization depends on the *average* of the retrieved batch. If a query returns BM25 scores of `[20, 18, 16]`, the mean is `18`. The document with score `20` gets a normalized score of `0.88`. BUT, if another document enters the database and the scores become `[100, 20, 18, 16]`, the mean shifts to `38.5`. Suddenly, that exact same document with score `20` gets pushed to a normalized score of `0.000...`. The math is highly unpredictable.
2. **Weight Tuning:** Even if successfully normalized, you now have to guess magic weights: `(BM25_Norm * 0.6) + (Vector_Norm * 0.4)`. Why 0.6? Why not 0.7? It's guesswork.

### The Solution: Reciprocal Rank Fusion (RRF)
Because mathematical normalization is so brittle, modern production RAG pipelines abandon scores entirely during the hybrid merge phase. They use **Reciprocal Rank Fusion (RRF)**, which only looks at the *rank* (1st, 2nd, 3rd) of the document, completely bypassing the Normalization Trap.

👉 **See the `rrf_tutorial.md` for the complete guide on implementing RRF.**

---

## Part 10: Benchmarking Your BM25 Against Naive RAG

Before adopting BM25, measure whether it actually improves over your Naive RAG baseline:

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Head-to-Head Test Setup ──
# Requires a labelled test set: {query: [list of relevant doc IDs]}
test_cases = {
    "bít tết wagyu":       ["item_001"],  # Should return Wagyu steak
    "phở bò":             ["item_002"],  # Should return pho
    "nước uống chanh dây": ["item_004"],  # Should return passion fruit juice
    "something warm":      ["item_002", "item_005"],  # Semantic — soup dishes
    "món ăn chay":         ["item_006", "item_007"],  # Vegetarian items
}


def recall_at_k(engine_results: list, relevant_ids: list, k: int) -> float:
    """Of the relevant docs, how many did we retrieve in top-K?"""
    retrieved_ids = [r.document.metadata.get("id") for r in engine_results[:k]]
    relevant_set  = set(relevant_ids)
    return len(set(retrieved_ids) & relevant_set) / len(relevant_set)


def precision_at_k(engine_results: list, relevant_ids: list, k: int) -> float:
    """Of the top-K retrieved docs, how many were actually relevant?"""
    retrieved_ids = [r.document.metadata.get("id") for r in engine_results[:k]]
    return len(set(retrieved_ids) & set(relevant_ids)) / k


# ── Compare BM25 vs Vector ──
embedding_model = HuggingFaceEmbeddings(model_name="AITeamVN/Vietnamese_Embedding")
faiss_store     = FAISS.from_documents(documents, embedding_model)
bm25_engine     = VietBM25()
bm25_engine.build_index(documents)

print(f"{'Query':<30} | {'BM25 R@3':^8} | {'Vector R@3':^10} | {'Winner':^8}")
print("-" * 65)

for query, relevant in test_cases.items():
    bm25_results = bm25_engine.search(query, k=3)

    faiss_raw = faiss_store.similarity_search_with_score(query, k=3)
    # Wrap in a compatible format
    from dataclasses import dataclass
    @dataclass
    class FaissResult:
        document: object
        score: float
        rank: int
    faiss_results = [FaissResult(doc, 1/(1+score), i+1) for i, (doc, score) in enumerate(faiss_raw)]

    bm25_recall   = recall_at_k(bm25_results,  relevant, k=3)
    vector_recall = recall_at_k(faiss_results, relevant, k=3)

    winner = "BM25" if bm25_recall > vector_recall else ("Vector" if vector_recall > bm25_recall else "Tie")
    print(f"{query:<30} | {bm25_recall:^8.2f} | {vector_recall:^10.2f} | {winner:^8}")
```

**What to observe:**
- Exact name queries (bít tết, phở bò, chanh dây): BM25 should win.
- Semantic queries (something warm, món ăn chay): Vector should win.
- This proves why you need BOTH — motivating Stage 3 (Hybrid Search).

---

## Part 11: When to Move Beyond BM25

| Question | If YES → consider... |
|:---|:---|
| Users search with concepts, not keywords? | → Add Vector Search (already in Stage 1) |
| Exact matches AND semantic needed simultaneously? | → Hybrid Search + RRF (Stage 3) |
| Results are retrieved but in wrong order? | → Cross-Encoder Re-Ranking (Stage 4) |
| Queries are vague or poorly worded? | → Query Rewriting (Stage 5) |
| Multi-step reasoning needed? | → Agentic RAG (Stage 6) |
| Do you even know if BM25 is helping? | → RAGAS Evaluation (Stage 7) |

---

## Summary

```
BM25
────────────────────────────────────────────────────────────────
Problem it solves:   Naive RAG misses exact product name matches (vocabulary gap)
Solution:            Score documents by term rarity (IDF) × term frequency (TF)
                     with saturation and length normalization

Key formula components:
  IDF(t):    Rare terms → high weight. Common terms → near-zero weight.
  TF_sat:    First occurrence matters most; 100th adds almost nothing (k₁ controls this)
  LenNorm:   Short dense docs win over long diluted docs (b controls this)

Defaults:
  k₁ = 1.5, b = 0.75 → works well for most text corpora
  For long docs: use BM25Plus from rank_bm25 (Lv & Zhai, 2011)

The Vietnamese tokenization rule:
  NEVER use .split() for Vietnamese text.
  ALWAYS use underthesea.word_tokenize() or pyvi.
  "bít tết" must stay as ONE token. Else "tết" matches New Year content.

Key failures:
  1. Semantic gap: "filling food" returns nothing. (Fix: add vector search)
  2. No negation: "không cay" same score as "cay". (Fix: metadata filtering)
  3. Long doc penalty: detailed descriptions lose to summaries. (Fix: BM25+)
  4. No word order: bag-of-words is position-blind. (Inherent limitation)

Key insight (BEIR 2021):
  BM25 is NOT dead. It outperforms dense retrieval on exact-match tasks
  even in 2021. The correct use is ALONGSIDE vector search, not instead of it.
  
→ Next Stage: Hybrid Search (Stage 3) — combine BM25 + Vector with RRF fusion.
```

---

## Resources

### Papers (in recommended reading order)

| # | Paper | Why read it |
|:--|:---|:---|
| 1 | **Spärck Jones (1972)** — "A Statistical Interpretation of Term Specificity." *J. of Documentation 28(1)*. | The origin of IDF — essential for understanding WHY BM25 works |
| 2 | **Robertson & Walker (1994)** — "Simple Effective Approximations to the 2-Poisson Model." *SIGIR 1994*. | The original BM25 paper — read Sections 3 and 4 |
| 3 | **Lv & Zhai (2011)** — "Lower-Bounding Term Frequency Normalization." *CIKM 2011*. ([dl.acm.org](https://dl.acm.org/doi/10.1145/2063576.2063584)) | BM25+ and BM25L — read if your corpus has long documents |
| 4 | **Thakur et al. (2021)** — "BEIR: Heterogeneous Benchmark for Zero-shot IR." *NeurIPS 2021*. ([arXiv:2104.08663](https://arxiv.org/abs/2104.08663)) | Where BM25 wins and loses vs neural search |
| 5 | **Gao et al. (2023)** — "RAG for LLMs: A Survey." ([arXiv:2312.10997](https://arxiv.org/abs/2312.10997)) | BM25's role in the modern RAG pipeline |

---

### Deep-Dive Articles

| Title | Link | Why |
|:---|:---|:---|
| "Practical BM25 — The Algorithm and Its Variables" | https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables | The best visual walkthrough with graphs |
| "BM25 — The Backbone of Search" | https://arpitbhayani.me/blogs/bm25 | Core intuition in plain English |
| Stanford IR Lecture Notes | https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html | Textbook-quality mathematical treatment |

---

### Libraries

| Library | Purpose | Install |
|:---|:---|:---|
| `rank_bm25` | BM25, BM25+, BM25L implementations | `pip install rank_bm25` |
| `underthesea` | Vietnamese NLP tokenizer | `pip install underthesea` |
| `pyvi` | Alternative Vietnamese tokenizer (faster, slightly less accurate) | `pip install pyvi` |
| `elasticsearch` | Production BM25 at scale (Lucene-based) | Docker or cloud |
