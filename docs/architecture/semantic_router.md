# Semantic Router — Deep Dive Tutorial

> A comprehensive guide to **Semantic Router**: what it is, how it works under the hood,
> how it integrates with LLMs and LangGraph, and — critically — when you actually need it.
>
> **Library Repository:** [aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)

---

## Part 1: Your Current Architecture — How `bind_tools()` Works

Before understanding Semantic Router, you must understand exactly what your AI Waiter
is doing right now.

### Your current `agent.py` (simplified)

```python
# ai_waiter_core/orchestrator/agent.py

llm = ChatOllama(model="llama3.1", temperature=0.1)
llm_with_tools = llm.bind_tools([search_menu, place_order, request_payment])

def call_model(state):
    sys_message = {"role": "system", "content": SYSTEM_PROMPT.format(table_id=...)}
    response = llm_with_tools.invoke([sys_message] + state["messages"])
    return {"messages": [response]}

# LangGraph flow:
# START → agent (call_model) → tools_condition → tools → agent → END
```

### What happens under the hood on EVERY request

When you call `llm.bind_tools([search_menu, place_order, request_payment])`, LangChain
reads the **docstrings** and **Pydantic type hints** of each tool and converts them into
JSON Schema. For example, `place_order` becomes:

```json
{
  "name": "place_order",
  "description": "Use this tool to place an order for the user.",
  "parameters": {
    "type": "object",
    "properties": {
      "dish_name": {"type": "string", "description": "The exact name of the food..."},
      "quantity":  {"type": "integer", "description": "Number of portions..."}
    },
    "required": ["dish_name"]
  }
}
```

**All 3 tool schemas are secretly injected into the system prompt on EVERY request.**
So when the user just says *"Xin chào"*, the LLM actually receives:

```
What the LLM receives internally:
──────────────────────────────────────────────────
[SYSTEM PROMPT]
You are a friendly AI Waiter... (your SYSTEM_PROMPT)

You have access to the following tools:
1. {"name": "search_menu", "description": "...", "parameters": {...}}
2. {"name": "place_order", "description": "...", "parameters": {...}}
3. {"name": "request_payment", "description": "...", "parameters": {...}}

If you want to use a tool, respond with JSON. Otherwise respond normally.

[USER MESSAGE]
Xin chào
──────────────────────────────────────────────────
```

The LLM spends **1–3 seconds** reading all 3 tool schemas, thinking about each one,
and then deciding: *"None of these tools are relevant. I'll just say Chào bạn."*

### How `bind_tools` handles multiple tool calls

The LLM does **not** execute your Python functions. It outputs a **JSON Array** of
tool call instructions. For example, if the user says *"Cho 1 phở bò và thanh toán"*:

```json
[
  {"name": "place_order", "arguments": {"dish_name": "phở bò", "quantity": 1}},
  {"name": "request_payment", "arguments": {"table_id": "T1"}}
]
```

LangGraph's `ToolNode` then loops through this array, executes both Python functions,
collects both results, and sends them back to the LLM to write the final human response.

### Your current cost per request (with 3 tools)

| Component | Token overhead | Latency |
|:---|:---|:---|
| `SYSTEM_PROMPT` | ~200 tokens | — |
| 3 tool schemas | ~150 tokens | — |
| LLM thinking time | — | 1–3 seconds |
| **Total per request** | **~350 tokens** | **1–3 seconds** |

With only 3 tools, this is **perfectly acceptable**. The overhead is small.

---

## Part 2: The Scaling Problem — When `bind_tools` Breaks Down

Your current system has 3 tools. But what happens as you expand?

### Realistic tool expansion for AI Waiter

| Phase | Tools | Examples |
|:---|:---|:---|
| **Now (3 tools)** | `search_menu`, `place_order`, `request_payment` | Current state |
| **Phase 2 (6 tools)** | + `modify_order`, `cancel_order`, `check_order_status` | Order management |
| **Phase 3 (10 tools)** | + `call_human_waiter`, `get_restaurant_info`, `reserve_table`, `submit_feedback` | Full service |
| **Phase 4 (15+ tools)** | + `navigate_robot`, `check_inventory`, `apply_coupon`, `split_bill`, `translate_menu` | Production robot |

### The math of scaling

| Tools | Schema tokens per request | LLM behavior |
|:---|:---|:---|
| 3 | ~150 tokens | ✅ Works perfectly |
| 6 | ~300 tokens | ✅ Still fine |
| 10 | ~500 tokens | ⚠️ LLM starts getting confused, occasionally picks wrong tool |
| 15 | ~750 tokens | ❌ Significant hallucination — LLM invents tool calls that don't exist |
| 20+ | ~1000+ tokens | ❌ Broken — LLM can't reliably parse the massive schema list |

### The three failure modes at scale

**1. Tool Hallucination:** The LLM invents a tool that doesn't exist.
```
User: "Bạn có wifi không?"
LLM output: {"name": "check_wifi_status", "arguments": {}}   ← This tool doesn't exist!
```

**2. Wrong Tool Selection:** With too many similar tools, the LLM picks the wrong one.
```
User: "Hủy phở bò"
LLM calls: place_order("phở bò")   ← Should have called cancel_order!
```

**3. Parameter Leakage:** The LLM mixes parameters between tools.
```
User: "Gọi 2 phở bò cho bàn 5"
LLM calls: search_menu(query="2 phở bò bàn 5")   ← Wrong tool, leaked order params into search
```

---

## Part 3: The Solution — Semantic Router as a Pre-Filter

**Semantic Router** is not a replacement for `bind_tools`. It is a **pre-filter** that
sits in front of the LLM and decides which *subset* of tools the LLM should see.

### The key insight

Instead of giving the LLM ALL 15 tools on every request, you:
1. Use Semantic Router (~5ms) to classify the intent category.
2. Only give the LLM the 1–3 tools relevant to that category.

```
WITHOUT Semantic Router (current):
    User → LLM reads 15 tool schemas → decides → executes
    Cost: 750+ tokens, 2-4 seconds, high error rate

WITH Semantic Router:
    User → Semantic Router (5ms) → "This is an ORDER"
         → LLM reads only 2 order tools → decides → executes
    Cost: 100 tokens, 1-2 seconds, high accuracy
```

---

## Part 4: What is Semantic Router?

Semantic Router uses **Embeddings** (vectors) to classify intent in milliseconds.
No LLM is involved in the routing decision.

### How it works:

1. **You define Routes** with 5–10 example sentences each. This is NOT training —
   you just write Python code.
2. **On server startup**, the library converts all examples into vectors (~few KB in RAM).
3. **On each request**, the library converts the user's input into a vector.
4. **Cosine similarity** determines which Route the input is closest to.
5. **If similarity passes a threshold**, it returns that Route. Otherwise returns `None`.

> **"Do I need to train it?"**
> No. You just add example strings to a Python list. On startup, Semantic Router
> converts them to vectors automatically. Zero epochs, zero PyTorch, zero GPUs.

### Available Encoder Models (runs on CPU via ONNX)

| Model Name | Size | Speed | Vietnamese | Best For |
|:---|:---|:---|:---|:---|
| `BAAI/bge-small-en-v1.5` | 133MB | ~5ms | ⚠️ Basic | English-first projects |
| `sentence-transformers/all-MiniLM-L6-v2` | 80MB | ~3ms | ❌ Weak | Ultra-fast English only |
| **`BAAI/bge-m3`** | 568MB | ~15ms | ✅ Excellent | **Best for Vietnamese** |
| `intfloat/multilingual-e5-small` | 118MB | ~5ms | ✅ Good | Fast multilingual balance |

---

## Part 5: What Semantic Router CANNOT Do

### ❌ It cannot handle multi-tool queries

If the user says: *"Cho tôi 1 phở bò VÀ thanh toán bàn 5"*

Semantic Router picks **one** Route — whichever scores highest. It cannot say
"This needs both ORDER and PAYMENT". For multi-tool queries, you must fall back
to the full `bind_tools` path.

### ❌ It cannot extract parameters

Semantic Router can tell you `Route = ORDER`. But it **cannot** extract:
`dish_name = "phở bò"`, `quantity = 2`, `special_requests = "không hành"`.
You still need an LLM with `bind_tools` for parameter extraction.

### ❌ It cannot reason

It cannot chain searches, filter results, or handle conditional logic.
That requires an LLM Agent (your current LangGraph setup).

---

## Part 6: Integration — Semantic Router + Your AI Waiter

### The Architecture

```
[User Message]
      │
      ▼
┌─────────────────────┐
│  Semantic Router    │  ← ~5ms, NO LLM used
│  (The Pre-Filter)   │
└────────┬────────────┘
         │
    ┌────┼──────────────────┐
    │    │                  │
    ▼    ▼                  ▼
 ORDER  CHAT            COMPLEX / None
    │    │                  │
    ▼    ▼                  ▼
┌──────┐ ┌──────┐   ┌───────────────┐
│ LLM  │ │ LLM  │   │ LLM Agent     │
│ with │ │ (no  │   │ with ALL      │
│ order│ │tools)│   │ tools via     │
│ tools│ │      │   │ bind_tools()  │
│ only │ │      │   │               │
└──────┘ └──────┘   └───────────────┘
```

### The Complete Code

```python
import time
from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.layer import RouteLayer
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════
# 1. DEFINE YOUR TOOLS (same as your current ai_waiter_core)
# ═══════════════════════════════════════════════════════════════

class OrderParams(BaseModel):
    dish_name: str = Field(description="The exact name of the food or drink")
    quantity: int = Field(default=1, description="Number of portions. Default 1.")

@tool(args_schema=OrderParams)
def place_order(dish_name: str, quantity: int) -> str:
    """Place an order for food or drinks at the restaurant."""
    return f"Ordered {quantity}x {dish_name}."

@tool
def search_menu(query: str) -> str:
    """Search the restaurant menu for dishes, prices, or ingredients."""
    return f"Found results for: {query}"

@tool
def request_payment(table_id: str) -> str:
    """Generate a payment QR code for the customer's bill."""
    return f"Payment requested for table {table_id}."

# ═══════════════════════════════════════════════════════════════
# 2. SETUP SEMANTIC ROUTER (The Pre-Filter)
# ═══════════════════════════════════════════════════════════════

order_route = Route(
    name="ORDER",
    utterances=[
        "Cho tôi 2 bát phở bò",
        "Gọi 1 cơm sườn",
        "Thêm nước cho bàn 5",
        "Tôi muốn gọi món",
        "Đặt 3 ly trà đá",
    ],
)

menu_route = Route(
    name="MENU",
    utterances=[
        "Có món chay không?",
        "Phở bò giá bao nhiêu?",
        "Cho tôi xem thực đơn",
        "Món nào ngon nhất?",
        "Có nước ép không?",
    ],
)

chat_route = Route(
    name="CHAT",
    utterances=[
        "Xin chào",
        "Cảm ơn rất nhiều",
        "Mấy giờ đóng cửa?",
        "Wifi password là gì?",
        "Nhà vệ sinh ở đâu?",
    ],
)

payment_route = Route(
    name="PAYMENT",
    utterances=[
        "Tính tiền giùm",
        "Thanh toán bàn 5",
        "Cho tôi bill",
        "Tôi muốn trả tiền",
    ],
)

complex_route = Route(
    name="COMPLEX",
    utterances=[
        "Cho tôi 1 phở bò VÀ thanh toán luôn",
        "Vừa gọi món vừa tính tiền",
        "Hủy phở rồi gọi cơm thay",
    ],
)

router = RouteLayer(
    encoder=FastEmbedEncoder(name="BAAI/bge-m3"),
    routes=[order_route, menu_route, chat_route, payment_route, complex_route],
)

# ═══════════════════════════════════════════════════════════════
# 3. SETUP LLM "WORKERS"
# ═══════════════════════════════════════════════════════════════

# The BASE model — loaded ONCE in Ollama, using ~8GB VRAM
base_llm = ChatOllama(model="llama3.1", temperature=0.1)

# Each "worker" is just a lightweight config (~2KB each in Python RAM).
# They ALL share the same Ollama model — zero extra GPU/RAM cost.
order_llm   = base_llm.bind_tools([place_order])          # 1 tool schema
menu_llm    = base_llm.bind_tools([search_menu])           # 1 tool schema
payment_llm = base_llm.bind_tools([request_payment])       # 1 tool schema
chat_llm    = base_llm                                     # 0 tool schemas (fastest)
agent_llm   = base_llm.bind_tools([place_order, search_menu, request_payment])  # all tools

# ═══════════════════════════════════════════════════════════════
# 4. THE PIPELINE — Semantic Router decides, LLM executes
# ═══════════════════════════════════════════════════════════════

SYSTEM_MSG = {"role": "system", "content": "You are a friendly AI Waiter at a restaurant."}

def process(user_input: str) -> str:

    # STEP 1: Pre-Filter (~5ms, zero LLM cost)
    start = time.time()
    route = router(user_input)
    router_ms = (time.time() - start) * 1000
    print(f"[Router] {route.name or 'None'} in {router_ms:.1f}ms")

    # STEP 2: Route to the focused LLM Worker
    if route.name == "ORDER":
        # LLM sees only 1 tool → fast + accurate parameter extraction
        response = order_llm.invoke([SYSTEM_MSG, {"role": "user", "content": user_input}])
        if response.tool_calls:
            params = response.tool_calls[0]["args"]
            return place_order.invoke(params)
        return response.content

    elif route.name == "MENU":
        # LLM sees only the search tool → focused retrieval
        response = menu_llm.invoke([SYSTEM_MSG, {"role": "user", "content": user_input}])
        if response.tool_calls:
            params = response.tool_calls[0]["args"]
            return search_menu.invoke(params)
        return response.content

    elif route.name == "PAYMENT":
        response = payment_llm.invoke([SYSTEM_MSG, {"role": "user", "content": user_input}])
        if response.tool_calls:
            params = response.tool_calls[0]["args"]
            return request_payment.invoke(params)
        return response.content

    elif route.name == "CHAT":
        # LLM sees ZERO tools → responds as fast as possible
        return chat_llm.invoke([SYSTEM_MSG, {"role": "user", "content": user_input}]).content

    elif route.name == "COMPLEX":
        # Multi-tool query → fall back to the full bind_tools agent
        return agent_llm.invoke([SYSTEM_MSG, {"role": "user", "content": user_input}]).content

    else:
        # Semantic Router returned None → out of domain or uncertain
        # Fall back to agent with all tools as a safety net
        return agent_llm.invoke([SYSTEM_MSG, {"role": "user", "content": user_input}]).content
```

---

## Part 7: The RAM Question — Does This Use Extra Memory?

**No.** Each LLM "worker" is NOT a separate model loaded into memory.

When you write:
```python
base_llm    = ChatOllama(model="llama3.1")
order_llm   = base_llm.bind_tools([place_order])
menu_llm    = base_llm.bind_tools([search_menu])
chat_llm    = base_llm
```

| Object | What it really is | RAM cost |
|:---|:---|:---|
| Ollama Server | The actual Llama 3.1 model weights | ~8GB (loaded **once**) |
| `base_llm` | A Python HTTP client pointing to `localhost:11434` | ~1 KB |
| `order_llm` | Same client + a dict `{"tools": [order schema]}` | ~2 KB |
| `menu_llm` | Same client + a dict `{"tools": [search schema]}` | ~2 KB |
| `chat_llm` | Same reference as `base_llm` | 0 KB |

**Total extra RAM for 5 workers: ~7 KB.** All workers send HTTP requests to the
**same** Ollama server. The only difference is the JSON payload in the request body.

You could create 100 workers and the total extra memory would still be negligible.

---

## Part 8: The Optimal Point — When to Adopt Semantic Router

This is the most important decision. Here is the data-driven guide:

### The Decision Matrix

| Your Situation | Recommendation | Why |
|:---|:---|:---|
| **3 tools** (your current state) | ❌ Don't add Semantic Router yet | Overhead is minimal (~150 tokens). `bind_tools` works perfectly. |
| **5–6 tools** | ⚠️ Consider it | This is the **tipping point**. LLM starts reading 300+ tokens of schemas per request. If latency matters, add Semantic Router. |
| **8–10 tools** | ✅ Strongly recommended | LLM confusion starts. Wrong tool selection occurs ~5-10% of the time. Semantic Router prevents this. |
| **15+ tools** | ✅ Mandatory | Without pre-filtering, the LLM will hallucinate non-existent tools regularly. |

### The Optimal Adoption Point

```
Tools:  1  2  3  4  5  6  7  8  9  10  11  12  ...  20
        ──────────────  ───────────────  ──────────────
        bind_tools      TIPPING POINT    Semantic Router
        works fine      (adopt here)     is mandatory

                        ▲
                        │
                  YOU ARE HERE (3 tools)
                  Plan for expansion → adopt at 5-6 tools
```

### What to do RIGHT NOW (with 3 tools)

1. **Keep your current `agent.py` as-is.** It is clean and correct.
2. **Delete the unused `ROUTER_PROMPT`** from `prompts.py` — it is dead code that is
   never imported in `agent.py`.
3. **When you add your 4th–5th tool**, revisit this document and implement Semantic Router.

### What to do at EXPANSION (5+ tools)

Group your tools into domains, and create one Semantic Router Route per domain:

```python
# Future tool groups:
TOOL_GROUPS = {
    "ORDER":    [place_order, modify_order, cancel_order],
    "MENU":     [search_menu, get_recommendations],
    "PAYMENT":  [request_payment, split_bill, apply_coupon],
    "SERVICE":  [call_human_waiter, submit_feedback],
    "ROBOT":    [navigate_robot, check_inventory],
}

# Each route maps to an LLM worker with only its domain's tools:
order_llm   = base_llm.bind_tools(TOOL_GROUPS["ORDER"])    # 3 tools, not 15
menu_llm    = base_llm.bind_tools(TOOL_GROUPS["MENU"])     # 2 tools, not 15
payment_llm = base_llm.bind_tools(TOOL_GROUPS["PAYMENT"])  # 3 tools, not 15
```

This keeps each LLM call focused and accurate, even at 15+ total tools.

---

## Part 9: Semantic Router vs `bind_tools` — Summary Table

| Feature | `bind_tools` only (current) | Semantic Router + `bind_tools` |
|:---|:---|:---|
| **Routing decision** | LLM decides (1–3s) | Embedding model decides (5ms) |
| **Tool schemas per request** | ALL tools sent every time | Only relevant tools sent |
| **Multi-tool queries** | ✅ LLM handles natively | Falls back to full agent |
| **Parameter extraction** | ✅ LLM extracts via Pydantic | ✅ Same (LLM still does this) |
| **Memory cost** | ~1 KB | +7 KB (negligible) |
| **Best at N tools** | 1–6 tools | 5+ tools |
| **Latency for "Xin chào"** | 1–3 seconds | ~50ms (no LLM call needed) |

### The Golden Rule

> **Use Semantic Router to decide WHAT to do.**
> **Use the LLM to decide HOW to do it.**

Semantic Router says: *"This is an order."*
The LLM says: *"The order is for 2 bowls of phở bò, no onions."*

---

## Resources

| Resource | Link | Why |
|:---|:---|:---|
| Semantic Router GitHub | https://github.com/aurelio-labs/semantic-router | Official repo + examples |
| FastEmbed Models List | https://qdrant.github.io/fastembed/examples/Supported_Models/ | All available encoder models |
| Semantic Router Docs | https://docs.aurelio.ai/semantic-router/ | Full API documentation |
| BAAI/bge-m3 (best multilingual) | https://huggingface.co/BAAI/bge-m3 | Recommended encoder for Vietnamese |
