# LLM Routing — Basic Tutorial
> Route user input to the right handler using an LLM as a classifier.

---

## 1. What is Routing?

Routing = **deciding what to do** with a user's message before processing it.

```
Customer: "Cho tôi 2 tô phở"         →  ORDER branch    →  call create_order()
Customer: "Có món chay không?"        →  MENU_QUERY      →  RAG search menu
Customer: "Nhà hàng mở mấy giờ?"     →  GENERAL_CHAT    →  LLM answers directly
```

Without routing, you'd send everything through the same pipeline — wasting time on RAG searches when the customer just wants to chat, or missing orders because the LLM treated them as questions.

---

## 2. The Simplest Router — LLM Classification

Use the LLM itself to classify user intent, then branch.

### Step 1: Define routes

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3", temperature=0)

# Router prompt
router_prompt = ChatPromptTemplate.from_template("""
Classify the following customer message into exactly ONE category.

Categories:
- ORDER: customer wants to order food or drinks
- MENU_QUERY: customer asks about menu items, prices, ingredients
- GENERAL_CHAT: greetings, restaurant info, or casual conversation

Message: "{input}"

Reply with ONLY the category name, nothing else.
""")

# Router chain
router_chain = router_prompt | llm | StrOutputParser()
```

### Step 2: Branch based on result

```python
def route_message(user_input: str):
    intent = router_chain.invoke({"input": user_input}).strip().upper()
    
    if intent == "ORDER":
        return handle_order(user_input)
    elif intent == "MENU_QUERY":
        return handle_menu_query(user_input)
    else:
        return handle_general_chat(user_input)

# Handler functions (simplified)
def handle_order(msg):
    return f"[ORDER] Processing: {msg}"

def handle_menu_query(msg):
    return f"[RAG] Searching menu for: {msg}"

def handle_general_chat(msg):
    return f"[CHAT] Responding to: {msg}"
```

### Step 3: Test it

```python
tests = [
    "Cho bàn 3 một tô phở bò",           # → ORDER
    "Bạn có món chay không?",              # → MENU_QUERY
    "Nhà hàng mở mấy giờ?",              # → GENERAL_CHAT
    "2 ly trà đá, 1 cơm sườn",           # → ORDER
    "Phở bò bao nhiêu tiền?",            # → MENU_QUERY
]

for t in tests:
    intent = router_chain.invoke({"input": t}).strip()
    print(f"{intent:15} ← {t}")
```

---

## 3. LangChain RunnableBranch (Built-in Router)

LangChain has a built-in `RunnableBranch` for conditional logic:

```python
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# Define handler chains
order_chain = ChatPromptTemplate.from_template(
    "Process this food order: {input}"
) | llm | StrOutputParser()

menu_chain = ChatPromptTemplate.from_template(
    "Search the menu and answer: {input}"
) | llm | StrOutputParser()

chat_chain = ChatPromptTemplate.from_template(
    "Have a friendly conversation: {input}"
) | llm | StrOutputParser()

# Classify first, then branch
def classify(info):
    intent = router_chain.invoke(info).strip().upper()
    return {**info, "intent": intent}

# Build the branching logic
branch = RunnableBranch(
    (lambda x: x["intent"] == "ORDER", order_chain),
    (lambda x: x["intent"] == "MENU_QUERY", menu_chain),
    chat_chain,  # default fallback
)

# Full pipeline: classify → branch
full_chain = RunnablePassthrough.assign(intent=lambda x: router_chain.invoke(x).strip().upper()) | branch

result = full_chain.invoke({"input": "Cho tôi một ly cà phê sữa"})
print(result)
```

---

## 4. LangGraph Router (Recommended for Complex Flows)

LangGraph gives you the most control — visual graph, state management, cycles.

```python
# pip install langgraph
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define state
class State(TypedDict):
    input: str
    intent: str
    output: str

# 2. Define nodes
def classify_intent(state: State) -> State:
    """LLM classifies the input into a category."""
    result = router_chain.invoke({"input": state["input"]})
    return {"intent": result.strip().upper()}

def handle_order(state: State) -> State:
    return {"output": f"🍜 Processing order: {state['input']}"}

def handle_menu(state: State) -> State:
    return {"output": f"📋 Searching menu: {state['input']}"}

def handle_chat(state: State) -> State:
    return {"output": f"💬 General response: {state['input']}"}

# 3. Define routing function
def route_by_intent(state: State) -> str:
    intent = state["intent"]
    if intent == "ORDER":
        return "order"
    elif intent == "MENU_QUERY":
        return "menu"
    else:
        return "chat"

# 4. Build graph
graph = StateGraph(State)

# Add nodes
graph.add_node("classify", classify_intent)
graph.add_node("order", handle_order)
graph.add_node("menu", handle_menu)
graph.add_node("chat", handle_chat)

# Set entry point
graph.set_entry_point("classify")

# Add conditional edges (router)
graph.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "order": "order",
        "menu": "menu",
        "chat": "chat",
    }
)

# All branches end
graph.add_edge("order", END)
graph.add_edge("menu", END)
graph.add_edge("chat", END)

# 5. Compile and run
app = graph.compile()

result = app.invoke({
    "input": "Bàn 5 gọi 2 tô phở bò tái",
    "intent": "",
    "output": ""
})
print(result["output"])
```

### Visualize the graph
```python
# Print ASCII graph
app.get_graph().print_ascii()

# Or save as image (requires graphviz)
# app.get_graph().draw_png("routing_graph.png")
```

```
         ┌──────────┐
         │ classify  │
         └────┬──────┘
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
 ┌───────┐ ┌──────┐ ┌──────┐
 │ order │ │ menu │ │ chat │
 └───┬───┘ └──┬───┘ └──┬───┘
     │        │        │
     └────────┼────────┘
              ▼
           [ END ]
```

---

## 5. Adding Sub-Routes

Your AI Waiter might need **nested routing** — e.g., within ORDER, distinguish between "new order" vs "modify order" vs "cancel order":

```python
def handle_order(state: State) -> State:
    """Sub-route within ORDER intent."""
    sub_prompt = ChatPromptTemplate.from_template("""
    Classify this order action:
    - NEW_ORDER: customer wants to order new items
    - MODIFY_ORDER: customer wants to change an existing order
    - CANCEL_ORDER: customer wants to cancel
    - GET_BILL: customer wants to pay
    
    Message: "{input}"
    Reply with ONLY the category name.
    """)
    sub_chain = sub_prompt | llm | StrOutputParser()
    sub_intent = sub_chain.invoke({"input": state["input"]}).strip()
    
    return {"output": f"🍜 [{sub_intent}] {state['input']}"}
```

---

## 6. Your AI Waiter Routing Architecture

```
                    ┌──────────────┐
  User Input  ────→ │  LLM Router  │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │   ORDER    │ │ MENU_QUERY │ │   CHAT     │
     │            │ │            │ │            │
     │ Tool Call: │ │ RAG:       │ │ Direct LLM │
     │ create_    │ │ ChromaDB   │ │ response   │
     │ order()    │ │ → context  │ │            │
     │ get_bill() │ │ → LLM     │ │            │
     └────────────┘ └────────────┘ └────────────┘
```

---

## 7. Summary

| Method | Complexity | Best For |
|---|---|---|
| Manual `if/else` after LLM classify | Simple | Quick prototype |
| `RunnableBranch` | Medium | Clean LangChain chains |
| **LangGraph** | Full control | Production, complex flows, sub-routes |

> **Problem:** All methods above use an LLM call for routing, which adds **1-3 seconds** of latency. See the **Advanced Routing** doc for faster alternatives.
