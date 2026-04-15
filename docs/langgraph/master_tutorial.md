# LangGraph Master Tutorial: Building the AI Waiter

This tutorial walks you through the complete lifecycle of building an agentic workflow using LangGraph. We will follow a "Bottom-Up" approach: from raw Python tools to a fully stateful, memory-aware AI Agent.

---

## Phase 1: Building the "Hands" (Tools)

Tools are how the AI interacts with the real world. In our case, this means searching the menu, placing orders, and generating payment links.

### 1.1 The Logic 
Write a clean Python function that performs an action. Use the `@tool` decorator. 
> [!IMPORTANT]
> The **Docstring** (the text in `""" ... """`) is the most important part. It is the instruction the LLM uses to decide when to call the tool.

```python
# ai_waiter_core/tools/order.py
from langchain_core.tools import tool

@tool
def place_order(item: str, quantity: int, table_id: str):
    """
    Use this tool to place an order in the restaurant database.
    Arguments:
    - item: Name of the dish.
    - quantity: Number of portions.
    - table_id: The ID of the table (e.g., 'Table_5').
    """
    # ... logic to save to SQL ...
    return f"Order for {quantity}x {item} saved for {table_id}."
```

### 1.2 The Registry
Group all your tools into a single list. This list is what we "give" to the LLM later.

```python
# ai_waiter_core/tools/__init__.py
from .order import place_order
from .search_hybrid import search_menu

CORE_TOOLS = [place_order, search_menu]
```

---

## Phase 2: Building the "Brain" & "Memory"

### 2.1 The Checkpointer (Memory)
Memory in LangGraph isn't magic—it's just a database that saves the conversation history for a specific `thread_id`.

```python
# ai_waiter_core/orchestrator/memory.py
from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer():
    return MemorySaver() # In-memory storage for now
```

### 2.2 The LLM Binder
We take our LLM (Ollama) and "bind" the tools to it. This tells the LLM: *"Hey, here are some extra buttons you can press if you need to."*

```python
from langchain_ollama import ChatOllama
from ai_waiter_core.tools import CORE_TOOLS

llm = ChatOllama(model="llama3.1")
llm_with_tools = llm.bind_tools(CORE_TOOLS)
```

---

## Phase 3: Building the "Loop" (The Graph)

LangGraph uses a flowchart logic. The **State** is the "clipboard" passed between nodes.

### 3.1 Define the State
```python
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 'add_messages' means "always append new messages, never delete old ones"
    messages: Annotated[List, add_messages]
    table_id: str
```

### 3.2 The Decision Node (The Agent)
This node calls the LLM. It looks at the history and decides to either:
1. Talk back to the user.
2. Call a tool.

```python
def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### 3.3 Assembling the Graph
We draw the arrows (Edges) between the players (Nodes).

```python
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Create the Graph blueprint
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("agent", call_model) # The Brain
workflow.add_node("tools", ToolNode(CORE_TOOLS)) # The Hands

# 3. Define Flow
workflow.add_edge(START, "agent") # Start with the brain

# 4. Add "Logic" Edges
# After "agent", check if a tool was called. If yes -> go to "tools".
workflow.add_conditional_edges("agent", tools_condition)

# After "tools" finishes, go back to the "agent" to explain the result.
workflow.add_edge("tools", "agent")

# 5. Compile
app = workflow.compile(checkpointer=get_checkpointer())
```

---

## Phase 4: Execution

To use the graph, you invoke it with a `config` that contains the `thread_id`. This is how LangGraph knows which table it is talking to!

```python
config = {"configurable": {"thread_id": "Table_01"}}
app.invoke({"messages": [("user", "Hi!")], "table_id": "T1"}, config=config)
```
