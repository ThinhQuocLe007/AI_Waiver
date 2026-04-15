# Tutorial: Building an AI Agent with LangChain & LangGraph

This document explains the core workflow of building an Agentic AI Waiter. It is broken down into three main parts: Tools, Memory & Prompts, and the LangGraph Orchestrator. 

An "Agent" is just an AI model that has been given access to **Tools** and a **Looping Mechanism** to decide when to use those tools.

---

## 1. Building and Registering Tools

Large Language Models (LLMs) are essentially just text engines. If you ask `llama3.1` to "Place an order for Table 5", it can generate the text "I placed the order", but it cannot *actually* save anything to a database. 

To give the AI physical capabilities, we build **Tools**.

### Step 1: Write a normal Python class
First, we write standard Python code that does something in the real world (like interacting with SQLite).
```python
# tools/order.py
import sqlite3

class OrderDB:
    def add_order(self, table_id, items, quantity):
        # connects to database and saves the order
        # ...
        return order_id
```

### Step 2: Register with LangChain (`@tool`)
The LLM doesn't understand raw Python classes. We must wrap our function with LangChain's `@tool` decorator. 

When you use `@tool`, LangChain automatically extracts the function name, its arguments, and its **docstring** (the comment inside the function). The docstring is extremely important because it is literally the instruction sent to the LLM on *when* and *how* to use the tool.

We do this in `tools/__init__.py` to neatly package all our tools into one list.

```python
# tools/__init__.py
from langchain_core.tools import tool
from .order import OrderDB

order_db = OrderDB()

@tool
def place_order(items_description: str, quantity: int, table_id: str) -> str:
    """
    Place a food or drink order into the system.
    Requires the item name, quantity, and the customer's table_id.
    """
    order_id = order_db.add_order(table_id, items_description, quantity)
    return f"Order #{order_id} has been placed."

# We group them into a list so the Agent can load them easily
CORE_TOOLS = [place_order]
```

---

## 2. Prompts and Memory

Once the tools are built, the AI needs a personality (Prompt) and a way to remember the conversation (Memory).

### The System Prompt
The System Prompt is a set of hard rules given to the LLM before every transaction. It tells the AI who it is and provides guidelines on tool usage.

```python
# orchestrator/prompts.py
SYSTEM_PROMPT = """
You are a friendly AI Waiter. 
- You are currently serving at Table: {table_id}
- Use the `place_order` tool when the customer confirms their order.
- Never mention the names of your tools to the customer.
"""
```

### The Memory (Checkpointer)
LLMs have amnesia. If Table 5 says "I want sushi" and then says "Actually, make that 2 portions", the LLM will reply "2 portions of what?" unless we send the *entire history* every time.

LangGraph solves this with a **Checkpointer**. A Checkpointer stores the conversation in "Threads". We use `table_id` as the thread ID. 

```python
# orchestrator/memory.py
from langgraph.checkpoint.memory import MemorySaver

# The checkpointer automatically saves and loads messages
def get_checkpointer():
    return MemorySaver()

# We use this config to tell LangGraph which 'Thread' (Table) to load
def create_config(table_id: str):
    return {"configurable": {"thread_id": table_id}}
```
When LangGraph gets a message with `thread_id="Table_5"`, it pulls all past messages for Table 5, appends the new message, and sends the whole package to the LLM.

---

## 3. The LangGraph Agent

Finally, we need a system that loops between the LLM and the Tools. This is **LangGraph**.
LangGraph treats the agent as a State Machine (a flowchart).

1. **State**: The variables that get passed around the graph (our chat history).
2. **Nodes**: The workers (The LLM deciding what to do, or the Tools executing code).
3. **Edges**: The logic deciding where to go next.

Here is how the flow works:

```python
# orchestrator/agent.py
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama

# 1. Define what data is passed around
class AgentState(TypedDict):
    messages: Annotated[List, add_messages] # Automatically appends new messages
    table_id: str

# 2. Setup the LLM and connect the tools we built in Step 1
llm = ChatOllama(model="llama3.1", temperature=0.1)
llm_with_tools = llm.bind_tools(CORE_TOOLS)

# 3. Create the "Brain" Node
def call_model(state: AgentState):
    # The LLM reads the history, looks at its tools, and generates a response
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 4. Draw the Flowchart
workflow = StateGraph(AgentState)

# Create two nodes: The Brain, and the Physical Hands (Tools)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(CORE_TOOLS))

# Step A: Start by giving the message to the Brain
workflow.add_edge(START, "agent")

# Step B: Does the Brain want to use a tool?
# If yes -> send to the 'tools' node. If no -> END the conversation.
workflow.add_conditional_edges("agent", tools_condition)

# Step C: After a tool finishes, send the result BACK to the Brain so it can answer the user
workflow.add_edge("tools", "agent")

# 5. Compile the flowchart and attach the memory
app = workflow.compile(checkpointer=get_checkpointer())
```

### The Loop in Action: A Visual Guide

To understand how the AI "thinks" and "acts," look at the flowchart below. This loop continues until the AI decides it has enough information to give you a final answer.

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart TD
    START((START)) --> Agent[<b>Agent Node</b><br/><i>(Brain)</i>]
    Agent --> Condition{Tools Needed?}
    
    Condition -- "Yes (Tool Call)" --> Tools[<b>Tools Node</b><br/><i>(Action)</i>]
    Tools --> Agent
    
    Condition -- "No (Final Text)" --> END((END))

    style Agent fill:#e65100,stroke:#ffb74d,color:#fff
    style Tools fill:#1b5e20,stroke:#81c784,color:#fff
    style Condition fill:#0d47a1,stroke:#64b5f6,color:#fff
```

#### Step-by-Step Example: Ordering Sushi

When you say **"I'll have 2 sushi,"** the following hidden conversation happens inside the graph:

| Step | Player | Action | Data Behind the Scenes |
| :--- | :--- | :--- | :--- |
| **1** | **User** | Sends Message | `"I'll have 2 sushi"` |
| **2** | **Agent** | Decides to act | *Brain thinks:* "I need the `place_order` tool." |
| **3** | **Graph** | Routes to Tools | `tools_condition` triggers the `tools` node. |
| **4** | **Tools** | Executes Code | Runs `place_order(item="sushi", quantity=2)`. |
| **5** | **System** | Gets Result | Tool returns: `"Order #5 saved successfully."` |
| **6** | **Agent** | Reads Result | *Brain thinks:* "The order is done. Now I tell the user." |
| **7** | **Agent** | Final Reply | `"Great! I've placed your order for 2 sushi (#5)."` |
| **8** | **Graph** | Ends | `tools_condition` sees no more tool calls and stops. |

> [!NOTE]
> Even though steps 2 through 6 are "hidden" from the user, they are all saved in the **Memory (Checkpointer)** so the AI never forgets that it already placed the order!
