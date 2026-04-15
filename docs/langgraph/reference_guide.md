# LangGraph: Core Concepts & Reference Guide

This document explains the "Why" and the terminology behind the LangGraph library. Use this as a map to understand the architecture of your AI Waiter.

---

## 1. What is a StateGraph?

A `StateGraph` is a specialized flowchart (a **Finite State Machine**) where a central "State" object is passed between different workers (Nodes). 

In LangGraph, the State is the **Single Source of Truth**. No node "owns" the data; every node just reads from and writes to the State.

### The `Annotated[List, add_messages]` Pattern
In your `AgentState`, you see this specific line:
```python
messages: Annotated[List, add_messages]
```
This is a **Reducer**.
- **Normal Python:** `state['messages'] = ['new message']` results in a list of 1 message (overwrites).
- **LangGraph Reducer:** `state['messages'] = ['new message']` results in the new message being *appended* to the old list.

This is critical because an LLM needs to see the **entire history** to keep context, and the Reducer handles that bookkeeping for you behind the scenes.

---

## 2. Nodes (The Workers)

Nodes are simply Python functions that perform a task. They take the current `State` as input and return an update to that `State`.

### Types of Nodes in your project:
1.  **Agent Node**: Calls the LLM. It returns a `BaseMessage` that gets appended to the `messages` list.
2.  **Tool Node**: Executes Python code. If the LLM generates a "Tool Call" (like `place_order`), this node runs the function, captures the return value, and writes it back to the `messages` list as a `ToolMessage`.

---

## 3. Edges (The Logic)

Edges define the direction of the flow.

### Normal Edges
`workflow.add_edge("A", "B")` means after A finishes, B **always** starts.

### Conditional Edges
`workflow.add_conditional_edges("A", logic_function)` means after A finishes, a separate function (`logic_function`) decides where to go next based on the data in the `State`.

In our project, we use `tools_condition`:
*   **If the Agent wants to use a Tool** -> Send the flow to the `tools` node.
*   **If the Agent is done talking** -> Send the flow to `END`.

---

## 4. The Checkpointer (Persistence Layer)

The Checkpointer is the "Filing Cabinet" where LangGraph stores your data after every single step.

### Key Terms:
1. **Thread ID (thread_id)**: This is a unique identifier for a conversation (we use `table_id`). It acts like a folder name. 
2. **Checkpoint**: A "save point" of the entire State. If your server crashes, LangGraph can reload the exact State from the last checkpoint using the `thread_id`.

### Why separate `app.invoke()` from `config`?
When you call the agent, you pass two things:
```python
app.invoke(
    {"messages": [...]}, # The current input (The "State")
    {"configurable": {"thread_id": "Table_5"}} # The pointer to the memory (The "Config")
)
```
This separation allows one single AI Agent (one `app`) to talk to **hundreds of tables at once** without getting their messages mixed up. Each table has its own `thread_id` in the checkpointer.

---

## 5. Summary: The Loop

1. **User** sends message with `thread_id="Table_5"`.
2. **Checkpointer** loads history for Table 5 and gives it to the **Agent Node**.
3. **Agent Node** (LLM) decides to call a Tool.
4. **Tool Node** runs the Python code and saves the result into the State.
5. **Checkpointer** saves a new checkpoint.
6. **Agent Node** (LLM) reads the result and gives a final reply.
7. **User** receives the reply.

---

## Further Reading
- [LangGraph Official Documentation](https://langchain-ai.github.io/langgraph/)
- [Conceptual Guide: Multi-Agent Systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
