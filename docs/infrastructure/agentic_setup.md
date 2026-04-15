# Agentic Architecture Setup (Hybrid Supervisor)
> How to implement the Fast Router + Parallel Tool Calling Agent for your AI Waiter using open-source models (Llama 3/Qwen).

---

## 1. Goal: The 1-Second Response Pipeline

We want to build the architecture discussed in `../architecture/analysis.md`.
1. **Supervisor (Fast Router):** Intercepts simple chats instantly.
2. **Tool Agent (LLM):** Handles complex orders by calling tools (sometimes in parallel).

### File Structure (`ai_waiter_core/orchestrator/`)
- `tools.py` — Your restaurant functions (order, search)
- `agent.py` — The core LangChain waitstaff agent 
- `supervisor.py` — The hybrid router entry point

---

## 2. Definining Your Tools (`tools.py`)

Using `@tool`. The docstrings must be extremely clear so `llama3` understands *when* to use them.

```python
from langchain_core.tools import tool

@tool
def search_menu(query: str) -> str:
    """
    Search the restaurant menu database.
    Use this to find dishes, prices, ingredients, or food suggestions.
    Input should be a short search term (e.g. "món chay", "giá phở bò").
    """
    # ... your ChromaDB or search logic here ...
    return f"Kết quả tìm kiếm cho '{query}': Phở bò 55k, Cơm sườn 50k, Salad chay 45k."

@tool
def place_order(table_id: int, item_name: str, quantity: int) -> str:
    """
    Place a food order into the system.
    ONLY use this when the customer explicitly asks to order or add an item.
    """
    # ... your Database/ROS 2 publisher code here ...
    return f"Đã ghi nhận đặt {quantity} {item_name} cho bàn {table_id}."
```

---

## 3. Creating the Parallel Agent (`agent.py`)

Here we set up `llama3.1` to act as an Agent capable of running multiple searches or orders simultaneously.

```python
from langchain_ollama import ChatOllama
# IMPORT NOTE: In LangChain v0.2+, AgentExecutor moved to langchain_classic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools import search_menu, place_order

# 1. Initialize Open-Source LLM
llm = ChatOllama(model="llama3.1", temperature=0.2)

# 2. Define the Agent's Brain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là AI Waiter. Bàn: {table_id}. Có thể dùng nhiều tool cùng lúc."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

tools = [search_menu, place_order]
agent_core = create_tool_calling_agent(llm, tools, prompt)

waiter_agent = AgentExecutor(agent=agent_core, tools=tools, verbose=True)

def run_agent_workflow(user_text: str, table_id: int):
    return waiter_agent.invoke({"input": user_text, "table_id": table_id})["output"]
```

---

## 4. Building the Fast Supervisor with LangGraph (`supervisor.py`)

Instead of manual embedding loops, use **LangGraph** to create a structured "decision tree."

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from .agent import run_agent_workflow

# 1. Define the Shared State
class State(TypedDict):
    input: str
    intent: str
    output: str

# 2. Fast Router Node (The "Traffic Cop")
def classify_intent(state: State) -> State:
    llm_fast = ChatOllama(model="llama3.1", temperature=0)
    result = llm_fast.invoke(
        f"Phân loại tin nhắn: '{state['input']}'. "
        "Nếu là chào hỏi/xã giao -> 'smalltalk'. "
        "Nếu là đặt món/hỏi menu -> 'heavy_agent'. "
        "Chỉ trả về 1 từ."
    )
    state["intent"] = result.content.strip().lower()
    return state

# 3. Handle Branches
def handle_smalltalk(state: State) -> State:
    state["output"] = "Dạ, em chào anh chị ạ! Anh chị muốn xem menu hay đặt món luôn không?"
    return state

def handle_heavy_agent(state: State) -> State:
    state["output"] = run_agent_workflow(state["input"], table_id=5)
    return state

# 4. Routing Logic
def route_decision(state: State):
    return "smalltalk" if "smalltalk" in state["intent"] else "heavy_agent"

# 5. Assemble the "Brain" (Graph)
workflow = StateGraph(State)
workflow.add_node("classify", classify_intent)
workflow.add_node("smalltalk", handle_smalltalk)
workflow.add_node("heavy_agent", handle_heavy_agent)

workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_decision)
workflow.add_edge("smalltalk", END)
workflow.add_edge("heavy_agent", END)

app = workflow.compile()
```

---

## 5. Why LangGraph is a game changer for Hardware:

1.  **Stateful Memory**: The `State` object acts as a shared whiteboard.
2.  **Explicit Routing**: You save GPU power by bypassing the "Heavy Agent" for simple questions.
3.  **Scalability**: You can add a `payment` or `feedback` node without breaking the rest of the flow.
