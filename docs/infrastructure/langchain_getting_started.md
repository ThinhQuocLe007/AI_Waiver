# Getting Started with LangChain
> Tailored for the **AI Waiter** project — Vietnamese restaurant ordering system using LLM + RAG + Tools.

---

## 1. What is LangChain?

LangChain is a Python framework that helps you **chain together LLM calls, tools, memory, and data retrieval** into complex AI pipelines.

Key abstractions:
| Concept | What it does |
|---|---|
| **Model** | Wraps an LLM (Llama 3, Gemini, OpenAI, etc.) |
| **Prompt Template** | Structures your system/user prompts |
| **Chain** | Connects model + prompt + output into one callable unit |
| **Tool** | A Python function the LLM can decide to call |
| **Retriever** | Fetches relevant documents (for RAG) |
| **Memory** | Stores conversation history |
| **Agent** | LLM that autonomously picks which tools to use |

---

## 2. Installation

```bash
# Core LangChain
pip install langchain langchain-core langchain-community

# Local LLM via Ollama (for Llama 3)
pip install langchain-ollama

# Vector store for RAG
pip install chromadb

# Optional: OpenAI / Gemini
pip install langchain-openai langchain-google-genai
```

Make sure Ollama is running locally:
```bash
ollama serve
ollama pull llama3
```

---

## 3. Your First Chain — Hello LangChain

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the model
llm = ChatOllama(model="llama3")

# 2. Define a prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly Vietnamese restaurant waiter. Answer in Vietnamese."),
    ("human", "{input}")
])

# 3. Build a chain (prompt → model)
chain = prompt | llm

# 4. Invoke
response = chain.invoke({"input": "Cho tôi xem thực đơn"})
print(response.content)
```

> **The `|` pipe operator** connects components — this is LangChain Expression Language (LCEL).

---

## 4. Adding an Output Parser

Parse the raw LLM message into a clean string:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Bạn có món chay không?"})
print(result)  # plain string, not a Message object
```

---

## 5. RAG — Menu Search (Your Main Use Case)

RAG = **Retrieve relevant menu items → inject into prompt → LLM answers**.

### Step 1: Load & embed your menu

```python
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load menu data
loader = JSONLoader(file_path="data/menu.json", jq_schema=".[]", text_content=False)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db/menu_chroma")
```

### Step 2: Build RAG chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

rag_prompt = ChatPromptTemplate.from_template("""
Bạn là nhân viên phục vụ nhà hàng. Dựa vào thực đơn dưới đây, hãy trả lời câu hỏi của khách.

Thực đơn:
{context}

Câu hỏi: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("Bạn có món bò lúc lắc không?")
print(answer)
```

---

## 6. Tools & Function Calling (Order API)

Define Python functions as **tools** the LLM can call:

```python
from langchain_core.tools import tool

@tool
def create_order(table_number: int, item_name: str, quantity: int) -> str:
    """Creates a new food order for a table. Use when customer wants to order food."""
    # Call your actual Order API here
    return f"Đã đặt {quantity} phần {item_name} cho bàn {table_number}."

@tool
def get_bill(table_number: int) -> str:
    """Gets the total bill for a table. Use when customer asks for the check."""
    # Call your Payment API here
    return f"Tổng tiền bàn {table_number}: 250,000 VND."

# Bind tools to the LLM
llm_with_tools = llm.bind_tools([create_order, get_bill])
```

---

## 7. Agent — LLM Decides Which Tool to Call

An agent lets the LLM reason and pick tools automatically. Modern open-source models (like `llama3`) can even execute **multiple tools in parallel**!

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

tools = [create_order, get_bill]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là nhân viên phục vụ nhà hàng AI. Gọi tool khi cần xử lý đơn hàng."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)

# AgentExecutor handles the actual loop: LLM -> Tool -> LLM -> Output
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,           # See the inner "thoughts" of the agent
    max_iterations=5,       # Stop runaway loops
    handle_parsing_errors=True
)

# Example 1: Single Tool
result = agent_executor.invoke({"input": "Cho bàn 3 một tô phở bò"})
print(result["output"])

# Example 2: Parallel Tool Calling
# The LLM will call create_order for table 5 AND getting the bill for table 2 at the same time
result = agent_executor.invoke({
    "input": "Cậu kêu 1 tô phở cho bàn 5, à quên tính tiền bàn 2 giúp mình với"
})
print(result["output"])
```

## 8. Conversation Memory

Keep track of what the customer said earlier:

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Session-based memory (one history per table/session)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,  # your chain from step 3
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Same session_id = LLM remembers previous turns
chain_with_memory.invoke(
    {"input": "Cho tôi một ly cà phê sữa"},
    config={"configurable": {"session_id": "table_5"}}
)
chain_with_memory.invoke(
    {"input": "Thêm một ly nữa nhé"},  # LLM knows "ly" = "cà phê sữa"
    config={"configurable": {"session_id": "table_5"}}
)
```

---

## 9. LangGraph — Intent Routing (Advanced)

Route between **RAG branch** (menu query) and **General Chat** based on user intent:

```python
# pip install langgraph
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    input: str
    intent: str
    output: str

def classify_intent(state: State) -> State:
    result = llm.invoke(f"Classify: '{state['input']}' → 'order', 'menu_query', or 'chat'. Reply with one word.")
    state["intent"] = result.content.strip().lower()
    return state

def handle_rag(state: State) -> State:
    state["output"] = rag_chain.invoke(state["input"])
    return state

def handle_order(state: State) -> State:
    state["output"] = agent_executor.invoke({"input": state["input"]})["output"]
    return state

def handle_chat(state: State) -> State:
    state["output"] = chain.invoke({"input": state["input"]}).content
    return state

def route(state: State):
    return state["intent"]

graph = StateGraph(State)
graph.add_node("classify", classify_intent)
graph.add_node("menu_query", handle_rag)
graph.add_node("order", handle_order)
graph.add_node("chat", handle_chat)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route, {
    "menu_query": "menu_query",
    "order": "order",
    "chat": "chat",
})
graph.add_edge("menu_query", END)
graph.add_edge("order", END)
graph.add_edge("chat", END)

app = graph.compile()
result = app.invoke({"input": "Bàn 2 muốn gọi bún bò", "intent": "", "output": ""})
print(result["output"])
```

---

## 10. Project Integration Map

```
AI Waiter Pipeline              LangChain Component
────────────────────────────────────────────────────
PhoWhisper text output      →   string input to chain
Intent Classification       →   LangGraph router node
Menu Query (RAG)            →   Chroma retriever + RAG chain
Order / Payment Action      →   @tool + AgentExecutor
General Chat                →   ChatPromptTemplate | ChatOllama
Conversation Context        →   RunnableWithMessageHistory
Llama 3 (local)             →   ChatOllama(model="llama3")
```

---

## 11. Useful References

| Resource | URL |
|---|---|
| LangChain Python Docs | https://python.langchain.com/docs/ |
| LangChain Expression Language (LCEL) | https://python.langchain.com/docs/expression_language/ |
| RAG Tutorial | https://python.langchain.com/docs/use_cases/question_answering/ |
| Agents & Tools | https://python.langchain.com/docs/modules/agents/ |
| LangGraph | https://langchain-ai.github.io/langgraph/ |
| Ollama Integration | https://python.langchain.com/docs/integrations/chat/ollama/ |
| LangSmith (debugging) | https://smith.langchain.com/ |
