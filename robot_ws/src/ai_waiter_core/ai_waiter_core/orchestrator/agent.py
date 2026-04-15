from typing import Annotated, TypedDict, List
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ai_waiter_core.tools import CORE_TOOLS
from ai_waiter_core.core.config import settings
from .prompts import SYSTEM_PROMPT
from .memory import get_checkpointer

# 1. Define the State
class AgentState(TypedDict):
    """
    State for the AI Waiter.
    'messages' is a list of ChatMessages (user, AI, or tool results).
    'table_id' helps us maintain context.
    """
    messages: Annotated[List, add_messages]
    table_id: str

# 2. Setup the LLM and Bind Tools
# We use llama3.1 as the brain of our agent.
llm = ChatOllama(model=settings.MODEL_NAME, temperature=0.1)
llm_with_tools = llm.bind_tools(CORE_TOOLS)

# 3. Define the Agent Logic
def call_model(state: AgentState):
    """
    Main node that decides what to say or which tool to call.
    """
    table_id = state.get("table_id", "T1")
    # Inject table_id into the system prompt for better context awareness
    sys_message = {"role": "system", "content": SYSTEM_PROMPT.format(table_id=table_id)}
    
    # We prepend the system prompt to the existing history
    response = llm_with_tools.invoke([sys_message] + state["messages"])
    return {"messages": [response]}

# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add our core nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(CORE_TOOLS))

# Define the flow
workflow.add_edge(START, "agent")

# If the agent calls a tool, route to 'tools' node, then back to 'agent'.
# Otherwise, we finish the conversation.
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# 5. Compile with Memory Checkpointer
app = workflow.compile(checkpointer=get_checkpointer())

def get_agent_app():
    """Returns the compiled agent application."""
    return app  