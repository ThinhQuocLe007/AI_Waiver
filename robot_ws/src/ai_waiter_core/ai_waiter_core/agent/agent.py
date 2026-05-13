from typing import Annotated, TypedDict, List, Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Updated Tool Imports
from ai_waiter_core.tools import (
    CORE_TOOLS, 
    search_menu, 
    verify_and_prepare_order, 
    confirm_order, 
    request_payment
)
from ai_waiter_core.core.config import settings
from .prompts import SYSTEM_PROMPT
from .memory import get_checkpointer
from .router import SemanticRouter

# 2. Router Initialization
router = SemanticRouter(model_name="BAAI/bge-m3")

# 3. Updated AgentState
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    table_id: str
    pending_cart: Optional[str]  # Added: Stores the cart JSON across turns

# 4. Setup Focused LLM Workers
base_llm = ChatOllama(model=settings.MODEL_NAME, temperature=0.1)

# Order worker now has both Verify and Confirm tools
order_worker   = base_llm.bind_tools([verify_and_prepare_order, confirm_order])
menu_worker    = base_llm.bind_tools([search_menu])
payment_worker = base_llm.bind_tools([request_payment])
chat_worker    = base_llm
full_agent     = base_llm.bind_tools(CORE_TOOLS)

def call_model(state: AgentState):
    """Router-aware node that picks the right worker."""
    table_id = state.get("table_id", "T1")
    last_message = state["messages"][-1].content
    
    # Inject current state into prompt
    sys_content = SYSTEM_PROMPT.format(table_id=table_id)
    if state.get("pending_cart"):
        sys_content += f"\n\nCURRENT PENDING CART: {state['pending_cart']}"
        
    sys_message = {"role": "system", "content": sys_content}
    
    # Route based on intent
    intent = router.route(last_message)
    
    if intent == "ORDER_CONFIRM":
        worker = order_worker
    elif intent == "MENU":
        worker = menu_worker
    elif intent == "PAYMENT":
        worker = payment_worker
    else:
        worker = full_agent # Complexity fallback
        
    response = worker.invoke([sys_message] + state["messages"])
    
    # Logic to capture the cart if a tool returned it (optional, 
    # but useful if you want to store it outside message history)
    return {"messages": [response]}

# 5. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(CORE_TOOLS))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=get_checkpointer())

def get_agent_app():
    return app
