import logging
from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

from ai_waiter_core.agent.state import AgentState
from ai_waiter_core.config import settings
from ai_waiter_core.utils import trace_latency
from ai_waiter_core.utils.prompt_utils import load_prompt

# Import the Menu constants from the path you created in Step 1
from ai_waiter_core.agent.tools.utils.extract_menu import MENU_NAMES

# Import the native ordering tools
from ai_waiter_core.agent.tools.ordering.sync_cart import sync_cart
from ai_waiter_core.agent.tools.ordering.confirmation import confirm_order

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Chain building (compiled once at module level)
# ------------------------------------------------------------

def _build_order_prompt() -> ChatPromptTemplate:
    skeleton = load_prompt("order_worker_agent.md")
    hospitality = load_prompt("hospitality.md", "skills")
    
    # We combine them into a single system prompt template
    system_template = f"{skeleton}\n\n{hospitality}\n\n{{context_block}}"
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages")
    ])

_llm = ChatOllama(
    model=settings.WORKER_MODEL,
    temperature=0.1,
    metadata={"ls_model_name": settings.WORKER_MODEL, "ls_provider": "ollama"}
).bind_tools([sync_cart, confirm_order])

_order_prompt = _build_order_prompt()
_order_chain = _order_prompt | _llm

# ------------------------------------------------------------
# Context Builder Helper
# ------------------------------------------------------------

def _build_context_block(state: AgentState) -> str:
    """Cleanly assembles dynamic context without messy string concatenation."""
    menu_str = "\n".join([f"- {name}" for name in MENU_NAMES])
    
    blocks = [
        "### RESTAURANT MENU:",
        "You must strictly match items to these exact names:",
        menu_str
    ]
    
    if state.get("feedback"):
        blocks.extend([
            "",
            "### SYSTEM FEEDBACK (MANDATORY FIX):",
            state["feedback"],
            "Politely apologize and clarify with the user."
        ])
        
    if state.get("active_cart"):
        blocks.extend([
            "",
            "### CURRENT ACTIVE CART:",
            str(state["active_cart"])
        ])
        
    return "\n".join(blocks)

# ------------------------------------------------------------
# The graph node
# ------------------------------------------------------------

@trace_latency("Order Worker Node", run_type="chain")
def order_worker_node(state: AgentState) -> Dict[str, Any]:
    """
    Decoupled LangGraph node that manages order taking natively using tools.
    """
    table_id = state.get("table_id", "T1")
    order_stage = state.get("order_stage", "IDLE")
    context_block = _build_context_block(state)
    
    try:
        ai_msg = _order_chain.invoke({
            "table_id": table_id,
            "order_stage": order_stage,
            "context_block": context_block,
            "messages": state["messages"]
        })
    except Exception as e:
        logger.error(f"Order Worker Failed: {e}")
        ai_msg = AIMessage(content="Xin lỗi, em xử lý thông tin bị lỗi. Anh/chị có thể nhắc lại được không ạ?")
        
    return {
        "messages": [ai_msg],
        "feedback": None  # Clear feedback once it has been processed
    }
