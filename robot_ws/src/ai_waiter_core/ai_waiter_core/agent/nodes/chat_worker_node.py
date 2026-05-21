from pathlib import Path
from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from ai_waiter_core.agent.state import AgentState
from ai_waiter_core.config import settings
from ai_waiter_core.utils import trace_latency

RESOURCES_DIR = settings.resources_dir

# Initialize ChatOllama for casual conversation (no tools bound)
_chat_model = ChatOllama(
    model=settings.WORKER_MODEL,
    temperature=0.1,
    metadata={"ls_model_name": settings.WORKER_MODEL, "ls_provider": "ollama"}
)

@trace_latency("Chat Worker Node", run_type="chain")
def chat_worker_node(state: AgentState) -> Dict[str, Any]:
    """
    Decoupled LangGraph node that manages small talk, greetings, wifi, and general restaurant inquiries.
    """
    table_id = state.get("table_id", "T1")
    
    # Load Prompts & Pluggable Skills
    system_prompt_path = RESOURCES_DIR / "system_prompts" / "waiter_agent.md"
    hospitality_skill_path = RESOURCES_DIR / "skills" / "hospitality.md"
    
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().format(table_id=table_id)
        
    with open(hospitality_skill_path, "r", encoding="utf-8") as f:
        hospitality_skill = f.read()
        
    sys_content = f"{system_prompt}\n\n{hospitality_skill}"
    sys_message = SystemMessage(content=sys_content)
    
    # Invoke LLM
    response = _chat_model.invoke([sys_message] + state["messages"])
    
    return {
        "messages": [response],
        "feedback": None
    }
