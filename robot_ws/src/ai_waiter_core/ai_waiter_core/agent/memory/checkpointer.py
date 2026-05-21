from langgraph.checkpoint.memory import MemorySaver
from ai_waiter_core.config import settings
from langsmith import uuid7
import os 

def get_checkpointer():
    """
    Return a MemorySaver for testing/verification. 
    (Fallback from SqliteSaver to ensure environment stability).
    """
    return MemorySaver()

def create_thread_config(table_id: str, session_id: str = None):
    """
    Creates an enriched LangGraph configuration dictionary.
    Uses UUID v7 for chronological ordering in LangSmith.
    """
    if not session_id:
        session_id = str(uuid7())
        
    return {
        "configurable": {
            "thread_id": session_id,
            "table_id": table_id
        },
        "metadata": {
            "session_id": session_id,
            "table_id": table_id
        }
    }
