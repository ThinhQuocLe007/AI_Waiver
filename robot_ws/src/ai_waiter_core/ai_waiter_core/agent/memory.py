from langgraph.checkpoint.sqlite import SqliteSaver
from ai_waiter_core.core.config import settings
import sqlite3
import os 

def get_checkpointer():
    """
    Return a SqliteSaver that persists conversation history to a SQLite database.
    """
    db_path = str(settings.CHECKPOINTS_DB_PATH)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread= False)
    return SqliteSaver(conn)

def create_config(table_id: str):
    """
    Creates a standard LangGraph configuration dictionary.
    This tells the agent which 'thread' (table) it is currently talking to.
    """
    return {"configurable": {"thread_id": table_id}}

