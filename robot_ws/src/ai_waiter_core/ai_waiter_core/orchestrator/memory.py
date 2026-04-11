from langgraph.checkpoint.memory import MemorySaver

def get_checkpointer():
    """
    Returns a LangGraph MemorySaver for in-memory persistence.
    We can swap this for a SQLite-based checkpointer later (e.g., SqliteSaver).
    """
    return MemorySaver()

def create_config(table_id: str):
    """
    Creates a standard LangGraph configuration dictionary.
    This tells the agent which 'thread' (table) it is currently talking to.
    """
    return {"configurable": {"thread_id": table_id}}

# You can add more complex memory-merging logic here if needed,
# such as summarizing long conversations to save context window.
