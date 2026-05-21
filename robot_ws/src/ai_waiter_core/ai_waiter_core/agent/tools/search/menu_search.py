from langchain_core.tools import tool
from .hybrid_retriever import RetrieverManager

# Initialize once
retriever = RetrieverManager()
retriever.load_database()

from pydantic import BaseModel, Field

class SearchMenuInput(BaseModel):
    query: str = Field(..., description="The name of the dish or keywords to search for.")

from ai_waiter_core.utils import trace_latency

@tool(args_schema=SearchMenuInput)
@trace_latency("Menu Search Tool", run_type="tool")
def search_menu(query: str) -> str:
    """
    Search the restaurant menu for food, drinks, prices, and ingredients.
    Use this for discovery and general questions about what we serve.
    """
    results = retriever.hybrid_search(query, k=3)
    if not results:
        return "No matching menu items found. Please try a different keywords."
    
    return "\n---\n".join([f"[{r.doc_type}] {r.document.page_content}" for r in results])
