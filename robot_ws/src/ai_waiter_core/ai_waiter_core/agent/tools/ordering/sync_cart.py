from typing import List
from langchain_core.tools import tool
from ai_waiter_core.schemas.order import OrderItem, Cart
from ai_waiter_core.utils import trace_latency

@tool
@trace_latency("Sync Cart Tool", run_type="tool")
def sync_cart(items: List[OrderItem]) -> str:
    """
    Synchronizes the active cart with the provided items.
    Use this to draft or update the cart. ALWAYS pass the ENTIRE updated list of items.
    """
    cart = Cart(items=items)
    
    # In a full implementation, we calculate price here via MenuManager
    # For now, we return the parsed cart as a JSON string flag.
    # The LangGraph will intercept this flag and update state["active_cart"] cleanly.
    return f"SYNC_CART_SUCCESS: {cart.model_dump_json()}"
