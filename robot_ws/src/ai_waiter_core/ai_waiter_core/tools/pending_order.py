import re
from typing import List
from langchain_core.tools import tool
from ai_waiter_core.core.schemas.order import (
    OrderItem, OrderItemInput, Cart, VerifyOrderInput
)
from .search.menu_manager import MenuManager
from .search.hybrid_retriever import RetrieverManager

# Initialize both for Exact + Fuzzy matching
menu = MenuManager()
retriever = RetrieverManager()
retriever.load_database()

@tool
def verify_and_prepare_order(items_request: List[OrderItemInput]) -> str:
    """
    Validates items against the menu. Supports partial names and fuzzy matching.
    Always returns the canonical name and price from the menu.
    """
    items_list = []
    missing_items = []
    suggestions = []

    for item in items_request:
        # 1. Try Exact Match first
        price = menu.get_price(item.name)
        
        if price is not None:
            # Exact match found
            items_list.append(OrderItem(
                name=item.name,
                quantity=item.quantity,
                price=price,
                special_requests=item.special_requests
            ))
        else:
            # 2. Backup: Ambiguity Logic
            # Find suggestions to let the LLM ask the user for clarification.
            search_results = retriever.hybrid_search(item.name, k=3)
            if search_results:
                suggested_names = [res.document.metadata.get('name', '') for res in search_results]
                # Filter out empty names
                suggested_names = [name for name in suggested_names if name]
                
                if suggested_names:
                    suggestions.append(f"'{item.name}' not found. Closest matches: {', '.join(suggested_names)}")
                else:
                    missing_items.append(item.name)
            else:
                missing_items.append(item.name)

    # 3. Return Errors / Ambiguities to the LLM
    if missing_items or suggestions:
        error_msg = "ERROR: Order verification failed.\n"
        if missing_items:
            error_msg += f"- Missing completely: {', '.join(missing_items)}\n"
        if suggestions:
            error_msg += f"- Ambiguous items:\n  * " + "\n  * ".join(suggestions) + "\n"
        error_msg += "Please clarify with the user or use the exact names from the suggestions."
        return error_msg

    cart = Cart(items=items_list, total_price=0.0)
    cart.total_price = cart.calculate_total()

    # We return the FIXED names so the AI knows exactly what it's confirming
    return f"PENDING_CART: {cart.model_dump_json()}"
