from typing import List
from langchain_core.tools import tool
from ai_waiter_core.schemas.order import OrderItem
from ai_waiter_core.agent.tools.ordering.order_db import OrderDB
from ai_waiter_core.utils import trace_latency

# Initialize the DB Manager
db_manager = OrderDB()

@tool
@trace_latency("Order Confirmation Tool", run_type="tool")
def confirm_order(table_id: str, items: List[OrderItem]) -> str:
    """
    Finalizes and saves the order to the database. 
    ONLY call this after the user has explicitly agreed to the final summary.
    
    table_id: The table ID (e.g., 'T1')
    items: The final list of items in the customer's cart.
    """
    try:
        # LangChain natively parses the items into Pydantic models.
        # We simply convert them back to dicts for the database.
        cart_data = {"items": [item.model_dump() for item in items]}
            
        order_id = db_manager.add_order(table_id, cart_data)
        
        if order_id:
            return f"CONFIRM_ORDER_SUCCESS: Order #{order_id} has been placed successfully for Table {table_id}."
        else:
            return "ERROR: There was a database error while saving your order."
    except Exception as e:
        return f"ERROR: Failed to process cart data: {str(e)}"
