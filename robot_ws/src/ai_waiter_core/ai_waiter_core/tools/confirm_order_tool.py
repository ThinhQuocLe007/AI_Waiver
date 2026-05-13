import json
import ast
import re
from langchain_core.tools import tool
from .ordering.order_db import OrderDB

# Initialize the DB Manager
db_manager = OrderDB()

@tool
def confirm_order(table_id: str, pending_cart_json: str) -> str:
    """
    Finalizes and saves the order to the database. 
    ONLY call this after the user has explicitly said 'Yes' or 'Confirm' to the summary.
    
    table_id: The table ID (e.g., 'T1')
    pending_cart_json: The JSON string of the cart (without the 'PENDING_CART:' prefix)
    """
    try:
        # 1. Clean up the string from LLM hallucinations
        cleaned_str = re.sub(r'<[^>]+>', '', pending_cart_json)
        cleaned_str = cleaned_str.replace("PENDING_CART:", "").strip()
        
        # 2. Try JSON parse first
        try:
            cart_data = json.loads(cleaned_str)
        except json.JSONDecodeError:
            # 3. Fallback for single quotes and python primitives (like None)
            cart_data = ast.literal_eval(cleaned_str)
            
        order_id = db_manager.add_order(table_id, cart_data)
        
        if order_id:
            return f"SUCCESS: Order #{order_id} has been placed successfully for Table {table_id}."
        else:
            return "ERROR: There was a database error while saving your order."
    except Exception as e:
        return f"ERROR: Failed to parse cart data: {str(e)}"
