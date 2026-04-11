from langchain_core.tools import tool
from .order import OrderDB
from .qr_payment import PaymentManager
from .search_hybrid import RetrieverManager

# Initialize Managers
order_db = OrderDB()
payment_mgr = PaymentManager()
retriever = RetrieverManager()

# Pre-load the search database
retriever.load_database()

@tool
def search_menu(query: str) -> str:
    """Search the menu for food/drinks/prices."""
    results = retriever.hybrid_search(query, k=3)
    if not results: return "No info found."
    return "\n---\n".join([f"[{r.doc_type}] {r.document.page_content}" for r in results])

@tool
def place_order(items_description: str, quantity: int, table_id: str) -> str:
    """Place an order for a specific table."""
    order_id = order_db.add_order(table_id, items_description, quantity)
    return f"Order #{order_id} placed!" if order_id else "Error placing order."

@tool
def request_payment(table_id: str, amount: float) -> str:
    """Generate a payment QR code link."""
    url = payment_mgr.generate_qr_payload(table_id, amount)
    return f"Scan to pay: {url}"

CORE_TOOLS = [search_menu, place_order, request_payment]
