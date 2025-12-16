from langchain.tools import tool 

from src.tool.rag.retriever import hybrid_search
from src.tool.orders.database import OrderDB

# intialize 
db = OrderDB()
@tool
def search(query: str) -> str:
    """
    Search for ANY information about the restaurant using hybrid BM25 + Vector search.
    
    This provides better performance for:
    - Exact keyword matching (BM25)
    - Semantic similarity (Vector search)
    - Vietnamese text processing
    
    Covers:
    - Menu items, dishes, prices, ingredients
    - Restaurant hours, location, contact info  
    - Services (delivery, takeaway, reservations)
    - Payment methods, parking, facilities
    - Chef information, awards, policies
    - Allergen info, dietary options
    
    Args:
        query: The customer's question about the restaurant
    """
    try:
        # Use hybrid search for better performance
        docs = hybrid_search(query, k=4)
        
        if not docs:
            return "Không tìm thấy thông tin liên quan. Xin lỗi quý khách."
        
        # Format results with better presentation
        results = []
        for doc in docs:
            content = doc.page_content.strip()
            doc_type = doc.metadata.get('type', 'info')
            
            # Add emoji based on content type
            if doc_type == 'menu_item':
                results.append(f"🍜 {content}")
            elif doc_type in ['hours', 'location', 'services', 'payment', 'facilities']:
                results.append(f"ℹ️ {content}")
            elif doc_type in ['chef_quality', 'awards']:
                results.append(f"⭐ {content}")
            elif doc_type in ['policies', 'dietary']:
                results.append(f"📋 {content}")
            else:
                results.append(f"• {content}")
        
        return "\n\n".join(results)
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        # Fallback to simple message
        return f"Có lỗi khi tìm kiếm thông tin về: {query}. Vui lòng thử lại."

@tool
def place_order(item: str, quantity: int) -> str:
    """
    Place an order for menu items.
    
    ONLY use this when customer explicitly wants to ORDER food.
    Examples: "Tôi muốn đặt...", "Cho tôi...", "Gọi cho tôi..."
    
    DO NOT use for browsing or asking questions.
    
    Args:
        item: Exact dish name from menu
        quantity: Number of items (must be positive integer)
    """
    try:
        # Save to SQLite
        order_id = db.add_order(item, quantity)
        return f"SUCCESS: Order saved! Ticket #{order_id}. Kitchen is preparing {quantity}x {item}."
    except Exception as e:
        return f"ERROR: Could not save order. {str(e)}"