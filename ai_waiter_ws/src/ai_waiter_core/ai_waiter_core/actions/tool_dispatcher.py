# from langchain.tools import tool
# import logging 

# from src.tool.rag.retriever import hybrid_search
# from src.tool.orders.database import OrderDB

# # intialize 
# db = OrderDB()

# logger = logging.getLogger(__name__)

# @tool
# def search(query: str) -> str:
#     """
#     Search for ANY information about the restaurant using hybrid BM25 + Vector search.
    
#     This provides better performance for:
#     - Exact keyword matching (BM25)
#     - Semantic similarity (Vector search)
#     - Vietnamese text processing
    
#     Covers:
#     - Menu items, dishes, prices, ingredients
#     - Restaurant hours, location, contact info  
#     - Services (delivery, takeaway, reservations)
#     - Payment methods, parking, facilities
#     - Chef information, awards, policies
#     - Allergen info, dietary options
    
#     Args:
#         query: The customer's question about the restaurant
#     """
#     try:
#         # Use hybrid search for better performance
#         search_results = hybrid_search(query, k=4)
        
#         # Check if results are empty
#         if not search_results or len(search_results) == 0:
#             return f"Xin lỗi quý khách, em không tìm thấy thông tin về '{query}' trong hệ thống của quán. Quý khách có thể:\n- Hỏi về những món ăn khác\n- Liên hệ trực tiếp với quán để biết thêm chi tiết\n- Đặt hàng những món quán có sẵn"
        
#         # Format results with scores and better presentation
#         results = []
#         for result in search_results:
#             # Access document from SearchResult object
#             content = result.document.page_content.strip()
#             doc_type = result.doc_type
#             score = result.score
            
#             # Format with score indicator
#             score_indicator = "⭐" if score > 0.8 else "✓"
            
#             # Add emoji based on content type
#             if doc_type == 'menu_item':
#                 results.append(f"{score_indicator} [{score:.2f}] {content}")
#             elif doc_type in ['hours', 'location', 'services', 'payment', 'facilities']:
#                 results.append(f"{score_indicator} [{score:.2f}] {content}")
#             elif doc_type in ['chef_quality', 'awards']:
#                 results.append(f"{score_indicator} [{score:.2f}] {content}")
#             elif doc_type in ['policies', 'dietary']:
#                 results.append(f"{score_indicator} [{score:.2f}] {content}")
#             else:
#                 results.append(f"• [{score:.2f}] {content}")
        
#         return "\n\n".join(results)
        
#     except Exception as e:
#         print(f"❌ Search error: {e}")
#         # Fallback to simple message
#         return f"Có lỗi khi tìm kiếm thông tin về: {query}. Vui lòng thử lại."

# @tool
# def place_order(item: str, quantity: int) -> str:
#     """
#     Place an order for menu items.
    
#     ONLY use this when customer explicitly wants to ORDER food.
#     Examples: "Tôi muốn đặt...", "Cho tôi...", "Gọi cho tôi..."
    
#     DO NOT use for browsing or asking questions.
    
#     Args:
#         item: Exact dish name from menu
#         quantity: Number of items (must be positive integer)
#     """
#     try:
#         # Validate input
#         if not item or not item.strip():
#             return "ERROR: Tên món không được để trống."
        
#         if quantity <= 0:
#             return f"ERROR: Số lượng phải lớn hơn 0. Nhận được: {quantity}"
        
#         # Save to database (SQLite now, MySQL in future)
#         order_id = db.add_order(item, quantity)
        
#         return f"SUCCESS: Order saved! Ticket #{order_id}. Kitchen is preparing {quantity}x {item}."
        
#     except Exception as e:
#         return f"ERROR: Could not save order. {str(e)}"
    

from langchain.tools import tool
from ai_waiter_core.storage.retriever import hybrid_search
from ai_waiter_core.storage.order_db import OrderDB

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
        search_results = hybrid_search(query, k=4)
        
        # Check if results are empty
        if not search_results or len(search_results) == 0:
            return f"Xin lỗi quý khách, em không tìm thấy thông tin về '{query}' trong hệ thống của quán. Quý khách có thể:\n- Hỏi về những món ăn khác\n- Liên hệ trực tiếp với quán để biết thêm chi tiết\n- Đặt hàng những món quán có sẵn"
        
        # Format results with scores and better presentation
        results = []
        for result in search_results:
            content = result.document.page_content.strip()
            score = result.score
            
            # Format with score indicator
            score_indicator = "⭐" if score > 0.8 else "✓"
            
            # Add content based on type
            if result.doc_type == 'menu_item':
                results.append(f"{score_indicator} Món ăn: {content}")
            elif result.doc_type in ['hours', 'location', 'services', 'payment', 'facilities']:
                results.append(f"{score_indicator} Thông tin: {content}")
            elif result.doc_type in ['chef_quality', 'awards']:
                results.append(f"{score_indicator} Danh tiếng: {content}")
            elif result.doc_type in ['policies', 'dietary']:
                results.append(f"{score_indicator} Chính sách: {content}")
            else:
                results.append(f"✓ {content}")
        
        if not results:
            return f"Xin lỗi quý khách, em không tìm thấy thông tin rõ ràng về '{query}'. Vui lòng hỏi chi tiết hơn hoặc liên hệ với quán."
        
        # print(results)
        return "Dựa trên thông tin của quán:\n\n" + "\n\n".join(results)
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return f"Có lỗi khi tìm kiếm thông tin về '{query}'. Quý khách vui lòng thử lại hoặc liên hệ trực tiếp với quán."

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
        # Validate input
        if not item or not item.strip():
            return "❌ Lỗi: Tên món không được để trống. Quý khách vui lòng cho biết muốn đặt món nào?"
        
        if quantity <= 0:
            return f"❌ Lỗi: Số lượng phải lớn hơn 0. Quý khách muốn gọi bao nhiêu phần?"
        
        # Save to database (SQLite now, MySQL in future)
        order_id = db.add_order(item, quantity)
        
        return f"✅ Đã lưu đơn hàng! Mã hóa đơn #{order_id}. Quán đang chuẩn bị {quantity}x {item} cho quý khách."
        
    except Exception as e:
        return f"❌ Lỗi: Không thể lưu đơn hàng. {str(e)}"