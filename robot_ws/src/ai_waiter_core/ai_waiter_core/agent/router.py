from ai_waiter_core.core.config import settings
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:
    def __init__(self, model_name: str = "BAAI/bge-m3", threshold: float = 0.75):
        self.model = SentenceTransformer(model_name, device= settings.DEVICE)
        self.threshold = threshold
        
        # Updated routes with confirmation examples
        self.routes = {
            "ORDER_CONFIRM": [
                "Cho tôi 2 bát phở bò",
                "Gọi 1 cơm sườn",
                "Đặt 3 ly trà đá",
                "Tôi muốn gọi món này",
                "Xác nhận đơn hàng",      # Added for step 2
                "Đúng rồi, đặt đi",      # Added for step 2
                "Tiến hành món này",       # Added for step 2
                "Đồng ý đặt món",         # Added for step 2
            ],
            "MENU": [
                "Có món chay không?",
                "Phở bò giá bao nhiêu?",
                "Cho tôi xem thực đơn",
                "Món nào ngon nhất?",
                "Hôm nay có gì đặc biệt?",
            ],
            "PAYMENT": [
                "Tính tiền giùm",
                "Thanh toán bàn 5",
                "Cho tôi xem hóa đơn",
                "Tôi muốn trả tiền",
            ],
            "CHAT": [
                "Xin chào",
                "Cảm ơn bạn",
                "Chào buổi sáng",
                "Bạn là ai?",
            ],
            "COMPLEX": [
                "Cho tôi 1 phở bò và thanh toán luôn",
                "Vừa gọi món vừa muốn xem menu",
                "Hủy phở rồi gọi cơm thay",
            ]
        }
        
        self.route_embeddings = {}
        for route_name, utterances in self.routes.items():
            self.route_embeddings[route_name] = self.model.encode(utterances)

    def route(self, query: str) -> Optional[str]:
        query_vec = self.model.encode([query])
        best_route = None
        max_sim = -1.0
        
        for route_name, embeddings in self.route_embeddings.items():
            similarities = cosine_similarity(query_vec, embeddings)[0]
            current_max = np.max(similarities)
            
            if current_max > max_sim:
                max_sim = current_max
                best_route = route_name
        
        if max_sim >= self.threshold:
            return best_route
        
        return None
