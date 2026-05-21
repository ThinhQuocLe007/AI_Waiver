import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ai_waiter_core.agent.state import AgentState
from ai_waiter_core.config import settings
from ai_waiter_core.utils import log_struct, trace_latency

UTTERANCES_PATH = settings.resources_dir / "few_shots" / "utterances.json"

class SemanticRouterNode:
    def __init__(self, utterances_path: str = None, model_name: str = "BAAI/bge-m3", threshold: float = 0.75):
        self.model = SentenceTransformer(model_name, device=settings.DEVICE)
        self.threshold = threshold
        
        path = utterances_path or UTTERANCES_PATH
        with open(path, "r", encoding="utf-8") as f:
            self.routes = json.load(f)
        
        self.route_embeddings = {}
        self._encode_all_routes()

    def _encode_all_routes(self):
        log_struct("Encoding semantic router utterances", route_count=len(self.routes))
        for route_name, utterances in self.routes.items():
            self.route_embeddings[route_name] = self.model.encode(utterances)

    def route(self, query: str) -> Dict[str, Any]:
        query_vec = self.model.encode([query])
        best_route = None
        max_sim = -1.0
        
        for route_name, embeddings in self.route_embeddings.items():
            similarities = cosine_similarity(query_vec, embeddings)[0]
            current_max = np.max(similarities)
            if current_max > max_sim:
                max_sim = current_max
                best_route = route_name
        
        return {
            "intent": best_route if max_sim >= self.threshold else None,
            "confidence": float(max_sim),
            "raw_intent": best_route
        }

# Pre-instantiate the router class once globally to prevent reloading models on every query
_router_instance = None

@trace_latency("Semantic Router Node", run_type="chain")
def semantic_router_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node that performs legacy vector-based intent classification.
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = SemanticRouterNode()
        
    from langchain_core.messages import HumanMessage
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), 
        ""
    )
    
    routing_result = _router_instance.route(last_user_message)
    return {
        "metadata": routing_result
    }
