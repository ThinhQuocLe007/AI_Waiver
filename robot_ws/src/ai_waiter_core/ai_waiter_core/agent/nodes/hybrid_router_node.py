import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from ai_waiter_core.agent.state import AgentState
from ai_waiter_core.agent.nodes.semantic_router_node import SemanticRouterNode
from ai_waiter_core.agent.nodes.slm_router_node import slm_router_node
from ai_waiter_core.utils import trace_latency

logger = logging.getLogger(__name__)

# Pre-instantiate globally to avoid loading embeddings on every turn
_semantic_router_instance = None

# Confidence threshold for fast-tracking. Lowered to 0.75 to catch ORDER queries
# with specific menu item names that previously fell below 0.82.
HYBRID_CONFIDENCE_THRESHOLD = 0.75

def is_potentially_complex_or_non_order(text: str) -> bool:
    """
    Lightweight heuristic to detect multi-intent (COMPLEX), past-tense comments,
    or tricky conditional structures that should be routed to the SLM safety net.
    """
    text_lower = text.lower()
    
    # 1. Past tense comments or casual observation
    if "bàn bên cạnh" in text_lower or "thấy bàn" in text_lower or "đã đặt" in text_lower:
        return True
        
    # 2. Sequential triggers combined with order
    if "rồi" in text_lower or "xong" in text_lower or "sau đó" in text_lower:
        return True
        
    # 3. Conditional triggers
    if "nếu" in text_lower or "nếu không" in text_lower:
        return True
        
    # 4. Payment indicators
    if any(kw in text_lower for kw in ["tính tiền", "thanh toán", "check bill", "in bill", "hóa đơn", "bill"]):
        return True
        
    # 5. Menu questions or comparison indicators
    if any(kw in text_lower for kw in ["bao nhiêu", "giá", "ngon không", "chay không", "mấy người", "hỏi"]):
        return True
        
    return False

@trace_latency("Hybrid Router Node", run_type="chain")
def hybrid_router_node(state: AgentState) -> Dict[str, Any]:
    """
    Combines high-speed Semantic Routing with high-accuracy SLM Routing.
    If the vector similarity is high (>= threshold) and the intent is not COMPLEX,
    it returns instantly. Otherwise, it delegates to the SLM.
    
    Returns routing_meta with confidence score and which engine decided.
    """
    global _semantic_router_instance
    if _semantic_router_instance is None:
        _semantic_router_instance = SemanticRouterNode()
        
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), 
        ""
    )
    
    # 1. Try Fast Semantic Routing
    semantic_result = _semantic_router_instance.route(last_user_message)
    sem_intent = semantic_result.get("raw_intent")
    sem_confidence = semantic_result.get("confidence", 0.0)
    
    logger.info(f"Semantic Router predicted {sem_intent} with confidence: {sem_confidence:.4f}")
    
    # If confidence is high and it's a simple intent (and has no complex features), use it!
    # We always pass COMPLEX to SLM because COMPLEX needs deeper structural understanding.
    if (
        sem_confidence >= HYBRID_CONFIDENCE_THRESHOLD 
        and sem_intent 
        and sem_intent != "COMPLEX" 
        and (sem_intent != "ORDER" or not is_potentially_complex_or_non_order(last_user_message))
    ):
        logger.info(f"Hybrid Router: Fast-tracking {sem_intent} (Conf: {sem_confidence:.2f} >= {HYBRID_CONFIDENCE_THRESHOLD})")
        return {
            "current_intent": sem_intent,
            "routing_meta": {
                "decided_by": "SEMANTIC",
                "semantic_confidence": round(sem_confidence, 4),
                "semantic_intent": sem_intent,
            }
        }
        
    # 2. Fallback to SLM Router
    logger.info(f"Hybrid Router: Delegating to SLM Router (Conf: {sem_confidence:.2f} < {HYBRID_CONFIDENCE_THRESHOLD}, COMPLEX, or matched heuristic)")
    slm_result = slm_router_node(state)
    slm_intent = slm_result.get("current_intent", "CHAT")
    
    return {
        "current_intent": slm_intent,
        "routing_meta": {
            "decided_by": "SLM",
            "semantic_confidence": round(sem_confidence, 4),
            "semantic_intent": sem_intent,
            "slm_intent": slm_intent,
        }
    }
