import re
from typing import List, Optional, Dict
from pydantic import BaseModel
from ai_waiter_core.schemas.menu_registry import MENU_NAMES

class ResolveResult(BaseModel):
    status: str # "exact", "auto_resolved", "ambiguous", "not_found"
    canonical_name: Optional[str] = None
    suggestions: List[str] = []

def resolve_menu_item(query: str) -> ResolveResult:
    """
    Tiered resolution: exact -> substring -> (placeholder for fuzzy)
    """
    query_clean = query.strip().lower()
    
    # 1. Exact Match
    for name in MENU_NAMES:
        if name.lower() == query_clean:
            return ResolveResult(status="exact", canonical_name=name)
            
    # 2. Substring / Prefix Match
    # e.g. "Phở Bò" -> "Phở Bò Đặc Biệt"
    matches = []
    for name in MENU_NAMES:
        if query_clean in name.lower():
            matches.append(name)
            
    if len(matches) == 1:
        return ResolveResult(status="auto_resolved", canonical_name=matches[0])
    elif len(matches) > 1:
        return ResolveResult(status="ambiguous", suggestions=matches)
        
    # 3. Not Found
    return ResolveResult(status="not_found")
