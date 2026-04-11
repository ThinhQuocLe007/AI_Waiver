import os
import sys
from ai_waiter_core.tools.search_hybrid import RetrieverManager
from ai_waiter_core.core.config import settings
from ai_waiter_core.tools.utils import print_search_results

def run_tests():
    """
    Run basic tests for RetrieverManager.
    """
    print("Initializing RetrieverManager...")
    manager = RetrieverManager(score_threshold=0.3)
    
    print("Building database...")
    success = manager.build_database()
    if not success:
        print("FAIL: Failed to build database")
        return
    
    print("SUCCESS: Database built")
    
    # Test cases
    test_cases = [
        {"query": "phở bò", "expected_type": "menu"},
        {"query": "giờ mở cửa", "expected_type": "info"},
        {"query": "món gì đó không có", "expected_type": None, "threshold": 0.99}
    ]
    
    for case in test_cases:
        query = case["query"]
        print(f"\nSearching for: '{query}'")
        threshold = case.get("threshold")
        results = manager.hybrid_search(query, k=5, threshold=threshold)
        
        print_search_results(results)
        
        if case["expected_type"]:
            found = any(res.doc_type == case["expected_type"] for res in results)
            if found:
                print(f"PASS: Found expected type '{case['expected_type']}'")
            else:
                print(f"FAIL: Expected type '{case['expected_type']}' not found")
        elif threshold is not None:
             if len(results) == 0:
                 print(f"PASS: No results found with threshold {threshold}")
             else:
                 print(f"FAIL: Results found despite high threshold {threshold}")

if __name__ == "__main__":
    run_tests()
