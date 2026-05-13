import sys
import os
from pathlib import Path

# Current file is in robot_ws/src/ai_waiter_core/tests/test_modular_retriever.py
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent # /home/lequocthinh/Desktop/pythonCode/AI_Waiver

from ai_waiter_core.tools.search_hybrid import RetrieverManager
from ai_waiter_core.core.config import settings

def test_retriever():
    print("--- Initializing RetrieverManager ---")
    retriever = RetrieverManager()
    
    # Path to your menu data
    data_path = project_root / "data" / "raw"
    
    print(f"--- Building Database from {data_path} ---")
    if retriever.build_database([str(data_path)]):
        print("Success: Database built.\n")
    else:
        print("Error: Database build failed.")
        return

    query = "Lẩu thái hải sản"
    print(f"--- Testing Query: '{query}' ---")

    # 1. Test RRF Mode
    print("\n[MODE: RRF]")
    results_rrf = retriever.hybrid_search(query, mode="rrf", k=3)
    for i, res in enumerate(results_rrf, 1):
        name = res.document.metadata.get('name', 'N/A')
        print(f"{i}. {name} (Score: {res.score:.4f})")
        print(f"   Tags: {res.document.metadata.get('tags', 'N/A')}")
        print(f"   Taste: {res.document.metadata.get('taste_profile', 'N/A')}")

    # 2. Test Weighted Mode
    print("\n[MODE: Weighted]")
    results_weighted = retriever.hybrid_search(query, mode="weighted", k=3, threshold=0.1)
    for i, res in enumerate(results_weighted, 1):
        name = res.document.metadata.get('name', 'N/A')
        print(f"{i}. {name} (Score: {res.score:.4f})")

if __name__ == "__main__":
    test_retriever()
