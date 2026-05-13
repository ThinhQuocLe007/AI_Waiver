import sys
import os
import uuid
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from ai_waiter_core.agent.agent import get_agent_app

def run_scenario(name, queries):
    app = get_agent_app()
    thread_id = f"test-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n=== SCENARIO: {name} ===")
    
    for query in queries:
        print(f"\n[USER]: {query}")
        state = {"messages": [{"role": "user", "content": query}], "table_id": "T1"}
        
        last_msg = None
        for event in app.stream(state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
        
        # Log tool calls for transparency
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"  🔧 TOOL: {tc['name']}({tc['args']})")
        
        print(f"[AI]: {last_msg.content}")

if __name__ == "__main__":
    # Test 1: Absolute Precision (Should Pass)
    run_scenario("Precision", [
        "Cho mình 1 Phở Bò Đặc Biệt"
    ])
    
    # Test 2: Partial Names / Nicknames (Likely to Fail currently)
    run_scenario("Partial Names", [
        "Lấy cho mình 1 Phở bò",
        "Thêm 1 Bạc xỉu nữa nhé"
    ])
    
    # Test 3: Typos (Likely to Fail currently)
    run_scenario("Typos", [
        "Cho mình 1 bát Phỏ bò đặc biệt"
    ])
    
    # Test 4: Missing Items (Should report error politely without hallucinating)
    run_scenario("Missing Items", [
        "Mình muốn gọi 1 Trà đá"
    ])

    # Test 5: Re-ordering after Error (Correction flow)
    run_scenario("Correction Flow", [
        "Cho mình 1 Trà đá",
        "À menu không có trà đá hả? Vậy cho mình 1 Trà Đào Cam Sả đi"
    ])
