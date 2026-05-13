import os
import sys
import uuid
from pathlib import Path

# Fix python path to find the package
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
sys.path.append(str(project_root))

from ai_waiter_core.agent.agent import get_agent_app
from ai_waiter_core.agent.memory import create_config

def test_full_ordering_flow():
    app = get_agent_app()
    table_id = "T8"
    thread_id = str(uuid.uuid4()) # Unique session
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"=== STARTING END-TO-END TEST (Session: {thread_id}) ===")

    # STAGE 1: Requesting the order
    user_msg_1 = "Cho mình 1 phở bò đặc biệt và 1 trà đá ít đường"
    print(f"\n[USER]: {user_msg_1}")
    
    state_1 = {"messages": [{"role": "user", "content": user_msg_1}], "table_id": table_id}
    
    # Run the agent
    print("--- Processing Order Request ---")
    for event in app.stream(state_1, config, stream_mode="values"):
        final_msg = event["messages"][-1]
    
    # Verify tool call (Preparation)
    if hasattr(final_msg, "tool_calls"):
        for tc in final_msg.tool_calls:
            print(f"DEBUG: Tool triggered -> {tc['name']}")

    print(f"[AI]: {final_msg.content}")

    # STAGE 2: User confirms the order
    user_msg_2 = "Đúng rồi, đặt luôn cho mình đi"
    print(f"\n[USER]: {user_msg_2}")
    
    state_2 = {"messages": [{"role": "user", "content": user_msg_2}]}
    
    print("--- Processing Confirmation ---")
    for event in app.stream(state_2, config, stream_mode="values"):
        final_msg_2 = event["messages"][-1]
        
    # Verify final tool call (Confirmation)
    if hasattr(final_msg_2, "tool_calls"):
        for tc in final_msg_2.tool_calls:
            print(f"DEBUG: Final Tool triggered -> {tc['name']}")

    print(f"[AI]: {final_msg_2.content}")

    # STAGE 3: Final check (Optional - check DB)
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    test_full_ordering_flow()
