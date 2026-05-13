import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from ai_waiter_core.agent.agent import get_agent_app

def start_chat():
    app = get_agent_app()
    table_id = "T1"
    thread_id = f"chat-{uuid.uuid4().hex[:4]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"--- AI WAITER CHAT SESSION: {thread_id} ---")
    print(f"--- Serving Table: {table_id} ---")
    print("Type 'exit' or 'quit' to stop.\n")

    # Local state for the conversation
    state = {"messages": [], "table_id": table_id, "pending_cart": None}

    while True:
        try:
            user_input = input("\033[94m[USER]: \033[0m")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Add user message to state
            state["messages"].append({"role": "user", "content": user_input})
            
            # Stream the response
            print("\033[90mThinking...\033[0m", end="\r")
            
            final_msg = None
            for event in app.stream(state, config, stream_mode="values"):
                final_msg = event["messages"][-1]
                # Sync state back if it was updated (e.g., pending_cart)
                if "pending_cart" in event:
                    state["pending_cart"] = event["pending_cart"]
            
            # Clear thinking line
            print(" " * 20, end="\r")

            # Log tool calls if any
            if hasattr(final_msg, "tool_calls") and final_msg.tool_calls:
                for tc in final_msg.tool_calls:
                    print(f"\033[93m  🔧 Tool: {tc['name']}({tc['args']})\033[0m")

            # Print AI response
            print(f"\033[92m[AI]: {final_msg.content}\033[0m\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    start_chat()
