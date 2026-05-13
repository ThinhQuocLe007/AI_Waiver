import sys
import uuid

# Add src to sys.path so we can import ai_waiter_core
from pathlib import Path
root_path = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
sys.path.append(str(root_path / "robot_ws" / "src" / "ai_waiter_core"))

from ai_waiter_core.agent.agent import get_agent_app
from ai_waiter_core.agent.memory import create_config

def run_interaction(app, config, user_message):
    print(f"\n[{config['configurable']['thread_id']}] User: {user_message}")
    print("-" * 50)
    
    # We use stream so we can see the internal steps (tool calls)
    for event in app.stream({"messages": [("user", user_message)]}, config=config, stream_mode="values"):
        last_msg = event["messages"][-1]
        
        # Only print AI messages and Tool messages that are NEW in this step
        if last_msg.type == "ai":
            if last_msg.tool_calls:
                print(f"🤖 [TOOL CALL] {last_msg.tool_calls[0]['name']}({last_msg.tool_calls[0]['args']})")
            elif last_msg.content:
                print(f"🤖 AI Waiter: {last_msg.content}")
        elif last_msg.type == "tool":
            print(f"🔧 [TOOL OUTPUT] {last_msg.content}")
            
    print("-" * 50)

def main():
    print("Initializing Agent...")
    app = get_agent_app()
    
    # Create a unique session ID for this test
    session_id = str(uuid.uuid4())
    config = create_config(table_id="TestTable_1")
    config["configurable"]["thread_id"] = session_id
    
    print("\n--- Starting Difficult Test Flow ---")
    
    # Step 1: User orders something very ambiguous
    run_interaction(app, config, "Cho tôi 1 phần bánh xèo")
    
    # Step 2: User clarifies what they meant (e.g., choosing from options the LLM provides)
    run_interaction(app, config, "Tôi lấy Bánh Xèo Miền Tây")
    
    # Step 3: User changes their mind and adds a special request
    run_interaction(app, config, "À không, cho tôi 2 cái đi, nhớ ghi chú là ít dầu mỡ nhé")
    
    # Step 4: Final confirmation
    run_interaction(app, config, "Đúng rồi, xác nhận đặt món")

if __name__ == "__main__":
    main()
