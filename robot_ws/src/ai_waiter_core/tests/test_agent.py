import uuid
import sys
import os

# 1. Add the project source to the path so we can import our package
sys.path.append(os.path.abspath("robot_ws/src/ai_waiter_core"))

from ai_waiter_core.orchestrator.agent import get_agent_app
from ai_waiter_core.orchestrator.memory import create_config

def run_test():
    app = get_agent_app()
    table_id = "Table_5"
    config = create_config(table_id)
    
    print(f"--- Session: {table_id} (Vietnamese Test) ---")
    
    # Lần 1: Hỏi thực đơn
    print("\n[USER]: Chào bạn, cho mình xem thực đơn hôm nay có gì ngon?")
    inputs = {"messages": [("user", "Chào bạn, cho mình xem thực đơn hôm nay có gì ngon?")], "table_id": table_id}
    for output in app.stream(inputs, config=config, stream_mode="values"):
        last_message = output["messages"][-1]
    print(f"[AI]: {last_message.content}")

    # Lần 2: Gọi món (Kiểm tra xem nó có nhớ Table_5 không)
    print("\n[USER]: Cho mình 2 phần cá hồi nướng nhé.")
    inputs = {"messages": [("user", "Cho mình 2 phần cá hồi nướng nhé.")]}
    for output in app.stream(inputs, config=config, stream_mode="values"):
        last_message = output["messages"][-1]
    print(f"[AI]: {last_message.content}")

if __name__ == "__main__":
    run_test()
