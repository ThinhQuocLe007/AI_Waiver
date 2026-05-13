import sys
import os
import uuid

# Add the source directory to path
sys.path.append(os.path.abspath("robot_ws/src/ai_waiter_core"))

from ai_waiter_core.agent.agent import get_agent_app
from ai_waiter_core.agent.memory import create_config

def run_test_query(app, query: str, table_id: str = "T5"):
    print(f"\n--- USER: {query} ---")
    
    config = create_config(table_id)
    # create_config returns {"configurable": {"thread_id": table_id}}
    
    # We pass the message history to the agent
    input_state = {
        "messages": [{"role": "user", "content": query}],
        "table_id": table_id
    }
    
    # Execute the agent
    for event in app.stream(input_state, config, stream_mode="values"):
        # We only care about the last message in the stream
        last_message = event["messages"][-1]
    
    # If the response is a tool call, the app will loop. 
    # LangGraph handles the tool execution automatically if bound.
    
    # Print the AI response
    if hasattr(last_message, "content") and last_message.content:
        print(f"AI: {last_message.content}")
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            print(f"TOOL CALLED: {tool_call['name']}({tool_call['args']})")

if __name__ == "__main__":
    app = get_agent_app()
    
    # Test cases to see routing in action
    test_queries = [
        "Xin chào, cho mình xem menu",      # Should route to MENU
        "Cho mình đặt 2 phở bò bàn 5",      # Should route to ORDER_CONFIRM
        "Mình muốn thanh toán tiền",        # Should route to PAYMENT
        "Cảm ơn bạn nhé",                   # Should route to CHAT
    ]
    
    for query in test_queries:
        run_test_query(app, query)
