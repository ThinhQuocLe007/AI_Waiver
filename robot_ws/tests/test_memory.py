from ai_waiter_core.orchestrator.memory import get_checkpointer, create_config
from ai_waiter_core.orchestrator.agent import get_agent_app


app = get_agent_app()
config = create_config("Table_1")

# Conversation 1 

print("--- Session 1 ---")
for chunk in app.stream({"messages": [("user", "Hi, I'm sitting at Table 1. My name is Thinh.")]}, config):
    print(chunk)

# Conversation 2 
print("--- Session 2 ---")
for chunk in app.stream({"messages": [("user", "Do you rememeber my name?")]}, config):
    print(chunk)
