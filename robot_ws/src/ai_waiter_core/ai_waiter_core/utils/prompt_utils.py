import json
from ai_waiter_core.config import settings

def load_prompt(filename: str, sub_dir: str = "system_prompts") -> str:
    """
    Loads a markdown or text prompt file from the resources directory.
    Example: load_prompt("router_agent.md") or load_prompt("hospitality.md", "skills")
    """
    path = settings.resources_dir / sub_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json_data(filename: str, sub_dir: str = "few_shots") -> list | dict:
    """
    Loads a JSON file (e.g. for few-shot examples).
    Example: load_json_data("router.json")
    """
    path = settings.resources_dir / sub_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
