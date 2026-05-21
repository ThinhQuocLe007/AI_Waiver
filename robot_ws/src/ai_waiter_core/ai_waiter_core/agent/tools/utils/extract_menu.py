import json
from typing import List
from ai_waiter_core.config import settings

def get_menu_names() -> List[str]:
    """Dynamically loads menu item names from the JSON asset."""
    # The menu.json is at assets/data/menu.json
    menu_path = settings.assets_dir / "data" / "menu.json"
    
    if not menu_path.exists():
        return []
        
    try:
        with open(menu_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [item['name'] for item in data]
    except Exception as e:
        print(f"Error loading menu for schema: {e}")
        return []

# Fetch names at module import time
MENU_NAMES = get_menu_names()
