import json
import os
from ai_waiter_core.config import settings
class MenuManager:
    def __init__(self):
        self.menu_map = {}
        self.load_menu()
    def load_menu(self):
        # Path to your menu.json
        menu_path = settings.assets_dir / "data" / "menu.json"
        with open(menu_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Create a clean lookup table: {"item name": price}
            self.menu_map = {item['name'].lower(): float(item['price']) for item in data}
    def get_price(self, item_name: str):
        return self.menu_map.get(item_name.lower())