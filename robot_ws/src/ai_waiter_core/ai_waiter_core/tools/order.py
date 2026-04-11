import sqlite3
import os
from datetime import datetime
from ai_waiter_core.core.config import settings
from ai_waiter_core.core.utils.logger import logger

class OrderDB:
    """
    Order database manager using SQLite.
    """
    def __init__(self, db_path=None):
        self.db_path = str(db_path or settings.ORDER_DB_PATH)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Create the table if it doesn't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_id TEXT NOT NULL,
                    items TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT DEFAULT 'CONFIRMED',
                    created_at TEXT NOT NULL
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize Order DB: {e}")
            raise
    
    def add_order(self, table_id, items_text, quantity):
        """Save a confirmed order to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
                INSERT INTO orders (table_id, items, quantity, created_at)
                VALUES (?, ?, ?, ?)
            ''', (table_id, items_text, quantity, current_time))
            
            order_id = cursor.lastrowid
            conn.commit()
            conn.close()
            logger.info(f"Order {order_id} saved for Table {table_id}")
            return order_id
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None
