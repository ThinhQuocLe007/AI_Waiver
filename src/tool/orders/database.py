import sqlite3
from datetime import datetime

class OrderDB:
    def __init__(self, db_path="orders.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Create the table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # SQL Command to create table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                items TEXT NOT NULL,
                quantity INTEGER,
                status TEXT DEFAULT 'CONFIRMED',
                created_at TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_order(self, items_text, quantity):
        """Save a confirmed order"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO orders (items, quantity, created_at)
            VALUES (?, ?, ?)
        ''', (items_text, quantity, current_time))
        
        order_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return order_id

    def get_recent_orders(self, limit=5):
        """View recent orders (for debugging)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM orders ORDER BY id DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows