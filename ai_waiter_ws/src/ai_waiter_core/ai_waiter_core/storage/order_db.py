import sqlite3
from datetime import datetime


class OrderDB:
    """
    Order database manager.
    Currently uses SQLite, but structured to support MySQL in the future.
    """
    
    def __init__(self, db_path="orders.db", db_type="sqlite"):
        """
        Initialize order database.
        
        Args:
            db_path: Database file path (for SQLite) or connection string (for MySQL)
            db_type: Database type ("sqlite" or "mysql" - future)
        """
        self.db_path = db_path
        self.db_type = db_type
        self.init_db()
    
    def init_db(self):
        """Create the table if it doesn't exist"""
        if self.db_type == "sqlite":
            self._init_sqlite()
        elif self.db_type == "mysql":
            self._init_mysql()  # TODO: Implement MySQL initialization
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # SQL Command to create table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    items TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT DEFAULT 'CONFIRMED',
                    created_at TEXT NOT NULL
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            raise
    
    def _init_mysql(self):
        """Initialize MySQL database (TODO: Implement when needed)"""
        # TODO: Implement MySQL initialization
        # Example structure:
        # import mysql.connector
        # self.conn = mysql.connector.connect(...)
        # cursor = self.conn.cursor()
        # cursor.execute('CREATE TABLE IF NOT EXISTS orders (...)')
        raise NotImplementedError("MySQL support coming soon")
    
    def add_order(self, items_text, quantity):
        """
        Save a confirmed order to database.
        
        Args:
            items_text: Name of the menu item
            quantity: Number of items
        
        Returns:
            order_id: The ID of the created order
        """
        if self.db_type == "sqlite":
            return self._add_order_sqlite(items_text, quantity)
        elif self.db_type == "mysql":
            return self._add_order_mysql(items_text, quantity)  # TODO: Implement
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _add_order_sqlite(self, items_text, quantity):
        """Save order to SQLite database"""
        try:
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
            
        except sqlite3.Error as e:
            raise
    
    def _add_order_mysql(self, items_text, quantity):
        """Save order to MySQL database (TODO: Implement when needed)"""
        # TODO: Implement MySQL insert
        # Example:
        # cursor = self.conn.cursor()
        # cursor.execute('INSERT INTO orders (items, quantity, created_at) VALUES (%s, %s, %s)', ...)
        # self.conn.commit()
        # return cursor.lastrowid
        raise NotImplementedError("MySQL support coming soon")
    
    def get_recent_orders(self, limit=5):
        """View recent orders (for debugging)"""
        if self.db_type == "sqlite":
            return self._get_recent_orders_sqlite(limit)
        elif self.db_type == "mysql":
            return self._get_recent_orders_mysql(limit)  # TODO: Implement
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _get_recent_orders_sqlite(self, limit):
        """Get recent orders from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM orders ORDER BY id DESC LIMIT ?', (limit,))
            rows = cursor.fetchall()
            conn.close()
            return rows
        except sqlite3.Error as e:
            return []
    
    def _get_recent_orders_mysql(self, limit):
        """Get recent orders from MySQL (TODO: Implement when needed)"""
        raise NotImplementedError("MySQL support coming soon")