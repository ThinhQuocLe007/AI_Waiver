import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # --- Project Root ---
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
    DATA_DIR = PROJECT_ROOT / "data"

    # --- LLM Settings ---
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    DEVICE = os.getenv("DEVICE", "cuda")

    # --- Data Paths ---
    MENU_FILE = DATA_DIR / "raw" / "menu.json"
    INFO_FILE = DATA_DIR / "raw" / "restaurant_info.txt"
    
    VECTOR_DB_PATH = DATA_DIR / "vector_db" / "faiss_index"
    BM25_PATH = DATA_DIR / "vector_db" / "bm25.pkl"
    ORDER_DB_PATH = DATA_DIR / "sqlite" / "orders.db"

    # --- Server Settings ---
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

# Create a singleton instance
settings = Settings()
