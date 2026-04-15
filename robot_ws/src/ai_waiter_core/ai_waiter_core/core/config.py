import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, DirectoryPath

class Settings(BaseSettings):
    # --- Project Root Calculation ---
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
    
    # --- LLM Settings ---
    MODEL_NAME: str = Field(default="qwen2.5:3b", env="MODEL_NAME")
    HF_TOKEN: str = Field(default="", env="HF_TOKEN")
    DEVICE: str = Field(default="cuda", env="DEVICE")


    # --- Data Paths ---
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent.parent.parent.resolve() / "data")    @property
    def VECTOR_DB_PATH(self) -> Path:
        return self.data_dir / "vector_db" / "faiss_index"

    @property
    def BM25_PATH(self) -> Path:
        return self.data_dir / "vector_db" / "bm25.pkl"

    @property
    def ORDER_DB_PATH(self) -> Path:
        return self.data_dir / "sqlite" / "orders.db"
        
    @property
    def CHECKPOINTS_DB_PATH(self) -> Path:
        return self.data_dir / "sqlite" / "checkpoints.db"


    # --- Server Settings ---
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT", ge=1, le=65535)

    # --- Config for Environment Loading ---
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8',
        extra="ignore" 
    )

# Create the singleton instance
settings = Settings()
