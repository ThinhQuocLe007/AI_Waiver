from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
PACKAGE_ROOT = Path(__file__).parent.parent.resolve()

class BaseSystemSettings(BaseSettings):
    PROJECT_ROOT: Path = ROOT
    resources_dir: Path = Field(default=PACKAGE_ROOT / "agent" / "resources")
    storage_dir: Path = Field(default=ROOT / "storage")
    assets_dir: Path = Field(default=ROOT / "assets")
    inputs_dir: Path = Field(default=ROOT / "inputs")
    
    # Server network settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT", ge=1, le=65535)

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8',
        extra="ignore" 
    )
