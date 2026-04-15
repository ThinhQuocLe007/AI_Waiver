# 🛠️ Mastering `pydantic-settings` for AI Applications

In modern Python development, handling configuration via `.env` files is standard. However, as applications grow (like your **AI Waiter**), managing these settings requires more than just strings—it requires **validation**. 

`pydantic-settings` is a library built on top of Pydantic that automatically parses, type-casts, and validates environment variables.

---

## 1. Installation

```bash
pip install pydantic-settings
```

---

## 2. Core Concepts: `BaseSettings`

Unlike a standard Pydantic `BaseModel`, a `BaseSettings` class automatically looks for environment variables that match its field names (case-insensitive by default).

### The "Standard" Way (Problems: no validation, manual casting)
```python
import os
PORT = int(os.getenv("PORT", 8000))
MODEL_NAME = os.getenv("MODEL_NAME")
```

### The `pydantic-settings` Way
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    # automatic type conversion: "8000" (str) -> 8000 (int)
    port: int = 8000 
    
    # if MODEL_NAME is missing in env, this raises a validation error immediately
    model_name: str 

    # Link to your .env file
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

config = AppConfig()
print(config.port) 
```

---

## 3. Advanced Validation

This is where the magic happens. You can ensure your configurations are not just the right *type*, but also have the right *value*.

```python
from pydantic import Field, DirectoryPath

class ServerSettings(BaseSettings):
    # Ensure port is within valid range
    port: int = Field(8000, gt=1024, le=65535)
    
    # Ensure this directory actually exists on the computer
    # If the path doesn't exist, the app crashes at startup!
    data_dir: DirectoryPath = "./data"
    
    # Ensure the string matches a specific pattern (e.g., lowercase only)
    environment: str = Field("development", pattern="^(development|production|staging)$")
```

---

## 4. Environment Prefixing

If your server has many apps, you might want to prefix your variables to avoid naming collisions (e.g., `WAITER_PORT` instead of just `PORT`).

```python
class WaiterSettings(BaseSettings):
    port: int = 8000
    
    model_config = SettingsConfigDict(env_prefix='WAITER_')

# Now it looks for "WAITER_PORT" in your .env or shell.
config = WaiterSettings()
```

---

## 5. Practical Example: Refactoring `ai_waiter_core`

Let's see how your current `config.py` would look if we migrated it:

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # --- Paths ---
    # We can use Pydantic to handle Path objects automatically
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent.parent.parent.resolve() / "data")

    # --- LLM Settings ---
    model_name: str = Field("qwen2.5:3b", env="MODEL_NAME")
    device: str = Field("cuda", env="DEVICE")
    
    # --- Server ---
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1, le=65535)

    # Tell Pydantic where the .env file is
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore" # Ignore extra variables in .env that aren't defined here
    )

settings = Settings()
```

---

## 💡 Summary of Benefits for AI Waiter

1.  **Fail-Fast**: If your `VECTOR_DB_PATH` is wrong, the app stops **immediately** with a clear message, rather than failing 2 minutes later when the first customer asks a question.
2.  **Type Safety**: Your IDE finally knows that `settings.PORT` is an `int`, not a `str`.
3.  **Default Handling**: Provide sensible defaults (like `device="cuda"`) while allowing easy overrides in `.env`.
4.  **Automatic Parsing**: No more `int(os.getenv(...))` boilerplate everywhere.
