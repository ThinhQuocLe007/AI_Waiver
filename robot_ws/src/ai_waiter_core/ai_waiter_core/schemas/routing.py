from pydantic import BaseModel, Field
from typing import Optional, Literal

IntentType = Literal["ORDER", "MENU", "PAYMENT", "CHAT", "COMPLEX"]

class IntentPrediction(BaseModel):
    """The result of the SLM intent routing."""
    intent: Optional[IntentType] = Field(
        description="Classified user intent. COMPLEX for multi‑action queries."
    )
    reasoning: str = Field(description="Brief rationale for the classification.")