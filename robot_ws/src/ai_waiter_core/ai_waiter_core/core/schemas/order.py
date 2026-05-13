from pydantic import BaseModel, Field
from typing import List, Optional

class OrderItem(BaseModel):
    name: str = Field(..., description="Name of the menu item")
    quantity: int = Field(..., description="Quantity of the item")
    price: float = Field(..., description="Price of the item")
    special_requests: Optional[str] = Field(None, description="Special requests (e.g., 'no onions')")

 
class Cart(BaseModel):
    items: List[OrderItem] = Field(..., description="List of items in the cart")
    total_price: float = Field(..., description="Total price of the cart")
    
    def calculate_total(self) -> float:
        """Helper to ensure total_price matches the sum of items"""
        return sum(item.price * item.quantity for item in self.items)


class OrderItemInput(BaseModel):
    name: str = Field(..., description="The exact name of the dish as per the menu")
    quantity: int = Field(..., ge=1, description="Number of items to order")
    special_requests: Optional[str] = Field(None, description="Any dietary preferences or modifications")


class VerifyOrderInput(BaseModel):
    items_request: List[OrderItemInput] = Field(..., description="The list of items the customer wants to order")