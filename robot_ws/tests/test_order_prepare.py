import sys
import os
from pathlib import Path

# Resolve project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent

from ai_waiter_core.tools.search.menu_manager import MenuManager
from ai_waiter_core.tools.verify_prepare_order_tool import verify_and_prepare_order

def test_menu_manager():
    """Test that MenuManager loads prices correctly"""
    print("--- Test 1: MenuManager Lookup ---")
    menu = MenuManager()
    
    # Test exact match
    price = menu.get_price("Phở Bò Đặc Biệt")
    print(f"  Phở Bò Đặc Biệt -> {price}")
    assert price == 75000.0, f"Expected 75000.0, got {price}"
    
    # Test case-insensitive
    price2 = menu.get_price("phở bò đặc biệt")
    assert price == price2, "Case-insensitive lookup failed"
    
    # Test non-existent
    price3 = menu.get_price("Pizza Hawaii")
    assert price3 is None, "Should return None for missing items"
    
    print("  PASSED ✓\n")

def test_valid_order():
    """Test the tool with valid menu items"""
    print("--- Test 2: Valid Order ---")
    result = verify_and_prepare_order.invoke({
        "items_request": [
            {"name": "Phở Bò Đặc Biệt", "quantity": 2},
            {"name": "Cà Phê Sữa Đá Sài Gòn", "quantity": 1, "special_requests": "ít đường"}
        ]
    })
    print(f"  Result: {result[:100]}...")
    assert "PENDING_CART" in result, "Should return a PENDING_CART"
    assert "150000" in result or "179000" in result, "Total should reflect real prices"
    print("  PASSED ✓\n")

def test_missing_item():
    """Test the tool with an item NOT in the menu"""
    print("--- Test 3: Missing Item ---")
    result = verify_and_prepare_order.invoke({
        "items_request": [
            {"name": "Pizza Hawaii", "quantity": 1}
        ]
    })
    print(f"  Result: {result}")
    assert "ERROR" in result, "Should return an ERROR for unknown items"
    assert "Pizza Hawaii" in result, "Should mention the missing item name"
    print("  PASSED ✓\n")

def test_special_requests():
    """Test that special_requests are preserved in the cart"""
    print("--- Test 4: Special Requests ---")
    result = verify_and_prepare_order.invoke({
        "items_request": [
            {"name": "Phở Bò Đặc Biệt", "quantity": 1, "special_requests": "không hành, thêm giá"}
        ]
    })
    print(f"  Result: {result[:120]}...")
    assert "không hành" in result, "Special requests should be in the output"
    print("  PASSED ✓\n")

if __name__ == "__main__":
    test_menu_manager()
    test_valid_order()
    test_missing_item()
    test_special_requests()
    print("=== All tests passed! ===")
