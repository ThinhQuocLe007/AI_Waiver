import json 
from datetime import datetime

class AI_Tools: 
    def __init__(self): 
        pass 

    def create_simple_order(self, dish_name, quatity=1): 
        """
        Create order  
        """
        order_id = f'ORD_1'
        total_price = quatity * 50000 
        return f'Đặt món thành công, mã đơn hàng: 1, tổng thanh toán {total_price}'
    
    def get_time(self): 
        """
        Get current time
        """
        now = datetime.now()
        return f'Bây giờ là: {now.strftime('%H%M')} '
    

TOOLS_DEFINITION = [
    {
        'type': 'function',
        'function': {
            'name': 'create_simple_order',
            'description': 'Đặt món khi khách muốn order',
            'parameters': {
                'type': 'object',
                'properties': {
                    'dish_name': {'type': 'string', 'description': 'Tên món ăn'},
                    'quantity': {'type': 'number', 'description': 'Số lượng', 'default': 1}
                },
                'required': ['dish_name']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_time',
            'description': 'Lấy thời gian hiện tại khi khách hỏi giờ',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            }
        }
    }
]