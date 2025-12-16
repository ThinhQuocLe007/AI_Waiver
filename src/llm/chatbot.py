import ollama 
import json 
import re 
from src.tool.actions import search, place_order 
from src.llm.prompts import SYSTEM_PROMPT

class AI_Waiter:
    def __init__(self, model_name="llama3.1"):
        """
        Initialize the AI Waiter
        """
        self.model_name = model_name
        self.pending_order = None  
        self.system_prompt_logic = SYSTEM_PROMPT

    @staticmethod
    def extract_json(text_response):
        """Extract JSON from Llama's response text"""
        try:
            match = re.search(r"\{.*\}", text_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except Exception as e:
            return None
        return None

    def execute_tool(self, action, params):
        """Execute tool based on action and parameters"""
        
        # === 1. SEARCH (Unchanged) ===
        if action == "search":
            query = params.get("query", "")
            print(f"🔍 [Search] Searching for: '{query}'...")
            result = search.invoke({"query": query})
            print(f"📄 [Search Result] Found:\n{result}")
            return result
            
        elif action == "draft_order":
            item = params.get("item", "")
            quantity = params.get("quantity", 1)
            
            # Save to memory
            self.pending_order = {"item": item, "quantity": quantity}
            
            print(f"📝 [Draft] Saved to notepad: {quantity}x {item}")
            return {"status": "drafted", "item": item, "quantity": quantity}

        elif action == "confirm_order":
            decision = params.get("decision")
            
            if decision == "yes":
                # Check if we actually have an order waiting
                if self.pending_order:
                    item = self.pending_order["item"]
                    quantity = self.pending_order["quantity"]
                    
                    print(f"🚀 [Confirm] Sending to Kitchen: {quantity}x {item}")
                    
                    result = place_order.invoke({"item": item, "quantity": quantity})
                    
                    # Clear the notepad
                    self.pending_order = None
                    print(f"✅ [Order Result] {result}")
                    return result
                else:
                    return "ERROR: Khách đồng ý nhưng không có đơn hàng nháp."
            
            elif decision == "no":
                print("❌ [Cancel] Order cancelled.")
                self.pending_order = None
                return "CANCELED_BY_USER"
            
        else:
            return "Hành động không được hỗ trợ."

    def chat(self, user_input):
        """Main chat method to process user input"""
        print(f"\n👤 Khách: {user_input}")
        
        # === STEP 1: DECISION MAKING ===
        messages = [
            {"role": "system", "content": self.system_prompt_logic},
            {"role": "user", "content": user_input}
        ]
        
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            options={"temperature": 0.1}
        )
        
        first_response_text = response['message']['content']
        
        # === STEP 2: CHECK & EXECUTE TOOLS ===
        tool_call = self.extract_json(first_response_text)
        
        if tool_call and tool_call.get("action"):
            action = tool_call["action"]
            params = tool_call.get("params", {})
            
            print(f"⚙️ [System] Using tool: '{action}' with params: {params}")
            
            # Execute tool
            tool_result = self.execute_tool(action, params)
            
            # === STEP 3: SYNTHESIS (Generating the polite reply) ===
            final_system_prompt = ""

            # --- Scenario A: Search Result ---
            if action == "search":
                final_system_prompt = f"""
                Bạn là AI Waiter. Trả lời dựa trên thông tin:
                {tool_result}
                Yêu cầu: Ngắn gọn, mời khách dùng món nếu phù hợp.
                """
            
            # --- Scenario B: Draft Order (Ask for confirmation) ### NEW ---
            elif action == "draft_order":
                item = tool_result['item']
                qty = tool_result['quantity']
                final_system_prompt = f"""
                Bạn vừa ghi lại: {qty} phần {item}.
                Yêu cầu: 
                1. Xác nhận lại với khách: "Có phải anh/chị muốn gọi {qty} {item} không ạ?"
                2. Chờ khách nói "Có" hoặc "Chốt" mới được gửi xuống bếp.
                """

            # --- Scenario C: Confirmation Result ### NEW ---
            elif action == "confirm_order":
                if tool_result == "CANCELED_BY_USER":
                    final_system_prompt = "Khách vừa hủy đơn. Hãy nói: 'Dạ không sao ạ, mình muốn xem thêm menu không ạ?'"
                elif "ERROR" in str(tool_result):
                    final_system_prompt = "Có lỗi xảy ra (khách đồng ý nhưng không có đơn). Hãy xin lỗi và hỏi lại khách muốn gọi gì."
                else:
                    # Success!
                    final_system_prompt = f"""
                    Đơn hàng đã được gửi xuống bếp thành công:
                    {tool_result}
                    Yêu cầu: Cảm ơn khách và chúc ngon miệng.
                    """
            
            # Generate final response
            final_messages = [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            final_output = ollama.chat(
                model=self.model_name, 
                messages=final_messages,
                options={"temperature": 0.7}
            )
            
            final_answer = final_output['message']['content']
            print(f"🤖 AI Waiter: {final_answer}")
            return final_answer
            
        else:
            # Simple conversation (no tools needed)
            print(f"🤖 AI Waiter: {first_response_text}")
            return first_response_text

    def initiaize_system(self): 
        # You can implement your DB setup here later
        print("System initialized.")