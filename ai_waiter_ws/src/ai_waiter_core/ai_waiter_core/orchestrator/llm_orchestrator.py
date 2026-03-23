import json
import logging
import re

import ollama

from ai_waiter_core.actions.tool_dispatcher import search, place_order
from .prompts import SYSTEM_PROMPT
from ai_waiter_core.storage.retriever import build_vector_db, get_retriever
from ai_waiter_core.storage.order_db import OrderDB

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )


logger = logging.getLogger(__name__)


class AI_Waiter:
    def __init__(self, model_name="llama3.1"):
        """
        Initialize the AI Waiter.
        """
        self.model_name = model_name
        self.pending_order = None  # store order information
        self.system_prompt_logic = SYSTEM_PROMPT # router prompt
        self.chat_history = []  # chat history

    # Utility: parsing tool calls
    @staticmethod
    def extract_json(text_response):
        """
        Extract a JSON object from LLM response.

        Current strategy:
        - Try to parse the whole string as JSON.
        - If that fails, fall back to finding the first {...} block.
        - Validate that it has an "action" and "params" dict.
        """
        text = text_response.strip()

        # 1) Try direct JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "action" in obj:
                if "params" not in obj or not isinstance(obj["params"], dict):
                    obj["params"] = {}
                return obj
        except json.JSONDecodeError:
            pass

        # 2) Fallback: find first JSON-like block
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            json_str = match.group(0)
            obj = json.loads(json_str)
            if not isinstance(obj, dict) or "action" not in obj:
                return None
            if "params" not in obj or not isinstance(obj["params"], dict):
                obj["params"] = {}
            return obj
        except Exception:
            return None

    # Tool execution
    def execute_tool(self, action, params):
        """
        Execute tool based on action and parameters.
        Returns either structured data or an error string.
        """
        # === 1. SEARCH ===
        if action == "search":
            query = params.get("query", "")
            logger.info("[Search] Searching for: %r", query)
            try:
                result = search.invoke({"query": query})
            except Exception as e:
                logger.error("[Search error] %s", e)
                return {"error": "SEARCH_FAILED"}
            logger.info("[Search Results] %s", result)
            return result

        # === 2. DRAFT ORDER (two-step flow, confirmation comes later) ===
        elif action == "draft_order":
            item = params.get("item", "")
            quantity = params.get("quantity", 1)

            # Save to memory (single pending order)
            self.pending_order = {"item": item, "quantity": quantity}

            logger.info("[Draft] Saved to notepad: %sx %s", quantity, item)
            return {"status": "drafted", "item": item, "quantity": quantity}

        # === 3. CONFIRM ORDER (two-step flow) ===
        elif action == "confirm_order":
            decision = params.get("decision")

            if decision == "yes":
                # Check if we actually have an order waiting
                if self.pending_order:
                    item = self.pending_order["item"]
                    quantity = self.pending_order["quantity"]

                    logger.info("[Confirm] Sending to Kitchen: %sx %s", quantity, item)

                    try:
                        result = place_order.invoke({"item": item, "quantity": quantity})
                    except Exception as e:
                        logger.error("[Order error] %s", e)
                        self.pending_order = None
                        return "ERROR: Lỗi khi gửi đơn hàng xuống bếp."

                    # Clear the notepad
                    self.pending_order = None
                    logger.info("[Order Result] %s", result)
                    return result
                else:
                    return "ERROR: Khách đồng ý nhưng không có đơn hàng nháp."

            elif decision == "no":
                logger.info("[Cancel] Order cancelled by user.")
                self.pending_order = None
                return "CANCELED_BY_USER"

        # Unsupported action
        return "Hành động không được hỗ trợ."

    # LLM calls
    def _call_llm(self, messages, temperature):
        """
        Helper to call Ollama chat and return the assistant content text.
        """
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    # def _route(self, user_input):
    #     """
    #     First LLM pass: decide whether to call tools or just chat.
    #     Returns (tool_call_or_None, raw_text_response).
    #     """
    #     messages = [
    #         {"role": "system", "content": self.system_prompt_logic},
    #         *self.chat_history,
    #         {"role": "user", "content": user_input},
    #     ]

    #     first_response_text = self._call_llm(messages, temperature=0.1)
    #     tool_call = self.extract_json(first_response_text)
    #     return tool_call, first_response_text

    def _route(self, user_input):
        """
        First LLM pass: decide whether to call tools or just chat.
        Returns (tool_call_or_None, raw_text_response).
        """
        messages = [
            {"role": "system", "content": self.system_prompt_logic},
            *self.chat_history,
            {"role": "user", "content": user_input},
        ]

        first_response_text = self._call_llm(messages, temperature=0.1)
        # print(f"[DEBUG] LLM Raw Response: {first_response_text}")  # Add this line
        
        tool_call = self.extract_json(first_response_text)
        # print(f"[DEBUG] Extracted tool_call: {tool_call}")  # Add this line
        
        return tool_call, first_response_text

    def _build_final_system_prompt(self, action, tool_result):
        """
        Build the system prompt for the second LLM pass, based on the tool action and result.
        """
        # --- Scenario A: Search Result ---
        if action == "search":
            return (
                "Bạn là AI Waiter. Trả lời dựa trên thông tin dưới đây:\n"
                f"{tool_result}\n"
                "Yêu cầu: Trả lời ngắn gọn, dễ hiểu, thân thiện; nếu phù hợp thì mời khách dùng món.\n"
            )

        # --- Scenario B: Draft Order (Ask for confirmation) ---
        if action == "draft_order":
            item = tool_result.get("item", "")
            qty = tool_result.get("quantity", 1)
            return (
                "Yêu cầu:\n"
                f"1. Xác nhận lại với khách: \"Mình chốt {qty} {item} đúng không ạ?\"\n"
                "2. Nếu khách trả lời với ý ĐỒNG Ý (ví dụ: \"ok\", \"làm đi\", "
                "\"mang lên\", \"chuẩn\", \"ừ\", \"được\"...), hãy coi đó là xác nhận chốt đơn.\n"
            )

        # --- Scenario C: Confirmation Result ---
        if action == "confirm_order":
            if tool_result == "CANCELED_BY_USER":
                return "Khách vừa hủy đơn. Hãy nói: 'Dạ không sao ạ, mình muốn xem thêm menu không ạ?'"
            if "ERROR" in str(tool_result):
                return "Có lỗi xảy ra (khách đồng ý nhưng không có đơn). Hãy xin lỗi và hỏi lại khách muốn gọi gì."
            # Success
            return (
                "Đơn hàng đã được gửi xuống bếp thành công:\n"
                f"{tool_result}\n"
                "Yêu cầu: Cảm ơn khách và chúc ngon miệng.\n"
            )

        # Fallback (should not normally happen)
        return "Bạn là AI Waiter, hãy trả lời thân thiện và ngắn gọn cho khách."

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def chat(self, user_input):
        """
        Main chat method to process user input.
        - First pass: route and maybe decide to call a tool.
        - If tool is used: second pass to generate final answer conditioned on tool_result.
        - If no tool: return first LLM response.
        """
        print(f"\n[ Khách ] : {user_input}")

        # 1) First LLM pass (router)
        tool_call, first_response_text = self._route(user_input)

        # 2) If a valid tool call is detected
        if tool_call and tool_call.get("action"):
            action = tool_call["action"]
            params = tool_call.get("params", {})
            print(f"[System] Using tool: '{action}' with params: {params}")

            tool_result = self.execute_tool(action, params)

            # 3) Build final system prompt given the tool outcome
            final_system_prompt = self._build_final_system_prompt(action, tool_result)

            # 4) Second LLM pass for final answer 
            final_messages = [
                {"role": "system", "content": final_system_prompt},
                *self.chat_history,
                {"role": "user", "content": user_input},
            ]

            final_answer = self._call_llm(final_messages, temperature=0.7)

            # 5) Update history
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": final_answer})

            print(f"🤖 AI Waiter: {final_answer}")
            return final_answer

        # 3) No tool needed → simple conversation
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": first_response_text})

        print(f"🤖 AI Waiter: {first_response_text}")
        return first_response_text

    def initialize_system(self):
        """
        Placeholder for any future initialization logic (e.g., DB setup).
        """
        try:
            ok = build_vector_db() 
            if ok: 
                _ = get_retriever() # Load retriver into mem
                print('[System] RAG database is ready')
            else:
                print('[System] Fail to build RAG database')
        except Exception as e: 
            print(f'[System] Error while intializing RAG: {e}')

        # try:
        #     OrderDB() 
        #     print(f'[System] Orders database is ready') 
        # except Exception as e: 
        #     print(f'[System] Error while itializing orders DB: {e}')

    def clear_history(self):
        """Clear chat history if needed."""
        self.chat_history = []
        print("Chat history cleared.")



