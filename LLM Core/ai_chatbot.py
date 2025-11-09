# Add this to a new cell in your notebook
import json
from datetime import datetime
from rag_system import RAGSystem
from llama3 import LlamaModel

class AIWaiter:
    def __init__(self, menu_file_path, model_name='llama3.1:latest'):
        """
        Initialize AI waiter chatbot (Vietnamese) with RAG + Llama 3
        Args:
            menu_file_path (str): path to menu json file
            model_name (str): Ollama model identifier
        """
        self.menu_file_path = menu_file_path
        self.model_name = model_name

        # Components
        self.rag = None
        self.llama = None

        # Chat history
        self.conversation_history = []

        print('Initialize Chatbot...')
        self._initialize_systems()

    def _initialize_systems(self):
        """
        Init RAG + Llama model using Ollama
        """
        try:
            # Load RAG 
            print('Loading RAG system...')
            self.rag = RAGSystem(
                menu_file_path=self.menu_file_path
            )
            # Load Llama system 
            self.llama = LlamaModel(model_name=self.model_name)
            success = self.llama.load_model()
            
            if not success:
                raise Exception("Failed to load Llama model via Ollama")

            print(f'\u2705 Chatbot initialization complete!')
        except Exception as e:
            print(f'247c Error initializing chatbot: {e}')
            raise

    def _get_relevant_context(self, user_message, top_k=3): 
        """
        Get relevant context from RAG system 
        """
        try: 
            context = self.rag.get_context_for_llms(user_message, top_k= top_k) 
            return context

        except Exception as e: 
            print(f'\u274c Error when loading context for LLms')
            return 'No information available at the moment '
    def _create_system_prompt(self, context):
        """
        Create system prompt with menu context
        """
        system_prompt = f"""Bạn là "Linh", một nhân viên phục vụ thân thiện tại nhà hàng Việt Nam.

        THÔNG TIN MENU CÓ SẴN:
        {context}

        NHIỆM VỤ:
        - Giúp khách hàng tìm hiểu menu và đặt món
        - Tư vấn món ăn dựa trên thông tin menu có sẵn
        - Trả lời các câu hỏi về món ăn, nguyên liệu, giá cả
        - Gợi ý món ăn phù hợp với sở thích khách hàng

        PHONG CÁCH:
        - Thân thiện, nhiệt tình 
        - Trả lời bằng tiếng Việt tự nhiên
        - Giữ câu trả lời ngắn gọn nhưng đầy đủ thông tin
        - Hỏi thêm để hiểu rõ nhu cầu khách hàng

        CHÚ Ý:
        - Chỉ sử dụng thông tin từ menu có sẵn ở trên
        - Nếu không có thông tin về món nào đó, hãy thành thật nói và gợi ý món khác
        - Luôn đề cập giá cả khi giới thiệu món
        - Hỏi về sở thích, ngân sách nếu cần để tư vấn tốt hơn"""

        return system_prompt

    def _clean_response(self, response):
        """
        Clean and format the AI response
        """
        if not response:
            return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Bạn có thể hỏi lại không?"
        
        # Remove any unwanted patterns or clean up
        cleaned = response.strip()
        
        # Ensure it's not too long
        if len(cleaned) > 500:
            sentences = cleaned.split('.')
            cleaned = '. '.join(sentences[:3]) + '.'
        
        return cleaned

    def chat(self, user_message, max_new_tokens=300, temperature=0.7):
        """
        Main chat function that combines RAG + LLM
        Args:
            user_message (str): User's message
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature

        Returns:
            str: chatbot response
        """
        try:
            # Get relevant context from RAG
            context = self._get_relevant_context(user_message, top_k=3)

            # Create system prompt with context
            system_prompt = self._create_system_prompt(context)

            # Get chatbot response using Ollama
            response = self.llama.chat(
                user_message=user_message,
                system_message=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            # Clean response
            response = self._clean_response(response)

            # Save to conversation history
            self.conversation_history.append({
                'user': user_message,
                'assistant': response,
                'timestamp': datetime.now().isoformat(),
                'context_used': context[:100] + '...' if len(context) > 100 else context
            })

            return response

        except Exception as e:
            print(f'247c Error in chat: {e}')
            return "Xin lỗi, tôi gặp sự cố kỹ thuật. Bạn có thể hỏi lại không? 😅"

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")

    def save_conversation(self, filename):
        """Save conversation history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"💾 Conversation saved to {filename}")
        except Exception as e:
            print(f"\247c Error saving conversation: {e}")

    def get_stats(self):
        """Get chatbot statistics"""
        rag_stats = self.rag.get_stats() if self.rag else {}
        return {
            'conversations': len(self.conversation_history),
            'model_name': self.model_name,
            'rag_stats': rag_stats
        }
