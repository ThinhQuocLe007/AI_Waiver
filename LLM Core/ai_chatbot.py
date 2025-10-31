import json 
import time 
from datetime import datetime
from rag_system import RAGSystem
from llama3 import LlamaModel 

class AIWaiter: 
    def __init__(self, menu_file_path, model_name, hf_token): 
        """
        Initialize AI waiter chatbot (vietnamese)  with RAG + Llama 3 
        Args: 
            menu_file_path (str): path to menu json file 
            model_name (str): Hugging face model identifier 
            hf_token(str): hugging face authentication token  
        """
        self.menu_file_path = menu_file_path
        self.model_name = model_name
        self.hf_token = hf_token

        # Components 
        self.rag = None 
        self.llama = None 

        # Chat history 
        self.conversation_history = [] 

        print('Initialize Chatbot')
        self._initialize_systems()
    
    def chat(self, user_message, max_new_tokens=300, temperature= 0.7): 
        """
        Main chat function that combines RAG + LLMS 
        Args: 
            user_message (str): User's message
            max_new_tokens (int): Maximum tokens to generate 
            temperature (float): Sampling temperature 

        Return 
            str: chatbot response 
        """
        try: 
            # Get relevant context from RAG 
            context = self._get_relevant_context(user_message, top_k= 3) 

            # Create system prompt with context 
            system_prompt = self._create_system_prompt(context)

            # Chatbot response 
            response = self.llama.chat(
                user_message= user_message, 
                system_message= system_prompt, 
                max_new_tokens= max_new_tokens, 
                temperature= temperature
            )

            # Clean response 
            response = self._clean_response(response) 

            # Save to conversation history 
            self.conversation_history.append({
                'user': user_message, 
                'assisstant': response, 
                'timestamp': datetime.now().isoformat(), 
                'context_used': context[:100] + '...' if len(context) > 100 else context
            }) 

            return response

        except Exception as e:  
            print(f'Error in chat: {e}')
            return "Sorry, have error. "
        
    def vietnamese_chatbot(self, user_message): 
        """Optimized chat for Vietnamese language"""
        enhanced_message = f'Vietnamese customer asking: {user_message}'
        return self.chat(enhanced_message)
    
    def clear_conversation_history(self): 
        """Clear conversation history"""
        self.conversation_history = [] 
        print("Conversation history cleared")

    def save_conversation(self, filename): 
        """Save converstaion history to file"""
        try: 
            with open(filename, 'w', encoding= 'utf-8') as f: 
                json.dump(self.conversation_history, f, ensure_ascii= False)
                print(f'Conversation saved to {filename}')
        except Exception as e: 
            print(f"Error when saving conversation: {e}")


    def _initialize_systems(self): 
        """
        Init RAG + LLMs model 
        """
        try:
            # Load Llama model  
            self.llama = LlamaModel(
                model_name= self.model_name, 
                hf_token= self.hf_token,
                device= 'auto', 
                quantize= True
            )
            self.llama.load_model() 

            # Load RAG system 
            self.rag = RAGSystem(
                menu_file_path= self.menu_file_path, 

            )

            print(f'\u2705 Chatbot initialization complete !')
        except Exception as e: 
            print(f'\u247c Error initializing chatbot: {e}')
            raise 


    def _get_relevant_context(self, user_message, top_k=3): 
        """
        Get relevant menu context from RAG system  
        """
        try: 
            context = self.rag.get_context_for_llms(user_message, top_k= top_k) 
            return context 
        except Exception as e: 
            print(f'Error when getting context: {e}')
            return f"No menu information avaiable for user request: {user_message}"
    
    def _create_system_prompt(self, context):
        """Create system prompt with menu context"""
        system_prompt = f"""You are "Linh", a friendly Vietnamese restaurant assistant.

                            MENU INFORMATION:
                            {context}

                            YOUR ROLE:
                            - Help customers with menu recommendations based on available dishes above
                            - Be warm, welcoming, and knowledgeable about Vietnamese cuisine
                            - Include prices in Vietnamese Dong (VND) when making recommendations
                            - You can speak both Vietnamese and English fluently
                            - Be enthusiastic about Vietnamese food culture

                            RESPONSE GUIDELINES:
                            - Keep responses concise but informative (2-3 sentences max)
                            - Always recommend specific dishes from the menu above
                            - Include prices when relevant
                            - If asked about unavailable items, politely redirect to available options
                            - Use friendly tone with occasional food emojis 🍜🥢

                            CONVERSATION STYLE:
                            - Natural and conversational
                            - Ask follow-up questions to help customers decide
                            - Provide brief descriptions of recommended dishes
                            - Be helpful and patient"""

        return system_prompt
    

    def _clean_response(self, response): 
        """Clean and format the response"""
        if not response: 
            return f'Error when cleaning response'
        
        response = response.strip() 

        if len(response) > 500: 
            response = response[: 497] + "..."

        return response
    

