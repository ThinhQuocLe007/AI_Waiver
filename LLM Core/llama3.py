import ollama
import json 
import re 
from datetime import datetime

class LlamaModel: 
    def __init__(self, model_name = 'llama3.1:latest'): 
        """
        Initialize the llama model using ollama 
        Args: 
            model_name (str): Ollama model identifier 
            host (str): Ollama host server  
        
        """
        self.model_name = model_name
        self.available_functions = {} 
        self.client = None 
        self.tools = [] # For function calling 

    def load_model(self): 
        """
        Initialize Ollama client and check model availability 
        """
        try: 
            print(f'Connecting to Ollama server')
            self.client = ollama.Client()

            # Check if model exists 
            list_response = self.client.list() 
            model_objects = list_response.models 
            available_models = [obj.model for obj in model_objects]
            if self.model_name not in available_models: 
                print(f'\u274c Moldel {self.model_name} not found.Available models: ')
                for model in available_models: 
                    print(f'    - {model}')
                print(f'To install the model, run: ollama pull {self.model_name}')
                return False 
            
            print(f'\u2705 Model {self.model_name} loaded sucessfully via Ollama')
            return True
            
        except Exception as e: 
            print(f'\u274c Error when connect to Ollama: {e}')
            return False  
        
    def generate_text(self, prompt, max_new_tokens=256, temperature= 0.7, top_p=0.9): 
        """
        Generate text 
        """
        if self.client is None: 
            print('\u274c Client is not initialized, please load_model() first ! ')
            return None 

        try: 
            options = {
                'temperature': temperature, 
                'top_p': top_p, 
                'num_predict': max_new_tokens
            } 

            response = self.client.generate(
                model= self.model_name, 
                prompt= prompt, 
                options= options, 
                stream= False 
            )

            return response['response']

        except Exception as e: 
            print(f'\u274c Error during text generation')
            return None 
        
    def chat(self, user_message, system_message = 'You are an waiter in restaurant', max_new_tokens= 256, temperature= 0.7, top_p=0.9): 
        """
        Chat using Ollama's chat API 
        """

        if self.client is None: 
            print('\u274c Client is not initialized, please load_model() first ! ')
            return None 

        try: 
            messages = [
                {'role': 'system', 'content': system_message}, 
                {'role': 'user', 'content': user_message}
            ]

            options = {
                'temperature': temperature, 
                'top_p': top_p, 
                'num_predict': max_new_tokens
            } 

            response = self.client.chat(
                model= self.model_name, 
                messages= messages,
                options= options, 
                stream= False 
            )

            return response['message']['content']

        except Exception as e: 
            print(f'\u274c Error during text generation')
            return None     

    # Function calling  
    def register_function(self, name, func, description, parameters):
        """
        Register a function for Ollama function calling.
        """
        self.available_functions[name] = func
        
        # Create Ollama tool definition
        tool = {
            'type': 'function',
            'function': {
                'name': name,
                'description': description,
                'parameters': parameters
            }
        }
        
        # Add to tools list if not already present
        existing_names = [t['function']['name'] for t in self.tools if t['type'] == 'function']
        if name not in existing_names:
            self.tools.append(tool)
            print(f"✅ Function '{name}' registered successfully")

    def chat_with_functions(self, user_message, system_message="You are a helpful assistant.", 
                        max_new_tokens=512, temperature=0.1, top_p=0.9):
        """
        Chat with native Ollama function calling support
        """
        if self.client is None:
            print("❌ Ollama client not initialized. Call load_model() first.")
            return None
            
        if not self.tools:
            print("⚠️ No functions registered. Using regular chat.")
            return self.chat(user_message, system_message, max_new_tokens, temperature, top_p)
        
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            options = {
                'temperature': temperature,
                'top_p': top_p,
                'num_predict': max_new_tokens,
            }
            
            # First call with function tools
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                options=options
            )
            
            # Check if the model wants to call functions
            if response['message'].get('tool_calls'):
                print("🔧 AI is calling functions...")
                
                # Add AI's response to conversation
                messages.append(response['message'])
                
                # Process each function call
                for call in response['message']['tool_calls']:
                    function_name = call['function']['name']
                    
                    # Handle arguments
                    if isinstance(call['function']['arguments'], str):
                        function_args = json.loads(call['function']['arguments'])
                    else:
                        function_args = call['function']['arguments']
                    
                    print(f"📞 Calling: {function_name}({function_args})")
                    
                    # Execute function
                    if function_name in self.available_functions:
                        try:
                            function_result = self.available_functions[function_name](**function_args)
                            print(f"📋 Result: {function_result[:100]}...")  # Truncate for display
                            
                            # Add function result to conversation
                            messages.append({
                                'role': 'tool',
                                'content': str(function_result),
                            })
                            
                        except Exception as e:
                            print(f"❌ Function execution error: {e}")
                            messages.append({
                                'role': 'tool', 
                                'content': f"Error executing function: {e}",
                            })
                    else:
                        print(f"❌ Function {function_name} not found")
                
                # Get final response
                final_response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options
                )
                
                return final_response['message']['content']
            
            else:
                # No function calls needed
                return response['message']['content']
                
        except Exception as e:
            print(f'❌ Error in chat_with_functions: {e}')
            return f"Error: {e}"