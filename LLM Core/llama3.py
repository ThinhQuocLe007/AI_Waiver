import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LlamaModel:
    def __init__(self, model_name, hf_token, device='auto', quantize=True):
        """
        Initialize the Llama model with optional quantization.
        Args:
            model_name (str): Hugging Face model identifier.
            hf_token (str): Hugging Face authentication token.
            device (str): Device map ('auto', 'cpu', 'cuda').
            quantize (bool): Whether to apply 4-bit quantization.
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.quantize = quantize
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Load a model and tokenizer from Hugging Face with quantization.
        """
        try:
            #TODO: DONT UNDERSTAND HERE 
            quantization_config = None
            if self.quantize:
                print("Setting up 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for better performance
                    bnb_4bit_use_double_quant=False,
                )

            # Load tokenizer
            print(f'Loading tokenizer for {self.model_name}...')
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token # Use 'token' instead of the deprecated 'use_auth_token'
            )

            if self.tokenizer.chat_template is None:
                print("⚠️ Chat template not found. Manually setting Llama 3.1 template.")
                # This is the official chat template for Llama 3.1
                self.tokenizer.chat_template = (
                    "{% set loop_messages = messages %}"
                    "{% for message in loop_messages %}"
                        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                        "{{ content }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                    "{% endif %}"
                )

            # Load model
            print(f'Loading model {self.model_name}...')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                dtype=torch.bfloat16, # bfloat16 is preferred for modern GPUs
                device_map=self.device,
                quantization_config=quantization_config, # Pass the quantization config here
                trust_remote_code=True,
            )

            # Set pad token if it does not exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print('✅ Model loaded successfully.')
        except Exception as e:
            print(f'Error loading model: {e}')

    def generate_text(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        Generate text base on loaded model 
        """
        if self.model is None and self.tokenizer is None: 
            print("Model and tokenizer are not loaded. Call load_model() first.")
            return None 

        try:
            # Tokenize input prompt 
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=4096)
            inputs = {k : v.to(self.model.device) for k,v in inputs.items()}

            # Generate text 
            with torch.no_grad(): 
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the original prompt from the generated text
            response = generated_text[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            print(f'Error during text generation: {e}')
            return None
        
    def chat(self, user_message, system_message="You are a helpful assistant.", max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        Simulate a chat interaction with system and user messages.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message    }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt=True
        )

        return self.generate_text(prompt, max_new_tokens= 512)
    
    def get_memory_usage(self): 
        """
        Check GPU memory usage if available.
        """
        if torch.cuda.is_available(): 
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"GPU Memory Allocated: {allocated:.2f} GB")
            print(f"GPU Memory Reserved: {reserved:.2f} GB")
        else: 
            print("CUDA is not available. Cannot check GPU memory usage.")