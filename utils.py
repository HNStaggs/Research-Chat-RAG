from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import time

class PerformanceMonitor:
    def __init__(self):
        self.times = {}
    
    def start(self, operation):
        self.times[operation] = time.time()
    
    def end(self, operation):
        if operation in self.times:
            elapsed = time.time() - self.times[operation]
            logging.info(f"{operation} took {elapsed:.2f} seconds")
            del self.times[operation]

def clear_memory():
    """Clear unused memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class CodeGenerator:
    def __init__(self):
        self.model_name = "./models/santacoder"
        
        # Initialize tokenizer with padding
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            padding_side="left",
            truncation_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Force CPU usage and optimize memory
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        # Move model to CPU explicitly
        self.model = self.model.to("cpu")
        
        logging.info("CodeGenerator initialized on CPU")

    def generate_code(self, prompt, max_length=500):
        try:
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to("cpu")
            
            # Generate with max_new_tokens instead of max_length
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_code = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up the generated code
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1]
            if "```" in generated_code:
                generated_code = generated_code.split("```")[0]
                
            return generated_code.strip()
            
        except Exception as e:
            logging.error(f"Error in code generation: {str(e)}")
            return f"Error generating code: {str(e)}"

    def generate_explanation(self, code):
        try:
            prompt = f"Explain this code:\n{code}\n\nExplanation:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=500,
                padding=True
            ).to("cpu")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1
            )
            
            return self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
        except Exception as e:
            logging.error(f"Error in explanation generation: {str(e)}")
            return f"Error generating explanation: {str(e)}"