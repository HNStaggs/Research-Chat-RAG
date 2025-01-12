import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import nullcontext
import logging
import time
import gc

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    def __init__(self):
        self.times = {}
    
    def start(self, operation):
        self.times[operation] = time.time()
    
    def end(self, operation):
        if operation in self.times:
            elapsed = time.time() - self.times[operation]
            logging.info(f"{operation} took {elapsed:.2f} seconds")
            del self.times[operation]

class GPUManager:
    """Manage GPU resources and memory"""
    @staticmethod
    def get_memory_info():
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(0)/1024**2,
                "reserved": torch.cuda.memory_reserved(0)/1024**2,
                "max": torch.cuda.get_device_properties(0).total_memory/1024**2
            }
        return None

    @staticmethod
    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def clear_memory():
    """Clear both CPU and GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class CodeGenerator:
    """Generate code using the model"""
    def __init__(self):
        self.model_name = "./models/santacoder"
        
        # CUDA setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        logging.info(f"Initializing CodeGenerator on {self.device}")
        if self.device == "cuda":
            logging.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            padding_side="left",
            truncation_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with CUDA optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        
        # Optimize for GPU if available
        if self.device == "cuda":
            self.model.to(self.device)
            torch.cuda.empty_cache()
            logging.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
            logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")

    def generate_code(self, prompt, max_length=500):
        try:
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            
            # Move inputs to correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use automatic mixed precision for faster inference
            with torch.cuda.amp.autocast() if self.device == "cuda" else nullcontext():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
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
            
            # Clean up generated code
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1]
            if "```" in generated_code:
                generated_code = generated_code.split("```")[0]
            
            return generated_code.strip()
            
        except Exception as e:
            logging.error(f"Error in code generation: {str(e)}")
            return f"Error generating code: {str(e)}"

    def generate_explanation(self, code):
        """Generate explanation for the code"""
        try:
            prompt = f"Explain this code:\n{code}\n\nExplanation:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=500,
                padding=True
            ).to(self.device)
            
            with torch.cuda.amp.autocast() if self.device == "cuda" else nullcontext():
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

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'device') and self.device == "cuda":
            torch.cuda.empty_cache()