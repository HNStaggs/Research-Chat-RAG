import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import nullcontext
import logging
import time

class GPUManager:
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

class CodeGenerator:
    def __init__(self):
        self.model_name = "./models/santacoder"
        
        # CUDA setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Initializing CodeGenerator on {self.device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'N/A'}")
        
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
        
        # Optimize for RTX A2000
        if self.device == "cuda":
            self.model.to(self.device)
            torch.cuda.empty_cache()
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")

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
            
            # Move inputs to GPU
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

    def __del__(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'device') and self.device == "cuda":
            torch.cuda.empty_cache()