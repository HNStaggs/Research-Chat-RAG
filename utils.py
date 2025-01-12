import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import nullcontext, contextmanager
import logging
import time
import gc
import warnings

# Configure logging
logging.basicConfig(
    filename='model_operations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@contextmanager
def gpu_memory_manager():
    """Context manager for GPU memory operations"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
        # Suppress warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        try:
            self.model_name = "./models/santacoder"
            
            # CUDA setup with error handling
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            logging.info(f"Initializing CodeGenerator on {self.device}")
            if self.device == "cuda":
                logging.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            
            # Initialize tokenizer with padding
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                padding_side="left",
                truncation_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
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
                
        except Exception as e:
            logging.error(f"Error in initialization: {str(e)}")
            raise

    def generate_code(self, prompt, max_length=500, language="python"):
        """Generate code with improved quality and formatting"""
        try:
            with gpu_memory_manager():
                formatted_prompt = f"""
Please provide clean, well-formatted {language} code for the following request.
Use proper indentation and add explanatory comments.

{prompt}

Response format:
```{language}
// Your code here
"""
                
                # Tokenize with proper truncation
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            
            # Move inputs to correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with optimized parameters
            with autocast('cuda') if self.device == "cuda" else nullcontext():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.4,
                    top_p=0.85,
                    top_k=50,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            generated_code = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract code from between backticks
            code_pattern = "```"
            if code_pattern in generated_code:
                code_blocks = generated_code.split(code_pattern)
                if len(code_blocks) >= 3:
                    generated_code = code_blocks[1]
                    if generated_code.startswith(('python', 'javascript', 'java', 'cpp', 'sql')):
                        generated_code = generated_code.split('\n', 1)[1]
            
        # Clean up and validate
        generated_code = generated_code.strip()
        if not ModelUtils.validate_output(generated_code):
            raise ValueError("Generated code appears to be invalid or too short")
            
        return generated_code

    except Exception as e:
        logging.error(f"Error in code generation: {str(e)}")
        return f"Error generating code: {str(e)}"

def generate_explanation(self, code):
    """Generate explanation for the code with improved clarity"""
    try:
        with gpu_memory_manager():
            prompt = f"""
            Please provide a clear, detailed explanation of the following code:
            {code}
            Explain:
            - What the code does
            - Key components and their purpose
            - Any important considerations
        Explanation:
        """
            
            # Tokenize with proper truncation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=500,
                padding=True
            ).to(self.device)
            
            # Generate with optimized parameters
            with autocast('cuda') if self.device == "cuda" else nullcontext():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    top_p=0.85,
                    top_k=50,
                    num_beams=3,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    num_return_sequences=1,
                    repetition_penalty=1.2
                )
            
            explanation = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up and validate
            explanation = explanation.replace("Explanation:", "").strip()
            if len(explanation) < 20:
                raise ValueError("Generated explanation appears to be too short")
            
            return explanation
            
    except Exception as e:
        logging.error(f"Error in explanation generation: {str(e)}")
        return f"Error generating explanation: {str(e)}"

def __del__(self):
    """Cleanup when object is destroyed"""
    try:
        if hasattr(self, 'device') and self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

class ModelUtils:
"""Utility functions for model operations"""
@staticmethod
def format_code(code: str, language: str) -> str:
"""Format code based on language"""
try:
code = code.strip().replace('\r\n', '\n')

if language.lower() == 'python':
            lines = code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.endswith(':'):
                    formatted_lines.append('    ' * indent_level + stripped_line)
                    indent_level += 1
                elif stripped_line.startswith(('return', 'break', 'continue')):
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append('    ' * indent_level + stripped_line)
                else:
                    formatted_lines.append('    ' * indent_level + stripped_line)
            
            return '\n'.join(formatted_lines)
        
        return code
        
    except Exception as e:
        logging.error(f"Error formatting code: {str(e)}")
        return code

@staticmethod
def validate_output(code: str, min_length: int = 10, min_lines: int = 2) -> bool:
    """Validate the generated code output"""
    try:
        if not code or not isinstance(code, str):
            return False
        if len(code.strip()) < min_length:
            return False
        if code.count('\n') < min_lines:
            return False
        error_indicators = ['error:', 'exception:', 'failed:', 'undefined']
        if any(indicator in code.lower() for indicator in error_indicators):
            return False
        return True
    except Exception as e:
        logging.error(f"Error in validate_output: {str(e)}")
        return False