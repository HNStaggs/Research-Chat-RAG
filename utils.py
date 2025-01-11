import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CodeGenerator:
    def __init__(self):
        if not os.path.exists('./models/santacoder'):
            raise RuntimeError(
                "Models not found. Please run setup.py first to download required models."
            )
        
        self.model_name = "./models/santacoder"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )

    def generate_code(self, prompt, max_length=500):
        formatted_prompt = f"""
# Given the following context and request, generate code
# Context:
{prompt}

# Solution:
```python
"""
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the generated code
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1]
        if "```" in generated_code:
            generated_code = generated_code.split("```")[0]
            
        return generated_code.strip()

    def generate_explanation(self, code):
        prompt = f"""
Explain the following code in detail:

{code}

Explanation:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
