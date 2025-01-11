from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os

def download_models():
    print("Downloading models...")
    
    # Create models directory
    os.makedirs('./models/santacoder', exist_ok=True)
    os.makedirs('./models/sentence-transformer', exist_ok=True)
    
    # Download code model
    print("Downloading SantaCoder...")
    model_name = "bigcode/santacoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save locally
    tokenizer.save_pretrained('./models/santacoder')
    model.save_pretrained('./models/santacoder')
    
    # Download embedding model
    print("Downloading Sentence Transformer...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embed_model.save('./models/sentence-transformer')
    
    print("Models downloaded successfully!")

if __name__ == "__main__":
    download_models()
