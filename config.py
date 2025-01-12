# config.py
import torch

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_memory': {0: "4GB"},
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'cache_dir': "./models/cache",
    'similarity_k': 3,
    'model_dtype': torch.float16,
}