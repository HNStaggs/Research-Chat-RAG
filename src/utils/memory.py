import gc
import torch
import psutil
import logging

def clear_memory():
    """Clear system memory and GPU cache"""
    try:
        # Clear Python garbage collector
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
    except Exception as e:
        logging.error(f"Error clearing memory: {e}")
        return False

def get_memory_usage():
    """Get current memory usage statistics"""
    try:
        memory_stats = {
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available": psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
            "ram_used": psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
        }
        
        if torch.cuda.is_available():
            memory_stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024 * 1024 * 1024),  # GB
                "gpu_memory_cached": torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # GB
            })
            
        return memory_stats
    except Exception as e:
        logging.error(f"Error getting memory stats: {e}")
        return None 