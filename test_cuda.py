import torch
import sys

def check_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test CUDA memory
    if torch.cuda.is_available():
        try:
            x = torch.rand(1000, 1000).cuda()
            print("Successfully allocated CUDA tensor")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error allocating CUDA tensor: {e}")

if __name__ == "__main__":
    check_cuda()