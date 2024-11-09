import os
import sys
import torch

def check_cuda_environment():
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA Environment variables
    cuda_home = os.environ.get('CUDA_HOME')
    cuda_path = os.environ.get('CUDA_PATH')
    ld_library_path = os.environ.get('LD_LIBRARY_PATH')
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    print("\nCUDA Environment Variables:")
    print(f"CUDA_HOME: {cuda_home}")
    print(f"CUDA_PATH: {cuda_path}")
    print(f"LD_LIBRARY_PATH: {ld_library_path}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # Check if CUDA is available
    print("\nPyTorch CUDA Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available")
        
    # Try to create a CUDA tensor
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        print("\nSuccessfully created CUDA tensor:", x)
    except Exception as e:
        print("\nError creating CUDA tensor:", str(e))

if __name__ == "__main__":
    check_cuda_environment()
