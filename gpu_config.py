"""
GPU Configuration and Optimization Module
Configures optimal settings for NVIDIA RTX 3050 GPU training runs.
"""

import os
import multiprocessing
import torch

def configure_gpu_performance(enable_benchmark: bool = True):
    """
    Configures PyTorch backend settings for optimal GPU acceleration.
    """
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner. Finds the best algorithms for the hardware when input sizes are constant.
        torch.backends.cudnn.benchmark = enable_benchmark
        # Ensure precision is set optimally for Tensor Cores.
        # TensorFloat-32 (TF32) is supported on Ampere (RTX 30-series) and later.
        torch.set_float32_matmul_precision('high')
        print(f"[GPU INFO] cuDNN benchmark set to {enable_benchmark}")
        print(f"[GPU INFO] float32 matmul precision set to high (TF32 allowed)")
    else:
        print("[GPU WARNING] CUDA is not available. Running on CPU.")

def get_device() -> torch.device:
    """
    Returns the target training device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_amp_settings():
    """
    Returns the optimal Automatic Mixed Precision (AMP) configuration.
    RTX 3050 (Ampere architecture) supports both float16 and bfloat16.
    """
    if not torch.cuda.is_available():
        return False, torch.float32
    
    # Check if bfloat16 is supported natively by the hardware
    # Ampere supports bfloat16 natively which prevents underflow/overflow problems without scaler.
    # Otherwise, fallback to float16 which requires GradScaler.
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16

def get_dataloader_kwargs(batch_size: int, num_workers: int = None) -> dict:
    """
    Generates optimal DataLoader arguments for the Lenovo LOQ system.
    Avoids multi-threading bottleneck and handles Windows IPC limits.
    """
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False, # Defaults to False, should be overriden as needed
    }
    
    if torch.cuda.is_available():
        # pin_memory speeds up data transfer from CPU memory to GPU VRAM
        kwargs["pin_memory"] = True
        
        # Determine optimal worker count
        if num_workers is None:
            # On Windows, PyTorch multi-processing can have high overhead or pagefile exhaustion.
            # A safe maximum is min(4, physical/logical CPU count).
            logical_cores = multiprocessing.cpu_count()
            # Default to 4 workers on our 12-thread system.
            kwargs["num_workers"] = min(4, logical_cores)
        else:
            kwargs["num_workers"] = num_workers
            
        # persistent_workers keeps worker processes alive between epochs
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = True
    else:
        kwargs["pin_memory"] = False
        kwargs["num_workers"] = 0
        
    return kwargs

if __name__ == "__main__":
    configure_gpu_performance()
    device = get_device()
    use_amp, amp_type = get_amp_settings()
    dl_kwargs = get_dataloader_kwargs(batch_size=32)
    
    print("\n--- GPU Configuration Summary ---")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"AMP Support: {use_amp} (dtype: {amp_type})")
    print(f"Optimal DataLoader parameters: {dl_kwargs}")
