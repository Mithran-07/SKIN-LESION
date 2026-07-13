"""
Deep Learning Benchmark Script
Runs a dummy CNN model to measure throughput, VRAM usage, and GPU capabilities.
Logs performance metrics to TensorBoard.
"""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import optimization configurations
from gpu_config import configure_gpu_performance, get_device, get_amp_settings

class DummyCNN(nn.Module):
    """
    A simple dummy convolutional network designed for benchmarking.
    Mimics typical operations of a lightweight classification network.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def run_benchmark(batch_size: int = 64, num_steps: int = 100, use_amp: bool = True):
    # Configure performance backends
    configure_gpu_performance(enable_benchmark=True)
    device = get_device()
    
    print(f"\n=========================================")
    print(f"Starting Benchmark on {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Capacity: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Parameters: Batch Size={batch_size}, Steps={num_steps}, AMP={use_amp}")
    print(f"=========================================\n")
    
    # Initialize TensorBoard writer
    log_dir = os.path.join("tensorboard", f"benchmark_amp_{use_amp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Setup dummy data and target label tensors
    # Typical dermoscopic classification image size is 224x224
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 7, (batch_size,), device=device)
    
    # Instantiate Model, Optimizer, and Loss
    model = DummyCNN(num_classes=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Setup Automatic Mixed Precision (AMP)
    amp_enabled, amp_dtype = get_amp_settings()
    # If AMP is requested but not supported, disable it
    if not use_amp or not amp_enabled:
        amp_enabled = False
        print("[BENCHMARK] Mixed precision is disabled or unsupported. Running in FP32.")
    else:
        print(f"[BENCHMARK] Mixed precision (AMP) enabled. Using: {amp_dtype}")
        
    # Standard Gradient Scaler is required only for float16. bfloat16 does not need scaling.
    scaler = torch.amp.GradScaler(device="cuda") if (amp_enabled and amp_dtype == torch.float16) else None
    
    # --- WARM-UP PHASE ---
    # Warm up CUDA driver and cuDNN autotuner to get accurate timings
    print("Warming up CUDA kernel and cuDNN benchmark...")
    for _ in range(10):
        optimizer.zero_grad()
        if amp_enabled:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print("Warm-up complete.\n")
    
    # --- BENCHMARK LOOP ---
    step_times = []
    
    print(f"Running benchmark for {num_steps} steps...")
    for step in range(num_steps):
        start_time = time.perf_counter()
        
        optimizer.zero_grad()
        
        if amp_enabled:
            # Autocast context for operations that can run faster in low precision
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        step_duration = time.perf_counter() - start_time
        step_times.append(step_duration)
        
        # Monitor memory allocation
        if torch.cuda.is_available():
            allocated_vram = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            reserved_vram = torch.cuda.memory_reserved(0) / (1024**2)    # MB
        else:
            allocated_vram = 0
            reserved_vram = 0
            
        throughput = batch_size / step_duration
        
        # Log to TensorBoard
        writer.add_scalar("Benchmark/Step_Time_ms", step_duration * 1000, step)
        writer.add_scalar("Benchmark/Throughput_Img_Sec", throughput, step)
        writer.add_scalar("Benchmark/VRAM_Allocated_MB", allocated_vram, step)
        writer.add_scalar("Benchmark/VRAM_Reserved_MB", reserved_vram, step)
        writer.add_scalar("Benchmark/Loss", loss.item(), step)
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | "
                  f"Step Time: {step_duration * 1000:.1f} ms | "
                  f"Throughput: {throughput:.1f} imgs/sec | "
                  f"Allocated VRAM: {allocated_vram:.1f} MB | "
                  f"Reserved VRAM: {reserved_vram:.1f} MB")
            
    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    avg_throughput = batch_size / avg_step_time
    
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Average Step Time: {avg_step_time * 1000:.2f} ms")
    print(f"Average Throughput: {avg_throughput:.2f} images/second")
    if torch.cuda.is_available():
        max_allocated = torch.cuda.max_memory_allocated(0) / (1024**2)
        max_reserved = torch.cuda.max_memory_reserved(0) / (1024**2)
        print(f"Peak VRAM Allocated: {max_allocated:.2f} MB")
        print(f"Peak VRAM Reserved: {max_reserved:.2f} MB")
        
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return {
        "avg_step_time_ms": avg_step_time * 1000,
        "avg_throughput": avg_throughput,
        "peak_vram_allocated_mb": max_allocated if torch.cuda.is_available() else 0.0,
        "peak_vram_reserved_mb": max_reserved if torch.cuda.is_available() else 0.0,
    }

if __name__ == "__main__":
    # Run FP32 benchmark
    print("\n>>> Running FP32 Benchmark <<<")
    run_benchmark(batch_size=64, num_steps=50, use_amp=False)
    
    # Run AMP Benchmark
    print("\n>>> Running AMP Benchmark <<<")
    run_benchmark(batch_size=64, num_steps=50, use_amp=True)
