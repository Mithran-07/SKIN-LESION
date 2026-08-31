# Deep Learning Workstation Environment Report

This report documents the configuration and performance baseline of your Lenovo LOQ workstation, prepared for your research project: **"Dual-Branch CNN Framework for Non-Melanoma Dermoscopic Classification"**.

---

## 1. System Hardware Verification

*   **Operating System:** Windows 11 (v10.0.26200)
*   **Processor (CPU):** Intel64 Family 6 Model 151 Stepping 2, GenuineIntel (8 Physical Cores, 12 Threads)
*   **System RAM:** 15.71 GB total (6.88 GB free)
*   **Storage (SSD):** 460.28 GB total partition (72.21 GB available for datasets/checkpoints)

---

## 2. NVIDIA GPU & CUDA Acceleration Details

*   **GPU Model:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere architecture)
*   **VRAM Capacity:** 6.00 GB (6144 MB)
*   **Driver Version:** NVIDIA Driver 610.62
*   **CUDA Support:** Driver supports up to CUDA 13.3; PyTorch runtime is linked with **CUDA 12.4**
*   **cuDNN Library:** v9.1.0 (90100)
*   **Compute Capability:** 8.6 (Supports native Float16 and BFloat16 Mixed Precision)

---

## 3. Python Environment Details

*   **Environment Type:** Isolated virtual environment (created with `uv` under `.venv`)
*   **Python Version:** CPython 3.12.13 (highly stable version for deep learning binaries on Windows)
*   **Python Executable Path:** `c:\Users\mithr\OneDrive\Documents\ADL\.venv\Scripts\python.exe`

### Installed Packages Summary

All required packages have been successfully installed and verified via automated imports:

| Package | Version | Status | Notes |
| :--- | :--- | :--- | :--- |
| **PyTorch** | `2.6.0+cu124` | [OK] | With CUDA 12.4 support |
| **torchvision** | `0.21.0+cu124` | [OK] | Precompiled for PyTorch 2.6.0 |
| **torchaudio** | `2.6.0+cu124` | [OK] | Precompiled for PyTorch 2.6.0 |
| **timm** | `1.0.27` | [OK] | PyTorch Image Models library |
| **albumentations** | `2.0.8` | [OK] | Fast image augmentation |
| **opencv-python** | `5.0.0` | [OK] | OpenCV bindings (`cv2`) |
| **scikit-learn** | `1.9.0` | [OK] | Machine Learning utilities |
| **pandas** | `3.0.3` | [OK] | Data analysis framework |
| **numpy** | `2.4.4` | [OK] | Array computation |
| **matplotlib** | `3.11.0` | [OK] | Plotting & visualization |
| **seaborn** | `0.13.2` | [OK] | Statistical visualizations |
| **torchmetrics** | `1.9.0` | [OK] | DL evaluation metrics |
| **tqdm** | `4.68.4` | [OK] | Progress bars |
| **pyyaml** | `6.0.3` | [OK] | YAML configuration loader |
| **pillow** | `12.2.0` | [OK] | PIL Image operations |
| **jupyterlab** | `4.6.1` | [OK] | Notebook interface |
| **tensorboard** | `2.21.0` | [OK] | Experiment logging |
| **pytest** | `9.1.1` | [OK] | Unit testing |
| **rich** | `15.0.0` | [OK] | Console output formatter |

---

## 4. Benchmark Performance Baseline

We executed a classification network benchmark (Dummy CNN mimicking a lightweight ResNet structure) on your GPU, processing random 224x224 images (dermoscopic dimensions) at a batch size of 64.

### Performance Comparison: FP32 vs. AMP (BFloat16)

| Metric | FP32 Precision | AMP Mixed Precision (BFloat16) | Speedup / Gain |
| :--- | :--- | :--- | :--- |
| **Average Step Time** | 716.49 ms | 204.14 ms | **3.51x Faster** |
| **Training Throughput** | 89.32 images/sec | 313.51 images/sec | **3.51x Throughput** |
| **Peak VRAM Allocated** | 4801.37 MB | 4801.37 MB | Equivalent |
| **Peak VRAM Reserved** | 6914.00 MB | 6914.00 MB | Cache buffer overhead |

> [!TIP]
> **Key Finding:** Enabling Automatic Mixed Precision (AMP) using `bfloat16` yields a **3.51x training throughput speedup** on the RTX 3050's Tensor Cores. Always train your dermoscopic classifier models using AMP.

---

## 5. Workstation Directory Structure

The workspace has been organized as follows for clean dataset and output management:

```
c:/Users/mithr/OneDrive/Documents/ADL/
├── checkpoints/       # Saved training model checkpoints (.pth)
├── results/           # Output statistics, metrics, and plots
├── logs/              # Plain text files for runtime logging
├── tensorboard/       # TensorBoard event logs (organized by run name)
├── cache/             # Preprocessed data cache (e.g. crop/scale arrays)
└── datasets/
    ├── HAM10000/      # Repository for HAM10000 images and metadata
    ├── ISIC2018/      # Repository for ISIC2018 datasets
    └── ISIC2019/      # Repository for ISIC2019 datasets
```

---

## 6. RTX 3050 Performance Best Practices

To extract maximum performance from the LOQ laptop GPU during training sessions:

1.  **Use AMP Mixed Precision:**
    Wrap your training loop forwards/backwards using `torch.amp.autocast`:
    ```python
    from gpu_config import get_amp_settings
    use_amp, amp_dtype = get_amp_settings()
    
    if use_amp:
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    ```
2.  **Enable cuDNN Benchmarking:**
    Call `configure_gpu_performance(enable_benchmark=True)` at the start of your script. This configures cuDNN to auto-tune convolution algorithms for the fixed dermoscopic image input size (224x224).
3.  **Optimize DataLoader Workers:**
    Use `get_dataloader_kwargs(batch_size)` to construct your DataLoader. On Windows:
    - `num_workers = 4` prevents CPU bottleneck without running into Windows multi-process memory leaks.
    - `pin_memory = True` allows fast memory copy from CPU page-locked memory to the GPU VRAM.
    - `persistent_workers = True` keeps the loader processes alive between epochs.
4.  **Save and Resume Checkpoints:**
    Use `CheckpointManager` from `checkpoint_manager.py` to save states after every epoch:
    ```python
    from checkpoint_manager import CheckpointManager
    manager = CheckpointManager()
    
    # Saving
    manager.save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss.item()
    }, is_best=is_best_loss)
    
    # Resuming
    manager.load_latest_checkpoint(model, optimizer)
    ```

---

## 7. Launching TensorBoard

To inspect your learning curves and baseline benchmarks in real-time, run the following command in a new terminal window:

```bash
& .\.venv\Scripts\tensorboard --logdir tensorboard
```

Open your browser and navigate to `http://localhost:6006`.
