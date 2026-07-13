"""
Environment Verification Script
Checks system specs, GPU details, and package installations.
Outputs a detailed verification dictionary.
"""

import sys
import os
import platform
import shutil
import json
import psutil

# Packages to verify
PACKAGES = {
    "torch": "PyTorch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "timm": "timm",
    "albumentations": "albumentations",
    "cv2": "opencv-python (cv2)",
    "sklearn": "scikit-learn (sklearn)",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "torchmetrics": "torchmetrics",
    "tqdm": "tqdm",
    "yaml": "pyyaml (yaml)",
    "PIL": "pillow (PIL)",
    "seaborn": "seaborn",
    "jupyterlab": "jupyterlab",
    "tensorboard": "tensorboard",
    "pytest": "pytest",
    "rich": "rich"
}

def verify_system_specs():
    specs = {
        "os": f"{platform.system()} {platform.release()} (v{platform.version()})",
        "cpu": platform.processor(),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        "disk_free_gb": shutil.disk_usage(".").free / (1024**3),
        "disk_total_gb": shutil.disk_usage(".").total / (1024**3),
    }
    return specs

def verify_gpu():
    import torch
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if gpu_info["cuda_available"]:
        for i in range(gpu_info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            device_detail = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_memory_gb": props.total_memory / (1024**3),
                "multi_processor_count": props.multi_processor_count
            }
            gpu_info["devices"].append(device_detail)
            
    return gpu_info

def verify_packages():
    import importlib.metadata
    import importlib.util
    
    report = {}
    for module_name, display_name in PACKAGES.items():
        try:
            # Try importing the package to verify it's working
            # For jupyterlab, we check if it is installed in environment metadata since it's an app
            if module_name == "jupyterlab":
                ver = importlib.metadata.version("jupyterlab")
                status = "Success"
            else:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    # Actually import it to verify import success
                    mod = importlib.import_module(module_name)
                    # Try to get version
                    if hasattr(mod, "__version__"):
                        ver = mod.__version__
                    else:
                        try:
                            ver = importlib.metadata.version(display_name.split(" ")[0])
                        except Exception:
                            ver = "Installed"
                    status = "Success"
                else:
                    status = "Not Found"
                    ver = "N/A"
        except Exception as e:
            status = f"Import Error: {str(e)}"
            ver = "N/A"
            
        report[display_name] = {"status": status, "version": ver}
        
    return report

def main():
    print("=========================================")
    print("Running Environment Verification...")
    print("=========================================")
    
    sys_specs = verify_system_specs()
    print("\n[System Specs]")
    print(f"OS: {sys_specs['os']}")
    print(f"CPU: {sys_specs['cpu']}")
    print(f"Total RAM: {sys_specs['ram_total_gb']:.2f} GB (Available: {sys_specs['ram_available_gb']:.2f} GB)")
    print(f"Disk Space: {sys_specs['disk_free_gb']:.2f} GB free out of {sys_specs['disk_total_gb']:.2f} GB")
    
    # Import PyTorch and check GPU
    try:
        import torch
        print(f"\n[PyTorch Installation]")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Python version: {sys.version}")
        
        gpu_info = verify_gpu()
        print("\n[GPU Details]")
        print(f"CUDA Available: {gpu_info['cuda_available']}")
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print(f"cuDNN Version: {gpu_info['cudnn_version']}")
        print(f"Devices Count: {gpu_info['device_count']}")
        for d in gpu_info['devices']:
            print(f"  - Device {d['id']}: {d['name']} | Compute: {d['capability']} | VRAM: {d['total_memory_gb']:.2f} GB")
    except ImportError:
        print("\n[PyTorch Installation] FAILED to import torch!")
        gpu_info = {"cuda_available": False, "devices": []}
        
    print("\n[Package Verification]")
    pkg_report = verify_packages()
    all_ok = True
    for name, info in pkg_report.items():
        status_symbol = "[OK] " if info["status"] == "Success" else "[FAIL]"
        if info["status"] != "Success":
            all_ok = False
        print(f"  {status_symbol} {name:<30} : {info['version']} ({info['status']})")
        
    if all_ok:
        print("\n[OK] All packages imported and verified successfully!")
    else:
        print("\n[FAIL] Some package installations have issues. Please review details above.")
        
    # Write to a JSON file to parse for report generation
    results = {
        "system": sys_specs,
        "gpu": gpu_info,
        "packages": pkg_report,
        "pytorch_version": torch.__version__ if 'torch' in sys.modules else "N/A",
        "python_version": sys.version
    }
    
    with open("verify_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nVerification results saved to verify_results.json")

if __name__ == "__main__":
    main()
