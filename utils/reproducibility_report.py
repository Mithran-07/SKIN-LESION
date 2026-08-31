"""
Reproducibility Report Generator.

Captures system state, Git commit, Python/PyTorch versions,
and configuration into docs/reproducibility_report.md
"""

import os
import sys
import subprocess
import torch
from datetime import datetime
from pathlib import Path
import yaml

def run_cmd(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception as e:
        return f"Error: {e}"

def generate_report(config_path: str = "config/config.yaml", output_path: str = "docs/reproducibility_report.md"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Gather system info
    git_commit = run_cmd("git rev-parse HEAD")
    git_branch = run_cmd("git rev-parse --abbrev-ref HEAD")
    
    py_version = sys.version
    pt_version = torch.__version__
    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if cuda_avail:
        device_info = f"CUDA (Count: {torch.cuda.device_count()})"
    elif mps_avail:
        device_info = "MPS (Apple Silicon)"
    else:
        device_info = "CPU"
        
    # Read Config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_str = f.read()
    else:
        config_str = "No config found."
        
    # Pip freeze
    pip_freeze = run_cmd(f"{sys.executable} -m pip freeze")
    
    report = f"""# Reproducibility Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Environment Details
- **Git Branch**: `{git_branch}`
- **Git Commit**: `{git_commit}`
- **Python Version**: {py_version.splitlines()[0]}
- **PyTorch Version**: {pt_version}
- **Compute Device**: {device_info}

## Experiment Configuration
```yaml
{config_str}
```

## Python Packages
<details>
<summary>Click to expand `pip freeze`</summary>

```text
{pip_freeze}
```
</details>
"""
    
    with open(output_path, "w") as f:
        f.write(report)
        
    print(f"Reproducibility report generated at {output_path}")

if __name__ == "__main__":
    generate_report()
