"""
Experiment Tracker Utility.

Automatically initializes experiment folders (EXP001, EXP002, ...)
and populates them with template files.
"""

import os
import json
import yaml
from pathlib import Path

def get_next_experiment_id(base_dir: str = "experiments") -> str:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("EXP") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not existing_dirs:
        return "EXP001"
        
    ids = [int(d.replace("EXP", "")) for d in existing_dirs if d.replace("EXP", "").isdigit()]
    next_id = max(ids) + 1 if ids else 1
    return f"EXP{next_id:03d}"

def init_experiment(base_dir: str = "experiments"):
    exp_id = get_next_experiment_id(base_dir)
    exp_dir = Path(base_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. config.yaml
    config_data = {"experiment_id": exp_id, "status": "initialized", "notes": "Enter configuration..."}
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_data, f)
        
    # 2. metrics.json
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump({"epochs": [], "final_metrics": {}}, f, indent=4)
        
    # 3. environment.json
    with open(exp_dir / "environment.json", "w") as f:
        json.dump({"hardware": "Lenovo LOQ", "packages": []}, f, indent=4)
        
    # 4. notes.md
    with open(exp_dir / "notes.md", "w") as f:
        f.write(f"# Experiment {exp_id}\\n\\n## Hypothesis\\n\\n## Observations\\n\\n## Conclusion\\n")
        
    print(f"Initialized {exp_id} at {exp_dir}")
    return exp_id

if __name__ == "__main__":
    init_experiment()
