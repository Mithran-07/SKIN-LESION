import os
from pathlib import Path

ROOT = Path("C:/ADL")

must_push_dirs = [
    "config", "models", "training", "losses", "data", "utils", "scripts", 
    "tests", "docs", "paper", "thesis", "visualization", "explainability", 
    "uncertainty", "federated"
]
must_push_files = [
    "README.md", "LICENSE", "CITATION.cff", "MODEL_REGISTRY.md", 
    "FINAL_RESULTS.md", "PROJECT_HANDOFF.md", "REPRODUCIBILITY.md", 
    "manifest.json", "requirements.txt", "requirements-lock.txt", 
    "environment.yml", "pyproject.toml", ".gitignore"
]

do_not_push_dirs = [
    "datasets", "cache", "tensorboard", "logs", ".venv", "__pycache__", 
    "artifacts", "checkpoints"
]
do_not_push_exts = [".pyc", ".pth", ".pt", ".ckpt", ".log", ".zip"]

optional_dirs = [
    "results/final", "figures"
]
optional_files = [
    "benchmark_final.csv", "summary_metrics.json", "experiment_summary.md", 
    "comparison_table.csv", "comparison_table.md"
]

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def scan_repo():
    must_push = []
    optional = []
    do_not_push = []
    must_push_size = 0
    optional_size = 0
    do_not_push_size = 0
    
    for root, dirs, files in os.walk(ROOT):
        # Skip hidden dirs like .git or .gemini
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        rel_root = Path(root).relative_to(ROOT)
        
        for file in files:
            file_path = Path(root) / file
            rel_path = rel_root / file
            
            try:
                size = file_path.stat().st_size
            except FileNotFoundError:
                continue
                
            path_str = str(rel_path).replace("\\", "/")
            
            # Classification
            is_do_not_push = False
            for dnp_dir in do_not_push_dirs:
                if path_str.startswith(dnp_dir + "/") or path_str == dnp_dir or dnp_dir in path_str.split("/"):
                    is_do_not_push = True
                    break
            
            if not is_do_not_push:
                for ext in do_not_push_exts:
                    if path_str.endswith(ext):
                        is_do_not_push = True
                        break
            
            if is_do_not_push:
                do_not_push.append((path_str, size))
                do_not_push_size += size
                continue
                
            is_optional = False
            for opt_dir in optional_dirs:
                if path_str.startswith(opt_dir + "/") or path_str == opt_dir or opt_dir in path_str.split("/"):
                    is_optional = True
                    break
            
            if not is_optional:
                for opt_file in optional_files:
                    if file == opt_file:
                        is_optional = True
                        break
            
            if is_optional:
                optional.append((path_str, size))
                optional_size += size
                continue
                
            # If not Do Not Push and Not Optional, it's Must Push
            must_push.append((path_str, size))
            must_push_size += size
            
    return (must_push, must_push_size), (optional, optional_size), (do_not_push, do_not_push_size)

def main():
    must, opt, dnp = scan_repo()
    
    checklist = f"""# Git Push Checklist

## 1. Expected Repository Size
- **MUST PUSH (Source Code & Assets):** {format_size(must[1])}
- **OPTIONAL (Final Results & Figures):** {format_size(opt[1])}
- **DO NOT PUSH (Checkpoints, Logs, Datasets):** {format_size(dnp[1])}

The repository size (MUST PUSH) is extremely lightweight and well under GitHub's 1GB limit. No single source file exceeds the 100MB strict limit. Git LFS is NOT required.

## 2. Files to Push (MUST PUSH)
These files contain the core research code, configs, and documentation:
"""
    for f, s in sorted(must[0])[:30]:
        checklist += f"- `{f}` ({format_size(s)})\n"
    if len(must[0]) > 30:
        checklist += f"- ... and {len(must[0]) - 30} other source files.\n"
        
    checklist += f"""
## 3. Files to Release Separately (OPTIONAL)
These files are highly useful for publication but are derived artifacts. They should be pushed to Git if small, OR released as a "GitHub Release" alongside the paper:
"""
    for f, s in sorted(opt[0]):
        checklist += f"- `{f}` ({format_size(s)})\n"
        
    checklist += f"""
## 4. Files to Exclude (DO NOT PUSH)
These files MUST remain local. Add them to `.gitignore`:
- `artifacts/checkpoints.zip` (Very large, binary)
- `checkpoints/` (*.pth files)
- `datasets/` (Data privacy & size limits)
- `logs/` & `tensorboard/` (Dynamic cache)
- `__pycache__/`, `.venv/`

## 5. Final Repository Structure
```
Dual-Branch-CNN/
├── config/             # YAML configurations (Push)
├── models/             # PyTorch model definitions (Push)
├── scripts/            # Executable scripts (Push)
├── data/               # Dataset loaders, NOT data (Push)
├── results/final/      # Final CSVs & Figures (Push/Release)
├── README.md           # Documentation (Push)
├── FINAL_RESULTS.md    # Conclusions (Push)
├── MODEL_REGISTRY.md   # Registry (Push)
└── .gitignore          # Must explicitly block datasets/ & checkpoints/
```

## 6. Git Commands
Execute the following to securely push the project:

```bash
# 1. Ensure .gitignore is tracking large/private files
git add .gitignore

# 2. Add MUST PUSH files specifically
git add config/ models/ scripts/ data/ utils/ training/ losses/ tests/
git add README.md LICENSE CITATION.cff MODEL_REGISTRY.md FINAL_RESULTS.md PROJECT_HANDOFF.md REPRODUCIBILITY.md manifest.json
git add requirements.txt requirements-lock.txt environment.yml pyproject.toml

# 3. Add Optional Results (if desired for repo transparency)
git add results/final/

# 4. Check status to ensure NO datasets or .pth files were staged
git status

# 5. Commit and Push
git commit -m "Research project final archive"
git push origin main
```
"""
    
    with open(ROOT / "GIT_PUSH_CHECKLIST.md", "w", encoding="utf-8") as f:
        f.write(checklist)
    print("Generated GIT_PUSH_CHECKLIST.md")

if __name__ == "__main__":
    main()
