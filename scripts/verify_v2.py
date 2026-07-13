import sys
import torch
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dual_branch_net import build_model_from_config
from gpu_config import get_device, get_amp_settings

def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def main():
    device = get_device()
    print(f"Device: {device}")
    
    with open(PROJECT_ROOT / "repo_temp" / "SKIN-LESION-main" / "config" / "config.yaml", "r") as f:
        repo_cfg = yaml.safe_load(f)
        
    model = build_model_from_config(repo_cfg)
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"Parameter count: {params['total']:,} (V1 was 10,669,639)")
    
    # Dummy input
    B = 2
    C, H, W = 3, 224, 224
    x = torch.randn(B, C, H, W, device=device)
    labels = torch.randint(0, 7, (B,), device=device)
    
    print("Running forward pass...")
    logits, tex_fmap, str_fmap = model(x)
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (B, 7), "Incorrect output shape"
    
    print("Running backward pass...")
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    loss.backward()
    print("Backward pass successful.")
    
    print("Checking AMP compatibility...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_amp, _, _ = model(x)
        loss_amp = loss_fn(logits_amp, labels)
    print("AMP forward pass successful.")

if __name__ == "__main__":
    main()
