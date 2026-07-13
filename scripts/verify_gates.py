import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter import load_splits
from data.dataloader import get_all_dataloaders
from models.dual_branch_net import DualBranchNet
from gpu_config import get_device

def main():
    device = get_device()
    train_df, val_df, test_df = load_splits()
    _, val_loader, _ = get_all_dataloaders(train_df, val_df, test_df, batch_size=8, num_workers=0)
    
    model = DualBranchNet(
        num_classes=7,
        shallow_channels=[256, 512, 1024],
        deep_base_channels=64,
        deep_bottleneck=256,
        deep_num_blocks=[2, 2, 3, 3],
        fusion_hidden_dim=512,
        fusion_output_dim=256,
        pretrained_init=False,
    ).to(device)
    model.eval()
    
    gate_out = {}
    h1 = model.fusion.gate_t.register_forward_hook(lambda m,i,o: gate_out.update({"t": o.detach().cpu()}))
    h2 = model.fusion.gate_s.register_forward_hook(lambda m,i,o: gate_out.update({"s": o.detach().cpu()}))
    
    tex_means, str_means = [], []
    
    print("Running untrained gate verification (10 batches)...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10: break
            images = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
            model(images)
            tex_means.append(gate_out["t"].squeeze().numpy())
            str_means.append(gate_out["s"].squeeze().numpy())
            
    import numpy as np
    tex_all = np.concatenate(tex_means)
    str_all = np.concatenate(str_means)
    
    print(f"Texture Gate Mean: {tex_all.mean():.4f}")
    print(f"Structure Gate Mean: {str_all.mean():.4f}")
    
    print("\nVerification complete. Gates should be roughly 0.5 initialized.")

if __name__ == "__main__":
    main()
