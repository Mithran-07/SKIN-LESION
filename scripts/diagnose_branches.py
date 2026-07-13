import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter  import load_splits
from data.dataloader import get_all_dataloaders
from models.dual_branch_net import build_model_from_config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(PROJECT_ROOT / "repo_temp" / "SKIN-LESION-main" / "config" / "config.yaml", "r") as f:
        repo_cfg = yaml.safe_load(f)

    train_df, val_df, test_df = load_splits()
    _, _, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=8,
        num_workers=0,
        image_size=224,
        images_root=PROJECT_ROOT / "datasets" / "HAM10000",
        pin_memory=False,
    )

    model = build_model_from_config(repo_cfg)
    ckpt_path = PROJECT_ROOT / "checkpoints" / "dual_branch_seed42" / "best_checkpoint.pth"
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # Lists to store intermediate outputs
    texture_vectors = []
    structure_vectors = []
    gate_weights = []

    # Hooks to capture branch outputs and fusion gates
    def hook_fn(module, input, output):
        # output is (texture_vec, last_fmap)
        texture_vectors.append(output[0].detach().cpu())

    def hook_fn_deep(module, input, output):
        # output is (structure_vec, last_fmap)
        structure_vectors.append(output[0].detach().cpu())

    def hook_fn_fusion(module, input, output):
        # output is gate weights (B, 1280)
        gate_weights.append(output.detach().cpu())

    h1 = model.shallow_branch.register_forward_hook(hook_fn)
    h2 = model.deep_branch.register_forward_hook(hook_fn_deep)
    h3 = model.fusion.gate.register_forward_hook(hook_fn_fusion)

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 20: # Use 160 samples for statistical significance
                break
            images = images.to(device)
            _ = model(images)

    h1.remove()
    h2.remove()
    h3.remove()

    tex_vecs = torch.cat(texture_vectors, dim=0).numpy()       # (N, 1024)
    struct_vecs = torch.cat(structure_vectors, dim=0).numpy()   # (N, 256)
    gates = torch.cat(gate_weights, dim=0).numpy()              # (N, 1280)

    # Compute statistics
    print("=== BRANCH ACTIVATION STATISTICS ===")
    print(f"Texture Branch Output (Shallow-Wide):")
    print(f"  Mean absolute activation: {np.mean(np.abs(tex_vecs)):.4f}")
    print(f"  Standard deviation:       {np.std(tex_vecs):.4f}")
    print(f"  Sparsity (% of near-zero activations < 1e-3): {np.mean(np.abs(tex_vecs) < 1e-3) * 100:.2f}%")
    print(f"  L2 Norm Mean:             {np.mean(np.linalg.norm(tex_vecs, axis=1)):.4f}")

    print(f"\nStructure Branch Output (Deep-Narrow):")
    print(f"  Mean absolute activation: {np.mean(np.abs(struct_vecs)):.4f}")
    print(f"  Standard deviation:       {np.std(struct_vecs):.4f}")
    print(f"  Sparsity (% of near-zero activations < 1e-3): {np.mean(np.abs(struct_vecs) < 1e-3) * 100:.2f}%")
    print(f"  L2 Norm Mean:             {np.mean(np.linalg.norm(struct_vecs, axis=1)):.4f}")

    print(f"\n=== FUSION ATTENTION GATE ANALYSIS ===")
    # Concatenated vector is [texture (1024) || structure (256)]
    tex_gates = gates[:, :1024]
    struct_gates = gates[:, 1024:]

    print(f"Texture Attention Weights (first 1024 dimensions):")
    print(f"  Mean gate weight:         {np.mean(tex_gates):.4f}")
    print(f"  Std of gate weights:      {np.std(tex_gates):.4f}")
    print(f"  Min/Max gate weight:      {np.min(tex_gates):.4f} / {np.max(tex_gates):.4f}")

    print(f"\nStructure Attention Weights (last 256 dimensions):")
    print(f"  Mean gate weight:         {np.mean(struct_gates):.4f}")
    print(f"  Std of gate weights:      {np.std(struct_gates):.4f}")
    print(f"  Min/Max gate weight:      {np.min(struct_gates):.4f} / {np.max(struct_gates):.4f}")

    # Check if one branch is suppressed
    ratio = np.mean(tex_gates) / np.mean(struct_gates)
    print(f"\nGate Weight Ratio (Texture / Structure): {ratio:.4f}")
    if ratio > 5.0:
        print("Warning: Texture branch dominates the fusion head.")
    elif ratio < 0.2:
        print("Warning: Structure branch dominates the fusion head.")
    else:
        print("Both branches are active and balanced in the fusion head.")

if __name__ == "__main__":
    main()
