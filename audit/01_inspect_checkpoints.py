"""
AUDIT 01 — Checkpoint inspection (read-only).
Loads every checkpoint from the LOQ archive (map_location=cpu), reports metadata,
parameter counts, architecture fingerprint, and history-vs-saved-epoch consistency.
Does NOT modify any checkpoint.
"""
import hashlib, json, sys
from pathlib import Path
import torch

LOQ = Path("/Users/mithran/Documents/My projects/ADL MAIN/ADL.  LOQ/checkpoints")
MAC = Path(__file__).resolve().parent.parent / "checkpoints"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

report = {}
paths = sorted(LOQ.glob("*/*.pth")) + [MAC / "efficientnet_b4" / "best_checkpoint.pth"]
for p in paths:
    key = f"{'MAC' if str(p).startswith(str(MAC)) else 'LOQ'}:{p.parent.name}/{p.name}"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    n_params = sum(v.numel() for k, v in sd.items() if v.dtype.is_floating_point and "running_" not in k and "num_batches" not in k)
    fusion_keys = {k: list(v.shape) for k, v in sd.items() if k.startswith("fusion.")}
    hist = ck.get("history", {})
    vb = hist.get("val_bal_acc", [])
    best_hist_epoch = (max(range(len(vb)), key=lambda i: vb[i]) + 1) if vb else None
    entry = {
        "sha256": sha256(p),
        "size_bytes": p.stat().st_size,
        "top_level_keys": sorted(ck.keys()) if isinstance(ck, dict) else None,
        "model_name": ck.get("model_name"),
        "saved_epoch_1idx": (ck.get("epoch") + 1) if isinstance(ck.get("epoch"), int) else None,
        "best_val_bal_acc_field": ck.get("best_val_bal_acc"),
        "n_learnable_params_in_state_dict": n_params,
        "n_tensors": len(sd),
        "history_len_epochs": len(vb),
        "history_best_val_bal_acc": max(vb) if vb else None,
        "history_best_epoch_1idx": best_hist_epoch,
        "val_bal_acc_at_saved_epoch": (vb[ck["epoch"]] if vb and isinstance(ck.get("epoch"), int) and ck["epoch"] < len(vb) else None),
        "fusion_param_shapes": fusion_keys or None,
        "first_keys": list(sd.keys())[:3],
    }
    report[key] = entry
    print(f"\n=== {key}")
    for k, v in entry.items():
        if k != "fusion_param_shapes":
            print(f"  {k}: {v}")
    if fusion_keys:
        print(f"  fusion: {fusion_keys}")

(OUT / "01_checkpoints.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nSaved -> {OUT/'01_checkpoints.json'}")
