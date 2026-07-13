"""
Sanity Check — Phase 2
Verifies the complete data pipeline end-to-end before training.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.splitter import load_splits, create_splits
from data.dataloader import get_all_dataloaders, compute_class_weights
from gpu_config import configure_gpu_performance, get_device

DATASETS_ROOT = PROJECT_ROOT / "datasets"
HAM10000_DIR  = DATASETS_ROOT / "HAM10000"
SPLITS_DIR    = PROJECT_ROOT / "splits"
NUM_CLASSES   = 7
LABEL_NAMES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def check_splits_exist():
    required = ["train.csv", "val.csv", "test.csv"]
    missing  = [f for f in required if not (SPLITS_DIR / f).exists()]
    if missing:
        print(f"[INFO ] Split CSVs not found — creating them now...")
        create_splits()
    else:
        print(f"[OK   ] Split CSVs found in {SPLITS_DIR}")


def sanity_check():
    print("\n" + "="*65)
    print("  DATA PIPELINE SANITY CHECK")
    print("="*65)

    configure_gpu_performance()
    device = get_device()
    print(f"[INFO ] Device: {device}")

    # ── Step 1: Load splits
    print("\n[1/8] Loading splits...")
    check_splits_exist()
    train_df, val_df, test_df = load_splits()
    print(f"[OK   ] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Step 2: Build DataLoaders (num_workers=0 for sanity check to avoid multiprocessing issues)
    print("\n[2/8] Building DataLoaders...")
    train_loader, val_loader, test_loader = get_all_dataloaders(
        train_df, val_df, test_df,
        batch_size=16,
        num_workers=0,        # single-process for sanity check
        images_root=HAM10000_DIR,
    )
    print(f"[OK   ] Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Step 3: Fetch one batch
    print("\n[3/8] Fetching first batch from train_loader...")
    t0 = time.perf_counter()
    images, labels = next(iter(train_loader))
    t1 = time.perf_counter()
    print(f"[OK   ] Batch loaded in {(t1-t0)*1000:.1f} ms")

    # ── Step 4: Check tensor shape
    print("\n[4/8] Checking tensor shape...")
    expected_shape = (16, 3, 224, 224)
    # Note: last batch may be smaller, so check (B, 3, 224, 224)
    assert images.shape[1:] == (3, 224, 224), \
        f"Expected (B, 3, 224, 224) but got {images.shape}"
    print(f"[OK   ] Image tensor shape: {tuple(images.shape)}")

    # ── Step 5: Check label range
    print("\n[5/8] Checking labels...")
    assert labels.min() >= 0, f"Negative label found: {labels.min()}"
    assert labels.max() < NUM_CLASSES, f"Label {labels.max()} >= num_classes {NUM_CLASSES}"
    unique_labels = labels.unique().tolist()
    print(f"[OK   ] Labels in batch: {sorted(unique_labels)} | Range: [{int(labels.min())}, {int(labels.max())}]")

    # ── Step 6: Check for NaN / Inf
    print("\n[6/8] Checking for NaN / Inf values...")
    assert not torch.any(torch.isnan(images)), "NaN found in image tensors!"
    assert not torch.any(torch.isinf(images)), "Inf found in image tensors!"
    print(f"[OK   ] No NaN or Inf values detected")
    print(f"[INFO ] Tensor stats — min: {images.min():.4f} | max: {images.max():.4f} | mean: {images.mean():.4f}")

    # ── Step 7: GPU transfer
    print("\n[7/8] GPU transfer test...")
    if torch.cuda.is_available():
        images_gpu = images.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)
        torch.cuda.synchronize()
        assert images_gpu.device.type == "cuda"
        vram_mb = torch.cuda.memory_allocated() / 1e6
        print(f"[OK   ] Batch on GPU: {tuple(images_gpu.shape)} | VRAM used: {vram_mb:.1f} MB")
        del images_gpu, labels_gpu
        torch.cuda.empty_cache()
    else:
        print("[SKIP ] CUDA not available — skipping GPU transfer test")

    # ── Quick val and test loader check
    print("\n       Also verifying val and test loaders...")
    v_images, v_labels = next(iter(val_loader))
    assert v_images.shape[1:] == (3, 224, 224), f"Val shape error: {v_images.shape}"
    print(f"[OK   ] Val batch: {tuple(v_images.shape)}")

    # ── Step 8: Class weights
    print("\n[8/8] Computing class weights...")
    labels_list = train_df["label"].tolist()
    weights = compute_class_weights(labels_list, NUM_CLASSES)
    assert weights.shape == (NUM_CLASSES,), f"Expected shape ({NUM_CLASSES},), got {weights.shape}"
    print("[OK   ] Class weights:")
    for i, (name, w) in enumerate(zip(LABEL_NAMES, weights)):
        count = labels_list.count(i)
        print(f"        {name:8}: weight={w:.4f} | count={count}")

    print("\n" + "="*65)
    print("  ✅ ALL SANITY CHECKS PASSED — Pipeline is ready for training!")
    print("="*65)
    return True


if __name__ == "__main__":
    sanity_check()
