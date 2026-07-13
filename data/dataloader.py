"""
DataLoader Factory — Phase 2
CUDA-optimized DataLoaders with WeightedRandomSampler for class balancing.
Handles Windows-specific multiprocessing constraints.
"""

import os
import sys
import multiprocessing
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DermDataset
from data.augmentations import get_train_transforms, get_val_transforms, get_test_transforms

# Windows-safe worker count
_MAX_WORKERS = min(4, multiprocessing.cpu_count())


def compute_class_weights(labels: list, num_classes: int = 7) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for cross-entropy loss.
    weight_i = n_total / (n_classes * n_i)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.clip(counts, 1, None)  # avoid division by zero
    n_total = float(len(labels))
    weights = np.sqrt(n_total / (num_classes * counts))
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: list, num_classes: int = 7) -> WeightedRandomSampler:
    """
    WeightedRandomSampler: each sample is drawn with probability inversely
    proportional to its class frequency -> balanced batch distribution.
    """
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = class_weights[labels]   # weight per sample
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def get_dataloader(
    df: pd.DataFrame,
    split: str,
    batch_size: int = 32,
    num_workers: int = _MAX_WORKERS,
    image_size: int = 224,
    images_root: Optional[Path] = None,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build an optimized DataLoader for the given split.

    Args:
        df: Split DataFrame (from splitter.py)
        split: One of 'train', 'val', 'test'
        batch_size: Batch size
        num_workers: Number of worker processes (capped at 4 for Windows)
        image_size: Target image resolution
        images_root: Root directory for image files (fallback)
        use_weighted_sampler: Use WeightedRandomSampler for train split
        pin_memory: Pin memory for fast GPU transfer
        prefetch_factor: Prefetch batches per worker

    Returns:
        DataLoader ready for GPU training
    """
    assert split in ("train", "val", "test"), f"Unknown split: {split}"

    # ── Select appropriate transform
    if split == "train":
        transform = get_train_transforms(image_size)
    else:
        transform = get_val_transforms(image_size)

    dataset = DermDataset(
        df=df,
        transform=transform,
        images_root=images_root,
        return_metadata=False,
    )

    # ── Sampler and shuffle logic
    sampler = None
    shuffle = False

    if split == "train":
        if use_weighted_sampler:
            sampler = make_weighted_sampler(dataset.labels, dataset.num_classes)
            # sampler and shuffle are mutually exclusive
            shuffle = False
        else:
            shuffle = True

    # ── Worker configuration (Windows-safe)
    num_workers = min(num_workers, _MAX_WORKERS)
    use_persistent = (num_workers > 0)
    use_pin_memory = pin_memory and torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "drop_last": (split == "train"),
        "sampler": sampler,
        "shuffle": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
        # Windows: use spawn context
        loader_kwargs["multiprocessing_context"] = "spawn"

    return DataLoader(dataset, **loader_kwargs)


def get_all_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = _MAX_WORKERS,
    image_size: int = 224,
    images_root: Optional[Path] = None,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function: returns (train_loader, val_loader, test_loader)."""
    train_loader = get_dataloader(
        train_df, "train", batch_size, num_workers, image_size, images_root,
        use_weighted_sampler=True, pin_memory=pin_memory
    )
    val_loader = get_dataloader(
        val_df, "val", batch_size, num_workers, image_size, images_root,
        use_weighted_sampler=False, pin_memory=pin_memory
    )
    test_loader = get_dataloader(
        test_df, "test", batch_size, num_workers, image_size, images_root,
        use_weighted_sampler=False, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader


def export_class_statistics(train_df: pd.DataFrame, output_path: Path):
    """Export class weights and sample counts to CSV."""
    labels = train_df["label"].tolist()
    weights = compute_class_weights(labels, num_classes=7)
    counts = np.bincount(labels, minlength=7)

    label_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    stats_df = pd.DataFrame({
        "label_int": list(range(7)),
        "label_str": label_names,
        "train_count": counts,
        "class_weight": weights.numpy(),
        "sample_fraction": counts / counts.sum(),
    })
    stats_df.to_csv(output_path, index=False)
    print(f"[OK   ] Class statistics -> {output_path}")
    return stats_df


def visualize_sampler_distribution(train_df: pd.DataFrame, output_path: Path,
                                   n_batches: int = 50, batch_size: int = 32):
    """
    Show class distribution before and after WeightedRandomSampler.
    Draws n_batches*batch_size samples with the sampler and plots distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = train_df["label"].tolist()
    label_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    # Original distribution
    raw_counts = np.bincount(labels, minlength=7)

    # Sampled distribution (simulate WeightedRandomSampler)
    weights = compute_class_weights(labels, num_classes=7)
    sample_weights = weights[torch.tensor(labels, dtype=torch.long)]
    sampler = WeightedRandomSampler(sample_weights, n_batches * batch_size, replacement=True)
    sampled_labels = [labels[i] for i in sampler]
    sampled_counts = np.bincount(sampled_labels, minlength=7)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Balance: Before vs After WeightedRandomSampler",
                 fontsize=13, fontweight="bold")

    x = np.arange(7)
    colors = plt.cm.Set2(np.linspace(0, 1, 7))

    for ax, counts, title in zip(
        axes,
        [raw_counts, sampled_counts],
        ["Original (imbalanced)", f"After Sampler ({n_batches} batches)"]
    ):
        bars = ax.bar(label_names, counts, color=colors, edgecolor="black")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    str(cnt), ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] Sampler distribution plot -> {output_path}")
