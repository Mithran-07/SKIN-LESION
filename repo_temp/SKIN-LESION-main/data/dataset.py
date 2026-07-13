"""
Dataset classes for HAM10000 and ISIC 2019 dermoscopic image datasets.

Handles:
- CSV metadata loading and label encoding
- Image loading from split directory structures
- Optional segmentation mask loading
- Patient-aware stratified splitting to prevent data leakage
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from data.augmentations import get_train_transforms, get_val_transforms
from data.sampler import ImbalancedDatasetSampler

logger = logging.getLogger(__name__)

# HAM10000 class label mapping
HAM10000_CLASSES = {
    "mel": 0,    # Melanoma
    "nv": 1,     # Melanocytic Nevi
    "bcc": 2,    # Basal Cell Carcinoma
    "akiec": 3,  # Actinic Keratosis / Intraepithelial Carcinoma
    "bkl": 4,    # Benign Keratosis-like Lesions
    "df": 5,     # Dermatofibroma
    "vasc": 6,   # Vascular Lesions
}

HAM10000_CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# ISIC 2019 adds SCC as class index 7
ISIC2019_CLASSES = {**HAM10000_CLASSES, "scc": 7}  # Squamous Cell Carcinoma
ISIC2019_CLASS_NAMES = HAM10000_CLASS_NAMES + ["SCC"]


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 dermoscopic image collection.

    The HAM10000 dataset contains 10,015 dermatoscopic images across 7
    diagnostic categories with ground truth established via histopathology,
    follow-up, expert consensus, or in-vivo confocal microscopy.

    Args:
        root (str): Root directory containing image subdirectories.
        metadata_csv (str): Path to HAM10000_metadata.csv.
        image_dirs (list): List of subdirectory names containing JPEG images.
        transform: Albumentations transform pipeline.
        mask_dir (str, optional): Directory containing binary segmentation masks.
        split (str): One of 'train', 'val', 'test'.
        val_frac (float): Fraction of data for validation.
        test_frac (float): Fraction of data for test.
        seed (int): Random seed for reproducible splits.
        patient_aware (bool): If True, split by patient_id to prevent leakage.
    """

    def __init__(
        self,
        root: str,
        metadata_csv: str,
        image_dirs: List[str],
        transform=None,
        mask_dir: Optional[str] = None,
        split: str = "train",
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        patient_aware: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.split = split
        self.class_to_idx = HAM10000_CLASSES
        self.class_names = HAM10000_CLASS_NAMES

        # ── Load and preprocess metadata ────────────────────────────────────
        df = pd.read_csv(metadata_csv)
        df["label"] = df["dx"].str.lower().map(self.class_to_idx)
        if df["label"].isna().any():
            unknown = df[df["label"].isna()]["dx"].unique()
            logger.warning(f"Unknown dx labels dropped: {unknown}")
            df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        # ── Build image path index ──────────────────────────────────────────
        image_map: Dict[str, str] = {}
        for img_dir in image_dirs:
            dir_path = self.root / img_dir
            for img_path in dir_path.glob("*.jpg"):
                image_map[img_path.stem] = str(img_path)
        df["filepath"] = df["image_id"].map(image_map)
        missing = df["filepath"].isna().sum()
        if missing > 0:
            logger.warning(f"{missing} images not found on disk — dropping.")
        df = df.dropna(subset=["filepath"])
        df = df.reset_index(drop=True)

        # ── Patient-aware stratified split ──────────────────────────────────
        df = self._split_dataset(df, val_frac, test_frac, seed, patient_aware)
        logger.info(
            f"[{split.upper()}] {len(df)} samples | "
            + " | ".join(
                f"{n}: {(df['label']==i).sum()}"
                for i, n in enumerate(self.class_names)
            )
        )
        self.df = df

    def _split_dataset(
        self,
        df: pd.DataFrame,
        val_frac: float,
        test_frac: float,
        seed: int,
        patient_aware: bool,
    ) -> pd.DataFrame:
        """Split by patient_id (if available) to prevent data leakage."""
        if patient_aware and "lesion_id" in df.columns:
            # Group by lesion_id to keep all images of the same lesion together
            unique_lesions = df["lesion_id"].unique()
            lesion_labels = df.groupby("lesion_id")["label"].first()

            train_lesions, test_lesions = train_test_split(
                unique_lesions,
                test_size=test_frac,
                random_state=seed,
                stratify=lesion_labels[unique_lesions],
            )
            train_lesions, val_lesions = train_test_split(
                train_lesions,
                test_size=val_frac / (1 - test_frac),
                random_state=seed,
                stratify=lesion_labels[train_lesions],
            )

            split_map = {"train": train_lesions, "val": val_lesions, "test": test_lesions}
            return df[df["lesion_id"].isin(split_map[self.split])].reset_index(drop=True)
        else:
            # Fallback: image-level stratified split
            train_idx, test_idx = train_test_split(
                df.index, test_size=test_frac, random_state=seed, stratify=df["label"]
            )
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_frac / (1 - test_frac),
                random_state=seed,
                stratify=df.loc[train_idx, "label"],
            )
            split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
            return df.loc[split_map[self.split]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["filepath"]).convert("RGB"))
        label = int(row["label"])

        # ── Load mask if available ──────────────────────────────────────────
        mask = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir / f"{row['image_id']}_segmentation.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert("L"))
                mask = (mask > 127).astype(np.float32)

        # ── Apply transforms ────────────────────────────────────────────────
        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"].unsqueeze(0)  # (1, H, W)
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        sample = {"image": image, "label": torch.tensor(label, dtype=torch.long)}
        if mask is not None:
            sample["mask"] = mask
        # Store metadata for debugging
        sample["image_id"] = row["image_id"]
        return sample

    def get_labels(self) -> List[int]:
        """Return all integer labels (used by the weighted sampler)."""
        return self.df["label"].tolist()


class ISIC2019Dataset(HAM10000Dataset):
    """
    Extension of HAM10000Dataset for ISIC 2019 data.

    ISIC 2019 introduces Squamous Cell Carcinoma (SCC) as an 8th class
    and includes cross-institutional data for better generalization.
    """

    def __init__(self, *args, **kwargs):
        # Temporarily set parent class mapping, then override
        super().__init__(*args, **kwargs)
        self.class_to_idx = ISIC2019_CLASSES
        self.class_names = ISIC2019_CLASS_NAMES


def create_dataloaders(
    cfg: dict,
    dataset_class=HAM10000Dataset,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from config dict.

    Args:
        cfg: Parsed config.yaml as a Python dict.
        dataset_class: Dataset class to instantiate.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_cfg = cfg["dataset"]
    split_cfg = cfg["split"]
    loader_cfg = cfg["dataloader"]
    aug_cfg = cfg["augmentation"]

    common_kwargs = dict(
        root=dataset_cfg["root"],
        metadata_csv=os.path.join(dataset_cfg["root"], dataset_cfg["metadata_csv"]),
        image_dirs=dataset_cfg["image_dirs"],
        mask_dir=dataset_cfg.get("mask_dir"),
        val_frac=split_cfg["val"],
        test_frac=split_cfg["test"],
        seed=cfg["seed"],
        patient_aware=split_cfg.get("patient_aware", True),
    )

    train_ds = dataset_class(
        **common_kwargs,
        transform=get_train_transforms(aug_cfg),
        split="train",
    )
    val_ds = dataset_class(
        **common_kwargs,
        transform=get_val_transforms(aug_cfg),
        split="val",
    )
    test_ds = dataset_class(
        **common_kwargs,
        transform=get_val_transforms(aug_cfg),
        split="test",
    )

    # Weighted sampler for train set to counteract class imbalance
    sampler = None
    if loader_cfg.get("use_weighted_sampler", True):
        sampler = ImbalancedDatasetSampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=loader_cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg.get("pin_memory", False),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=loader_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg.get("pin_memory", False),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=loader_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg.get("pin_memory", False),
    )

    return train_loader, val_loader, test_loader
