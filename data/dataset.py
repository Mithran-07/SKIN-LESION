"""
Dermoscopy Dataset — PyTorch Dataset class — Phase 2
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

HAM_LABEL_MAP = {
    "akiec": 0, "bcc": 1, "bkl": 2,
    "df": 3, "mel": 4, "nv": 5, "vasc": 6,
}
HAM_LABEL_NAMES = {v: k for k, v in HAM_LABEL_MAP.items()}


class DermDataset(Dataset):
    """
    PyTorch Dataset for dermoscopy images.

    Expects a DataFrame with columns:
      - image_id  : str, base filename without extension
      - img_path  : str, full path to image file
      - label     : int, numeric class label (0-6)
      - dx        : str, string class label
      - patient_id: str/int, patient identifier
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        images_root: Optional[Path] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            df: Split DataFrame from splitter.py
            transform: Albumentations or torchvision transform
            images_root: Fallback directory to search for images
            return_metadata: If True, return (image, label, patient_id)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images_root = images_root
        self.return_metadata = return_metadata
        self._missing = []

        # Pre-validate paths (warn about missing files upfront)
        self._validate_paths()

    def _validate_paths(self):
        n_missing = 0
        for idx, row in self.df.iterrows():
            path = self._resolve_path(row)
            if path is None or not Path(path).exists():
                n_missing += 1
                self._missing.append(row.get("image_id", str(idx)))
        if n_missing > 0:
            print(f"[WARN ] DermDataset: {n_missing} missing image files "
                  f"(will return zero tensor for those)")

    def _resolve_path(self, row) -> Optional[str]:
        """Try multiple strategies to find the image file."""
        # 1. img_path column
        img_path = row.get("img_path", "")
        if img_path and Path(str(img_path)).exists():
            return str(img_path)

        # 2. images_root fallback
        if self.images_root is not None:
            img_id = row.get("image_id", "")
            candidate = Path(self.images_root) / f"{img_id}.jpg"
            if candidate.exists():
                return str(candidate)
            # Search recursively (for nested HAM10000 folder)
            found = list(Path(self.images_root).rglob(f"{img_id}.jpg"))
            if found:
                return str(found[0])

        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label"])
        patient_id = str(row.get("patient_id", ""))

        # ── Load image
        img_path = self._resolve_path(row)
        if img_path is None or not Path(img_path).exists():
            # Return zero tensor for missing images
            img_tensor = torch.zeros(3, 224, 224)
            if self.return_metadata:
                return img_tensor, label, patient_id
            return img_tensor, label

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_np = np.array(img)
        except (UnidentifiedImageError, OSError, Exception):
            img_tensor = torch.zeros(3, 224, 224)
            if self.return_metadata:
                return img_tensor, label, patient_id
            return img_tensor, label

        # ── Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]   # ToTensorV2 returns CHW float tensor
        else:
            # Fallback: basic resize + to tensor
            import torchvision.transforms.functional as TF
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img_np).resize((224, 224))
            img_tensor = TF.to_tensor(pil)

        if self.return_metadata:
            return img_tensor, label, patient_id

        return img_tensor, label

    @property
    def class_counts(self) -> dict:
        """Return {label_int: count} for WeightedRandomSampler."""
        return self.df["label"].value_counts().to_dict()

    @property
    def labels(self) -> list:
        """Return list of all integer labels (same order as dataset)."""
        return self.df["label"].tolist()

    @property
    def num_classes(self) -> int:
        return len(HAM_LABEL_MAP)
