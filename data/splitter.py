"""
Patient-Level Stratified Train/Val/Test Splitter — Phase 2
Ensures:
  - No patient leaks across splits
  - Class distribution is preserved per split
  - 70% train / 15% val / 15% test
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
SPLITS_DIR    = PROJECT_ROOT / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

HAM10000_DIR = DATASETS_ROOT / "HAM10000"

HAM_LABEL_MAP = {
    "akiec": 0,
    "bcc":   1,
    "bkl":   2,
    "df":    3,
    "mel":   4,
    "nv":    5,
    "vasc":  6,
}

# HAM10000 uses 'lesion_id' for patient grouping (multiple images per lesion/patient)
PATIENT_COL = "lesion_id"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────
def patient_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    label_col:   str   = "dx",
    patient_col: str   = PATIENT_COL,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at PATIENT level so no patient appears in multiple splits.
    Preserves class distribution as best as possible.
    
    Strategy:
      1. Get the dominant diagnosis per patient (most frequent label).
      2. Stratify patients by dominant label.
      3. First split into train vs. (val+test).
      4. Then split (val+test) equally into val and test.
    """
    set_seed(seed)

    # Get dominant label per patient
    patient_labels = (
        df.groupby(patient_col)[label_col]
        .agg(lambda x: x.value_counts().index[0])   # mode
        .reset_index()
        .rename(columns={label_col: "dominant_label"})
    )

    patients = patient_labels[patient_col].values
    strata   = patient_labels["dominant_label"].values

    # ── Split 1: train vs rest (val+test)
    splitter1 = GroupShuffleSplit(
        n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed
    )
    train_idx, rest_idx = next(splitter1.split(patients, strata, groups=patients))
    train_patients = set(patients[train_idx])
    rest_patients_arr = patients[rest_idx]
    rest_strata_arr   = strata[rest_idx]

    # ── Split 2: val vs test from rest
    val_frac = val_ratio / (val_ratio + test_ratio)
    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=(1 - val_frac), random_state=seed + 1
    )
    val_idx, test_idx = next(splitter2.split(
        rest_patients_arr, rest_strata_arr, groups=rest_patients_arr
    ))
    val_patients  = set(rest_patients_arr[val_idx])
    test_patients = set(rest_patients_arr[test_idx])

    # ── Validate no patient overlap
    assert len(train_patients & val_patients)  == 0, "Patient leak: train/val"
    assert len(train_patients & test_patients) == 0, "Patient leak: train/test"
    assert len(val_patients   & test_patients) == 0, "Patient leak: val/test"

    # ── Assign labels
    df = df.copy()
    df["split"] = df[patient_col].map(
        lambda p: "train" if p in train_patients
                  else ("val" if p in val_patients else "test")
    )

    # Add image path column
    def find_path(img_id: str) -> str:
        # HAM10000 images live in HAM10000_images_part_1 or _part_2
        for subdir in ["HAM10000_images_part_1", "HAM10000_images_part_2", ""]:
            p = HAM10000_DIR / subdir / f"{img_id}.jpg" if subdir else HAM10000_DIR / f"{img_id}.jpg"
            if p.exists():
                return str(p)
        # Fallback: recursive search
        found = list(HAM10000_DIR.rglob(f"{img_id}.jpg"))
        return str(found[0]) if found else ""
    df["img_path"] = df["image_id"].map(find_path)

    # Add numeric label
    df["label"] = df[label_col].map(HAM_LABEL_MAP)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df


def print_split_report(train_df, val_df, test_df, label_col="dx"):
    total = len(train_df) + len(val_df) + len(test_df)
    print("\n" + "="*65)
    print(f"  SPLIT SUMMARY — {total} total images")
    print("="*65)
    print(f"{'Split':<8} {'Images':>8} {'%':>6} {'Lesions':>10}")
    print("-"*35)
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pct = len(df) / total * 100
        n_patients = df[PATIENT_COL].nunique()
        print(f"{name:<8} {len(df):>8,} {pct:>5.1f}% {n_patients:>10,}")

    print("\n  Class distribution per split:")
    all_classes = sorted(train_df[label_col].unique())
    header = f"  {'Class':<8}" + "".join([f"{'Train':>8}", f"{'Val':>6}", f"{'Test':>6}"])
    print(header)
    print("  " + "-"*30)
    for cls in all_classes:
        t = (train_df[label_col] == cls).sum()
        v = (val_df[label_col] == cls).sum()
        s = (test_df[label_col] == cls).sum()
        print(f"  {cls:<8} {t:>8,} {v:>6,} {s:>6,}")

    # Verify no patient overlap
    tp = set(train_df[PATIENT_COL])
    vp = set(val_df[PATIENT_COL])
    sp = set(test_df[PATIENT_COL])
    leaks = len(tp & vp) + len(tp & sp) + len(vp & sp)
    print(f"\n  Patient leaks: {leaks}")
    status = "✓ PASS" if leaks == 0 else "✗ FAIL"
    print(f"  Integrity check: {status}")
    print("="*65)


def create_splits(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load HAM10000 metadata and create patient-stratified splits."""
    meta_paths = list(HAM10000_DIR.glob("*metadata*.csv"))
    if not meta_paths:
        raise FileNotFoundError(
            f"HAM10000 metadata CSV not found in {HAM10000_DIR}. "
            "Run download_datasets.py first."
        )

    df = pd.read_csv(meta_paths[0])
    print(f"[OK   ] Loaded metadata: {len(df)} rows from {meta_paths[0].name}")

    # Check required columns
    for col in ["image_id", "dx", PATIENT_COL]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in metadata. Available: {list(df.columns)}")

    # Remove rows where image file doesn't exist
    def img_exists(img_id: str) -> bool:
        direct = HAM10000_DIR / f"{img_id}.jpg"
        if direct.exists():
            return True
        return len(list(HAM10000_DIR.rglob(f"{img_id}.jpg"))) > 0

    print("[INFO ] Verifying image file existence...")
    df["img_exists"] = df["image_id"].map(img_exists)
    n_missing = (~df["img_exists"]).sum()
    if n_missing > 0:
        print(f"[WARN ] {n_missing} rows have missing image files — excluding from splits")
    df = df[df["img_exists"]].drop(columns=["img_exists"]).reset_index(drop=True)

    print(f"[INFO ] Creating patient-stratified 70/15/15 splits...")
    train_df, val_df, test_df = patient_stratified_split(df, seed=seed)

    # Print summary
    print_split_report(train_df, val_df, test_df)

    # Save
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    # Save combined
    df_all = pd.concat([train_df, val_df, test_df])
    df_all.to_csv(SPLITS_DIR / "all_splits.csv", index=False)

    print(f"\n[OK   ] Splits saved to: {SPLITS_DIR}")
    print(f"         train.csv : {len(train_df):,} rows")
    print(f"         val.csv   : {len(val_df):,} rows")
    print(f"         test.csv  : {len(test_df):,} rows")

    return train_df, val_df, test_df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-computed split CSVs."""
    train_path = SPLITS_DIR / "train.csv"
    val_path   = SPLITS_DIR / "val.csv"
    test_path  = SPLITS_DIR / "test.csv"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            "Split CSVs not found. Run scripts/validate_datasets.py first."
        )

    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


if __name__ == "__main__":
    create_splits()
