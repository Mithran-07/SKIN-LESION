"""
Data Validator — Phase 2
Validates downloaded datasets, reports quality issues, and generates dataset_report.md.
"""

import os
import hashlib
import io
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, UnidentifiedImageError

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
RESULTS_DIR   = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HAM10000_DIR = DATASETS_ROOT / "HAM10000"
ISIC2019_DIR = DATASETS_ROOT / "ISIC2019"
ISIC2018_DIR = DATASETS_ROOT / "ISIC2018"

# HAM10000 7-class label map
HAM_LABEL_NAMES = {
    "akiec": "Actinic keratoses / Bowen",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}
VALID_LABELS = set(HAM_LABEL_NAMES.keys())


# ─────────────────────────────────────────────────────────
def validate_ham10000() -> dict:
    """Run all validation checks on HAM10000 dataset."""
    print("\n" + "="*60)
    print("  Validating HAM10000")
    print("="*60)

    report = {
        "name": "HAM10000",
        "total_images": 0,
        "metadata_rows": 0,
        "missing_images": [],
        "extra_images": [],
        "corrupted": [],
        "duplicates": [],
        "invalid_labels": [],
        "class_distribution": {},
        "patient_distribution": {},
        "errors": [],
    }

    # ── 1. Find metadata CSV
    meta_paths = list(HAM10000_DIR.glob("*metadata*.csv"))
    if not meta_paths:
        report["errors"].append("HAM10000_metadata.csv not found")
        print("[ERROR] metadata CSV missing")
        return report

    meta_path = meta_paths[0]
    df = pd.read_csv(meta_path)
    print(f"[OK   ] Metadata: {meta_path.name} — {len(df)} rows")
    report["metadata_rows"] = len(df)

    # Required columns
    for col in ["image_id", "dx", "lesion_id"]:
        if col not in df.columns:
            report["errors"].append(f"Missing column: {col}")
            print(f"[WARN ] Missing column: {col} (available: {list(df.columns)})")
    # Use lesion_id as patient proxy
    patient_col = "lesion_id" if "lesion_id" in df.columns else "image_id"

    # ── 2. Find all images on disk (search both part dirs)
    all_imgs = {}
    for subdir in ["HAM10000_images_part_1", "HAM10000_images_part_2", ""]:
        search_dir = HAM10000_DIR / subdir if subdir else HAM10000_DIR
        for p in search_dir.glob("*.jpg"):
            all_imgs[p.stem] = p
    report["total_images"] = len(all_imgs)
    print(f"[INFO ] Images on disk: {len(all_imgs)}")

    # ── 3. Missing images
    for img_id in df["image_id"]:
        if img_id not in all_imgs:
            report["missing_images"].append(img_id)
    if report["missing_images"]:
        print(f"[WARN ] Missing images: {len(report['missing_images'])}")
    else:
        print("[OK   ] No missing images")

    # ── 4. Extra images not in metadata
    meta_ids = set(df["image_id"])
    for img_id in all_imgs:
        if img_id not in meta_ids:
            report["extra_images"].append(img_id)
    if report["extra_images"]:
        print(f"[INFO ] Extra images (not in metadata): {len(report['extra_images'])}")

    # ── 5. Duplicate filenames
    all_names = [p.stem for p in HAM10000_DIR.glob("**/*.jpg")]
    counts = Counter(all_names)
    report["duplicates"] = [k for k, v in counts.items() if v > 1]
    if report["duplicates"]:
        print(f"[WARN ] Duplicate filenames: {len(report['duplicates'])}")
    else:
        print("[OK   ] No duplicate filenames")

    # ── 6. Invalid labels
    for label in df["dx"]:
        if label not in VALID_LABELS:
            report["invalid_labels"].append(label)
    if report["invalid_labels"]:
        print(f"[WARN ] Invalid labels: {set(report['invalid_labels'])}")
    else:
        print("[OK   ] All labels valid")

    # ── 7. Class distribution
    class_counts = df["dx"].value_counts().to_dict()
    report["class_distribution"] = class_counts
    print("\n[INFO ] Class distribution:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"        {cls:8}: {cnt:5} ({cnt/len(df)*100:.1f}%)")

    # ── 8. Patient distribution (using lesion_id)
    patient_counts = df.groupby(patient_col)["dx"].count().describe().to_dict()
    report["patient_distribution"] = {
        "unique_patients": df[patient_col].nunique(),
        "max_per_patient": int(df.groupby(patient_col)["dx"].count().max()),
        "min_per_patient": int(df.groupby(patient_col)["dx"].count().min()),
        "mean_per_patient": round(df.groupby(patient_col)["dx"].count().mean(), 2),
    }
    print(f"\n[INFO ] Unique lesions (patient proxy): {report['patient_distribution']['unique_patients']}")
    print(f"[INFO ] Images per lesion — max: {report['patient_distribution']['max_per_patient']}, "
          f"mean: {report['patient_distribution']['mean_per_patient']:.1f}")

    # ── 9. Corrupted images (sample check — check 200 random images for speed)
    import random
    random.seed(42)
    sample_paths = random.sample(list(all_imgs.values()), min(200, len(all_imgs)))
    for p in sample_paths:
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            report["corrupted"].append(str(p))
    if report["corrupted"]:
        print(f"[WARN ] Corrupted images found: {len(report['corrupted'])}")
    else:
        print("[OK   ] No corrupted images detected in sample")

    return report


# ─────────────────────────────────────────────────────────
def validate_isic2019() -> dict:
    """Validate ISIC 2019 dataset."""
    print("\n" + "="*60)
    print("  Validating ISIC 2019")
    print("="*60)

    report = {
        "name": "ISIC2019",
        "total_images": 0,
        "metadata_rows": 0,
        "class_distribution": {},
        "errors": [],
    }

    meta_path = ISIC2019_DIR / "ISIC_2019_Training_GroundTruth.csv"
    if not meta_path.exists():
        report["errors"].append("ISIC_2019_Training_GroundTruth.csv not found")
        print("[ERROR] ISIC 2019 metadata CSV missing")
        return report

    df = pd.read_csv(meta_path)
    report["metadata_rows"] = len(df)
    print(f"[OK   ] Metadata: {len(df)} rows")

    # Count images
    images_dir = ISIC2019_DIR / "images"
    if images_dir.exists():
        imgs = list(images_dir.glob("*.jpg"))
        report["total_images"] = len(imgs)
        print(f"[INFO ] Images on disk: {len(imgs)}")
    else:
        print("[WARN ] images/ directory not found")

    # Class distribution (ISIC 2019 format: one-hot columns)
    cls_cols = [c for c in df.columns if c not in ["image", "UNK"]]
    if cls_cols and len(cls_cols) > 0:
        cls_counts = {}
        for col in cls_cols:
            if col in df.columns:
                cls_counts[col] = int(df[col].sum()) if df[col].dtype != object else df[col].value_counts().get(1, 0)
        report["class_distribution"] = cls_counts
        print("[INFO ] Class distribution:")
        for cls, cnt in sorted(cls_counts.items()):
            print(f"        {cls:8}: {cnt:5}")

    return report


# ─────────────────────────────────────────────────────────
def plot_class_distributions(ham_report: dict, output_path: Path):
    """Generate class distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Dataset Class Distributions", fontsize=16, fontweight="bold")

    # HAM10000 distribution
    ax = axes[0]
    ham_dist = ham_report.get("class_distribution", {})
    if ham_dist:
        classes = sorted(ham_dist.keys())
        counts = [ham_dist[c] for c in classes]
        colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
        bars = ax.bar(classes, counts, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_title("HAM10000 Class Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Diagnosis Class")
        ax.set_ylabel("Number of Images")
        ax.tick_params(axis="x", rotation=30)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    str(cnt), ha="center", va="bottom", fontsize=9)

        # Imbalance indicator
        total = sum(counts)
        majority = max(counts)
        minority = min(counts)
        ax.set_xlabel(f"Classes | Imbalance Ratio: {majority/minority:.1f}x")
        ax.grid(axis="y", alpha=0.3)

    # HAM10000 pie chart
    ax2 = axes[1]
    if ham_dist:
        labels = [f"{c}\n({ham_dist[c]})" for c in sorted(ham_dist.keys())]
        sizes = [ham_dist[c] for c in sorted(ham_dist.keys())]
        wedge_props = dict(width=0.5, edgecolor="white")
        ax2.pie(sizes, labels=labels, autopct="%1.1f%%", pctdistance=0.75,
                wedgeprops=wedge_props, colors=plt.cm.Set2(np.linspace(0, 1, len(classes))))
        ax2.set_title("HAM10000 Class Share (%)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] Distribution chart saved -> {output_path}")


def generate_sample_grid(df: pd.DataFrame, images_dir: Path, output_path: Path,
                          n_per_class: int = 5):
    """Generate a grid showing n_per_class sample images per class."""
    import random
    random.seed(42)

    classes = sorted(df["dx"].unique())
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class,
                              figsize=(n_per_class * 3, n_classes * 3))
    fig.suptitle("Sample Images per Class (HAM10000)", fontsize=16, fontweight="bold", y=1.01)

    for row_idx, cls in enumerate(classes):
        cls_df = df[df["dx"] == cls]
        samples = cls_df.sample(min(n_per_class, len(cls_df)), random_state=42)

        for col_idx in range(n_per_class):
            ax = axes[row_idx][col_idx] if n_classes > 1 else axes[col_idx]
            ax.axis("off")

            if col_idx < len(samples):
                row = samples.iloc[col_idx]
                img_id = row["image_id"]
                patient_id = row.get("patient_id", "N/A")

                img_path = images_dir / f"{img_id}.jpg"
                if not img_path.exists():
                    # Search subdirectories
                    found = list(images_dir.rglob(f"{img_id}.jpg"))
                    img_path = found[0] if found else None

                if img_path and img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            ax.imshow(img)
                            ax.set_title(
                                f"{cls}\nPID:{str(patient_id)[:8]}\n{w}×{h}",
                                fontsize=7, pad=2
                            )
                    except Exception:
                        ax.text(0.5, 0.5, "Error", ha="center", va="center")
                else:
                    ax.text(0.5, 0.5, f"{img_id}\n(missing)", ha="center", va="center",
                            fontsize=7, color="red")

            if col_idx == 0:
                ax.set_ylabel(f"{cls}\n({HAM_LABEL_NAMES.get(cls, cls)})",
                               fontsize=9, labelpad=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK   ] Sample images grid saved -> {output_path}")


# ─────────────────────────────────────────────────────────
def generate_dataset_report(ham_report: dict, isic19_report: dict = None) -> str:
    """Generate results/dataset_report.md."""
    lines = []
    lines.append("# Dataset Validation Report\n")
    lines.append(f"*Generated automatically — Dual-Branch CNN Dermoscopy Project*\n")

    import datetime
    lines.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("---\n")

    # ── HAM10000
    lines.append("## 1. HAM10000\n")
    lines.append(f"| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| Total images on disk | {ham_report.get('total_images', 0):,} |\n")
    lines.append(f"| Metadata rows | {ham_report.get('metadata_rows', 0):,} |\n")
    lines.append(f"| Missing images | {len(ham_report.get('missing_images', []))} |\n")
    lines.append(f"| Extra images | {len(ham_report.get('extra_images', []))} |\n")
    lines.append(f"| Corrupted images (sample) | {len(ham_report.get('corrupted', []))} |\n")
    lines.append(f"| Duplicate filenames | {len(ham_report.get('duplicates', []))} |\n")
    lines.append(f"| Invalid labels | {len(ham_report.get('invalid_labels', []))} |\n")

    pd_dist = ham_report.get("patient_distribution", {})
    if pd_dist:
        lines.append(f"| Unique patients | {pd_dist.get('unique_patients', 'N/A')} |\n")
        lines.append(f"| Max images/patient | {pd_dist.get('max_per_patient', 'N/A')} |\n")
        lines.append(f"| Mean images/patient | {pd_dist.get('mean_per_patient', 'N/A')} |\n")
    lines.append("\n")

    # Class distribution table
    cls_dist = ham_report.get("class_distribution", {})
    if cls_dist:
        total = sum(cls_dist.values())
        lines.append("### Class Distribution\n\n")
        lines.append("| Class | Full Name | Count | % |\n")
        lines.append("|-------|-----------|-------|---|\n")
        for cls in sorted(cls_dist):
            cnt = cls_dist[cls]
            pct = cnt / total * 100
            name = HAM_LABEL_NAMES.get(cls, cls)
            lines.append(f"| {cls} | {name} | {cnt:,} | {pct:.1f}% |\n")
        lines.append(f"| **TOTAL** | | **{total:,}** | **100%** |\n")

        imbalance = max(cls_dist.values()) / min(cls_dist.values())
        lines.append(f"\n> **Class Imbalance Ratio**: {imbalance:.1f}x "
                     f"(majority/minority)\n")

    lines.append("\n![Class Distribution](class_distribution.png)\n")
    lines.append("\n---\n")

    # ── ISIC 2019
    if isic19_report:
        lines.append("## 2. ISIC 2019\n")
        lines.append(f"| Metric | Value |\n|--------|-------|\n")
        lines.append(f"| Total images on disk | {isic19_report.get('total_images', 0):,} |\n")
        lines.append(f"| Metadata rows | {isic19_report.get('metadata_rows', 0):,} |\n")
        if isic19_report.get("errors"):
            lines.append(f"| Errors | {'; '.join(isic19_report['errors'])} |\n")
        lines.append("\n")

        cls_dist_19 = isic19_report.get("class_distribution", {})
        if cls_dist_19:
            lines.append("### Class Distribution\n\n")
            lines.append("| Class | Count |\n|-------|-------|\n")
            for cls, cnt in sorted(cls_dist_19.items()):
                lines.append(f"| {cls} | {cnt:,} |\n")
        lines.append("\n---\n")

    # ── Errors section
    all_errors = ham_report.get("errors", [])
    if all_errors:
        lines.append("## ⚠️ Errors Found\n\n")
        for e in all_errors:
            lines.append(f"- {e}\n")

    # ── Sample images
    lines.append("## Sample Images\n\n")
    lines.append("![Sample Images per Class](sample_images.png)\n")

    report_text = "".join(lines)
    out_path = RESULTS_DIR / "dataset_report.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"[OK   ] Dataset report -> {out_path}")
    return report_text


# ─────────────────────────────────────────────────────────
def run_full_validation() -> Tuple[dict, dict]:
    """Run complete validation pipeline and generate all outputs."""
    print("\n" + "="*60)
    print("  FULL DATASET VALIDATION")
    print("="*60)

    ham_report = validate_ham10000()
    try:
        isic19_report = validate_isic2019()
    except Exception as e:
        print(f"[WARN ] ISIC 2019 validation skipped: {e}")
        isic19_report = {"name": "ISIC2019", "total_images": 0, "metadata_rows": 0,
                         "class_distribution": {}, "errors": [str(e)]}

    # Load HAM10000 metadata for visualizations
    meta_paths = list(HAM10000_DIR.glob("*metadata*.csv"))
    if meta_paths and ham_report.get("total_images", 0) > 0:
        df = pd.read_csv(meta_paths[0])
        try:
            plot_class_distributions(ham_report, RESULTS_DIR / "class_distribution.png")
        except Exception as e:
            print(f"[WARN ] Chart error: {e}")
        try:
            generate_sample_grid(df, HAM10000_DIR,
                                 RESULTS_DIR / "sample_images.png", n_per_class=5)
        except Exception as e:
            print(f"[WARN ] Sample grid error: {e}")

    try:
        generate_dataset_report(ham_report, isic19_report)
    except Exception as e:
        print(f"[WARN ] Report error: {e}")

    print("\n[DONE ] Validation complete. See results/dataset_report.md")
    return ham_report, isic19_report


if __name__ == "__main__":
    run_full_validation()
