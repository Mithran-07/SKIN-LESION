"""
Validate Datasets — Phase 2 CLI entrypoint
Runs full validation + creates patient-level splits.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.validator import run_full_validation
from data.splitter  import create_splits
from data.dataloader import export_class_statistics, visualize_sampler_distribution

RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR  = PROJECT_ROOT / "splits"
RESULTS_DIR.mkdir(exist_ok=True)
SPLITS_DIR.mkdir(exist_ok=True)


def main():
    print("\n" + "="*60)
    print("  PHASE 2 — DATASET VALIDATION & SPLIT CREATION")
    print("="*60)

    # ── 1. Validate datasets
    print("\n[STEP 1] Validating datasets...")
    ham_report, isic19_report = run_full_validation()

    # ── 2. Create patient-level splits
    print("\n[STEP 2] Creating patient-stratified train/val/test splits...")
    train_df, val_df, test_df = create_splits(seed=42)

    # ── 3. Export class statistics
    print("\n[STEP 3] Exporting class statistics...")
    export_class_statistics(train_df, RESULTS_DIR / "class_statistics.csv")

    # ── 4. Visualize WeightedRandomSampler effect
    print("\n[STEP 4] Visualizing sampler effect...")
    visualize_sampler_distribution(
        train_df,
        RESULTS_DIR / "sampler_distribution.png",
        n_batches=100, batch_size=32
    )

    print("\n" + "="*60)
    print("  ✅ PHASE 2 COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Splits : {SPLITS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
