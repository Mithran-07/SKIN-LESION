"""
Dataset download helpers for HAM10000 and ISIC 2019.

Supports two download methods:
1. Kaggle CLI (recommended — fastest)
2. ISIC API (fallback — slower, image-by-image)

Usage:
    python data/download.py --dataset ham10000 --output data/raw/
    python data/download.py --dataset isic2019 --output data/raw/
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


KAGGLE_DATASETS = {
    "ham10000": "kmader/skin-cancer-mnist-ham10000",
    "isic2019": "andrewmvd/isic-2019",
}

ISIC_API_COLLECTIONS = {
    "ham10000": "https://api.isic-archive.com/api/v2/images/?collections=54",
    "isic2019": "https://api.isic-archive.com/api/v2/images/?collections=57",
}


def download_via_kaggle(dataset: str, output_dir: Path) -> None:
    """
    Download a dataset using the Kaggle CLI.

    Requires:
        - kaggle CLI installed: pip install kaggle
        - API credentials at ~/.kaggle/kaggle.json
    """
    slug = KAGGLE_DATASETS.get(dataset)
    if not slug:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(KAGGLE_DATASETS)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading '{dataset}' via Kaggle CLI to {output_dir} ...")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", slug,
        "-p", str(output_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
    logger.info(f"Download complete. Files saved to: {output_dir}")


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def print_manual_instructions(dataset: str, output_dir: str) -> None:
    """Print manual download instructions if automatic download fails."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    if dataset == "ham10000":
        print("""
1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click 'Download' (requires free Kaggle account)
3. Extract the ZIP file
4. Place the following files/folders inside:
   {output}/HAM10000/
   ├── HAM10000_metadata.csv
   ├── HAM10000_images_part_1/   (contains *.jpg)
   └── HAM10000_images_part_2/   (contains *.jpg)

Alternatively, set up Kaggle CLI:
   pip install kaggle
   # Place your kaggle.json at ~/.kaggle/kaggle.json
   python data/download.py --dataset ham10000 --output {output}
""".format(output=output_dir))
    elif dataset == "isic2019":
        print("""
1. Go to: https://challenge.isic-archive.com/data/#2019
2. Download 'ISIC_2019_Training_Input.zip' and 'ISIC_2019_Training_GroundTruth.csv'
3. Extract to: {output}/ISIC2019/
""".format(output=output_dir))
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download dermoscopic datasets for the ADL project"
    )
    parser.add_argument(
        "--dataset",
        choices=["ham10000", "isic2019"],
        default="ham10000",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--method",
        choices=["kaggle", "manual"],
        default="kaggle",
        help="Download method",
    )
    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.method == "kaggle":
        if not check_kaggle_credentials():
            logger.warning("Kaggle credentials not found at ~/.kaggle/kaggle.json")
            print_manual_instructions(args.dataset, args.output)
            sys.exit(1)
        try:
            download_via_kaggle(args.dataset, output_dir)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            print_manual_instructions(args.dataset, args.output)
            sys.exit(1)
    else:
        print_manual_instructions(args.dataset, args.output)


if __name__ == "__main__":
    main()
