"""
Download Datasets — Phase 2 CLI entrypoint
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.downloader import download_ham10000, download_isic2019, download_isic2018, verify_downloads
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download dermoscopy datasets")
    parser.add_argument("--datasets", nargs="+",
                        choices=["ham10000", "isic2019", "isic2018", "all"],
                        default=["ham10000"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["ham10000", "isic2019", "isic2018"]

    if "ham10000" in datasets:
        download_ham10000(force=args.force)
    if "isic2019" in datasets:
        download_isic2019(force=args.force)
    if "isic2018" in datasets:
        download_isic2018(force=args.force)

    verify_downloads()

if __name__ == "__main__":
    main()
