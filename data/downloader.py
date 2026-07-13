"""
Dataset Downloader — Phase 2
Downloads HAM10000 (Kaggle) and ISIC 2019/2018 (isic-cli/API) into the correct directory structure.
All downloads are resumable.
"""

import os
import sys
import json
import subprocess
import zipfile
import shutil
import requests
from pathlib import Path
from tqdm import tqdm


# ── Directory Layout ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
HAM10000_DIR  = DATASETS_ROOT / "HAM10000"
ISIC2019_DIR  = DATASETS_ROOT / "ISIC2019"
ISIC2018_DIR  = DATASETS_ROOT / "ISIC2018"


# ── Kaggle Credentials ────────────────────────────────────
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
KAGGLE_HAM10000_DATASET = "kmader/skin-cancer-mnist-ham10000"


# ── ISIC API (no auth needed for public data) ──────────────
ISIC_API_BASE = "https://api.isic-archive.com/api/v2"


# ─────────────────────────────────────────────────────────
def verify_kaggle_credentials() -> bool:
    if not KAGGLE_JSON.exists():
        print(f"[ERROR] Kaggle credentials not found at {KAGGLE_JSON}")
        print("[INFO ] Create your token at: https://www.kaggle.com → Account → API")
        return False
    try:
        creds = json.loads(KAGGLE_JSON.read_text())
        if "key" not in creds:
            print("[ERROR] kaggle.json is malformed — missing 'key' field")
            return False
        print(f"[OK   ] Kaggle credentials verified (user: {creds.get('username', 'N/A')})")
        return True
    except Exception as e:
        print(f"[ERROR] Could not read kaggle.json: {e}")
        return False


def _run(cmd: list[str], cwd: Path = None, env: dict = None) -> int:
    """Run a subprocess command, streaming output."""
    full_env = {**os.environ, **(env or {})}
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(cwd or PROJECT_ROOT), env=full_env
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    return process.returncode


def _extract_zip(zip_path: Path, dest: Path):
    print(f"[INFO ] Extracting {zip_path.name} → {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    zip_path.unlink()
    print(f"[OK   ] Extracted successfully")


# ─────────────────────────────────────────────────────────
#  HAM10000  (Kaggle)
# ─────────────────────────────────────────────────────────
def download_ham10000(force: bool = False):
    """Download HAM10000 from Kaggle. Skips if images already present."""
    HAM10000_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(HAM10000_DIR.glob("**/*.jpg"))
    if not force and len(existing_images) >= 10000:
        print(f"[SKIP ] HAM10000 already present ({len(existing_images)} images found)")
        return True

    if not verify_kaggle_credentials():
        return False

    # Try using installed kaggle package
    python = sys.executable
    print(f"\n[INFO ] Downloading HAM10000 from Kaggle (~3.5 GB)...")
    print(f"[INFO ] This may take 10–30 minutes depending on connection speed.\n")

    rc = _run([
        python, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_HAM10000_DATASET,
        "-p", str(HAM10000_DIR),
        "--unzip"
    ])

    if rc != 0:
        print(f"[WARN ] kaggle module download failed (exit {rc}), trying kaggle CLI...")
        # Fallback: try kaggle executable in PATH
        rc = _run([
            "kaggle", "datasets", "download",
            "-d", KAGGLE_HAM10000_DATASET,
            "-p", str(HAM10000_DIR),
            "--unzip"
        ])

    if rc != 0:
        print(f"[ERROR] HAM10000 download failed. Check Kaggle credentials.")
        return False

    # Flatten structure if nested (kaggle sometimes adds subdirs)
    _flatten_images(HAM10000_DIR)

    count = len(list(HAM10000_DIR.glob("**/*.jpg")))
    print(f"[OK   ] HAM10000 downloaded: {count} images in {HAM10000_DIR}")
    return True


def _flatten_images(directory: Path):
    """Move all .jpg files to directory root if in subdirectories."""
    for img in directory.rglob("*.jpg"):
        if img.parent != directory:
            target = directory / img.name
            if not target.exists():
                shutil.move(str(img), str(target))

    # Remove empty subdirectories
    for sub in sorted(directory.iterdir()):
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()


# ─────────────────────────────────────────────────────────
#  ISIC 2019  (ISIC CLI + metadata CSV)
# ─────────────────────────────────────────────────────────
def download_isic2019(force: bool = False):
    """Download ISIC 2019 challenge dataset via ISIC CLI."""
    ISIC2019_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = ISIC2019_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    meta_csv = ISIC2019_DIR / "ISIC_2019_Training_GroundTruth.csv"
    existing = list(images_dir.glob("*.jpg"))

    if not force and meta_csv.exists() and len(existing) > 1000:
        print(f"[SKIP ] ISIC 2019 already present ({len(existing)} images)")
        return True

    python = sys.executable
    print(f"\n[INFO ] Downloading ISIC 2019 metadata CSV...")

    # Download ground truth CSV directly
    _download_isic2019_metadata(meta_csv)

    print(f"\n[INFO ] Downloading ISIC 2019 images via isic-cli (~8 GB)...")
    print(f"[INFO ] This is a large download — may take 1–3 hours.\n")

    # Use isic-cli to download images
    rc = _run([
        python, "-m", "isic_cli", "image", "download",
        "--search", "attribution:\"ISIC 2019\"",
        str(images_dir)
    ])

    if rc != 0:
        # Fallback: use isic executable directly
        rc = _run([
            "isic", "image", "download",
            "--search", "attribution:\"ISIC 2019\"",
            str(images_dir)
        ])

    count = len(list(images_dir.glob("*.jpg")))
    if count > 0:
        print(f"[OK   ] ISIC 2019: {count} images in {images_dir}")
        return True
    else:
        print(f"[WARN ] ISIC 2019 image download may have issues — check {images_dir}")
        return False


def _download_isic2019_metadata(dest: Path):
    """Download ISIC 2019 ground truth CSV."""
    # Primary: official ISIC challenge URL
    urls = [
        "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
        "https://challenge.isic-archive.com/data/2019/ISIC_2019_Training_GroundTruth.csv",
    ]

    for url in urls:
        try:
            print(f"[INFO ] Fetching: {url}")
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                print(f"[OK   ] Metadata saved → {dest}")
                return
        except Exception as e:
            print(f"[WARN ] Failed ({e}), trying next URL...")

    print(f"[WARN ] Could not download ISIC 2019 metadata CSV. Manual download may be needed.")


# ─────────────────────────────────────────────────────────
#  ISIC 2018  (Segmentation metadata — deferred images)
# ─────────────────────────────────────────────────────────
def download_isic2018(force: bool = False, images: bool = False):
    """Download ISIC 2018 task-3 metadata. Images optional."""
    ISIC2018_DIR.mkdir(parents=True, exist_ok=True)

    meta_csv = ISIC2018_DIR / "ISIC2018_Task3_Training_GroundTruth.csv"

    if not force and meta_csv.exists():
        print(f"[SKIP ] ISIC 2018 metadata already present")
        return True

    print(f"\n[INFO ] Downloading ISIC 2018 Task 3 ground truth...")
    urls = [
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.csv",
        "https://challenge.isic-archive.com/data/2018/ISIC2018_Task3_Training_GroundTruth.csv",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                with open(meta_csv, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                print(f"[OK   ] ISIC 2018 metadata → {meta_csv}")
                break
        except Exception as e:
            print(f"[WARN ] {e}")

    if images:
        images_dir = ISIC2018_DIR / "images"
        images_dir.mkdir(exist_ok=True)
        python = sys.executable
        print(f"\n[INFO ] Downloading ISIC 2018 images (~10 GB)...")
        _run([python, "-m", "isic_cli", "image", "download",
              "--search", "attribution:\"ISIC 2018\"", str(images_dir)])

    return True


# ─────────────────────────────────────────────────────────
#  Quick verification of downloads
# ─────────────────────────────────────────────────────────
def verify_downloads() -> dict:
    """Return a summary of what is available."""
    results = {}

    # HAM10000
    ham_imgs = list(HAM10000_DIR.glob("**/*.jpg"))
    ham_meta = HAM10000_DIR / "HAM10000_metadata.csv"
    results["HAM10000"] = {
        "images": len(ham_imgs),
        "metadata": ham_meta.exists(),
        "path": str(HAM10000_DIR),
        "ok": len(ham_imgs) >= 9000 and ham_meta.exists()
    }

    # ISIC 2019
    isic19_imgs = list((ISIC2019_DIR / "images").glob("*.jpg")) if (ISIC2019_DIR / "images").exists() else []
    isic19_meta = ISIC2019_DIR / "ISIC_2019_Training_GroundTruth.csv"
    results["ISIC2019"] = {
        "images": len(isic19_imgs),
        "metadata": isic19_meta.exists(),
        "path": str(ISIC2019_DIR),
        "ok": len(isic19_imgs) > 0 and isic19_meta.exists()
    }

    # ISIC 2018
    isic18_meta = ISIC2018_DIR / "ISIC2018_Task3_Training_GroundTruth.csv"
    results["ISIC2018"] = {
        "metadata": isic18_meta.exists(),
        "path": str(ISIC2018_DIR),
        "ok": isic18_meta.exists()
    }

    print("\n" + "="*60)
    print("  DATASET STATUS")
    print("="*60)
    for name, info in results.items():
        status = "✓" if info["ok"] else "✗"
        imgs = info.get("images", "N/A")
        print(f"  {status} {name:12} | Images: {str(imgs):>6} | Meta: {info['metadata']}")
    print("="*60)

    return results


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download dermoscopy datasets")
    parser.add_argument("--datasets", nargs="+",
                        choices=["ham10000", "isic2019", "isic2018", "all"],
                        default=["all"])
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--isic2018-images", action="store_true", help="Also download ISIC 2018 images")
    args = parser.parse_args()

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["ham10000", "isic2019", "isic2018"]

    if "ham10000" in datasets:
        download_ham10000(force=args.force)

    if "isic2019" in datasets:
        download_isic2019(force=args.force)

    if "isic2018" in datasets:
        download_isic2018(force=args.force, images=args.isic2018_images)

    verify_downloads()
