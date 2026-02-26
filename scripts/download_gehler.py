#!/usr/bin/env python3
"""
Download and prepare the Gehler/Shi (Reprocessed) dataset for FFCC benchmarking.

The dataset consists of 568 linear RGB images with measured ground-truth
illuminants, commonly used for evaluating color constancy algorithms.

Source: Shi & Funt (2010), reprocessed by Barron (2015).

Usage:
    python scripts/download_gehler.py --output-dir data/GehlerShiThumb
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


# The preprocessed GehlerShiThumb data from the FFCC MATLAB repo
GEHLER_URLS = {
    "gehler_shi_thumb": (
        "https://github.com/google/ffcc/raw/master/data/shi_gehler/"
        "preprocessed/GehlerShiThumb.zip"
    ),
}

# Alternative: manual download instructions
MANUAL_INSTRUCTIONS = """
If automatic download fails, you can prepare the dataset manually:

1. Clone the Google FFCC MATLAB repo:
   git clone https://github.com/google/ffcc.git

2. The GehlerShiThumb data is at:
   ffcc/data/shi_gehler/preprocessed/GehlerShiThumb/

3. Copy that folder to data/GehlerShiThumb/ in this project.

The folder should contain:
  - 568 PNG files (000001.png ... 000568.png)
  - 568 TXT files (000001.txt ... 000568.txt) with ground-truth RGB illuminants
  - cvfolds.txt with 3-fold cross-validation assignments

Alternative: prepare your own data
  - Place linear RGB PNG images (uint8 or uint16) in the folder
  - For each image, create a .txt file with 3 values (R G B illuminant)
  - Create cvfolds.txt with one integer per line (1, 2, or 3)
"""


def download_with_progress(url, dest):
    """Download a file with a simple progress indicator."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  Progress: {pct}%", end="", flush=True)

    urlretrieve(url, dest, reporthook)
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Download and prepare the Gehler/Shi dataset")
    ap.add_argument("--output-dir", type=str, default="data/GehlerShiThumb",
                    help="Output directory for the dataset")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    # Check if already exists
    existing_pngs = list(output_dir.glob("[0-9]*.png")) if output_dir.exists() else []
    if len(existing_pngs) >= 568:
        print(f"Dataset already exists at {output_dir} ({len(existing_pngs)} images)")
        return

    print("=" * 60)
    print("Gehler/Shi Dataset Preparation")
    print("=" * 60)

    # Try automatic download
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir.parent / "GehlerShiThumb.zip"

        download_with_progress(GEHLER_URLS["gehler_shi_thumb"], str(zip_path))

        print("Extracting...")
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(output_dir.parent))

        zip_path.unlink()  # Clean up zip
        print(f"Dataset ready at {output_dir}")

    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    # Verify
    pngs = list(output_dir.glob("[0-9]*.png"))
    txts = list(output_dir.glob("[0-9]*.txt"))
    cvfolds = output_dir / "cvfolds.txt"

    print(f"\nVerification:")
    print(f"  PNG images: {len(pngs)}")
    print(f"  GT files:   {len(txts)}")
    print(f"  cvfolds.txt: {'exists' if cvfolds.exists() else 'MISSING'}")

    if len(pngs) >= 568 and cvfolds.exists():
        print("\nDataset ready for benchmarking!")
        print(f"Run: python scripts/benchmark_gehler.py --data-dir {output_dir}")
    else:
        print("\nWARNING: Dataset may be incomplete.")


if __name__ == "__main__":
    main()
