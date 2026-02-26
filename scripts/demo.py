#!/usr/bin/env python3
"""
Quick demo: estimate illuminant of a single image using a pre-trained FFCC model.

Usage:
    python scripts/demo.py --image path/to/image.png --model models/gehler_fold1.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffcc import FFCCModel, featurize_image, uv_to_rgb_gains


def main():
    ap = argparse.ArgumentParser(description="FFCC single-image illuminant estimation")
    ap.add_argument("--image", type=str, required=True, help="Input image path (linear RGB)")
    ap.add_argument("--model", type=str, required=True, help="Pre-trained model (.npz)")
    args = ap.parse_args()

    # Load image
    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: cannot read {args.image}")
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to float [0, 1]
    if img.dtype == np.uint16:
        img = img.astype(np.float64) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0

    # Load model
    model = FFCCModel()
    model.load(args.model)

    # Predict
    X = featurize_image(img)
    mu_uv, P = model.forward(X)
    rgb_gains = uv_to_rgb_gains(mu_uv)

    print(f"Predicted illuminant (UV):  u={mu_uv[0]:.4f}, v={mu_uv[1]:.4f}")
    print(f"Predicted illuminant (RGB): R={rgb_gains[0]:.4f}, G={rgb_gains[1]:.4f}, B={rgb_gains[2]:.4f}")
    print(f"White balance gains:        R={1/rgb_gains[0]:.4f}, G={1/rgb_gains[1]:.4f}, B={1/rgb_gains[2]:.4f}")


if __name__ == "__main__":
    main()
