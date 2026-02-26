"""
ffcc — Fast Fourier Color Constancy (Python)

A faithful Python reimplementation of the FFCC algorithm
(Barron & Tsai, CVPR 2017) for automatic white balance / illuminant estimation.

Reference:
    Barron, J.T. & Tsai, Y.-T. (2017). Fast Fourier Color Constancy.
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    https://github.com/google/ffcc
"""

from ffcc.core import (
    NUM_BINS,
    BIN_SIZE,
    STARTING_UV,
    MIN_INTENSITY,
    VON_MISES_DIAGONAL_EPS,
    FFCCModel,
    featurize_image,
    ffcc_forward,
    uv_to_rgb_gains,
    gt_rgb_to_uv,
    angular_error,
    angular_error_uv,
    train_ffcc,
)

__version__ = "0.1.0"

__all__ = [
    "NUM_BINS",
    "BIN_SIZE",
    "STARTING_UV",
    "MIN_INTENSITY",
    "VON_MISES_DIAGONAL_EPS",
    "FFCCModel",
    "featurize_image",
    "ffcc_forward",
    "uv_to_rgb_gains",
    "gt_rgb_to_uv",
    "angular_error",
    "angular_error_uv",
    "train_ffcc",
]
