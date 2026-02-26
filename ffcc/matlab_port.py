"""
FFCC MATLAB-style featurization port.

A line-by-line port of the Google FFCC MATLAB reference implementation,
preserving integer arithmetic paths and MATLAB 1-based indexing conventions
for maximum fidelity to the original results.

This module is used by the reproduction benchmarks to ensure pixel-exact
match with the MATLAB codebase.  For new projects, ``ffcc.core.featurize_image``
is recommended instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np


def _matlab_round(x: np.ndarray) -> np.ndarray:
    """MATLAB-compatible round: halves away from zero."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ====================================================================
# Parameter dataclasses (matching MATLAB struct fields)
# ====================================================================
@dataclass
class HistogramParams:
    NUM_BINS: int = 64
    BIN_SIZE: float = 1.0 / 32.0
    STARTING_UV: float = -0.4375
    MINIMUM_INTENSITY: float = 1.0 / 256.0
    VON_MISES_DIAGONAL_MODE: str = "pad"


@dataclass
class HyperParams:
    VON_MISES_DIAGONAL_EPS: float = 1.0


@dataclass
class TrainingParams:
    FORCE_ISOTROPIC_VON_MISES: bool = False


@dataclass
class DebugParams:
    GRAY_WORLD_UNWRAPPING: bool = False


def default_params() -> SimpleNamespace:
    """Minimal params subset required for FFCC inference."""
    return SimpleNamespace(
        HISTOGRAM=HistogramParams(),
        HYPERPARAMS=HyperParams(),
        TRAINING=TrainingParams(),
        DEBUG=DebugParams(),
    )


def enumerate_bins(params: SimpleNamespace) -> Tuple[np.ndarray, np.ndarray]:
    h = params.HISTOGRAM
    bins = h.STARTING_UV + np.arange(h.NUM_BINS, dtype=np.float64) * h.BIN_SIZE
    extents = np.array([bins[0] - h.BIN_SIZE / 2.0,
                        bins[-1] + h.BIN_SIZE / 2.0], dtype=np.float64)
    return bins, extents


# ====================================================================
# Image channelization
# ====================================================================
def _pad1(z: np.ndarray) -> np.ndarray:
    return np.pad(z, ((1, 1), (1, 1), (0, 0)), mode="edge")


def masked_local_absolute_deviation(im: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """MATLAB MaskedLocalAbsoluteDeviation (supports uint8/uint16/float)."""
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    im_class = im.dtype
    if im_class == np.uint16:
        im_work = im.astype(np.int32)
        mask_work = mask.astype(np.int32)
        out_cast = np.uint16
        is_float = False
    elif im_class == np.uint8:
        im_work = im.astype(np.int16)
        mask_work = mask.astype(np.int16)
        out_cast = np.uint8
        is_float = False
    elif np.issubdtype(im_class, np.floating):
        im_work = im.astype(np.float64, copy=False)
        mask_work = mask.astype(np.float64)
        out_cast = np.float64
        is_float = True
    else:
        raise TypeError(f"Unsupported dtype: {im.dtype}")

    im_pad = _pad1(im_work)
    mask_pad = np.pad(mask_work, ((1, 1), (1, 1)), mode="edge")
    h, w, c = im_work.shape

    outs = []
    for ch in range(c):
        numer = np.zeros((h, w), dtype=im_work.dtype if not is_float else np.float64)
        denom = np.zeros((h, w), dtype=im_work.dtype if not is_float else np.float64)
        center = im_work[:, :, ch]
        for oi in (-1, 0, 1):
            for oj in (-1, 0, 1):
                if oi == 0 and oj == 0:
                    continue
                im_shift = im_pad[1 + oi:1 + oi + h, 1 + oj:1 + oj + w, ch]
                mask_shift = mask_pad[1 + oi:1 + oi + h, 1 + oj:1 + oj + w]
                numer = numer + mask_work * mask_shift * np.abs(im_shift - center)
                denom = denom + mask_work * mask_shift

        if is_float:
            out = np.divide(numer, denom,
                            out=np.zeros_like(numer, dtype=np.float64),
                            where=denom != 0)
        else:
            numer_i = (numer.astype(np.int64) << 3)
            denom_i = denom.astype(np.int64)
            quot = np.zeros_like(numer_i)
            valid = denom_i != 0
            quot[valid] = numer_i[valid] // denom_i[valid]
            out = (quot >> 3)
        outs.append(out)

    im_edge = np.stack(outs, axis=2)
    if out_cast is np.float64:
        return im_edge.astype(np.float64)
    return im_edge.astype(out_cast)


def _channelize_image(im: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    im1 = (im.astype(np.float64) * mask[:, :, None]) \
        if np.issubdtype(im.dtype, np.floating) else (im * mask[:, :, None])
    im2 = masked_local_absolute_deviation(im, mask)
    return [im1.astype(im.dtype, copy=False), im2.astype(im.dtype, copy=False)]


def _psplat2(u, v, c, bin_lo, bin_step, n_bins):
    """MATLAB Psplat2.m — splatting with MATLAB 1-based indexing internally."""
    ub = 1 + np.mod(_matlab_round((u - bin_lo) / bin_step).astype(np.int64), n_bins)
    vb = 1 + np.mod(_matlab_round((v - bin_lo) / bin_step).astype(np.int64), n_bins)
    lin = (ub - 1) + (vb - 1) * n_bins
    accum = np.bincount(lin.ravel(), weights=c.ravel().astype(np.float64),
                        minlength=n_bins * n_bins)
    return accum.reshape((n_bins, n_bins), order="F")


def featurize_image(im: np.ndarray, mask: Optional[np.ndarray] = None,
                    params: Optional[SimpleNamespace] = None
                    ) -> tuple[np.ndarray, list[np.ndarray]]:
    """MATLAB FeaturizeImage replication.

    Returns:
        X: (NUM_BINS, NUM_BINS, 2) normalized histogram
        im_channels: list of channelized images
    """
    if params is None:
        params = default_params()
    if mask is None:
        mask = np.ones(im.shape[:2], dtype=bool)
    else:
        mask = mask.astype(bool)

    im_channels = _channelize_image(im, mask)
    X_channels = []
    for im_channel in im_channels:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_im = np.log(im_channel.astype(np.float64))
            u = log_im[:, :, 1] - log_im[:, :, 0]
            v = log_im[:, :, 1] - log_im[:, :, 2]

        valid = np.isfinite(u) & np.isfinite(v) & mask
        if np.issubdtype(im.dtype, np.floating):
            min_val = params.HISTOGRAM.MINIMUM_INTENSITY
        else:
            min_val = np.iinfo(im.dtype).max * params.HISTOGRAM.MINIMUM_INTENSITY
        valid = valid & np.all(im_channel >= min_val, axis=2)

        if np.any(valid):
            Xc = _psplat2(u[valid], v[valid],
                          np.ones(int(valid.sum()), dtype=np.float64),
                          params.HISTOGRAM.STARTING_UV,
                          params.HISTOGRAM.BIN_SIZE,
                          params.HISTOGRAM.NUM_BINS)
            s = Xc.sum()
            Xc = Xc / max(np.finfo(np.float64).eps, s)
        else:
            Xc = np.zeros((params.HISTOGRAM.NUM_BINS, params.HISTOGRAM.NUM_BINS),
                           dtype=np.float64)
        X_channels.append(Xc)

    X = np.stack(X_channels, axis=2)
    return X, im_channels


# ====================================================================
# Inference utilities (MATLAB-compatible 1-based indexing)
# ====================================================================
def uv_to_idx(Y, params=None):
    if params is None:
        params = default_params()
    Y = np.asarray(Y, dtype=np.float64)
    h = params.HISTOGRAM
    idx0 = _matlab_round((Y - h.STARTING_UV) / h.BIN_SIZE)
    return np.mod(idx0, h.NUM_BINS).astype(np.int64) + 1


def idx_to_uv(idx, params=None, Sigma_idx=None):
    if params is None:
        params = default_params()
    idx = np.asarray(idx, dtype=np.float64)
    h = params.HISTOGRAM
    uv = (idx - 1.0) * h.BIN_SIZE + h.STARTING_UV
    if Sigma_idx is None:
        return uv
    return uv, np.asarray(Sigma_idx, dtype=np.float64) * (h.BIN_SIZE ** 2)


def uv_to_rgb(uv):
    """Convert UV to unit-norm RGB (MATLAB convention)."""
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim == 1:
        uv = uv.reshape(2, 1)
    rgb = np.vstack([np.exp(-uv[0, :]), np.ones(uv.shape[1]), np.exp(-uv[1, :])])
    return rgb / np.sqrt(np.sum(rgb * rgb, axis=0, keepdims=True))


def fit_bivariate_von_mises(P):
    """MATLAB FitBivariateVonMises (1-based indexing output)."""
    P = np.asarray(P, dtype=np.float64)
    n = P.shape[0]

    angle_scale = n / (2.0 * np.pi)
    angle_step = 1.0 / angle_scale
    angles = np.arange(0.0, 2.0 * np.pi, angle_step, dtype=np.float64)
    if angles.shape[0] > n:
        angles = angles[:n]

    P1 = np.sum(P, axis=1)
    P2 = np.sum(P, axis=0)

    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    y1 = np.sum(P1 * sin_angles)
    x1 = np.sum(P1 * cos_angles)
    y2 = np.sum(P2 * sin_angles)
    x2 = np.sum(P2 * cos_angles)

    mu1 = np.mod(np.arctan2(y1, x1), 2.0 * np.pi) * angle_scale
    mu2 = np.mod(np.arctan2(y2, x2), 2.0 * np.pi) * angle_scale
    mu = np.array([mu1, mu2], dtype=np.float64) + 1.0  # 1-based

    bins = np.arange(1, n + 1, dtype=np.float64)
    mu_round = _matlab_round(mu).astype(np.int64)
    wrapped1 = np.mod((bins - mu_round[0]) + n / 2.0 - 1.0, n) + 1.0
    wrapped2 = np.mod((bins - mu_round[1]) + n / 2.0 - 1.0, n) + 1.0

    E1 = np.sum(P1 * wrapped1)
    E2 = np.sum(P2 * wrapped2)
    Sigma1 = np.sum(P1 * wrapped1 * wrapped1) - E1 * E1
    Sigma2 = np.sum(P2 * wrapped2 * wrapped2) - E2 * E2
    Sigma12 = np.sum(P * (wrapped1[:, None] * wrapped2[None, :])) - E1 * E2
    Sigma = np.array([[Sigma1, Sigma12], [Sigma12, Sigma2]], dtype=np.float64)
    return mu, Sigma


def evaluate_model_inference(F_fft, B, X, X_fft=None, params=None):
    """Inference-only subset of MATLAB EvaluateModel."""
    if params is None:
        params = default_params()
    if X_fft is None:
        X_fft = np.fft.fft2(X, axes=(0, 1))
    if B is None:
        B = np.zeros(X.shape[:2], dtype=np.float64)

    from ffcc.matlab_port import fit_bivariate_von_mises as _fvm

    FX_fft = np.sum(X_fft * F_fft, axis=2)
    FX = np.real(np.fft.ifft2(FX_fft))
    H = FX + B

    H_shifted = H - H.max()
    expH = np.exp(H_shifted)
    P = expH / expH.sum()

    mu_idx, Sigma_idx = _fvm(P)

    if params.TRAINING.FORCE_ISOTROPIC_VON_MISES:
        Sigma_idx = np.eye(2, dtype=np.float64) * np.mean(np.diag(Sigma_idx))

    if params.HISTOGRAM.VON_MISES_DIAGONAL_MODE == "clamp":
        epsv = params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS
        if Sigma_idx[0, 0] <= epsv:
            Sigma_idx = np.eye(2, dtype=np.float64) * epsv
    elif params.HISTOGRAM.VON_MISES_DIAGONAL_MODE == "pad":
        Sigma_idx = Sigma_idx + params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS * np.eye(2)

    mu_uv, Sigma_uv = idx_to_uv(mu_idx, params=params, Sigma_idx=Sigma_idx)
    state_obs = {"mu": mu_uv, "Sigma": Sigma_uv}

    entropy = 0.5 * np.log(np.linalg.det(Sigma_uv))
    min_entropy = 0.5 * np.log(
        np.linalg.det(params.HYPERPARAMS.VON_MISES_DIAGONAL_EPS
                      * (params.HISTOGRAM.BIN_SIZE ** 2) * np.eye(2)))
    meta = {
        "P": P,
        "entropy": float(entropy),
        "minimum_entropy": float(min_entropy),
        "entropy_confidence": float(np.exp(min_entropy - entropy)),
    }
    return state_obs, meta


def load_npz_model(npz_path: str) -> dict:
    """Load model weights from .npz file."""
    d = np.load(npz_path)
    F_fft = d["F_fft_real"] + 1j * d["F_fft_imag"]
    B = d["B"].astype(np.float64)
    return {"F_fft": F_fft, "B": B}
