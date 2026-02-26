"""
FFCC: Fast Fourier Color Constancy — Core Implementation

A faithful Python reimplementation of the FFCC algorithm
(Barron & Tsai, CVPR 2017) for automatic white balance / illuminant estimation.

This module provides:
  - UV histogram featurization (2-channel: pixel + edge)
  - FFT-based convolution forward pass
  - Bivariate Von Mises distribution fitting on the torus
  - L-BFGS training with cross-entropy + Von Mises loss annealing
  - Standard color-constancy angular error metrics

Reference MATLAB code: https://github.com/google/ffcc
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize


# ====================================================================
# Constants (FFCC defaults, matching the MATLAB reference)
# ====================================================================
NUM_BINS: int = 64
BIN_SIZE: float = 1.0 / 32.0       # 0.03125
STARTING_UV: float = -0.4375
MIN_INTENSITY: float = 1.0 / 256.0
VON_MISES_DIAGONAL_EPS: float = 1.0


def _matlab_round(x):
    """MATLAB-compatible round: halves away from zero."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ====================================================================
# UV Space & Histogram
# Matches MATLAB: RgbToUv.m, Psplat2.m, FeaturizeImage.m,
# ChannelizeImage.m, MaskedLocalAbsoluteDeviation.m
# ====================================================================
def rgb_to_uv(image: np.ndarray):
    """Convert linear RGB image to log-chroma (u, v).

    u = log(G/R), v = log(G/B).  Matches MATLAB RgbToUv.m.

    Returns:
        u, v: (H, W) arrays of log-chroma values (NaN where invalid)
        mask: (H, W) boolean validity mask
    """
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    intensity = (R + G + B) / 3.0
    mask = (intensity >= MIN_INTENSITY) & (R > 0) & (G > 0) & (B > 0)

    u = np.full_like(R, np.nan)
    v = np.full_like(R, np.nan)
    u[mask] = np.log(G[mask] / R[mask])
    v[mask] = np.log(G[mask] / B[mask])

    return u, v, mask


def rgb_to_uv_vec(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB illuminant vector (3,) to UV."""
    return np.array([np.log(rgb[1] / rgb[0]), np.log(rgb[1] / rgb[2])])


def psplat2(u, v, weights, starting_uv=None,
            bin_size=BIN_SIZE, num_bins=NUM_BINS):
    """Splat (u,v) onto a 2D histogram with periodic wrapping.

    Matches MATLAB Psplat2.m exactly.
    """
    if starting_uv is None:
        starting_uv = STARTING_UV
    ub = np.mod(_matlab_round((u - starting_uv) / bin_size).astype(int), num_bins)
    vb = np.mod(_matlab_round((v - starting_uv) / bin_size).astype(int), num_bins)
    hist = np.zeros((num_bins, num_bins), dtype=np.float64)
    np.add.at(hist, (ub, vb), weights)
    return hist


def masked_local_absolute_deviation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel Masked Local Absolute Deviation (3x3 window).

    Matches MATLAB MaskedLocalAbsoluteDeviation.m exactly.
    For each channel, computes the masked average absolute difference
    from 8 neighbors, producing a 3-channel edge image.
    """
    H, W, C = image.shape
    mask_f = mask.astype(np.float64)

    im_pad = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    mask_pad = np.pad(mask_f, ((1, 1), (1, 1)), mode='edge')

    edge_image = np.zeros_like(image, dtype=np.float64)
    for c in range(C):
        numer = np.zeros((H, W), dtype=np.float64)
        denom = np.zeros((H, W), dtype=np.float64)
        for oi in range(-1, 2):
            for oj in range(-1, 2):
                if oi == 0 and oj == 0:
                    continue
                im_shift = im_pad[1 + oi:H + 1 + oi, 1 + oj:W + 1 + oj, c]
                mask_shift = mask_pad[1 + oi:H + 1 + oi, 1 + oj:W + 1 + oj]
                numer += mask_f * mask_shift * np.abs(im_shift - image[:, :, c])
                denom += mask_f * mask_shift
        safe_denom = np.where(denom > 0, denom, 1.0)
        edge_image[:, :, c] = numer / safe_denom

    return edge_image


def featurize_image(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Extract 2-channel UV histogram feature from an image.

    Matches MATLAB FeaturizeImage.m + ChannelizeImage.m exactly.

    Channel 0: log-chroma histogram from the original (masked) image.
    Channel 1: log-chroma histogram from the MaskedLocalAbsoluteDeviation
               edge image — independent UV computed from the 3-channel edge image.

    Args:
        image: (H, W, 3) float64 linear RGB image in [0, 1]
        mask:  (H, W) boolean mask (default: all True)

    Returns:
        X: (NUM_BINS, NUM_BINS, 2) float64 normalized histogram features
    """
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)

    ch1 = image * mask[:, :, np.newaxis].astype(np.float64)
    ch2 = masked_local_absolute_deviation(image, mask)

    channels = [ch1, ch2]
    histograms = []

    for im_ch in channels:
        R = im_ch[:, :, 0].astype(np.float64)
        G = im_ch[:, :, 1].astype(np.float64)
        B = im_ch[:, :, 2].astype(np.float64)

        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.log(G) - np.log(R)
            v = np.log(G) - np.log(B)

        valid = (~np.isinf(u) & ~np.isinf(v) &
                 ~np.isnan(u) & ~np.isnan(v) & mask)
        valid &= np.all(im_ch >= MIN_INTENSITY, axis=2)

        n_valid = np.count_nonzero(valid)
        if n_valid > 0:
            Xc = psplat2(u[valid], v[valid], np.ones(n_valid))
        else:
            Xc = np.zeros((NUM_BINS, NUM_BINS), dtype=np.float64)

        total = Xc.sum()
        if total > 0:
            Xc /= total
        histograms.append(Xc)

    return np.stack(histograms, axis=-1)


# ====================================================================
# FFCC Forward Pass & Distribution Fitting
# Matches MATLAB: EvaluateModel.m, FitBivariateVonMises.m,
# SoftmaxForward.m, SoftmaxBackward.m, UvToRgb.m
# ====================================================================
def uv_to_idx(uv):
    """UV value to 0-indexed bin index."""
    return np.mod(_matlab_round((uv - STARTING_UV) / BIN_SIZE), NUM_BINS)


def idx_to_uv(idx):
    """0-indexed bin index to UV value."""
    return idx * BIN_SIZE + STARTING_UV


def softmax_forward(H):
    """Softmax over entire 2D array. Matches MATLAB SoftmaxForward.m."""
    H_shifted = H - H.max()
    expH = np.exp(H_shifted)
    expH_sum = expH.sum()
    P = expH / expH_sum
    meta = {'P': P, 'expH': expH, 'expH_sum': expH_sum, 'H_shifted': H_shifted}
    return P, meta


def softmax_backward(dP, meta):
    """Backprop through softmax. Matches MATLAB SoftmaxBackward.m."""
    d_expH = dP / meta['expH_sum']
    d_sum = np.sum(d_expH * meta['P'])
    dH = (d_expH - d_sum) * meta['expH']
    return dH


def fit_bivariate_von_mises(P, compute_grad=False):
    """Fit bivariate Von Mises distribution to 2D PDF.

    Matches MATLAB FitBivariateVonMises.m exactly.

    Returns:
        mu_idx: (2,) mean in 0-indexed bin coordinates
        Sigma_idx: (2,2) covariance matrix in bin coordinates
        dmu_P: (optional) partial derivatives of mu w.r.t. P
        dSigma_P: (optional) partial derivatives of Sigma w.r.t. P
    """
    n = P.shape[0]
    angle_scale = n / (2 * np.pi)
    angle_step = 1.0 / angle_scale
    angles = np.arange(0, 2 * np.pi, angle_step)

    P1 = P.sum(axis=1)
    P2 = P.sum(axis=0)

    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)

    y1 = np.sum(P1 * sin_angles)
    x1 = np.sum(P1 * cos_angles)
    y2 = np.sum(P2 * sin_angles)
    x2 = np.sum(P2 * cos_angles)

    mu1 = np.mod(np.arctan2(y1, x1), 2 * np.pi) * angle_scale
    mu2 = np.mod(np.arctan2(y2, x2), 2 * np.pi) * angle_scale
    mu_idx = np.array([mu1, mu2])

    bins = np.arange(n)
    wrap = lambda x: np.mod(x + n // 2, n)
    wrapped1 = wrap(bins - int(_matlab_round(mu1)))
    wrapped2 = wrap(bins - int(_matlab_round(mu2)))

    E1 = np.sum(P1 * wrapped1)
    E2 = np.sum(P2 * wrapped2)
    Sigma1 = np.sum(P1 * wrapped1**2) - E1**2
    Sigma2 = np.sum(P2 * wrapped2**2) - E2**2
    Sigma12 = np.sum(P * np.outer(wrapped1, wrapped2)) - E1 * E2

    Sigma_idx = np.array([[Sigma1, Sigma12],
                          [Sigma12, Sigma2]])

    if not compute_grad:
        return mu_idx, Sigma_idx

    r_sq_1 = x1**2 + y1**2
    r_sq_2 = x2**2 + y2**2
    dmu1_P1 = ((x1 * sin_angles - y1 * cos_angles) / r_sq_1) * angle_scale
    dmu2_P2 = ((x2 * sin_angles - y2 * cos_angles) / r_sq_2) * angle_scale

    dSigma1_P1 = wrapped1 * (wrapped1 - 2 * E1)
    dSigma2_P2 = wrapped2 * (wrapped2 - 2 * E2)
    dSigma12_P = np.outer(wrapped1 - E1, wrapped2 - E2) - (E1 * E2)

    return mu_idx, Sigma_idx, (dmu1_P1, dmu2_P2), (dSigma1_P1, dSigma12_P, dSigma2_P2)


def circular_mean(P):
    """Compute circular mean of 2D probability on torus, returning UV."""
    mu_idx, _ = fit_bivariate_von_mises(P)
    return np.array([idx_to_uv(mu_idx[0]), idx_to_uv(mu_idx[1])])


def ffcc_forward(X, F_fft, B):
    """FFCC forward pass: predict illuminant UV from histogram features.

    Matches MATLAB EvaluateModel.m.

    Args:
        X: (NUM_BINS, NUM_BINS, 2) histogram features
        F_fft: (NUM_BINS, NUM_BINS, 2) learned FFT-domain filters
        B: (NUM_BINS, NUM_BINS) learned bias

    Returns:
        mu_uv: (2,) predicted illuminant in UV space
        P: (NUM_BINS, NUM_BINS) posterior probability over UV
    """
    X_fft = np.fft.fft2(X, axes=(0, 1))
    FX_fft = np.sum(X_fft * F_fft, axis=2)
    H = np.real(np.fft.ifft2(FX_fft)) + B

    P, _ = softmax_forward(H)

    mu_idx, Sigma_idx = fit_bivariate_von_mises(P)
    Sigma_idx += VON_MISES_DIAGONAL_EPS * np.eye(2)

    mu_uv = idx_to_uv(mu_idx)
    return mu_uv, P


def uv_to_rgb_gains(mu_uv: np.ndarray) -> np.ndarray:
    """Convert UV illuminant estimate to unit-norm RGB.

    Matches MATLAB UvToRgb.m: rgb = [exp(-u); 1; exp(-v)] / norm.
    """
    u, v = mu_uv[0], mu_uv[1]
    rgb = np.array([np.exp(-u), 1.0, np.exp(-v)])
    return rgb / np.sqrt(np.sum(rgb**2))


# ====================================================================
# Error Metrics
# ====================================================================
def angular_error(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    """Angular error in degrees between two RGB illuminants."""
    pred_norm = pred_rgb / (np.linalg.norm(pred_rgb) + 1e-10)
    gt_norm = gt_rgb / (np.linalg.norm(gt_rgb) + 1e-10)
    cos_angle = np.clip(np.dot(pred_norm, gt_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def angular_error_uv(pred_uv: np.ndarray, gt_uv: np.ndarray) -> float:
    """Angular error from UV-space predictions."""
    return angular_error(uv_to_rgb_gains(pred_uv), uv_to_rgb_gains(gt_uv))


def gt_rgb_to_uv(gt_rgb: np.ndarray) -> np.ndarray:
    """Convert GT illuminant RGB to UV space: u = log(G/R), v = log(G/B)."""
    R, G, B = gt_rgb[0], gt_rgb[1], gt_rgb[2]
    if R <= 0 or G <= 0 or B <= 0:
        return np.array([0.0, 0.0])
    return np.array([np.log(G / R), np.log(G / B)])


def make_gt_soft_label(gt_uv: np.ndarray) -> np.ndarray:
    """Create soft GT probability map via bilinear interpolation.

    Matches MATLAB UvToP.m.
    """
    P_gt = np.zeros((NUM_BINS, NUM_BINS), dtype=np.float64)

    y_ij = (gt_uv - STARTING_UV) / BIN_SIZE
    y_lo = np.floor(y_ij)
    w_1 = y_ij - y_lo
    w_0 = 1 - w_1

    wrap = lambda x: int(np.mod(x, NUM_BINS))

    i_lo = wrap(y_lo[0])
    i_hi = wrap(y_lo[0] + 1)
    j_lo = wrap(y_lo[1])
    j_hi = wrap(y_lo[1] + 1)

    P_gt[i_lo, j_lo] = w_0[0] * w_0[1]
    P_gt[i_lo, j_hi] = w_0[0] * w_1[1]
    P_gt[i_hi, j_lo] = w_1[0] * w_0[1]
    P_gt[i_hi, j_hi] = w_1[0] * w_1[1]

    return P_gt


def ll_multivariate_normal(X, mu, Sigma):
    """Log-likelihood of multivariate normal.

    Matches MATLAB LLMultivariateNormal.m.
    Returns: LL, dLL_mu, dLL_Sigma
    """
    Xc = X - mu
    iSigma = np.linalg.inv(Sigma)
    Xc_iSigma = iSigma @ Xc

    logZ = 0.5 * (2 * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)))
    LL = -logZ - 0.5 * np.dot(Xc_iSigma, Xc)

    dLL_mu = Xc_iSigma
    dLL_Sigma = (np.outer(Xc_iSigma, Xc_iSigma) - iSigma) / 2

    return LL, dLL_mu, dLL_Sigma


def compute_error_metrics(errors: np.ndarray) -> dict:
    """Standard color-constancy error statistics.

    Matches MATLAB error-reporting conventions.
    """
    e = np.sort(np.asarray(errors, dtype=np.float64).ravel())
    n = len(e)
    p25, p50, p75, p95 = np.percentile(e, [25, 50, 75, 95])
    return {
        "mean": float(np.mean(e)),
        "median": float(p50),
        "trimean": float((p25 + 2 * p50 + p75) / 4.0),
        "best25": float(np.mean(e[e <= p25])),
        "worst25": float(np.mean(e[e >= p75])),
        "worst5": float(np.mean(e[e >= p95])),
        "max": float(np.max(e)),
        "count": n,
    }


# ====================================================================
# FFCCModel — Trainable wrapper
# ====================================================================
class FFCCModel:
    """Trainable FFCC model with filter F and bias B."""

    def __init__(self, n_channels: int = 2, init_mode: str = 'random',
                 rng_seed: int = 0):
        self.n_channels = n_channels
        n = NUM_BINS
        if init_mode == 'zeros':
            self.F = np.zeros((n, n, n_channels), dtype=np.float64)
        else:
            rng = np.random.default_rng(rng_seed)
            self.F = rng.standard_normal((n, n, n_channels)) * 0.001
        self.B = np.zeros((n, n), dtype=np.float64)
        self._update_fft()

    def _update_fft(self):
        self.F_fft = np.zeros((NUM_BINS, NUM_BINS, self.n_channels),
                              dtype=np.complex128)
        for c in range(self.n_channels):
            self.F_fft[:, :, c] = np.fft.fft2(self.F[:, :, c])

    def get_params(self) -> np.ndarray:
        """Flatten all parameters to a single vector for L-BFGS."""
        parts = []
        for c in range(self.n_channels):
            parts.append(np.real(self.F_fft[:, :, c]).ravel())
            parts.append(np.imag(self.F_fft[:, :, c]).ravel())
        parts.append(self.B.ravel())
        return np.concatenate(parts)

    def set_params(self, params: np.ndarray):
        """Restore parameters from flat vector."""
        n = NUM_BINS
        n2 = n * n
        offset = 0
        for c in range(self.n_channels):
            real_part = params[offset:offset + n2].reshape(n, n)
            offset += n2
            imag_part = params[offset:offset + n2].reshape(n, n)
            offset += n2
            self.F_fft[:, :, c] = real_part + 1j * imag_part
            self.F[:, :, c] = np.real(np.fft.ifft2(self.F_fft[:, :, c]))
        self.B = params[offset:offset + n2].reshape(n, n)

    def forward(self, X: np.ndarray):
        """Single-sample forward pass → (mu_uv, P)."""
        return ffcc_forward(X, self.F_fft, self.B)

    def predict(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """End-to-end: image → RGB illuminant estimate."""
        X = featurize_image(image, mask)
        mu_uv, _ = self.forward(X)
        return uv_to_rgb_gains(mu_uv)

    def save(self, path: str):
        """Save model weights to .npz file."""
        np.savez(path, F_fft_real=np.real(self.F_fft),
                 F_fft_imag=np.imag(self.F_fft), B=self.B)

    def load(self, path: str):
        """Load model weights from .npz file."""
        data = np.load(path)
        self.F_fft = data['F_fft_real'] + 1j * data['F_fft_imag']
        self.B = data['B']
        self.n_channels = self.F_fft.shape[2]
        self.F = np.zeros((NUM_BINS, NUM_BINS, self.n_channels), dtype=np.float64)
        for c in range(self.n_channels):
            self.F[:, :, c] = np.real(np.fft.ifft2(self.F_fft[:, :, c]))

    def get_model_dict(self) -> dict:
        return {'F_fft': self.F_fft, 'B': self.B}


# ====================================================================
# Training — L-BFGS with Cross-Entropy + Von Mises Loss Annealing
# Matches MATLAB: TrainModel.m, TrainModelLossfun.m
# ====================================================================
def _ffcc_loss_and_grad(params_vec, X_fft_all, P_gt_all, gt_uv_all, N,
                        n_channels, tv, loss_mult_crossent, loss_mult_vonmises,
                        filter_precond, bias_precond):
    """Combined loss and gradient for L-BFGS.

    Implements proper backprop through Von Mises fitting,
    matching MATLAB TrainModelLossfun.m + EvaluateModel.m.
    """
    n = NUM_BINS
    n2 = n * n

    F_fft = np.zeros((n, n, n_channels), dtype=np.complex128)
    offset = 0
    for c in range(n_channels):
        real_part = params_vec[offset:offset + n2].reshape(n, n)
        offset += n2
        imag_part = params_vec[offset:offset + n2].reshape(n, n)
        offset += n2
        F_fft[:, :, c] = real_part + 1j * imag_part
    B = params_vec[offset:offset + n2].reshape(n, n)

    total_loss = 0.0
    grad_F_fft = np.zeros_like(F_fft)
    grad_B = np.zeros((n, n), dtype=np.float64)

    for i in range(N):
        FX_fft = np.sum(X_fft_all[i] * F_fft, axis=2)
        H = np.real(np.fft.ifft2(FX_fft)) + B

        P, P_meta = softmax_forward(H)
        d_loss_H = np.zeros_like(H)

        # Cross-entropy loss
        if loss_mult_crossent > 0:
            log_P = P_meta['H_shifted'] - np.log(P_meta['expH_sum'])
            ce_loss = -np.sum(P_gt_all[i] * log_P)
            ce_loss = max(0.0, ce_loss)
            total_loss += loss_mult_crossent * ce_loss
            d_loss_H += loss_mult_crossent * (P - P_gt_all[i])

        # Von Mises likelihood loss
        if loss_mult_vonmises > 0:
            mu_idx, Sigma_idx, dmu_P, dSigma_P = \
                fit_bivariate_von_mises(P, compute_grad=True)

            Sigma_idx_padded = Sigma_idx + VON_MISES_DIAGONAL_EPS * np.eye(2)
            mu_uv = idx_to_uv(mu_idx)
            Sigma_uv = Sigma_idx_padded * (BIN_SIZE ** 2)

            LL, dLL_mu, dLL_Sigma = ll_multivariate_normal(
                gt_uv_all[i], mu_uv, Sigma_uv)

            vm_loss = -LL
            dvm_mu = -dLL_mu
            dvm_Sigma = -dLL_Sigma

            vm_loss_min = np.log(2 * np.pi * VON_MISES_DIAGONAL_EPS * BIN_SIZE**2)
            vm_loss = max(0.0, vm_loss - vm_loss_min)
            total_loss += loss_mult_vonmises * vm_loss

            dvm_mu_idx = dvm_mu * BIN_SIZE
            dvm_Sigma_idx = dvm_Sigma * (BIN_SIZE ** 2)
            dvm1_P1, dvm2_P2 = dmu_P
            dS1_P1, dS12_P, dS2_P2 = dSigma_P

            d_vonmises_P = np.zeros((n, n), dtype=np.float64)
            d_vonmises_P += dvm_mu_idx[0] * dvm1_P1[:, np.newaxis]
            d_vonmises_P += dvm_mu_idx[1] * dvm2_P2[np.newaxis, :]
            d_vonmises_P += dvm_Sigma_idx[0, 0] * dS1_P1[:, np.newaxis]
            d_vonmises_P += dvm_Sigma_idx[1, 1] * dS2_P2[np.newaxis, :]
            d_vonmises_P += 2 * dvm_Sigma_idx[0, 1] * dS12_P

            d_loss_H += loss_mult_vonmises * softmax_backward(d_vonmises_P, P_meta)

        # Backprop: 1/n^2 from ifft2 chain rule
        d_loss_H_fft = np.fft.fft2(d_loss_H / (n * n))
        for c in range(n_channels):
            grad_F_fft[:, :, c] += np.conj(X_fft_all[i, :, :, c]) * d_loss_H_fft
        grad_B += d_loss_H

    # TV regularization in Fourier domain
    reg_scale = float(N)
    for c in range(n_channels):
        if isinstance(filter_precond, (list, tuple)):
            fp_c = filter_precond[c]
        elif isinstance(filter_precond, np.ndarray) and filter_precond.ndim >= 2:
            fp_c = filter_precond
        else:
            fp_c = float(filter_precond) * tv
        F_mag_sq = np.real(F_fft[:, :, c])**2 + np.imag(F_fft[:, :, c])**2
        total_loss += 0.5 * reg_scale * np.sum(fp_c * F_mag_sq)
        grad_F_fft[:, :, c] += reg_scale * fp_c * F_fft[:, :, c]

    B_fft = np.fft.fft2(B)
    if isinstance(bias_precond, np.ndarray) and bias_precond.ndim >= 2:
        bp = bias_precond
    else:
        bp = float(bias_precond) * tv
    B_mag_sq = np.real(B_fft)**2 + np.imag(B_fft)**2
    total_loss += 0.5 * reg_scale * np.sum(bp * B_mag_sq)
    grad_B += reg_scale * np.real(np.fft.ifft2(bp * B_fft))

    grad_vec = []
    for c in range(n_channels):
        grad_vec.append(np.real(grad_F_fft[:, :, c]).ravel())
        grad_vec.append(np.imag(grad_F_fft[:, :, c]).ravel())
    grad_vec.append(grad_B.ravel())

    return float(total_loss), np.concatenate(grad_vec)


def train_ffcc(train_data: list, val_data: list | None = None,
               n_epochs: int = 50, hyperparams: dict | None = None,
               init_mode: str = 'random', rng_seed: int = 0,
               verbose: bool = True) -> FFCCModel:
    """Train FFCC model using L-BFGS with loss annealing.

    Matches MATLAB TrainModel.m.  Two-phase annealing:
      Phase 1: Pure cross-entropy (convex, warm-up)
      Phase 2: Annealed mix → pure Von Mises likelihood (non-convex)

    Args:
        train_data: list of (X_histogram, gt_uv) tuples
        val_data: optional validation data (same format)
        n_epochs: controls the annealing schedule length
        hyperparams: optional dict of tuned hyperparameters
        init_mode: 'random' or 'zeros'
        rng_seed: random seed for initialization
        verbose: print training progress

    Returns:
        Trained FFCCModel
    """
    model = FFCCModel(n_channels=2, init_mode=init_mode, rng_seed=rng_seed)
    N = len(train_data)
    if verbose:
        print(f"Training FFCC on {N} samples...")

    X_all = np.array([d[0] for d in train_data])
    gt_uv_all = np.array([d[1] for d in train_data])
    P_gt_all = np.array([make_gt_soft_label(gt) for gt in gt_uv_all])

    X_fft_all = np.zeros((N, NUM_BINS, NUM_BINS, 2), dtype=np.complex128)
    for c in range(2):
        X_fft_all[:, :, :, c] = np.fft.fft2(X_all[:, :, :, c], axes=(1, 2))

    n = NUM_BINS
    tv_u = np.abs(np.fft.fft2(np.array([[-1], [1]]) / np.sqrt(8), s=(n, n))) ** 2
    tv_v = np.abs(np.fft.fft2(np.array([[-1, 1]]) / np.sqrt(8), s=(n, n))) ** 2
    tv = tv_u + tv_v

    filter_precond = 1e-6
    bias_precond = 1e-6
    crossent_mult_base = 1.0
    vonmises_mult_base = 1.0

    if hyperparams is not None:
        crossent_mult_base = float(hyperparams.get('CROSSENT_MULTIPLIER', crossent_mult_base))
        vonmises_mult_base = float(hyperparams.get('VONMISES_MULTIPLIER', vonmises_mult_base))

        filt_mult = hyperparams.get('FILTER_MULTIPLIERS', None)
        filt_shift = hyperparams.get('FILTER_SHIFTS', None)
        if filt_mult is not None and filt_shift is not None:
            filt_mult = np.asarray(filt_mult, dtype=np.float64).reshape(-1)
            filt_shift = np.asarray(filt_shift, dtype=np.float64).reshape(-1)
            filter_precond = [filt_mult[c] * tv + filt_shift[c] for c in range(2)]

        bias_mult = hyperparams.get('BIAS_MULTIPLIER', None)
        bias_shift = hyperparams.get('BIAS_SHIFT', None)
        if bias_mult is not None and bias_shift is not None:
            bias_precond = float(bias_mult) * tv + float(bias_shift)

    best_val_err = float('inf')
    best_params = None

    n_anneal = max(2, min(n_epochs // 10, 4))
    lbfgs_iters_init = 16
    lbfgs_iters_final = 64

    for i_anneal in range(n_anneal):
        if n_anneal == 1:
            vm_weight = 1.0
        else:
            vm_weight = i_anneal / (n_anneal - 1)

        loss_mult_crossent = (1 - vm_weight) * crossent_mult_base
        loss_mult_vonmises = vm_weight * vonmises_mult_base

        if n_anneal == 1:
            num_iters = lbfgs_iters_final
        else:
            iter_weight = i_anneal / (n_anneal - 1)
            num_iters = int(round(np.exp(
                np.log(lbfgs_iters_init) * (1 - iter_weight) +
                np.log(lbfgs_iters_final) * iter_weight)))

        if verbose:
            print(f"Anneal {i_anneal + 1}/{n_anneal}: "
                  f"CE={loss_mult_crossent:.3f}, VM={loss_mult_vonmises:.3f}, "
                  f"L-BFGS iters={num_iters}")

        params_vec = model.get_params()

        def objective(x):
            return _ffcc_loss_and_grad(
                x, X_fft_all, P_gt_all, gt_uv_all, N, 2, tv,
                loss_mult_crossent, loss_mult_vonmises,
                filter_precond, bias_precond)

        result = scipy_minimize(
            objective, params_vec,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': num_iters,
                'maxcor': num_iters,
                'ftol': 0,
                'gtol': 0,
                'disp': False,
            }
        )

        model.set_params(result.x)
        if verbose:
            print(f"  Loss: {result.fun:.4f} (converged: {result.success})")

        if val_data:
            errors = []
            for X, gt_uv in val_data:
                mu, _ = model.forward(X)
                errors.append(angular_error_uv(mu, gt_uv))
            val_err = np.mean(errors)
            if verbose:
                print(f"  Val mean AE: {val_err:.2f} deg")
            if val_err < best_val_err:
                best_val_err = val_err
                best_params = model.get_params().copy()

    if best_params is not None:
        model.set_params(best_params)
        if verbose:
            print(f"Restored best model (val_mean_ae={best_val_err:.2f} deg)")

    return model
