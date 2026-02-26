#!/usr/bin/env python3
"""
Reproduce FFCC on GehlerShi (Reprocessed) — 3-fold Cross-Validation.

This script reproduces the FFCC results on the Gehler/Shi dataset using the
same hyperparameters, fold splits, and evaluation protocol as the original
Google FFCC MATLAB codebase.

Usage:
    python scripts/benchmark_gehler.py --data-dir data/GehlerShiThumb

Expected results (angular error in degrees):
    mean ~2.00   median ~1.14   trimean ~1.34   best25 ~0.34   worst25 ~5.12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize as scipy_minimize

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffcc.core import (
    NUM_BINS, BIN_SIZE, STARTING_UV, VON_MISES_DIAGONAL_EPS,
    softmax_forward, softmax_backward, fit_bivariate_von_mises,
    idx_to_uv, ffcc_forward, uv_to_rgb_gains, gt_rgb_to_uv,
    make_gt_soft_label, ll_multivariate_normal,
)
from ffcc.matlab_port import (
    default_params,
    featurize_image as featurize_image_matlab,
)


# ====================================================================
# Error metrics (MATLAB-compatible)
# ====================================================================
def angular_error_deg(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred = pred_rgb / (np.linalg.norm(pred_rgb) + 1e-12)
    gt = gt_rgb / (np.linalg.norm(gt_rgb) + 1e-12)
    c = float(np.clip(np.dot(pred, gt), -1.0, 1.0))
    return math.degrees(math.acos(c))


def error_metrics(errors: np.ndarray) -> dict:
    e = np.asarray(errors, dtype=np.float64).ravel()
    p25, p50, p75, p95 = np.percentile(e, [25, 50, 75, 95])
    return {
        "mean": float(np.mean(e)),
        "mean2": float(np.sqrt(np.mean(e**2))),
        "mean4": float(np.mean(e**4) ** 0.25),
        "median": float(p50),
        "trimean": float((p25 + 2 * p50 + p75) / 4.0),
        "best25": float(np.mean(e[e <= p25])),
        "worst25": float(np.mean(e[e >= p75])),
        "worst5": float(np.mean(e[e >= p95])),
        "max": float(np.max(e)),
        "count": int(e.size),
    }


# ====================================================================
# FFT latent-space utilities (MATLAB preconditioned L-BFGS)
# ====================================================================
def fft2_to_vec(F, apply_scaling=True):
    """MATLAB lib_fft/Fft2ToVec.m."""
    F = np.asarray(F, dtype=np.complex128).copy()
    n = F.shape[0]
    Fr = F.real.copy()
    Fi = F.imag.copy()
    h = n // 2
    if apply_scaling:
        Fr[np.ix_([0, h], [0, h])] /= np.sqrt(2.0)
    parts = [
        Fr[:h + 1, 0], Fr[:h + 1, h],
        Fr[:, 1:h].reshape(-1, order="F"),
        Fi[1:h, 0], Fi[1:h, h],
        Fi[:, 1:h].reshape(-1, order="F"),
    ]
    V = np.concatenate(parts).astype(np.float64)
    if apply_scaling:
        V *= np.sqrt(2.0)
    return V


def vec_to_fft2(V, apply_scaling=True):
    """MATLAB lib_fft/VecToFft2.m."""
    V = np.asarray(V, dtype=np.float64).copy().reshape(-1)
    n = int(round(np.sqrt(V.size)))
    h = n // 2
    if apply_scaling:
        V /= np.sqrt(2.0)

    Fr = np.zeros((n, n), dtype=np.float64)
    Fi = np.zeros((n, n), dtype=np.float64)

    off = 0
    l1 = h + 1
    Fr[:h + 1, 0] = V[off:off + l1]; off += l1
    Fr[:h + 1, h] = V[off:off + l1]; off += l1
    l3 = n * (h - 1)
    Fr[:, 1:h] = V[off:off + l3].reshape((n, h - 1), order="F"); off += l3
    if apply_scaling:
        Fr[np.ix_([0, h], [0, h])] *= np.sqrt(2.0)
    l4 = h - 1
    Fi[1:h, 0] = V[off:off + l4]; off += l4
    Fi[1:h, h] = V[off:off + l4]; off += l4
    l6 = n * (h - 1)
    Fi[:, 1:h] = V[off:off + l6].reshape((n, h - 1), order="F"); off += l6

    F = Fr + 1j * Fi
    F[0, h + 1:] = np.conj(F[0, 1:h][::-1])
    F[1:, h + 1:] = np.rot90(np.conj(F[1:, 1:h]), 2)
    F[h + 1:, [0, h]] = np.flipud(np.conj(F[1:h, [0, h]]))
    return F


def fft2_regularizer_to_preconditioner(F_reg):
    """MATLAB lib_fft/Fft2RegularizerToPreconditioner.m."""
    F_reg = np.asarray(F_reg, dtype=np.float64)
    n = F_reg.shape[0]
    v = fft2_to_vec(F_reg + 1j * F_reg, apply_scaling=True)
    v_pre = np.sqrt(np.sqrt(2.0) / v)
    idx = np.array([0, n // 2, n // 2 + 1, n + 1], dtype=np.int64)
    v_pre[idx] /= np.sqrt(np.sqrt(2.0))
    return v_pre


# ====================================================================
# Loss + gradient (MATLAB preconditioned latent space)
# ====================================================================
def _loss_and_grad_precond(z_pre_vec, X_fft_all, P_gt_all, gt_uv_all,
                           precond_F, precond_B,
                           ce_mult, vm_mult, vm_diag_eps):
    n = NUM_BINS
    n2 = n * n
    n_channels = 2
    N = X_fft_all.shape[0]

    off = 0
    zF_pre = z_pre_vec[off:off + n2 * n_channels].reshape(n2, n_channels)
    off += n2 * n_channels
    zB_pre = z_pre_vec[off:off + n2]

    zF_lat = precond_F * zF_pre
    zB_lat = precond_B * zB_pre
    F_fft = np.zeros((n, n, n_channels), dtype=np.complex128)
    for c in range(n_channels):
        F_fft[:, :, c] = vec_to_fft2(zF_lat[:, c])
    B_fft = vec_to_fft2(zB_lat)
    B = np.real(np.fft.ifft2(B_fft))

    total_loss = 0.0
    grad_F_fft = np.zeros_like(F_fft)
    grad_B = np.zeros((n, n), dtype=np.float64)

    for i in range(N):
        FX_fft = np.sum(X_fft_all[i] * F_fft, axis=2)
        H = np.real(np.fft.ifft2(FX_fft)) + B
        P, P_meta = softmax_forward(H)
        d_loss_H = np.zeros_like(H)

        if ce_mult > 0:
            log_P = P_meta['H_shifted'] - np.log(P_meta['expH_sum'])
            ce_loss = max(0.0, -np.sum(P_gt_all[i] * log_P))
            total_loss += ce_mult * ce_loss
            d_loss_H += ce_mult * (P - P_gt_all[i])

        if vm_mult > 0:
            mu_idx, Sigma_idx, dmu_P, dSigma_P = \
                fit_bivariate_von_mises(P, compute_grad=True)
            Sigma_padded = Sigma_idx + vm_diag_eps * np.eye(2)
            mu_uv = idx_to_uv(mu_idx)
            Sigma_uv = Sigma_padded * (BIN_SIZE ** 2)

            LL, dLL_mu, dLL_Sigma = ll_multivariate_normal(
                gt_uv_all[i], mu_uv, Sigma_uv)
            vm_loss = max(0.0, -LL - np.log(2 * np.pi * vm_diag_eps * BIN_SIZE**2))
            total_loss += vm_mult * vm_loss

            dvm_mu_idx = -dLL_mu * BIN_SIZE
            dvm_Sigma_idx = -dLL_Sigma * (BIN_SIZE ** 2)
            dvm1_P1, dvm2_P2 = dmu_P
            dS1_P1, dS12_P, dS2_P2 = dSigma_P

            d_vm_P = np.zeros((n, n), dtype=np.float64)
            d_vm_P += dvm_mu_idx[0] * dvm1_P1[:, np.newaxis]
            d_vm_P += dvm_mu_idx[1] * dvm2_P2[np.newaxis, :]
            d_vm_P += dvm_Sigma_idx[0, 0] * dS1_P1[:, np.newaxis]
            d_vm_P += dvm_Sigma_idx[1, 1] * dS2_P2[np.newaxis, :]
            d_vm_P += 2.0 * dvm_Sigma_idx[0, 1] * dS12_P
            d_loss_H += vm_mult * softmax_backward(d_vm_P, P_meta)

        d_loss_H_fft = np.fft.fft2(d_loss_H / (n * n))
        for c in range(n_channels):
            grad_F_fft[:, :, c] += np.conj(X_fft_all[i, :, :, c]) * d_loss_H_fft
        grad_B += d_loss_H

    grad_B_fft = np.fft.fft2(grad_B / (n * n))
    grad_F_lat = np.zeros((n2, n_channels), dtype=np.float64)
    for c in range(n_channels):
        grad_F_lat[:, c] = fft2_to_vec(grad_F_fft[:, :, c])
    grad_B_lat = fft2_to_vec(grad_B_fft)

    grad_F_pre = precond_F * grad_F_lat
    grad_B_pre = precond_B * grad_B_lat

    data_mass = float(N)
    total_loss += 0.5 * data_mass * (np.sum(zF_pre ** 2) + np.sum(zB_pre ** 2))
    grad_F_pre += data_mass * zF_pre
    grad_B_pre += data_mass * zB_pre

    grad_vec = np.concatenate([grad_F_pre.reshape(-1), grad_B_pre.reshape(-1)])
    return float(total_loss), grad_vec


# ====================================================================
# Training
# ====================================================================
def train_ffcc_precond(train_data, hparams, n_epochs=20):
    """Train FFCC with MATLAB-style preconditioned L-BFGS."""
    n = NUM_BINS
    n2 = n * n
    n_channels = 2
    N = len(train_data)

    X_all = np.array([d[0] for d in train_data], dtype=np.float64)
    gt_uv_all = np.array([d[1] for d in train_data], dtype=np.float64)
    P_gt_all = np.array([make_gt_soft_label(gt) for gt in gt_uv_all])
    X_fft_all = np.zeros((N, n, n, n_channels), dtype=np.complex128)
    for c in range(n_channels):
        X_fft_all[:, :, :, c] = np.fft.fft2(X_all[:, :, :, c], axes=(1, 2))

    tv_u = np.abs(np.fft.fft2(np.array([[-1.0], [1.0]]) / np.sqrt(8.0), s=(n, n))) ** 2
    tv_v = np.abs(np.fft.fft2(np.array([[-1.0, 1.0]]) / np.sqrt(8.0), s=(n, n))) ** 2
    tv = tv_u + tv_v

    f_mult = np.asarray(hparams["FILTER_MULTIPLIERS"], dtype=np.float64).reshape(-1)
    f_shift = np.asarray(hparams["FILTER_SHIFTS"], dtype=np.float64).reshape(-1)
    b_mult = float(hparams["BIAS_MULTIPLIER"])
    b_shift = float(hparams["BIAS_SHIFT"])
    ce_mult_base = float(hparams["CROSSENT_MULTIPLIER"])
    vm_mult_base = float(hparams["VONMISES_MULTIPLIER"])
    vm_diag_eps = float(hparams.get("VON_MISES_DIAGONAL_EPS", 1.0))

    reg_F = np.stack([f_mult[c] * tv + f_shift[c] for c in range(n_channels)], axis=2)
    reg_B = b_mult * tv + b_shift

    precond_F = np.stack([fft2_regularizer_to_preconditioner(reg_F[:, :, c])
                          for c in range(n_channels)], axis=1)
    precond_B = fft2_regularizer_to_preconditioner(reg_B)

    z_pre = np.zeros(n2 * n_channels + n2, dtype=np.float64)

    n_anneal = max(2, min(n_epochs // 10, 4))
    lbfgs_init = 16
    lbfgs_final = 64

    for ia in range(n_anneal):
        vm_w = ia / (n_anneal - 1) if n_anneal > 1 else 1.0
        iter_w = ia / (n_anneal - 1) if n_anneal > 1 else 1.0
        ce_m = (1.0 - vm_w) * ce_mult_base
        vm_m = vm_w * vm_mult_base
        n_iter = int(round(np.exp(
            np.log(lbfgs_init) * (1 - iter_w) + np.log(lbfgs_final) * iter_w)))

        print(f"  Anneal {ia + 1}/{n_anneal}: CE={ce_m:.6g}, VM={vm_m:.6g}, "
              f"iters={n_iter}")

        result = scipy_minimize(
            lambda x: _loss_and_grad_precond(
                x, X_fft_all, P_gt_all, gt_uv_all,
                precond_F, precond_B, ce_m, vm_m, vm_diag_eps),
            z_pre, method="L-BFGS-B", jac=True,
            options={"maxiter": n_iter, "maxcor": n_iter,
                     "maxfun": 4 + 2 * n_iter, "ftol": 0, "gtol": 0})
        z_pre = result.x.astype(np.float64, copy=True)
        print(f"    Loss: {result.fun:.4f}")

    zF_pre = z_pre[:n2 * n_channels].reshape(n2, n_channels)
    zB_pre = z_pre[n2 * n_channels:]
    zF_lat = precond_F * zF_pre
    zB_lat = precond_B * zB_pre
    F_fft = np.zeros((n, n, n_channels), dtype=np.complex128)
    for c in range(n_channels):
        F_fft[:, :, c] = vec_to_fft2(zF_lat[:, c])
    B = np.real(np.fft.ifft2(vec_to_fft2(zB_lat)))
    return {"F_fft": F_fft, "B": B}


# ====================================================================
# Hyperparameter parsing (from MATLAB .m files)
# ====================================================================
def _parse_scalar(expr):
    expr = expr.replace("...", "").strip().rstrip(",").replace("^", "**")
    return float(eval(expr, {"__builtins__": {}}, {}))


def parse_hyperparams_m(path):
    """Parse tuned hyperparams from GehlerShiThumbHyperparams.m."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"params\.HYPERPARAMS\s*=\s*struct\((.*?)\);\s*\n", text, re.S)
    if not m:
        return {}
    block = m.group(1)
    out = {}
    for key in ["VONMISES_MULTIPLIER", "CROSSENT_MULTIPLIER",
                "BIAS_MULTIPLIER", "BIAS_SHIFT", "VON_MISES_DIAGONAL_EPS"]:
        mk = re.search(rf"'{key}',\s*([^\n]+)", block)
        if mk:
            out[key] = _parse_scalar(mk.group(1))
    for key in ["FILTER_MULTIPLIERS", "FILTER_SHIFTS"]:
        mk = re.search(rf"'{key}',\s*\[([^\]]+)\]", block, re.S)
        if mk:
            vals = [v.strip() for v in mk.group(1).split(",") if v.strip()]
            out[key] = np.array([_parse_scalar(v) for v in vals], dtype=np.float64)
    return out


def parse_repo_metrics(path):
    """Parse reference metrics from GehlerShiThumbHyperparams.m."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    start = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "'rgb_err'" in line:
            start = i
            break
    if start is None:
        return {}
    block_lines = []
    for line in lines[start:start + 40]:
        block_lines.append(line)
        if "'uv_err'" in line:
            break
    block = "\n".join(block_lines)
    keys = ["mean", "mean2", "mean4", "median", "tri", "b25", "w25", "w05", "max"]
    out = {}
    for k in keys:
        mk = re.search(rf"'{k}',\s*([-\d\.eE\+]+)", block)
        if mk:
            out[k] = float(mk.group(1))
    return out


# ====================================================================
# Dataset loading
# ====================================================================
def load_gehler_thumb(folder):
    """Load GehlerShiThumb dataset: PNG images + TXT illuminants + fold splits."""
    folder = Path(folder)
    pngs = sorted(folder.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    folds = np.loadtxt(folder / "cvfolds.txt", dtype=np.int64).reshape(-1)
    if len(pngs) != len(folds):
        raise RuntimeError(f"Image count {len(pngs)} != fold count {len(folds)}")

    gt_list = []
    for p in pngs:
        gt = np.loadtxt(p.with_suffix(".txt"), dtype=np.float64).reshape(-1)
        gt_list.append(gt)
    gt = np.stack(gt_list, axis=0)
    return pngs, gt, folds


def build_or_load_features(folder, cache_path, force=False):
    """Compute or load cached MATLAB-port histogram features."""
    pngs, gt_rgb, folds = load_gehler_thumb(folder)
    params = default_params()
    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        d = np.load(cache_path, allow_pickle=False)
        X = d["X"]; gt_uv = d["gt_uv"]
        folds_c = d["folds"]; names = [str(x) for x in d["names"]]
        if X.shape[0] == len(pngs) and np.array_equal(folds_c, folds):
            print(f"Loaded cached features ({cache_path})")
            return X, gt_uv, gt_rgb, folds, names

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    Xs, gt_uvs, names = [], [], []
    t0 = time.time()
    for i, p in enumerate(pngs, 1):
        if i == 1 or i % 50 == 0 or i == len(pngs):
            print(f"  Featurizing {i}/{len(pngs)}: {p.name}")
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X, _ = featurize_image_matlab(img, None, params)
        Xs.append(X.astype(np.float64))
        gt_uvs.append(gt_rgb_to_uv(gt_rgb[i - 1]))
        names.append(p.name)

    X = np.stack(Xs); gt_uv = np.stack(gt_uvs)
    np.savez_compressed(cache_path, X=X, gt_uv=gt_uv, gt_rgb=gt_rgb,
                        folds=folds, names=np.array(names, dtype="<U32"))
    print(f"  Saved cache ({time.time() - t0:.1f}s)")
    return X, gt_uv, gt_rgb, folds, names


# ====================================================================
# Cross-validation
# ====================================================================
def cross_validate(X, gt_uv, gt_rgb, folds, names, hparams, epochs=20):
    fold_ids = sorted(int(x) for x in np.unique(folds))
    all_errors = np.full(len(names), np.nan)
    fold_summaries = []

    for fid in fold_ids:
        test_idx = np.where(folds == fid)[0]
        train_idx = np.where(folds != fid)[0]
        train_data = [(X[i], gt_uv[i]) for i in train_idx]

        print(f"\n=== Fold {fid} (train={len(train_idx)}, test={len(test_idx)}) ===")
        t0 = time.time()
        model = train_ffcc_precond(train_data, hparams, n_epochs=epochs)
        dt = time.time() - t0

        fold_errs = []
        for j in test_idx:
            mu, _ = ffcc_forward(X[j], model["F_fft"], model["B"])
            pred_rgb = uv_to_rgb_gains(mu)
            err = angular_error_deg(pred_rgb, gt_rgb[j])
            all_errors[j] = err
            fold_errs.append(err)

        m = error_metrics(np.asarray(fold_errs))
        m["fold"] = fid; m["train_time_sec"] = dt
        fold_summaries.append(m)
        print(f"  mean={m['mean']:.4f}  median={m['median']:.4f}  "
              f"trimean={m['trimean']:.4f}  worst25={m['worst25']:.4f}  ({dt:.1f}s)")

    valid = np.isfinite(all_errors)
    overall = error_metrics(all_errors[valid])
    overall["evaluated"] = int(valid.sum())
    overall["total"] = len(valid)
    return overall, fold_summaries, all_errors


# ====================================================================
# Main
# ====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Reproduce FFCC on GehlerShiThumb (3-fold CV)")
    ap.add_argument("--data-dir", type=str, required=True,
                    help="Path to GehlerShiThumb folder")
    ap.add_argument("--hyperparams-m", type=str, default=None,
                    help="Path to GehlerShiThumbHyperparams.m (optional)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cache-dir", type=str, default="cache")
    ap.add_argument("--out-json", type=str, default="results/gehler_results.json")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    # Try to load tuned hyperparams from .m file or use defaults
    if args.hyperparams_m and Path(args.hyperparams_m).exists():
        hparams = parse_hyperparams_m(args.hyperparams_m)
        ref_metrics = parse_repo_metrics(args.hyperparams_m)
    else:
        # Default hyperparams (from MATLAB GehlerShiThumb tuning)
        hparams = {
            "CROSSENT_MULTIPLIER": 2**-4.25,
            "VONMISES_MULTIPLIER": 2**5.5,
            "FILTER_MULTIPLIERS": np.array([2**-8.25, 2**-7.25]),
            "FILTER_SHIFTS": np.array([2**-12, 2**-12]),
            "BIAS_MULTIPLIER": 2**-6.5,
            "BIAS_SHIFT": 2**-12,
            "VON_MISES_DIAGONAL_EPS": 1.0,
        }
        ref_metrics = {
            "mean": 1.97937, "median": 1.04998, "tri": 1.31196,
            "b25": 0.299816, "w25": 5.10577,
        }

    print("Reference metrics (MATLAB repo):")
    for k in ["mean", "median", "tri", "b25", "w25"]:
        if k in ref_metrics:
            print(f"  {k}: {ref_metrics[k]:.5f}")

    cache_path = Path(args.cache_dir) / "gehler_features.npz"
    X, gt_uv, gt_rgb, folds, names = build_or_load_features(
        args.data_dir, cache_path, args.force_recompute)

    overall, fold_summaries, all_errors = cross_validate(
        X, gt_uv, gt_rgb, folds, names, hparams, epochs=args.epochs)

    print("\n" + "=" * 60)
    print("Reproduced CV Metrics (angular error, degrees)")
    print("=" * 60)
    for k in ["mean", "median", "trimean", "best25", "worst25", "worst5", "max"]:
        print(f"  {k:>10s}: {overall[k]:.4f}")
    print(f"  {'count':>10s}: {overall['count']}")

    if ref_metrics:
        print("\nDelta vs MATLAB repo:")
        mapping = {"mean": "mean", "median": "median", "tri": "trimean",
                   "b25": "best25", "w25": "worst25"}
        for mk, ok in mapping.items():
            if mk in ref_metrics:
                print(f"  {mk}: {overall[ok] - ref_metrics[mk]:+.5f}")

    out = {
        "reference_metrics": ref_metrics,
        "reproduced_metrics": overall,
        "fold_summaries": fold_summaries,
        "errors_deg": [float(e) for e in all_errors],
        "names": names,
        "epochs": args.epochs,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
