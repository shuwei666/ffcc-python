"""Unit tests for FFCC core functionality."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ffcc.core import (
    NUM_BINS, BIN_SIZE, STARTING_UV,
    _matlab_round, psplat2, featurize_image, ffcc_forward,
    uv_to_rgb_gains, gt_rgb_to_uv, angular_error, angular_error_uv,
    make_gt_soft_label, softmax_forward, fit_bivariate_von_mises,
    idx_to_uv, uv_to_idx, FFCCModel, compute_error_metrics,
)


class TestMatlabRound:
    def test_positive_half(self):
        assert _matlab_round(0.5) == 1.0

    def test_negative_half(self):
        assert _matlab_round(-0.5) == -1.0

    def test_array(self):
        x = np.array([1.5, 2.5, -1.5, -2.5])
        expected = np.array([2.0, 3.0, -2.0, -3.0])
        np.testing.assert_array_equal(_matlab_round(x), expected)


class TestUVConversions:
    def test_idx_to_uv_roundtrip(self):
        uv = np.array([0.0, 0.1])
        idx = uv_to_idx(uv)
        uv_back = idx_to_uv(idx)
        np.testing.assert_allclose(uv_back, uv, atol=BIN_SIZE / 2)

    def test_gt_rgb_to_uv(self):
        rgb = np.array([1.0, 1.0, 1.0])
        uv = gt_rgb_to_uv(rgb)
        np.testing.assert_allclose(uv, [0.0, 0.0])

    def test_uv_to_rgb_gains_neutral(self):
        uv = np.array([0.0, 0.0])
        rgb = uv_to_rgb_gains(uv)
        np.testing.assert_allclose(rgb, np.ones(3) / np.sqrt(3), atol=1e-10)


class TestHistogram:
    def test_psplat2_single_point(self):
        u = np.array([0.0])
        v = np.array([0.0])
        w = np.array([1.0])
        hist = psplat2(u, v, w)
        assert hist.shape == (NUM_BINS, NUM_BINS)
        assert hist.sum() == pytest.approx(1.0)
        assert hist.max() == pytest.approx(1.0)

    def test_psplat2_normalization(self):
        rng = np.random.default_rng(42)
        u = rng.uniform(-0.3, 0.3, 100)
        v = rng.uniform(-0.3, 0.3, 100)
        w = np.ones(100)
        hist = psplat2(u, v, w)
        assert hist.sum() == pytest.approx(100.0)

    def test_featurize_shape(self):
        rng = np.random.default_rng(42)
        img = rng.uniform(0.01, 1.0, (32, 32, 3))
        X = featurize_image(img)
        assert X.shape == (NUM_BINS, NUM_BINS, 2)
        assert np.all(X >= 0)

    def test_featurize_normalized(self):
        rng = np.random.default_rng(42)
        img = rng.uniform(0.01, 1.0, (32, 32, 3))
        X = featurize_image(img)
        for c in range(2):
            total = X[:, :, c].sum()
            if total > 0:
                assert total == pytest.approx(1.0, abs=1e-10)


class TestSoftmax:
    def test_sums_to_one(self):
        rng = np.random.default_rng(42)
        H = rng.standard_normal((NUM_BINS, NUM_BINS))
        P, meta = softmax_forward(H)
        assert P.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(P >= 0)

    def test_numerically_stable(self):
        H = np.ones((NUM_BINS, NUM_BINS)) * 1000
        P, _ = softmax_forward(H)
        assert np.all(np.isfinite(P))
        assert P.sum() == pytest.approx(1.0, abs=1e-10)


class TestVonMises:
    def test_delta_peak(self):
        P = np.zeros((NUM_BINS, NUM_BINS))
        P[20, 30] = 1.0
        mu_idx, Sigma_idx = fit_bivariate_von_mises(P)
        assert mu_idx[0] == pytest.approx(20.0, abs=0.5)
        assert mu_idx[1] == pytest.approx(30.0, abs=0.5)

    def test_covariance_positive(self):
        rng = np.random.default_rng(42)
        P = rng.uniform(0, 1, (NUM_BINS, NUM_BINS))
        P /= P.sum()
        _, Sigma = fit_bivariate_von_mises(P)
        assert Sigma[0, 0] > 0
        assert Sigma[1, 1] > 0


class TestForward:
    def test_forward_returns_valid_uv(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (NUM_BINS, NUM_BINS, 2))
        X[:, :, 0] /= X[:, :, 0].sum()
        X[:, :, 1] /= X[:, :, 1].sum()
        F_fft = np.zeros((NUM_BINS, NUM_BINS, 2), dtype=np.complex128)
        B = np.zeros((NUM_BINS, NUM_BINS))
        mu_uv, P = ffcc_forward(X, F_fft, B)
        assert mu_uv.shape == (2,)
        assert P.shape == (NUM_BINS, NUM_BINS)
        assert np.all(np.isfinite(mu_uv))


class TestAngularError:
    def test_identical(self):
        rgb = np.array([0.5, 0.5, 0.7])
        assert angular_error(rgb, rgb) < 0.01  # floating-point precision

    def test_symmetric(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert angular_error(a, b) == pytest.approx(angular_error(b, a))

    def test_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert angular_error(a, b) == pytest.approx(90.0, abs=0.01)


class TestSoftLabel:
    def test_sums_to_one(self):
        uv = np.array([0.1, -0.1])
        P = make_gt_soft_label(uv)
        assert P.sum() == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self):
        uv = np.array([0.0, 0.0])
        P = make_gt_soft_label(uv)
        assert np.all(P >= 0)


class TestFFCCModel:
    def test_save_load_roundtrip(self, tmp_path):
        model = FFCCModel(init_mode='random', rng_seed=42)
        path = str(tmp_path / "test_model.npz")
        model.save(path)

        model2 = FFCCModel(init_mode='zeros')
        model2.load(path)

        np.testing.assert_allclose(model.F_fft, model2.F_fft)
        np.testing.assert_allclose(model.B, model2.B)

    def test_predict_shape(self):
        rng = np.random.default_rng(42)
        model = FFCCModel(init_mode='zeros')
        img = rng.uniform(0.01, 1.0, (32, 32, 3))
        rgb = model.predict(img)
        assert rgb.shape == (3,)
        assert np.all(np.isfinite(rgb))
        assert np.linalg.norm(rgb) == pytest.approx(1.0, abs=1e-10)


class TestErrorMetrics:
    def test_metrics_keys(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_error_metrics(errors)
        for k in ["mean", "median", "trimean", "best25", "worst25", "max", "count"]:
            assert k in m

    def test_count(self):
        errors = np.ones(10)
        m = compute_error_metrics(errors)
        assert m["count"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
