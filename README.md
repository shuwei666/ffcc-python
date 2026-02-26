# FFCC-Python: Fast Fourier Color Constancy

A faithful Python reimplementation of the **FFCC** algorithm (Barron & Tsai, CVPR 2017) for automatic white balance and illuminant estimation.

This implementation reproduces the results from the original [Google FFCC MATLAB codebase](https://github.com/google/ffcc), validated on the Gehler/Shi (Reprocessed) benchmark dataset.

## Highlights

- Pure Python/NumPy — no MATLAB required
- Reproduces MATLAB reference results on GehlerShi within < 0.1 deg mean angular error
- Clean, well-documented API for research and integration
- L-BFGS training with cross-entropy + Von Mises loss annealing
- MATLAB-compatible featurization for exact reproduction

## Quick Start

### Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy scipy
```

### Predict illuminant of an image

```python
import numpy as np
from ffcc import FFCCModel, featurize_image, uv_to_rgb_gains

# Load a pre-trained model
model = FFCCModel()
model.load("models/gehler_model.npz")

# Predict illuminant (image should be linear RGB float64 in [0, 1])
image = ...  # (H, W, 3) float64
rgb_illuminant = model.predict(image)

# Apply white balance
white_balanced = image / rgb_illuminant[np.newaxis, np.newaxis, :]
```

### Train on your own dataset

```python
from ffcc import FFCCModel, featurize_image, gt_rgb_to_uv, train_ffcc

# Prepare training data: list of (histogram_features, gt_uv) tuples
train_data = []
for img, gt_rgb in your_dataset:
    X = featurize_image(img)  # (64, 64, 2) histogram
    gt_uv = gt_rgb_to_uv(gt_rgb)
    train_data.append((X, gt_uv))

# Train
model = train_ffcc(train_data, n_epochs=50)

# Save
model.save("my_model.npz")
```

## Algorithm Overview

FFCC estimates a scene's global illuminant by:

1. **Log-chroma histograms**: Convert image to UV space (`u = log(G/R)`, `v = log(G/B)`) and compute 2D histograms (64x64 bins) for two channels:
   - Channel 0: Original pixel chromaticities
   - Channel 1: Edge chromaticities (Masked Local Absolute Deviation)

2. **FFT convolution**: Apply learned frequency-domain filters to the histograms, producing a log-probability map over possible illuminants.

3. **Von Mises fitting**: Fit a bivariate Von Mises distribution (circular Gaussian on the torus) to the posterior, extracting the mean as the illuminant estimate.

4. **Training**: L-BFGS optimization with annealed loss:
   - Phase 1: Cross-entropy (convex warm-up)
   - Phase 2: Von Mises negative log-likelihood (non-convex, more accurate)

```
Image → UV Histograms → FFT Conv → Softmax → Von Mises → UV → RGB gains
         (64×64×2)      (learned)   (P map)    (fit)     (μ)
```

## Benchmark Results

### Gehler/Shi (Reprocessed) — 3-fold Cross-Validation

| Metric   | MATLAB (reference) | This repo | Delta  |
|----------|-------------------|-----------|--------|
| Mean     | 1.979             | 2.005     | +0.026 |
| Median   | 1.050             | 1.139     | +0.089 |
| Trimean  | 1.312             | 1.344     | +0.032 |
| Best 25% | 0.300             | 0.338     | +0.038 |
| Worst 25%| 5.106             | 5.124     | +0.018 |

568 images, 3 folds, 20 epochs. Results are angular error in degrees (lower is better).

### Reproduce the benchmark

The Gehler/Shi dataset (568 images, ~3 MB) is included in `data/GehlerShiThumb/`. Just run:

```bash
python scripts/benchmark_gehler.py --data-dir data/GehlerShiThumb

# Results saved to results/gehler_results.json
```

## Project Structure

```
ffcc-python/
├── ffcc/                       # Core package
│   ├── __init__.py             # Public API
│   ├── core.py                 # FFCC algorithm (featurize, forward, train)
│   └── matlab_port.py          # MATLAB-compatible featurization
├── scripts/
│   ├── benchmark_gehler.py     # GehlerShi 3-fold CV reproduction
│   ├── demo.py                 # Single-image prediction demo
│   └── download_gehler.py      # Dataset download helper
├── tests/
│   └── test_core.py            # Unit tests
├── data/GehlerShiThumb/        # Gehler/Shi dataset (568 images, included)
├── pyproject.toml              # Package metadata
├── requirements.txt            # Runtime dependencies
└── README.md
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `featurize_image(image, mask)` | Extract 2-channel UV histogram (64×64×2) |
| `ffcc_forward(X, F_fft, B)` | Forward pass: histogram → illuminant UV |
| `uv_to_rgb_gains(mu_uv)` | Convert UV illuminant to unit-norm RGB |
| `gt_rgb_to_uv(gt_rgb)` | Convert GT RGB illuminant to UV space |
| `angular_error(pred, gt)` | Angular error between two RGB vectors (degrees) |
| `train_ffcc(data, ...)` | Train FFCC model with L-BFGS |
| `compute_error_metrics(errors)` | Compute mean/median/trimean/best25/worst25 |

### FFCCModel Class

```python
model = FFCCModel(n_channels=2, init_mode='random')
model.predict(image)          # End-to-end: image → RGB illuminant
model.forward(X)              # Histogram → (mu_uv, P)
model.save("model.npz")      # Save weights
model.load("model.npz")      # Load weights
```

## Dependencies

- **Required**: NumPy ≥ 1.21, SciPy ≥ 1.7
- **Benchmark**: OpenCV-Python ≥ 4.5 (for image loading)
- **Testing**: pytest ≥ 7.0

## Author

**Shuwei Yue / 岳书威** — shuwei_yue@szpu.edu.cn

WeChat Official Account: **ColorWorld花花世界**

## Citation

If you use this code, please cite the original FFCC paper:

```bibtex
@inproceedings{barron2017fast,
  title={Fast Fourier Color Constancy},
  author={Barron, Jonathan T and Tsai, Yun-Ta},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```

If you find this Python reimplementation useful, a star or acknowledgment is appreciated:

```bibtex
@misc{yue2025ffccpython,
  title={FFCC-Python: A faithful Python reimplementation of Fast Fourier Color Constancy},
  author={Yue, Shuwei},
  year={2025},
  url={https://github.com/shuwei666/ffcc-python}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Google FFCC MATLAB codebase](https://github.com/google/ffcc) — the reference implementation
- Gehler & Rosenhahn (2008), Shi & Funt (2010) — benchmark dataset
