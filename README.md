# Siamese Bearing Identification

Few-shot bearing fault identification and classification using Siamese Neural Networks on the **Paderborn University Bearing Dataset**.

---

## Overview

The objective of this project is to classify known mechanical faults (Healthy, Inner Race damage, Outer Race damage) on unseen bearings using only a few labeled reference recordings.

### Operating Condition
* **Condition Code**: `N15_M07_F10`
* **Rotational Speed**: 1,500 rpm
* **Load Torque**: 0.7 Nm
* **Radial Force**: 1,000 N
* **Sampling Rate**: 64 kHz vibration (`vibration_1` channel)
* **Duration**: 20 four-second recordings per bearing (~256,000 samples/file)

---

## Dataset Structure

The recordings are organized into `data/raw/` by fault type and bearing code:

```
data/raw/
├── train/
│   ├── K001/    # Healthy    (Files 1–5: references, 6–20: fitting)
│   ├── KI01/    # Inner race (Files 1–5: references, 6–20: fitting)
│   └── KA01/    # Outer race (Files 1–5: references, 6–20: fitting)
├── val/
│   ├── K002/    # Healthy    (Held-out bearing, all 20 files)
│   ├── KI03/    # Inner race (Held-out bearing, all 20 files)
│   └── KA03/    # Outer race (Held-out bearing, all 20 files)
└── test/
    ├── K003/    # Healthy    (Held-out bearing, all 20 files)
    ├── K004/    # Healthy    (Held-out bearing, all 20 files)
    ├── KI05/    # Inner race (Held-out bearing, all 20 files)
    ├── KI07/    # Inner race (Held-out bearing, all 20 files)
    ├── KA05/    # Outer race (Held-out bearing, all 20 files)
    └── KA07/    # Outer race (Held-out bearing, all 20 files)
```

---

## Setup & Dependencies

1. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install requirements** (if not already installed):
   ```bash
   pip install -r src/requirements.txt
   ```

Key dependencies:
* `torch` (with CUDA support recommended)
* `scipy`
* `numpy`
* `scikit-learn`
* `matplotlib`

---

## Preprocessing Pipeline

The preprocessing module converts raw vibration time series into standardized time-frequency representations:

1. **Segmentation**:
   * Window size: **8,192 samples** (~0.128 seconds).
   * Step size: **4,096 samples** (50% overlap) for fitting files (train files 6–20).
   * Step size: **8,192 samples** (no overlap) for references (train files 1–5) and validation/test files.
   * Trailing incomplete segments are discarded.
   * Segment mean is subtracted.

2. **STFT Spectrogram**:
   * Hann window of length **512**, overlap **448** (step 64, 87.5% overlap, no zero padding).
   * Frequency range: 0 to 32 kHz (Nyquist range, 257 bins).
   * Log-power magnitude: $\log_{10}(\text{magnitude}^2 + 10^{-12})$.
   * Resized to $(128 \times 128)$ via bilinear interpolation.

3. **Standardization**:
   * Scalar mean $\mu$ and standard deviation $\sigma$ are calculated **strictly on fitting segments**.
   * Frozen statistics are applied across all splits: $(X - \mu) / \sigma$.

### How to Run the Pipeline

Run the data pipeline directly from the repository root:

```bash
python src/data_pipeline.py
```

This will parse all 240 recordings, extract segment maps, apply frozen normalization, and output a summary of extracted segments and verified shapes.

---

## Data Visualization

A comprehensive visualization suite is provided in `src/visualize.py` to inspect the raw vibrations, frequency spectra, spectrogram representations, and preprocessing pipeline.

### Generated Figures (Saved to `figures/`):

1. **`01_raw_waveforms.png`**: Time-domain vibration signals comparing Healthy (`K001`), Inner Race (`KI01`), and Outer Race (`KA01`) damage. Shows full 4-second recordings alongside zoomed 0–100 ms impact dynamics, annotated with RMS, Peak, Crest Factor, and Kurtosis.
2. **`02_frequency_spectra.png`**: Power Spectral Density (PSD) using Welch's method. Includes full 0–32 kHz spectrum (high-frequency structural resonance) and low-frequency zoom 0–1,000 Hz (shaft rotational frequency $f_r = 25\text{ Hz}$ and fault harmonics).
3. **`03_stft_spectrograms.png`**: 2D STFT log-power spectrograms ($128 \times 128$) across the 3 fault conditions showing distinct time-frequency energy patterns.
4. **`04_preprocessing_pipeline_steps.png`**: Step-by-step visual walkthrough of feature extraction (Zero-mean segment $\to$ Raw STFT $\to$ Bilinear resize $\to$ Frozen standardization).
5. **`05_cross_bearing_comparison.png`**: $3 \times 3$ grid demonstrating domain shift across Train, Validation, and Test bearing units.
6. **`06_dataset_statistics.png`**: Dataset breakdown (segment counts per split, pixel intensity distribution before/after standardization, and RMS distribution across all 12 bearings).

### How to Run Visualizations

Generate all publication-quality figures at once:

```bash
python src/visualize.py --all
```

Or generate individual figures as needed:

```bash
# Time-domain waveforms
python src/visualize.py --waveforms

# Frequency-domain spectra (PSD)
python src/visualize.py --spectra

# STFT spectrograms
python src/visualize.py --spectrograms

# Preprocessing pipeline step-by-step
python src/visualize.py --pipeline

# Cross-bearing domain shift comparison
python src/visualize.py --cross-bearing

# Dataset partition counts & feature statistics
python src/visualize.py --stats
```

Export formats such as `--format pdf` or `--format svg` and custom resolution via `--dpi 300` are also supported.

