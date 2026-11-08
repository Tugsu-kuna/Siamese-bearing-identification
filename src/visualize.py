"""
src/visualize.py
Data Visualization Suite for Project 2: Siamese Bearing Identification.

Provides visualization utilities for:
  1. Time-Domain Waveforms (Full 4.0s recording & zoomed 0.1s impact pulse)
  2. Frequency-Domain Spectra (Power Spectral Density: 0-32 kHz & 0-1 kHz zoom)
  3. STFT Spectrograms (Time-Frequency maps for Healthy, Inner race, Outer race)
  4. Preprocessing Pipeline Transformation (Step-by-step feature extraction)
  5. Cross-Bearing Domain Shift (Train vs Val vs Test bearing spectrograms)
  6. Dataset Statistics & Normalization (Segment counts, pixel distributions, RMS)

Usage:
  # Generate all figures into figures/
  python src/visualize.py --all

  # Generate specific figures:
  python src/visualize.py --waveforms
  python src/visualize.py --spectra
  python src/visualize.py --spectrograms
  python src/visualize.py --pipeline
  python src/visualize.py --cross-bearing
  python src/visualize.py --stats
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.io as sio
from scipy.signal import welch, stft
from scipy.stats import kurtosis
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Adjust import path so it works whether run from root or src
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.data_pipeline import (
    read_paderborn_mat,
    segment_signal,
    compute_stft_map,
    SAMPLING_RATE,
    SEGMENT_LEN,
    N_FFT,
    N_OVERLAP,
    BEARING_SPLITS,
    extract_file_index
)

# Visual styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLOR_HEALTHY = "#1f77b4"     # Blue
COLOR_INNER = "#d95f02"       # Orange / Amber
COLOR_OUTER = "#2ca02c"       # Green
COLOR_ACCENT = "#7570b3"      # Purple

CLASS_COLORS = {
    "Healthy": COLOR_HEALTHY,
    "Inner_race": COLOR_INNER,
    "Outer_race": COLOR_OUTER
}


def compute_vibration_stats(signal: np.ndarray) -> Dict[str, float]:
    """Compute standard statistical indicators for bearing vibration signals."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest_factor = float(peak / (rms + 1e-12))
    kurt = float(kurtosis(signal, fisher=False))  # Pearson kurtosis (Normal = 3)
    variance = float(np.var(signal))
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "variance": variance
    }


def find_bearing_file(raw_dir: Path, split: str, bearing: str, file_idx: int = 1) -> Path:
    """Find the specific MAT file path for a bearing recording."""
    bearing_dir = raw_dir / split / bearing
    candidates = list(bearing_dir.glob(f"*{bearing}_{file_idx}.mat"))
    if not candidates:
        all_mat = sorted(list(bearing_dir.glob("*.mat")), key=lambda p: extract_file_index(p.name))
        if all_mat:
            return all_mat[0]
        raise FileNotFoundError(f"No MAT files found for bearing {bearing} in {bearing_dir}")
    return candidates[0]


def plot_time_domain_waveforms(raw_dir: Path, out_path: Path, dpi: int = 300) -> None:
    """
    Plot 3 rows of raw vibration signals (Healthy, Inner race, Outer race):
      - Left column: Full 4-second recording showing global vibration envelope
      - Right column: Zoomed-in 0.1-second slice showing impact transients
    """
    print(f"Generating time-domain waveform visualization -> {out_path}...")
    samples = [
        ("Healthy", "K001", "train", COLOR_HEALTHY),
        ("Inner Race Damage", "KI01", "train", COLOR_INNER),
        ("Outer Race Damage", "KA01", "train", COLOR_OUTER)
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharey="row")
    fig.subplots_adjust(hspace=0.35, wspace=0.18)

    zoom_samples = int(0.1 * SAMPLING_RATE)  # 0.1 second = 6400 samples

    for row_idx, (label, bearing, split, color) in enumerate(samples):
        mat_path = find_bearing_file(raw_dir, split, bearing, file_idx=1)
        sig = read_paderborn_mat(mat_path)
        stats = compute_vibration_stats(sig)

        # Full 4s signal
        t_full = np.arange(len(sig)) / SAMPLING_RATE
        ax_full = axes[row_idx, 0]
        ax_full.plot(t_full, sig, color=color, linewidth=0.4, alpha=0.9)
        ax_full.set_ylabel(f"{bearing}\n({label})\nAcc [V / m/s²]", fontsize=10, fontweight="bold")
        ax_full.set_xlim(0, t_full[-1])
        ax_full.grid(True, linestyle="--", alpha=0.5)

        # Annotations text box
        stat_text = (
            f"RMS: {stats['rms']:.3f}\n"
            f"Peak: {stats['peak']:.3f}\n"
            f"Crest Factor: {stats['crest_factor']:.2f}\n"
            f"Kurtosis: {stats['kurtosis']:.2f}"
        )
        ax_full.text(
            0.98, 0.92, stat_text,
            transform=ax_full.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.85)
        )

        # Zoomed 0.1s slice
        sig_zoom = sig[:zoom_samples]
        t_zoom = (np.arange(len(sig_zoom)) / SAMPLING_RATE) * 1000  # in ms
        ax_zoom = axes[row_idx, 1]
        ax_zoom.plot(t_zoom, sig_zoom, color=color, linewidth=0.8)
        ax_zoom.set_xlim(0, t_zoom[-1])
        ax_zoom.grid(True, linestyle="--", alpha=0.5)

        if row_idx == 0:
            ax_full.set_title("Full 4-Second Recording (Envelope View)", fontsize=11, fontweight="bold", pad=8)
            ax_zoom.set_title("Zoomed View (0–100 ms Impact Dynamics)", fontsize=11, fontweight="bold", pad=8)

        if row_idx == 2:
            ax_full.set_xlabel("Time [s]", fontsize=10, fontweight="bold")
            ax_zoom.set_xlabel("Time [ms]", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Paderborn Bearing Vibration in Time Domain (Operating Condition: N15_M07_F10)",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_frequency_spectra(raw_dir: Path, out_path: Path, dpi: int = 300) -> None:
    """
    Plot Frequency Spectrum (Power Spectral Density) for Healthy, Inner Race, and Outer Race:
      - Top: Full Nyquist range (0 to 32 kHz) showing high-frequency structural resonances
      - Bottom: Low-frequency zoom (0 to 1,000 Hz) showing rotational speed (25 Hz) & harmonics
    """
    print(f"Generating frequency spectra visualization -> {out_path}...")
    samples = [
        ("Healthy (K001)", "K001", "train", COLOR_HEALTHY),
        ("Inner Race (KI01)", "KI01", "train", COLOR_INNER),
        ("Outer Race (KA01)", "KA01", "train", COLOR_OUTER)
    ]

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.35)

    nperseg = 8192
    f_shaft = 1500.0 / 60.0  # 25 Hz rotational frequency

    for label, bearing, split, color in samples:
        mat_path = find_bearing_file(raw_dir, split, bearing, file_idx=1)
        sig = read_paderborn_mat(mat_path)

        f, psd = welch(sig, fs=SAMPLING_RATE, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
        psd_db = 10 * np.log10(psd + 1e-12)

        # Full spectrum (0 to 32 kHz) in kHz
        ax_full.plot(f / 1000.0, psd_db, label=label, color=color, linewidth=1.1, alpha=0.85)

        # Zoomed low frequency (0 to 1 kHz) in Hz
        mask_zoom = f <= 1000
        ax_zoom.plot(f[mask_zoom], psd_db[mask_zoom], label=label, color=color, linewidth=1.2, alpha=0.9)

    # Styling Full Spectrum
    ax_full.set_title("Full Spectrum PSD (0 to 32 kHz Nyquist Range) - High-Frequency Resonances", fontsize=11, fontweight="bold")
    ax_full.set_xlabel("Frequency [kHz]", fontsize=10, fontweight="bold")
    ax_full.set_ylabel("Power Spectral Density [dB / Hz]", fontsize=10, fontweight="bold")
    ax_full.set_xlim(0, SAMPLING_RATE / 2000.0)
    ax_full.legend(loc="upper right", framealpha=0.9)
    ax_full.grid(True, linestyle="--", alpha=0.5)

    # Styling Low Frequency Zoom
    ax_zoom.set_title("Low-Frequency Spectrum Zoom (0 to 1,000 Hz) - Rotational & Fault Harmonics", fontsize=11, fontweight="bold")
    ax_zoom.set_xlabel("Frequency [Hz]", fontsize=10, fontweight="bold")
    ax_zoom.set_ylabel("Power Spectral Density [dB / Hz]", fontsize=10, fontweight="bold")
    ax_zoom.set_xlim(0, 1000)
    ax_zoom.legend(loc="upper right", framealpha=0.9)
    ax_zoom.grid(True, linestyle="--", alpha=0.5)

    # Annotate rotational frequency harmonics
    for harmonic in range(1, 5):
        h_freq = harmonic * f_shaft
        ax_zoom.axvline(h_freq, color="gray", linestyle=":", alpha=0.7)
        ax_zoom.text(
            h_freq, ax_zoom.get_ylim()[0] + 5, f"{harmonic}×fr",
            rotation=90, fontsize=8, color="#555555", verticalalignment="bottom"
        )

    fig.suptitle(
        "Frequency Domain Analysis: Power Spectral Density (Welch's Method)",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_stft_spectrograms(raw_dir: Path, out_path: Path, dpi: int = 300) -> None:
    """
    Plot 2D STFT Spectrograms for Healthy, Inner Race, and Outer Race side-by-side.
    Time: 0 to 128 ms, Frequency: 0 to 32 kHz, Colormap: magma.
    """
    print(f"Generating STFT spectrograms visualization -> {out_path}...")
    samples = [
        ("Healthy", "K001", "train"),
        ("Inner Race Damage", "KI01", "train"),
        ("Outer Race Damage", "KA01", "train")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.subplots_adjust(wspace=0.15)

    maps = []
    for label, bearing, split in samples:
        mat_path = find_bearing_file(raw_dir, split, bearing, file_idx=1)
        sig = read_paderborn_mat(mat_path)
        seg = segment_signal(sig, step=8192)[0]
        stft_map = compute_stft_map(seg)
        maps.append((label, bearing, stft_map))

    # Shared color limits for fair visual comparison
    vmin = min(m[2].min() for m in maps)
    vmax = max(m[2].max() for m in maps)

    duration_ms = (SEGMENT_LEN / SAMPLING_RATE) * 1000.0  # 128 ms
    freq_khz = (SAMPLING_RATE / 2.0) / 1000.0             # 32 kHz
    extent = [0, duration_ms, 0, freq_khz]

    for idx, (label, bearing, stft_map) in enumerate(maps):
        ax = axes[idx]
        im = ax.imshow(
            stft_map,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"{bearing}: {label}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Time [ms]", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Frequency [kHz]", fontsize=10, fontweight="bold")
        ax.grid(False)

    # Colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Log-Power Magnitude $\log_{10}(|Z_{xx}|^2 + 10^{-12})$", fontsize=9, fontweight="bold")

    fig.suptitle(
        "STFT Time-Frequency Log-Power Spectrograms (128 × 128 Resized Features)",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_pipeline_transformation(raw_dir: Path, out_path: Path, bearing: str = "KI01", dpi: int = 300) -> None:
    """
    Step-by-step visual demonstration of the preprocessing and feature extraction pipeline:
      Step 1: Raw 8,192-sample segment in time domain (zero-mean)
      Step 2: Computed STFT log-power spectrogram (257 freq bins × 121 time frames)
      Step 3: Resized STFT map (128 × 128) via bilinear interpolation
      Step 4: Standardized feature map (X - mean) / std with frozen statistics
    """
    print(f"Generating pipeline transformation visualization for {bearing} -> {out_path}...")
    from scipy.signal import stft as raw_stft
    from scipy.ndimage import zoom

    mat_path = find_bearing_file(raw_dir, "train", bearing, file_idx=1)
    sig = read_paderborn_mat(mat_path)
    seg = segment_signal(sig, step=8192)[0]

    # Step 1: Zero-mean segment
    seg_time_ms = (np.arange(len(seg)) / SAMPLING_RATE) * 1000

    # Step 2: Raw STFT
    f, t, Zxx = raw_stft(
        seg,
        fs=SAMPLING_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_OVERLAP,
        nfft=N_FFT,
        padded=False,
        boundary=None
    )
    raw_log_map = np.log10(np.abs(Zxx) ** 2 + 1e-12)

    # Step 3: Resized to (128, 128)
    zoom_factors = (128 / raw_log_map.shape[0], 128 / raw_log_map.shape[1])
    resized_map = zoom(raw_log_map, zoom_factors, order=1).astype(np.float32)

    # Step 4: Standardized (using estimated training statistics ~ -3.95, 0.82)
    sample_mean = -3.94688
    sample_std = 0.82221
    standardized_map = (resized_map - sample_mean) / sample_std

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    fig.subplots_adjust(wspace=0.28)

    # Panel 1: Time domain segment
    axes[0].plot(seg_time_ms, seg, color=COLOR_INNER, linewidth=0.7)
    axes[0].set_title(f"Step 1: Raw Segment\n(8,192 samples, ~128 ms, zero-mean)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Time [ms]", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Vibration [V / m/s²]", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, seg_time_ms[-1])
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Raw STFT Spectrogram
    im2 = axes[1].imshow(
        raw_log_map,
        origin="lower",
        aspect="auto",
        cmap="magma"
    )
    axes[1].set_title(f"Step 2: STFT Spectrogram\n({raw_log_map.shape[0]} freq bins × {raw_log_map.shape[1]} time bins)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Time Frame Index", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Frequency Bin Index", fontsize=9, fontweight="bold")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Resized Spectrogram
    im3 = axes[2].imshow(
        resized_map,
        origin="lower",
        aspect="auto",
        cmap="magma"
    )
    axes[2].set_title("Step 3: Bilinear Resizing\n(Uniform 128 × 128 grid)", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Time Pixel Index", fontsize=9, fontweight="bold")
    axes[2].set_ylabel("Frequency Pixel Index", fontsize=9, fontweight="bold")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Standardized Feature Map
    im4 = axes[3].imshow(
        standardized_map,
        origin="lower",
        aspect="auto",
        cmap="coolwarm"
    )
    axes[3].set_title(r"Step 4: Standardized Feature Map" + "\n" + r"$(X - \mu) / \sigma$ (Frozen Stats)", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Time Pixel Index", fontsize=9, fontweight="bold")
    axes[3].set_ylabel("Frequency Pixel Index", fontsize=9, fontweight="bold")
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Feature Preprocessing Pipeline: Raw Signal to Siamese Input Map ({bearing} Inner Race)",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_cross_bearing_comparison(raw_dir: Path, out_path: Path, dpi: int = 300) -> None:
    """
    Plot 3x3 matrix showing STFT Spectrograms across different physical bearings:
      - Rows: Healthy, Inner Race, Outer Race
      - Columns: Train split bearing, Validation split bearing, Test split bearing
    This visually demonstrates the physical domain shift across distinct bearing units,
    justifying why few-shot Siamese metric learning is necessary.
    """
    print(f"Generating cross-bearing domain shift visualization -> {out_path}...")
    matrix_layout = [
        # (Class label, [ (Split, Bearing) ])
        ("Healthy", [("train", "K001"), ("val", "K002"), ("test", "K003")]),
        ("Inner Race", [("train", "KI01"), ("val", "KI03"), ("test", "KI05")]),
        ("Outer Race", [("train", "KA01"), ("val", "KA03"), ("test", "KA05")])
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    duration_ms = (SEGMENT_LEN / SAMPLING_RATE) * 1000.0  # 128 ms
    freq_khz = (SAMPLING_RATE / 2.0) / 1000.0             # 32 kHz
    extent = [0, duration_ms, 0, freq_khz]

    for row_idx, (class_name, col_bearings) in enumerate(matrix_layout):
        for col_idx, (split, bearing) in enumerate(col_bearings):
            ax = axes[row_idx, col_idx]
            mat_path = find_bearing_file(raw_dir, split, bearing, file_idx=1)
            sig = read_paderborn_mat(mat_path)
            seg = segment_signal(sig, step=8192)[0]
            stft_map = compute_stft_map(seg)

            im = ax.imshow(
                stft_map,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=-7.0,
                vmax=-1.0
            )

            # Titles and labels
            col_split_title = f"{split.capitalize()} Bearing: {bearing}"
            if row_idx == 0:
                ax.set_title(col_split_title, fontsize=11, fontweight="bold", pad=8)

            if col_idx == 0:
                ax.set_ylabel(f"{class_name}\nFrequency [kHz]", fontsize=10, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel("Time [ms]", fontsize=10, fontweight="bold")

            ax.grid(False)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Log-Power Magnitude $\log_{10}(|Z_{xx}|^2 + 10^{-12})$", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Cross-Bearing Domain Shift: Intra-Class Spectrogram Variation Across Physical Bearings",
        fontsize=13,
        fontweight="bold",
        y=0.96
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_dataset_statistics(raw_dir: Path, out_path: Path, dpi: int = 300) -> None:
    """
    Plot dataset summary charts:
      - Subplot A: Segment counts across partitions (Fitting, Reference, Val, Test)
      - Subplot B: Pixel intensity distribution before vs after frozen standardization
      - Subplot C: RMS vibration energy across all 12 bearing recordings
    """
    print(f"Generating dataset statistics visualization -> {out_path}...")
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    ax_counts = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_rms = fig.add_subplot(gs[0, 2])

    # 1. Segment Counts Bar Chart
    split_counts = {
        "Fitting Pool\n(Train 6-20)": 2749,
        "Reference Pool\n(Train 1-5)": 465,
        "Validation\n(Val 1-20)": 1875,
        "Test Pool\n(Test 1-20)": 3726
    }
    bar_colors = ["#386cb0", "#7fc97f", "#beaed4", "#fdc086"]
    bars = ax_counts.bar(split_counts.keys(), split_counts.values(), color=bar_colors, edgecolor="black", width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax_counts.annotate(f"{h:,}",
                           xy=(bar.get_x() + bar.get_width() / 2, h),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_counts.set_title("Extracted Segments by Split (Total: 8,815)", fontsize=11, fontweight="bold")
    ax_counts.set_ylabel("Segment Count", fontsize=10, fontweight="bold")
    ax_counts.set_ylim(0, 4200)
    ax_counts.tick_params(axis="x", labelsize=8)

    # 2. Pixel Intensity Distribution (Simulated from actual sample statistics)
    sample_maps = []
    for bearing, split in [("K001", "train"), ("KI01", "train"), ("KA01", "train")]:
        mat_path = find_bearing_file(raw_dir, split, bearing, file_idx=6)
        sig = read_paderborn_mat(mat_path)
        seg = segment_signal(sig, step=8192)[0]
        sample_maps.append(compute_stft_map(seg))

    combined_raw = np.concatenate([m.flatten() for m in sample_maps])
    sample_mean = -3.94688
    sample_std = 0.82221
    standardized = (combined_raw - sample_mean) / sample_std

    ax_hist.hist(combined_raw, bins=50, density=True, alpha=0.55, color="#1f77b4", label="Raw Log-STFT", edgecolor="none")
    ax_hist.hist(standardized, bins=50, density=True, alpha=0.55, color="#d95f02", label="Standardized (μ=0, σ=1)", edgecolor="none")
    ax_hist.axvline(0, color="red", linestyle="--", alpha=0.7, label="Standardized Mean (0.0)")
    ax_hist.set_title("Spectrogram Pixel Intensity Distribution", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Value", fontsize=10, fontweight="bold")
    ax_hist.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax_hist.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # 3. Vibration RMS Comparison Across All 12 Bearings
    all_bearings_info = [
        ("K001", "Healthy", "train"), ("K002", "Healthy", "val"),
        ("K003", "Healthy", "test"), ("K004", "Healthy", "test"),
        ("KI01", "Inner", "train"),   ("KI03", "Inner", "val"),
        ("KI05", "Inner", "test"),    ("KI07", "Inner", "test"),
        ("KA01", "Outer", "train"),   ("KA03", "Outer", "val"),
        ("KA05", "Outer", "test"),    ("KA07", "Outer", "test"),
    ]

    bearing_names = []
    rms_vals = []
    colors_list = []
    for b_code, f_type, split in all_bearings_info:
        mat_p = find_bearing_file(raw_dir, split, b_code, file_idx=1)
        s = read_paderborn_mat(mat_p)
        rms = float(np.sqrt(np.mean(s ** 2)))
        bearing_names.append(b_code)
        rms_vals.append(rms)
        if f_type == "Healthy":
            colors_list.append(COLOR_HEALTHY)
        elif f_type == "Inner":
            colors_list.append(COLOR_INNER)
        else:
            colors_list.append(COLOR_OUTER)

    x_indices = np.arange(len(bearing_names))
    ax_rms.bar(x_indices, rms_vals, color=colors_list, width=0.6, edgecolor="black")
    ax_rms.set_xticks(x_indices)
    ax_rms.set_xticklabels(bearing_names, rotation=45, ha="right", fontsize=8)
    ax_rms.set_title("Vibration RMS Across All 12 Bearings", fontsize=11, fontweight="bold")
    ax_rms.set_ylabel("RMS Amplitude [V / m/s²]", fontsize=10, fontweight="bold")

    # Legend for conditions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_HEALTHY, edgecolor="black", label="Healthy"),
        Patch(facecolor=COLOR_INNER, edgecolor="black", label="Inner Race"),
        Patch(facecolor=COLOR_OUTER, edgecolor="black", label="Outer Race")
    ]
    ax_rms.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Paderborn Bearing Dataset Summary, Partition Counts & Feature Statistics",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def generate_all_visualizations(
    raw_dir: str = "data/raw",
    out_dir: str = "figures",
    dpi: int = 300,
    img_format: str = "png"
) -> List[Path]:
    """Generate the complete suite of visualizations and save to output directory."""
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("Siamese Bearing Identification: Data Visualizer")
    print(f"Raw data directory: {raw_path.resolve()}")
    print(f"Output directory:   {out_path.resolve()}")
    print(f"Format: {img_format}, DPI: {dpi}")
    print("==================================================\n")

    generated = []

    # 1. Waveforms
    p1 = out_path / f"01_raw_waveforms.{img_format}"
    plot_time_domain_waveforms(raw_path, p1, dpi=dpi)
    generated.append(p1)

    # 2. Frequency spectra
    p2 = out_path / f"02_frequency_spectra.{img_format}"
    plot_frequency_spectra(raw_path, p2, dpi=dpi)
    generated.append(p2)

    # 3. STFT Spectrograms
    p3 = out_path / f"03_stft_spectrograms.{img_format}"
    plot_stft_spectrograms(raw_path, p3, dpi=dpi)
    generated.append(p3)

    # 4. Pipeline transformation
    p4 = out_path / f"04_preprocessing_pipeline_steps.{img_format}"
    plot_pipeline_transformation(raw_path, p4, bearing="KI01", dpi=dpi)
    generated.append(p4)

    # 5. Cross-bearing comparison
    p5 = out_path / f"05_cross_bearing_comparison.{img_format}"
    plot_cross_bearing_comparison(raw_path, p5, dpi=dpi)
    generated.append(p5)

    # 6. Dataset statistics
    p6 = out_path / f"06_dataset_statistics.{img_format}"
    plot_dataset_statistics(raw_path, p6, dpi=dpi)
    generated.append(p6)

    print("\n==================================================")
    print(f"Successfully generated {len(generated)} publication-quality figures in:")
    print(f"  {out_path.resolve()}")
    print("==================================================")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Siamese Bearing Identification Data Visualization Suite")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to data/raw directory")
    parser.add_argument("--out-dir", type=str, default="figures", help="Directory where figures will be saved")
    parser.add_argument("--dpi", type=int, default=300, help="DPI resolution for exported figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output image format")

    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    parser.add_argument("--waveforms", action="store_true", help="Generate time-domain waveforms")
    parser.add_argument("--spectra", action="store_true", help="Generate frequency-domain spectra (PSD)")
    parser.add_argument("--spectrograms", action="store_true", help="Generate STFT spectrogram comparison")
    parser.add_argument("--pipeline", action="store_true", help="Generate pipeline transformation walkthrough")
    parser.add_argument("--cross-bearing", action="store_true", help="Generate cross-bearing domain shift comparison")
    parser.add_argument("--stats", action="store_true", help="Generate dataset summary & statistics charts")

    args = parser.parse_args()

    raw_path = Path(args.data_dir)
    out_path = Path(args.out_dir)

    # Default to generating all if no specific plot was selected
    selected_any = any([
        args.all, args.waveforms, args.spectra, args.spectrograms,
        args.pipeline, args.cross_bearing, args.stats
    ])

    if not selected_any or args.all:
        generate_all_visualizations(raw_dir=args.data_dir, out_dir=args.out_dir, dpi=args.dpi, img_format=args.format)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        if args.waveforms:
            plot_time_domain_waveforms(raw_path, out_path / f"01_raw_waveforms.{args.format}", dpi=args.dpi)
        if args.spectra:
            plot_frequency_spectra(raw_path, out_path / f"02_frequency_spectra.{args.format}", dpi=args.dpi)
        if args.spectrograms:
            plot_stft_spectrograms(raw_path, out_path / f"03_stft_spectrograms.{args.format}", dpi=args.dpi)
        if args.pipeline:
            plot_pipeline_transformation(raw_path, out_path / f"04_preprocessing_pipeline_steps.{args.format}", dpi=args.dpi)
        if args.cross_bearing:
            plot_cross_bearing_comparison(raw_path, out_path / f"05_cross_bearing_comparison.{args.format}", dpi=args.dpi)
        if args.stats:
            plot_dataset_statistics(raw_path, out_path / f"06_dataset_statistics.{args.format}", dpi=args.dpi)


if __name__ == "__main__":
    main()
