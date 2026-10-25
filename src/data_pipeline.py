"""
src/data_pipeline.py
Data preprocessing and feature extraction pipeline for Project 2: Siamese Bearing Identification.

Folder configuration:
  - data/raw/train/: K001, KI01, KA01 (Files 1-5 are references, 6-20 are fitting)
  - data/raw/val/:   K002, KI03, KA03 (Held-out bearings, use all 20 files)
  - data/raw/test/:  K003, K004, KI05, KI07, KA05, KA07 (Held-out bearings, use all 20 files)

Pipeline steps:
  1. Parse 64 kHz vibration from Paderborn .mat files (channel 'vibration_1').
  2. Split signal into 8192-sample segments:
       - Step 4096 for fitting (train files 6 to 20, 50% overlap).
       - Step 8192 for references (train files 1 to 5) and validation/test files (no overlap).
       - Incomplete segments are discarded; segment mean is removed.
  3. Compute STFT log-power spectrogram:
       - Hann window: length 512, overlap 448 (step 64, 87.5% overlap, no padding).
       - Frequency range: 0 to 32 kHz (Nyquist, 257 frequency bins).
       - Log magnitude: log10(squared magnitudes + 1e-12).
       - Resized to (128, 128) via bilinear interpolation.
  4. Standardise:
       - Compute scalar mean and standard deviation across fitting maps only.
       - Freeze statistics and apply (map - mean) / std to all maps.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.io as sio
from scipy.signal import stft
from scipy.ndimage import zoom

# Fault classes
CLASSES = ["Healthy", "Inner_race", "Outer_race"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Dataset folder layout
BEARING_SPLITS = {
    "train": {
        "Healthy": ["K001"],
        "Inner_race": ["KI01"],
        "Outer_race": ["KA01"]
    },
    "val": {
        "Healthy": ["K002"],
        "Inner_race": ["KI03"],
        "Outer_race": ["KA03"]
    },
    "test": {
        "Healthy": ["K003", "K004"],
        "Inner_race": ["KI05", "KI07"],
        "Outer_race": ["KA05", "KA07"]
    }
}

SAMPLING_RATE = 64000
SEGMENT_LEN = 8192
FIT_STEP = 4096
TEST_STEP = 8192
N_FFT = 512
N_OVERLAP = 448  # Step = 64, overlap = 87.5%
TARGET_SHAPE = (128, 128)

def extract_file_index(filename: str) -> int:
    """Extract recording run index from filename (e.g. N15_M07_F10_K001_14.mat -> 14)."""
    m = re.search(r"_(\d+)\.mat$", filename)
    if m:
        return int(m.group(1))
    return -1

def read_paderborn_mat(mat_path: Path) -> np.ndarray:
    """
    Read 64 kHz vibration signal from a Paderborn .mat file.
    Extracts the channel named 'vibration_1'.
    """
    mat = sio.loadmat(str(mat_path))
    var_keys = [k for k in mat.keys() if not k.startswith("__")]
    if not var_keys:
        raise ValueError(f"No variable found in {mat_path}")
    main_obj = mat[var_keys[0]]
    y_channels = main_obj[0, 0]["Y"]
    
    for i in range(y_channels.shape[1]):
        chan = y_channels[0, i]
        name = chan["Name"][0] if len(chan["Name"]) > 0 else ""
        if name == "vibration_1":
            data = chan["Data"][0].flatten()
            return data.astype(np.float64)
            
    # Fallback: last channel if vibration_1 is not named explicitly
    return y_channels[0, -1]["Data"][0].flatten().astype(np.float64)

def segment_signal(signal: np.ndarray, step: int, seg_len: int = SEGMENT_LEN) -> List[np.ndarray]:
    """
    Slice signal into segments of length seg_len.
    Discards incomplete trailing segments and subtracts segment mean.
    """
    n_samples = len(signal)
    segments = []
    start = 0
    while start + seg_len <= n_samples:
        seg = signal[start : start + seg_len].copy()
        seg = seg - np.mean(seg)  # Remove mean
        segments.append(seg)
        start += step
    return segments

def compute_stft_map(segment: np.ndarray, target_shape: Tuple[int, int] = TARGET_SHAPE) -> np.ndarray:
    """
    Compute STFT log-power spectrogram:
      - Hann window: 512
      - Overlap: 448 (step 64, 87.5%)
      - FFT: 512, no zero padding
      - Frequency range: 0 to 32 kHz (Nyquist)
      - log10(squared magnitudes + 1e-12)
      - Resized to (128, 128)
    """
    f, t, Zxx = stft(
        segment,
        fs=SAMPLING_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_OVERLAP,
        nfft=N_FFT,
        padded=False,
        boundary=None
    )
    # Squared magnitude
    sq_mag = np.abs(Zxx) ** 2
    # log10(squared magnitudes + 1e-12)
    log_map = np.log10(sq_mag + 1e-12)  # shape: (257, 121)

    # Resize to (128, 128) via bilinear interpolation
    zoom_factors = (target_shape[0] / log_map.shape[0], target_shape[1] / log_map.shape[1])
    resized_map = zoom(log_map, zoom_factors, order=1)
    return resized_map.astype(np.float32)

class BearingDatasetContainer:
    """
    Manages dataset loading, partitioning, STFT processing, and standardization.
    """
    def __init__(self, raw_dir: str = "data/raw", condition: str = "N15_M07_F10"):
        self.raw_dir = Path(raw_dir)
        self.condition = condition
        
        # Segment storage: list of dicts
        # Each item: {'map': np.ndarray, 'class': int, 'bearing': str, 'file_idx': int, 'seg_idx': int}
        self.fit_segments: List[dict] = []       # Train bearings, files 6 to 20
        self.ref_pool_segments: List[dict] = []  # Train bearings, files 1 to 5
        self.val_segments: List[dict] = []       # Val bearings, files 1 to 20
        self.test_segments: List[dict] = []      # Test bearings, files 1 to 20
        
        self.mean: float = 0.0
        self.std: float = 1.0
        self.is_standardized: bool = False

    def load_and_preprocess(self):
        """Load and process all bearing recordings according to the partition rules."""
        print(f"Loading recordings from: {self.raw_dir.resolve()}")
        
        for split, class_bearings in BEARING_SPLITS.items():
            split_dir = self.raw_dir / split
            for cls_name, bearing_list in class_bearings.items():
                cls_idx = CLASS_TO_IDX[cls_name]
                for bearing in bearing_list:
                    bearing_dir = split_dir / bearing
                    mat_files = sorted(
                        list(bearing_dir.glob(f"*{self.condition}*.mat")),
                        key=lambda p: extract_file_index(p.name)
                    )
                    
                    if not mat_files:
                        print(f"  Warning: No files found for {bearing} in {bearing_dir}")
                        continue
                        
                    for mat_p in mat_files:
                        f_idx = extract_file_index(mat_p.name)
                        sig = read_paderborn_mat(mat_p)
                        
                        if split == "train":
                            if 1 <= f_idx <= 5:
                                # References: step 8192 (use first segment during evaluation)
                                segs = segment_signal(sig, step=TEST_STEP)
                                for s_idx, seg in enumerate(segs):
                                    stft_map = compute_stft_map(seg)
                                    self.ref_pool_segments.append({
                                        "map": stft_map,
                                        "class": cls_idx,
                                        "bearing": bearing,
                                        "file_idx": f_idx,
                                        "seg_idx": s_idx
                                    })
                            else:
                                # Fitting: files 6 to 20, step 4096 (50% overlap)
                                segs = segment_signal(sig, step=FIT_STEP)
                                for s_idx, seg in enumerate(segs):
                                    stft_map = compute_stft_map(seg)
                                    self.fit_segments.append({
                                        "map": stft_map,
                                        "class": cls_idx,
                                        "bearing": bearing,
                                        "file_idx": f_idx,
                                        "seg_idx": s_idx
                                    })
                        elif split == "val":
                            # Val bearings: files 1 to 20, step 8192
                            segs = segment_signal(sig, step=TEST_STEP)
                            for s_idx, seg in enumerate(segs):
                                stft_map = compute_stft_map(seg)
                                self.val_segments.append({
                                    "map": stft_map,
                                    "class": cls_idx,
                                    "bearing": bearing,
                                    "file_idx": f_idx,
                                    "seg_idx": s_idx
                                })
                        elif split == "test":
                            # Test bearings: files 1 to 20, step 8192
                            segs = segment_signal(sig, step=TEST_STEP)
                            for s_idx, seg in enumerate(segs):
                                stft_map = compute_stft_map(seg)
                                self.test_segments.append({
                                    "map": stft_map,
                                    "class": cls_idx,
                                    "bearing": bearing,
                                    "file_idx": f_idx,
                                    "seg_idx": s_idx
                                })

        print(f"\nExtracted Segments Summary:")
        print(f"  Fitting pool (Train files 6-20, step 4096): {len(self.fit_segments)} segments")
        print(f"  Reference pool (Train files 1-5, step 8192): {len(self.ref_pool_segments)} segments")
        print(f"  Validation pool (Val files 1-20, step 8192): {len(self.val_segments)} segments")
        print(f"  Test pool (Test files 1-20, step 8192): {len(self.test_segments)} segments")

        self.compute_and_apply_standardization()

    def compute_and_apply_standardization(self):
        """
        Compute scalar mean and std across fitting maps only, freeze, and apply:
          standardized = (map - mean) / std
        """
        if not self.fit_segments:
            raise ValueError("No fitting segments available to compute standardization.")

        all_fit_maps = np.stack([item["map"] for item in self.fit_segments], axis=0)
        self.mean = float(np.mean(all_fit_maps))
        self.std = float(np.std(all_fit_maps))
        if self.std < 1e-8:
            self.std = 1.0

        print(f"\nStandardization Statistics (Fitting Maps Only):")
        print(f"  Mean: {self.mean:.5f}")
        print(f"  Std:  {self.std:.5f}")

        # Standardize all segments with the frozen statistics
        for seg_list in [self.fit_segments, self.ref_pool_segments, self.val_segments, self.test_segments]:
            for item in seg_list:
                item["map"] = ((item["map"] - self.mean) / self.std).astype(np.float32)

        self.is_standardized = True
        print("  Frozen standardization applied to all sets.")

    def get_reference_first_segments(self) -> List[dict]:
        """Return the first segment (seg_idx == 0) for each reference file."""
        return [item for item in self.ref_pool_segments if item["seg_idx"] == 0]

if __name__ == "__main__":
    print("Running data pipeline self-test on local dataset...")
    container = BearingDatasetContainer(raw_dir="data/raw")
    container.load_and_preprocess()

    ref_first = container.get_reference_first_segments()
    print(f"\nSelf-Test Verification:")
    print(f"  Reference first segments (files 1-5): {len(ref_first)} items")
    if container.fit_segments:
        sample_map = container.fit_segments[0]["map"]
        print(f"  Map shape: {sample_map.shape} (Expected: (128, 128))")
        print(f"  Map dtype: {sample_map.dtype}")
        print(f"  Sample standardized mean: {sample_map.mean():.4f}, std: {sample_map.std():.4f}")
    print("\nSelf-test complete!")
