"""
src/train.py
Training routine for Project 2: Siamese Bearing Identification.

Trains:
  1. Siamese Network:
     - Pair-based learning with Contrastive Loss (margin=1.0)
     - Balanced pair generator across training bearings
     - Adam optimizer (lr=0.001), batch size 32, 30 epochs
     - Evaluates at each epoch on held-out validation set using 5 fixed reference files/class
     - Saves checkpoint with best validation macro-F1: models/best_siamese_model.pth
     - Reloads and verifies best model
  2. Baseline Classifier:
     - Same encoder backbone with 3 outputs and Cross-Entropy loss
     - Equal training budget (30 epochs, Adam lr=0.001, batch size 32)
     - Evaluates on validation set and saves best model: models/best_baseline_model.pth
     - Reloads and verifies best model
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

# Adjust path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.data_pipeline import BearingDatasetContainer, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from src.model import BearingEncoder, SiameseNetwork, ContrastiveLoss, BaselineClassifier, count_parameters


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BalancedPairDataset(Dataset):
    """
    Generates balanced pairs from the fitting segments pool:
      - 50% positive pairs (same class, matching target = 1)
      - 50% negative pairs (different class, non-matching target = 0)
      - Balanced across training bearings (K001, K002, KI01, KI03, KA01, KA03)
      - Reference and held-out files are strictly excluded.
    """
    def __init__(self, fit_segments: List[dict], pairs_per_epoch: int = 5120):
        self.pairs_per_epoch = pairs_per_epoch
        
        # Organize segments by class and by bearing
        self.class_bearing_segs: Dict[int, Dict[str, List[np.ndarray]]] = {0: {}, 1: {}, 2: {}}
        self.class_segs: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        
        for item in fit_segments:
            c = item["class"]
            b = item["bearing"]
            m = item["map"]  # (128, 128)
            if b not in self.class_bearing_segs[c]:
                self.class_bearing_segs[c][b] = []
            self.class_bearing_segs[c][b].append(m)
            self.class_segs[c].append(m)

        self.classes = [0, 1, 2]
        self.pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []
        self.generate_pairs()

    def generate_pairs(self):
        """Pre-generate balanced pairs for one epoch."""
        self.pairs = []
        half = self.pairs_per_epoch // 2

        # 1. Positive pairs (target = 1)
        for _ in range(half):
            c = random.choice(self.classes)
            bearings = list(self.class_bearing_segs[c].keys())
            b1 = random.choice(bearings)
            b2 = random.choice(bearings)
            s1 = random.choice(self.class_bearing_segs[c][b1])
            s2 = random.choice(self.class_bearing_segs[c][b2])
            self.pairs.append((s1, s2, 1))

        # 2. Negative pairs (target = 0)
        for _ in range(half):
            c1, c2 = random.sample(self.classes, 2)
            b1 = random.choice(list(self.class_bearing_segs[c1].keys()))
            b2 = random.choice(list(self.class_bearing_segs[c2].keys()))
            s1 = random.choice(self.class_bearing_segs[c1][b1])
            s2 = random.choice(self.class_bearing_segs[c2][b2])
            self.pairs.append((s1, s2, 0))

        random.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s1, s2, target = self.pairs[idx]
        t1 = torch.from_numpy(s1).unsqueeze(0).float()  # (1, 128, 128)
        t2 = torch.from_numpy(s2).unsqueeze(0).float()  # (1, 128, 128)
        return t1, t2, torch.tensor(target, dtype=torch.float32)


class FittingClassificationDataset(Dataset):
    """Dataset for training baseline classifier on fitting segments."""
    def __init__(self, fit_segments: List[dict]):
        self.items = fit_segments

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        x = torch.from_numpy(item["map"]).unsqueeze(0).float()
        y = torch.tensor(item["class"], dtype=torch.long)
        return x, y


def evaluate_siamese_on_val(
    model: SiameseNetwork,
    fixed_ref_segments: List[dict],
    val_segments: List[dict],
    device: torch.device
) -> Tuple[float, float, float, np.ndarray]:
    """
    Evaluate Siamese network on validation bearings using 5 fixed reference files per class:
      - Compute reference embeddings and class average templates
      - For each validation file, average Euclidean distances across its segments
      - Choose class with minimum average distance
      - Return accuracy, macro-F1, macro-recall, and confusion matrix
    """
    model.eval()

    # 1. Compute class templates from fixed reference files
    ref_by_class: Dict[int, List[torch.Tensor]] = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for item in fixed_ref_segments:
            x = torch.from_numpy(item["map"]).unsqueeze(0).unsqueeze(0).float().to(device)
            emb = model.forward_one(x)
            ref_by_class[item["class"]].append(emb.squeeze(0))

    class_templates: Dict[int, torch.Tensor] = {}
    for c in [0, 1, 2]:
        stacked = torch.stack(ref_by_class[c], dim=0)
        # Average embeddings by class
        centroid = torch.mean(stacked, dim=0)
        class_templates[c] = centroid

    # 2. Group validation segments by file
    val_files: Dict[str, List[dict]] = {}
    for item in val_segments:
        key = f"{item['bearing']}_{item['file_idx']}"
        if key not in val_files:
            val_files[key] = []
        val_files[key].append(item)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for file_key, seg_list in val_files.items():
            true_class = seg_list[0]["class"]
            # Stack all segments in this file
            batch_x = torch.stack([
                torch.from_numpy(s["map"]).unsqueeze(0).float() for s in seg_list
            ], dim=0).to(device)

            embs = model.forward_one(batch_x)  # (N_segs, 32)

            # Compute average distance from segments to each class template
            class_dists = []
            for c in [0, 1, 2]:
                template = class_templates[c].unsqueeze(0)  # (1, 32)
                # Euclidean distance
                dists = torch.norm(embs - template, p=2, dim=1)  # (N_segs,)
                avg_dist = dists.mean().item()
                class_dists.append(avg_dist)

            pred_class = int(np.argmin(class_dists))
            y_true.append(true_class)
            y_pred.append(pred_class)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return acc, macro_f1, macro_rec, cm


def evaluate_baseline_on_val(
    model: BaselineClassifier,
    val_segments: List[dict],
    device: torch.device
) -> Tuple[float, float, float, np.ndarray]:
    """Evaluate baseline classifier on validation files by averaging predicted logits."""
    model.eval()

    val_files: Dict[str, List[dict]] = {}
    for item in val_segments:
        key = f"{item['bearing']}_{item['file_idx']}"
        if key not in val_files:
            val_files[key] = []
        val_files[key].append(item)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for file_key, seg_list in val_files.items():
            true_class = seg_list[0]["class"]
            batch_x = torch.stack([
                torch.from_numpy(s["map"]).unsqueeze(0).float() for s in seg_list
            ], dim=0).to(device)

            logits = model(batch_x)  # (N_segs, 3)
            probs = torch.softmax(logits, dim=1)
            avg_probs = probs.mean(dim=0).cpu().numpy()

            pred_class = int(np.argmax(avg_probs))
            y_true.append(true_class)
            y_pred.append(pred_class)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return acc, macro_f1, macro_rec, cm


def train_siamese(
    container: BearingDatasetContainer,
    device: torch.device,
    save_path: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    margin: float = 1.0,
    pairs_per_epoch: int = 5120
) -> Dict:
    """Train the Siamese Neural Network using Contrastive Loss."""
    print("\n" + "=" * 60)
    print("Training Siamese Network (Twin Architecture)")
    print(f"Device: {device}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, margin={margin}")
    print("=" * 60)

    # 1. Prepare fixed 5 reference files per class (using first segment, seg_idx == 0)
    # References are files 1 to 5 from training bearings (K001, KI01, KA01)
    ref_first_segments = container.get_reference_first_segments()
    fixed_val_refs = [
        item for item in ref_first_segments
        if item["bearing"] in ["K001", "KI01", "KA01"] and 1 <= item["file_idx"] <= 5
    ]
    print(f"Fixed validation references count: {len(fixed_val_refs)} (Expected: 15, 5/class)")

    # 2. Instantiate model, loss, optimizer
    encoder = BearingEncoder(embedding_dim=32)
    model = SiameseNetwork(encoder=encoder).to(device)
    criterion = ContrastiveLoss(margin=margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initial weights verification
    init_param_norm = sum(p.norm().item() for p in model.parameters())
    print(f"Initial model parameters norm: {init_param_norm:.4f}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    pair_dataset = BalancedPairDataset(container.fit_segments, pairs_per_epoch=pairs_per_epoch)
    dataloader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)

    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_macro_f1": []}
    best_val_f1 = -1.0
    best_epoch = -1
    start_time = time.time()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        pair_dataset.generate_pairs()  # Regenerate balanced pairs for new epoch
        epoch_loss = 0.0
        num_batches = 0

        for x1, x2, target in dataloader:
            x1, x2, target = x1.to(device), x2.to(device), target.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(x1, x2)
            loss = criterion(emb1, emb2, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Validation evaluation
        val_acc, val_f1, val_rec, cm = evaluate_siamese_on_val(
            model, fixed_val_refs, container.val_segments, device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "margin": margin,
                "embedding_dim": 32
            }, save_path)

        flag = " [BEST -> Saved]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.4f}{flag}")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s.")
    print(f"Best Validation Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Checkpoint Reload Verification
    print("\nVerifying checkpoint reload...")
    checkpoint = torch.load(save_path, map_location=device)
    reload_model = SiameseNetwork(encoder=BearingEncoder(embedding_dim=32)).to(device)
    reload_model.load_state_dict(checkpoint["model_state_dict"])
    re_acc, re_f1, _, _ = evaluate_siamese_on_val(
        reload_model, fixed_val_refs, container.val_segments, device
    )
    print(f"Reload verification: Val Acc: {re_acc*100:.1f}%, Val Macro-F1: {re_f1:.4f} (Matches best: {abs(re_f1 - best_val_f1) < 1e-5})")

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "training_time": training_time,
        "history": history
    }


def train_baseline(
    container: BearingDatasetContainer,
    device: torch.device,
    save_path: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001
) -> Dict:
    """Train the Baseline Classifier under equal budget."""
    print("\n" + "=" * 60)
    print("Training Baseline Classifier (Cross-Entropy)")
    print(f"Device: {device}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print("=" * 60)

    encoder = BearingEncoder(embedding_dim=32)
    model = BaselineClassifier(encoder=encoder, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    fit_dataset = FittingClassificationDataset(container.fit_segments)
    dataloader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True)

    print(f"Fitting segments count: {len(fit_dataset):,}")
    print(f"Baseline trainable parameters: {count_parameters(model):,}")

    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_macro_f1": []}
    best_val_f1 = -1.0
    best_epoch = -1
    start_time = time.time()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        val_acc, val_f1, val_rec, cm = evaluate_baseline_on_val(
            model, container.val_segments, device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "num_classes": 3
            }, save_path)

        flag = " [BEST -> Saved]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.4f}{flag}")

    training_time = time.time() - start_time
    print(f"\nBaseline training completed in {training_time:.2f}s.")
    print(f"Best Baseline Validation Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Checkpoint Reload Verification
    checkpoint = torch.load(save_path, map_location=device)
    reload_model = BaselineClassifier(encoder=BearingEncoder(embedding_dim=32), num_classes=3).to(device)
    reload_model.load_state_dict(checkpoint["model_state_dict"])
    re_acc, re_f1, _, _ = evaluate_baseline_on_val(reload_model, container.val_segments, device)
    print(f"Baseline reload verification: Val Acc: {re_acc*100:.1f}%, Val Macro-F1: {re_f1:.4f}")

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "training_time": training_time,
        "history": history
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution environment device: {device}")

    # Load preprocessed dataset container
    container = BearingDatasetContainer(raw_dir="data/raw")
    container.load_and_preprocess()

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    siamese_save_path = models_dir / "best_siamese_model.pth"
    baseline_save_path = models_dir / "best_baseline_model.pth"

    # Train Siamese network
    siamese_res = train_siamese(
        container=container,
        device=device,
        save_path=siamese_save_path,
        epochs=30,
        batch_size=32,
        lr=0.001,
        margin=1.0,
        pairs_per_epoch=5120
    )

    # Train Baseline classifier
    baseline_res = train_baseline(
        container=container,
        device=device,
        save_path=baseline_save_path,
        epochs=30,
        batch_size=32,
        lr=0.001
    )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_summary.json", "w") as f:
        json.dump({
            "siamese": siamese_res,
            "baseline": baseline_res
        }, f, indent=2)

    print("\nTraining and validation complete! Checkpoints and summary saved.")


if __name__ == "__main__":
    main()
