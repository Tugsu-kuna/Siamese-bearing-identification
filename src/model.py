"""
src/model.py
Neural network architectures and loss functions for Project 2: Siamese Bearing Identification.

Implements:
  - BearingEncoder: 3x3 blocks (16, 32, 64), ReLU, MaxPool, Global Average Pooling,
                    unit-length 32-number embedding.
  - SiameseNetwork: Shared encoder called twice with twin branches.
  - ContrastiveLoss: Attracts matching pairs (same-class) and separates others with margin 1.
  - BaselineClassifier: Same encoder backbone with 3 outputs and Cross-Entropy loss.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BearingEncoder(nn.Module):
    """
    3x3 convolutional encoder for 128x128 single-channel STFT maps:
      - Block 1: Conv2d(1, 16, 3, padding=1), BatchNorm2d, ReLU, MaxPool2d(2, 2)   -> (16, 64, 64)
      - Block 2: Conv2d(16, 32, 3, padding=1), BatchNorm2d, ReLU, MaxPool2d(2, 2)  -> (32, 32, 32)
      - Block 3: Conv2d(32, 64, 3, padding=1), BatchNorm2d, ReLU, MaxPool2d(2, 2)  -> (64, 16, 16)
      - Global Average Pooling: (64, 1, 1) -> 64
      - Fully Connected: Linear(64, 32)
      - L2 Normalization: Unit-length 32-dimensional embedding
    """
    def __init__(self, in_channels: int = 1, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Block 1: 1 -> 16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 16 -> 32
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 32 -> 64
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection to 32-dim embedding
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, 128, 128)
            normalize: If True, apply L2 normalization to output unit-length embedding.
        Returns:
            Embedding tensor of shape (B, 32)
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if normalize:
            out = F.normalize(out, p=2, dim=1)
        return out


class SiameseNetwork(nn.Module):
    """
    Siamese Network wrapper around a shared BearingEncoder.
    Calls the shared encoder twice to generate unit-length embeddings for a pair of maps.
    """
    def __init__(self, encoder: Optional[BearingEncoder] = None, embedding_dim: int = 32):
        super().__init__()
        if encoder is None:
            self.encoder = BearingEncoder(embedding_dim=embedding_dim)
        else:
            self.encoder = encoder

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single batch of maps into unit-length embeddings."""
        return self.encoder(x, normalize=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode both map batches using the same shared weights."""
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss:
      L(e1, e2, y) = 0.5 * y * D^2 + 0.5 * (1 - y) * max(0, margin - D)^2
    where D = ||e1 - e2||_2 is the Euclidean distance,
    y = 1 indicates same-class (matching pair, attracted together),
    y = 0 indicates different-class (separated by margin).
    """
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1: Unit-length embeddings (B, D)
            emb2: Unit-length embeddings (B, D)
            target: Binary indicator tensor (B,) or (B, 1). 1 = same class, 0 = different class.
        """
        target = target.view(-1).float()
        # Pairwise Euclidean distance
        euclidean_dist = F.pairwise_distance(emb1, emb2, p=2)

        # Positive pairs: attract (minimize distance squared)
        pos_loss = target * torch.pow(euclidean_dist, 2)

        # Negative pairs: separate with margin
        neg_dist = F.relu(self.margin - euclidean_dist)
        neg_loss = (1.0 - target) * torch.pow(neg_dist, 2)

        loss = 0.5 * (pos_loss + neg_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BaselineClassifier(nn.Module):
    """
    Baseline model: Same encoder architecture with 3-class classification head,
    trained with standard Cross-Entropy Loss for equal budget comparison.
    """
    def __init__(self, encoder: Optional[BearingEncoder] = None, num_classes: int = 3, embedding_dim: int = 32):
        super().__init__()
        if encoder is None:
            self.encoder = BearingEncoder(embedding_dim=embedding_dim)
        else:
            self.encoder = encoder
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits (B, 3)."""
        emb = self.encoder(x, normalize=False)
        logits = self.classifier(emb)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract unit-length 32-number embedding."""
        return self.encoder(x, normalize=True)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Self-testing model architectures...")
    encoder = BearingEncoder()
    siamese = SiameseNetwork(encoder)
    baseline = BaselineClassifier(encoder)

    dummy_input1 = torch.randn(4, 1, 128, 128)
    dummy_input2 = torch.randn(4, 1, 128, 128)
    dummy_target = torch.tensor([1, 0, 1, 0])

    emb1, emb2 = siamese(dummy_input1, dummy_input2)
    print("Encoder output shape:", emb1.shape)
    print("Unit-norm check:", torch.norm(emb1, dim=1).tolist())

    criterion = ContrastiveLoss(margin=1.0)
    loss = criterion(emb1, emb2, dummy_target)
    print("Contrastive loss:", float(loss))

    logits = baseline(dummy_input1)
    print("Baseline logits shape:", logits.shape)
    print(f"Total encoder parameters: {count_parameters(encoder):,}")
    print(f"Total baseline parameters: {count_parameters(baseline):,}")
    print("Model tests passed successfully!")
