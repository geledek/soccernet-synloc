"""
SimCC-style coordinate head for improved keypoint localization.

SimCC (Simple Coordinate Classification) predicts 1D coordinate distributions
instead of 2D heatmaps, which is more efficient and can achieve better accuracy.

Reference:
    Li et al., "SimCC: A Simple Coordinate Classification Perspective
    for Human Pose Estimation", ECCV 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SimCCHead(nn.Module):
    """SimCC coordinate classification head.

    Predicts 1D distributions for x and y coordinates separately.
    This is more efficient than 2D heatmaps and can achieve sub-pixel accuracy.

    Args:
        in_channels: Input feature channels.
        num_keypoints: Number of keypoints to predict.
        input_size: (W, H) input image size.
        simcc_split_ratio: Ratio to split feature resolution for SimCC.
        use_dark: Whether to use DARK for sub-pixel refinement.
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int = 2,
        input_size: Tuple[int, int] = (640, 640),
        simcc_split_ratio: float = 2.0,
        use_dark: bool = True
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.use_dark = use_dark

        # SimCC output dimensions (upscaled coordinates)
        self.W = int(input_size[0] * simcc_split_ratio)
        self.H = int(input_size[1] * simcc_split_ratio)

        # Feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Coordinate classifiers (predict 1D distributions)
        self.mlp_x = nn.Linear(256, num_keypoints * self.W)
        self.mlp_y = nn.Linear(256, num_keypoints * self.H)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            feats: Feature tensor (B, C, H, W) from neck.

        Returns:
            Tuple of (x_logits, y_logits):
                - x_logits: (B, K, W) x-coordinate logits
                - y_logits: (B, K, H) y-coordinate logits
        """
        B = feats.shape[0]

        # Project features
        feats = self.projection(feats)

        # Global average pooling
        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)

        # Predict x and y distributions
        x_logits = self.mlp_x(feats).view(B, self.num_keypoints, self.W)
        y_logits = self.mlp_y(feats).view(B, self.num_keypoints, self.H)

        return x_logits, y_logits

    def decode(
        self,
        x_logits: torch.Tensor,
        y_logits: torch.Tensor,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode logits to coordinates.

        Args:
            x_logits: (B, K, W) x-coordinate logits.
            y_logits: (B, K, H) y-coordinate logits.
            normalize: Whether to normalize coordinates to [0, 1].

        Returns:
            Tuple of (keypoints, scores):
                - keypoints: (B, K, 2) keypoint coordinates
                - scores: (B, K) keypoint confidences
        """
        B, K = x_logits.shape[:2]

        # Softmax to get distributions
        x_probs = F.softmax(x_logits, dim=-1)
        y_probs = F.softmax(y_logits, dim=-1)

        # Get argmax positions
        x_coords = torch.argmax(x_probs, dim=-1).float()
        y_coords = torch.argmax(y_probs, dim=-1).float()

        # Get confidence as max probability
        x_max = x_probs.max(dim=-1).values
        y_max = y_probs.max(dim=-1).values
        scores = (x_max + y_max) / 2

        # DARK-style sub-pixel refinement
        if self.use_dark:
            x_coords = self._dark_refine(x_coords, x_probs, self.W)
            y_coords = self._dark_refine(y_coords, y_probs, self.H)

        # Scale to input resolution
        if normalize:
            x_coords = x_coords / self.W
            y_coords = y_coords / self.H
        else:
            x_coords = x_coords / self.simcc_split_ratio
            y_coords = y_coords / self.simcc_split_ratio

        keypoints = torch.stack([x_coords, y_coords], dim=-1)

        return keypoints, scores

    def _dark_refine(
        self,
        coords: torch.Tensor,
        probs: torch.Tensor,
        size: int
    ) -> torch.Tensor:
        """DARK-style sub-pixel refinement.

        Uses Taylor expansion around the peak for sub-pixel accuracy.

        Args:
            coords: (B, K) integer coordinates.
            probs: (B, K, L) probability distributions.
            size: Distribution size.

        Returns:
            Refined coordinates (B, K).
        """
        B, K, L = probs.shape

        # Pad for border handling
        probs_pad = F.pad(probs, (1, 1), mode='replicate')

        # Get neighbors
        idx = coords.long().clamp(0, size - 1)
        idx_expand = idx.unsqueeze(-1) + 1  # Account for padding

        # Gather p-1, p, p+1
        batch_idx = torch.arange(B, device=probs.device)[:, None].expand(B, K)
        kpt_idx = torch.arange(K, device=probs.device)[None, :].expand(B, K)

        p_minus = probs_pad[batch_idx, kpt_idx, idx_expand.squeeze(-1)]
        p_center = probs_pad[batch_idx, kpt_idx, idx_expand.squeeze(-1) + 1]
        p_plus = probs_pad[batch_idx, kpt_idx, idx_expand.squeeze(-1) + 2]

        # Second-order Taylor expansion
        # x* = x + 0.5 * (p+ - p-) / (2*p - p- - p+)
        eps = 1e-6
        denominator = 2 * p_center - p_minus - p_plus + eps
        offset = 0.5 * (p_plus - p_minus) / denominator
        offset = offset.clamp(-0.5, 0.5)

        return coords + offset


class SimCCLoss(nn.Module):
    """SimCC loss using KL divergence.

    Creates Gaussian target distributions and computes KL divergence.

    Args:
        simcc_split_ratio: Coordinate split ratio.
        sigma: Gaussian sigma for target distribution.
    """

    def __init__(
        self,
        simcc_split_ratio: float = 2.0,
        sigma: float = 6.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.simcc_split_ratio = simcc_split_ratio
        self.sigma = sigma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        x_logits: torch.Tensor,
        y_logits: torch.Tensor,
        targets: torch.Tensor,
        target_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute SimCC loss.

        Args:
            x_logits: (B, K, W) x-coordinate logits.
            y_logits: (B, K, H) y-coordinate logits.
            targets: (B, K, 2) target coordinates in [0, 1].
            target_weights: (B, K) visibility weights.

        Returns:
            Loss value.
        """
        B, K, W = x_logits.shape
        H = y_logits.shape[-1]
        device = x_logits.device

        # Scale targets to SimCC resolution
        target_x = targets[..., 0] * W
        target_y = targets[..., 1] * H

        # Generate Gaussian target distributions
        x_range = torch.arange(W, device=device).float()
        y_range = torch.arange(H, device=device).float()

        # (B, K, W/H) Gaussian distributions
        x_target = self._gaussian(x_range, target_x.unsqueeze(-1), self.sigma)
        y_target = self._gaussian(y_range, target_y.unsqueeze(-1), self.sigma)

        # Label smoothing
        if self.label_smoothing > 0:
            x_target = x_target * (1 - self.label_smoothing) + self.label_smoothing / W
            y_target = y_target * (1 - self.label_smoothing) + self.label_smoothing / H

        # KL divergence (with log_softmax for numerical stability)
        x_log_probs = F.log_softmax(x_logits, dim=-1)
        y_log_probs = F.log_softmax(y_logits, dim=-1)

        x_loss = F.kl_div(x_log_probs, x_target, reduction='none').sum(-1)
        y_loss = F.kl_div(y_log_probs, y_target, reduction='none').sum(-1)

        loss = x_loss + y_loss

        # Apply visibility weights
        if target_weights is not None:
            loss = loss * target_weights

        return loss.mean()

    def _gaussian(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """Generate normalized Gaussian distribution."""
        gauss = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)
