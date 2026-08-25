from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import mean_flat


class AlignmentLoss(ABC, nn.Module):
    """Base class for alignment loss functions.

    All losses receive tensors in canonical ``(B, L, D)`` format:
    ``pred`` and ``target`` are spatially aligned token sequences,
    ``mask`` (optional) is ``(B, L)`` or ``(B, L, 1)``.
    """

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


class CosineSimilarityLoss(AlignmentLoss):
    """Negative cosine similarity with mask-aware reduction."""

    def forward(self, pred, target, mask=None):
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        per_token = -(pred_n * target_n).sum(dim=-1, keepdim=True)  # (B, L, 1)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
        return mean_flat(per_token.squeeze(-1)).mean()


class MSEAlignmentLoss(AlignmentLoss):
    """Per-token MSE with mask-aware reduction."""

    def forward(self, pred, target, mask=None):
        per_token = (pred - target).pow(2).mean(dim=-1, keepdim=True)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
        return mean_flat(per_token.squeeze(-1)).mean()


class SmoothL1AlignmentLoss(AlignmentLoss):
    """Smooth L1 with mask-aware reduction."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, mask=None):
        per_token = F.smooth_l1_loss(
            pred, target, beta=self.beta, reduction='none',
        ).mean(dim=-1, keepdim=True)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
        return mean_flat(per_token.squeeze(-1)).mean()


class DistmatMarginLoss(AlignmentLoss):
    """Spatial similarity-matrix margin loss (VA-VAE loss_1).

    Compares the pairwise spatial similarity structure of ``pred`` and
    ``target`` and penalises differences that exceed ``margin``.
    """

    def __init__(self, margin: float = 0.25):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target, mask=None):
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        pred_sim = torch.bmm(pred_n, pred_n.transpose(1, 2))
        target_sim = torch.bmm(target_n, target_n.transpose(1, 2))
        diff = (pred_sim - target_sim).abs()
        loss = F.relu(diff - self.margin).mean()
        del pred_sim, target_sim, diff
        return loss


class CosineMarginLoss(AlignmentLoss):
    """Per-token cosine margin loss (VA-VAE loss_2).

    Penalises when the per-position cosine similarity between ``pred``
    and ``target`` drops below ``1 - margin``.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target, mask=None):
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        return F.relu(1.0 - self.margin - cos_sim).mean()
