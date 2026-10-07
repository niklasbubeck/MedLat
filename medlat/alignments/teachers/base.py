"""Universal teacher model interface for alignment modules."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class TeacherModel(ABC, nn.Module):
    """Abstract base for teacher models used in alignment.

    Provides a unified interface for feature extraction and optional
    attention-logit capture across model sources (timm, open_clip,
    HuggingFace, custom).

    Subclasses MUST implement :pyattr:`embed_dim` and
    :pymeth:`extract_features`.  Attention-related methods are optional
    — override them and return ``True`` from :pyattr:`supports_attention`
    to enable attention distillation (HASTE).
    """

    def __init__(self):
        super().__init__()

    def train(self, mode=True):
        return super().train(False)

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        ...

    @property
    def num_prefix_tokens(self) -> int:
        return 0

    @property
    def input_transform(self) -> Optional[nn.Module]:
        """Return an ``nn.Module`` that normalises raw ``[0, 1]`` images
        to the range expected by this teacher.  Called once during
        alignment construction — the returned module is stored on the
        alignment, not on the teacher."""
        return None

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` normalised image → ``(B, L, D)`` patch tokens.

        Prefix tokens (CLS, register, …) must be stripped before return.
        Resize to the teacher's expected resolution internally.
        """
        ...

    # ------------------------------------------------------------------
    # Attention distillation (optional — for HASTE and similar)
    # ------------------------------------------------------------------

    @property
    def supports_attention(self) -> bool:
        return False

    @property
    def num_blocks(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not support attention distillation."
        )

    def get_qkv_module(self, block_idx: int) -> nn.Module:
        """Return the module to hook for attention-logit capture."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support attention distillation."
        )

    def get_num_heads(self, block_idx: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not support attention distillation."
        )

    def get_head_dim(self, block_idx: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not support attention distillation."
        )

    def make_attn_hook(self, block_idx: int):
        """Return a callable hook with ``.logits`` and ``.clear()``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support attention distillation."
        )
