"""CustomTeacher — wraps a user-provided model and extraction function."""

from typing import Callable, Optional

import torch
import torch.nn as nn

from .base import TeacherModel


class CustomTeacher(TeacherModel):
    """Teacher wrapping an arbitrary model with a user-supplied extractor.

    Example::

        teacher = CustomTeacher(
            model=my_vit,
            embed_dim=768,
            extract_fn=lambda m, x: m.get_patch_tokens(x),
        )
        align = REPAAlignment(hidden_dim=1152, teacher=teacher)

    Args:
        model: Any ``nn.Module`` — frozen automatically.
        embed_dim: Feature dimension of the extracted tokens.
        extract_fn: ``(model, x) → (B, L, D)`` patch tokens.
        input_transform: Optional normalisation module.
        num_prefix_tokens: Number of prefix tokens (for info only —
            stripping must happen inside ``extract_fn``).
    """

    def __init__(
        self,
        model: nn.Module,
        embed_dim: int,
        extract_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        input_transform: Optional[nn.Module] = None,
        num_prefix_tokens: int = 0,
    ):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self._embed_dim = embed_dim
        self._extract_fn = extract_fn
        self._custom_input_transform = input_transform
        self._num_prefix_tokens = num_prefix_tokens

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def num_prefix_tokens(self):
        return self._num_prefix_tokens

    @property
    def input_transform(self):
        return self._custom_input_transform

    def extract_features(self, x):
        return self._extract_fn(self.model, x)
