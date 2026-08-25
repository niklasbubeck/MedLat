from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from medlat.modules.metrics import MetricLoggerMixin
from .losses import AlignmentLoss, CosineSimilarityLoss


class AlignmentModule(MetricLoggerMixin, ABC, nn.Module):
    """Almighty ABC for all alignment modules.

    Handles the shared alignment pipeline:

    1. **Input normalisation** — an optional ``input_transform`` (any
       ``nn.Module``) is applied to the raw image before the teacher
       model sees it.
    2. **Composable losses** — ``losses`` is a list of
       ``(AlignmentLoss, weight)`` tuples.  The base ``forward``
       computes the weighted sum and logs each component.
    3. **Spatial alignment** — if the predicted and target token
       sequences have different lengths the base class reshapes them
       to square grids and aligns them using ``spatial_mode``
       (``"bilinear"`` | ``"avgpool"`` | ``"maxpool"``).  Masks are
       always aligned with nearest-neighbour interpolation.

    Subclasses must implement :meth:`compute_target` and
    :meth:`decode_projection`.

    Two intermediate base classes provide the projection logic:

    - :class:`TokenizerAlignment` — for tokenizer/VAE-space alignment
      (Axis A), with an optional decoder pipeline.
    - :class:`GeneratorAlignment` — for generator-space alignment
      (Axis B), with an MLP projection head.
    """

    def __init__(
        self,
        name: str,
        input_transform: Optional[nn.Module] = None,
        spatial_mode: str = "bilinear",
        losses: Optional[List[Tuple[AlignmentLoss, float]]] = None,
    ):
        super().__init__()
        self.name = name
        self.spatial_mode = spatial_mode
        self.input_transform = input_transform

        if losses is None:
            losses = [(CosineSimilarityLoss(), 1.0)]
        self.loss_modules = nn.ModuleList([fn for fn, _ in losses])
        self.loss_weights = [w for _, w in losses]

    @abstractmethod
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        """Return target features from the (already-transformed) input image.

        The result may be ``(B, L, D)`` token sequences or ``(B, C, H, W)``
        feature maps — the base class normalises both to ``(B, L, D)``
        before loss computation.
        """
        ...

    @abstractmethod
    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        """Project input representations to the target feature space."""
        ...

    def ensure_projection_dim(self, target_dim: int):
        """Override in subclasses that need dynamic projection resizing."""
        pass

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        quant: torch.Tensor,
        input_image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_image is None:
            raise ValueError("AlignmentModule requires input_image to compute target features")

        with torch.no_grad():
            target_input = (
                self.input_transform(input_image)
                if self.input_transform is not None
                else input_image
            )
            target = self.compute_target(target_input)

        self.ensure_projection_dim(self._feature_dim(target))

        pred = self.decode_projection(quant)

        pred = self._to_tokens(pred)
        target = self._to_tokens(target)

        if pred.shape[1] != target.shape[1]:
            pred = self._align_tokens(pred, target)
            if mask is not None:
                mask = self._align_mask(mask, target)
        elif mask is not None and mask.shape[1] != pred.shape[1]:
            mask = self._align_mask(mask, pred)

        total_loss = torch.zeros((), device=pred.device)
        for fn, w in zip(self.loss_modules, self.loss_weights):
            component = fn(pred, target, mask)
            self.log_metric(f"alignment/{type(fn).__name__}", component.detach())
            total_loss = total_loss + w * component

        self.log_metric("alignment_loss", total_loss.detach())
        return total_loss, pred

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _feature_dim(x: torch.Tensor) -> int:
        if x.dim() == 3:
            return x.size(-1)
        if x.dim() == 4:
            return x.size(1)
        raise ValueError(f"Expected 3D (B, L, D) or 4D (B, C, H, W), got {x.dim()}D")

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        """Convert to canonical ``(B, L, D)`` token format."""
        if x.dim() == 3:
            return x
        if x.dim() == 4:
            b, c, h, w = x.shape
            return x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        raise ValueError(f"Expected 3D (B, L, D) or 4D (B, C, H, W), got {x.dim()}D")

    def _infer_grid_hw(self, seq_len: int) -> Tuple[int, int]:
        side = int(math.sqrt(seq_len))
        if side * side != seq_len:
            raise ValueError(
                f"Cannot infer square grid from seq_len={seq_len}. "
                "AlignmentModule assumes square token grids. For non-square grids "
                "override _align_tokens in the subclass."
            )
        return side, side

    def _align_tokens(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, lp, c = pred.shape
        lt = target.shape[1]
        ph, pw = self._infer_grid_hw(lp)
        th, tw = self._infer_grid_hw(lt)
        pred_map = pred.view(b, ph, pw, c).permute(0, 3, 1, 2)  # (B, C, H, W)

        if self.spatial_mode == "bilinear":
            pred_map = F.interpolate(pred_map, size=(th, tw), mode='bilinear', align_corners=False)
        elif self.spatial_mode == "avgpool":
            pred_map = F.adaptive_avg_pool2d(pred_map, (th, tw))
        elif self.spatial_mode == "maxpool":
            pred_map = F.adaptive_max_pool2d(pred_map, (th, tw))
        else:
            raise ValueError(f"Unknown spatial_mode '{self.spatial_mode}'")

        return pred_map.permute(0, 2, 3, 1).reshape(b, lt, c)

    def _align_mask(self, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        b, lp, c = mask.shape
        lt = target.shape[1]
        ph, pw = self._infer_grid_hw(lp)
        th, tw = self._infer_grid_hw(lt)
        mask_map = mask.view(b, ph, pw, c).permute(0, 3, 1, 2)
        mask_map = F.interpolate(mask_map, size=(th, tw), mode='nearest')
        return mask_map.permute(0, 2, 3, 1).reshape(b, lt, c)


# ====================================================================== #
# Intermediate base classes
# ====================================================================== #


class TokenizerAlignment(AlignmentModule):
    """Base class for tokenizer-space (Axis A) alignments.

    Adds an optional decoder pipeline:
    ``post_quant_conv → decoder → to_pixel``.

    Without a decoder the quantised tokens are returned as-is from
    :meth:`decode_projection` (subclasses may override).

    **Decoder interface contract**::

        decoder(x, interpolate_zq, H, W, D) -> Tensor  # (B, L, embed_dim)
    """

    def __init__(
        self,
        name: str,
        decoder: Optional[nn.Module] = None,
        codebook_embed_dim: Optional[int] = None,
        target_dim: Optional[int] = None,
        input_transform: Optional[nn.Module] = None,
        spatial_mode: str = "bilinear",
        losses: Optional[List[Tuple[AlignmentLoss, float]]] = None,
    ):
        super().__init__(
            name=name,
            input_transform=input_transform,
            spatial_mode=spatial_mode,
            losses=losses,
        )

        if decoder is not None:
            if codebook_embed_dim is None or target_dim is None:
                raise ValueError(
                    "codebook_embed_dim and target_dim are required "
                    "when a decoder is provided"
                )
            self.decoder = decoder
            self.post_quant_conv = nn.Linear(codebook_embed_dim, decoder.embed_dim)
            self.to_pixel = nn.Linear(decoder.embed_dim, target_dim)
        else:
            self.decoder = None
            self.post_quant_conv = None
            self.to_pixel = None

    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        if self.decoder is not None:
            quant = self._to_tokens(quant)
            x = self.post_quant_conv(quant)
            dec = self.decoder(x, interpolate_zq=None, H=None, W=None, D=None)
            return self.to_pixel(dec)
        return quant

    def ensure_projection_dim(self, target_dim: int):
        if self.to_pixel is not None and self.to_pixel.out_features != target_dim:
            requires_grad = self.to_pixel.weight.requires_grad
            self.to_pixel = nn.Linear(
                self.decoder.embed_dim, target_dim,
            ).to(self.to_pixel.weight.device)
            self.to_pixel.requires_grad_(requires_grad)


class GeneratorAlignment(AlignmentModule):
    """Base class for generator-space (Axis B) alignments.

    Adds an MLP projection head that maps generator hidden states
    to the teacher feature space.
    """

    def __init__(
        self,
        name: str,
        hidden_dim: int,
        target_dim: int,
        input_transform: Optional[nn.Module] = None,
        spatial_mode: str = "bilinear",
        losses: Optional[List[Tuple[AlignmentLoss, float]]] = None,
        proj_depth: int = 2,
    ):
        super().__init__(
            name=name,
            input_transform=input_transform,
            spatial_mode=spatial_mode,
            losses=losses,
        )

        layers = []
        for _ in range(proj_depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, target_dim))
        self.projection = nn.Sequential(*layers)

    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        return self.projection(self._to_tokens(quant))
