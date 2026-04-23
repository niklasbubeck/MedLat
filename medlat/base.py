"""
Abstract base classes that define the contracts for first-stage and generator models.
All registered models should inherit from the appropriate base class.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable
import torch
import torch.nn as nn

__all__ = [
    "FirstStageModel",
    "ContinuousFirstStage",
    "DiscreteFirstStage",
    "TokenFirstStage",
    "GeneratorModel",
    "AutoregressiveGenerator",
    "NonAutoregressiveGenerator",
    "GenerativeScheduler",
]


@runtime_checkable
class GenerativeScheduler(Protocol):
    """Protocol for generative noise schedulers (diffusion, flow matching, etc.).

    Any class that implements ``training_losses`` and ``p_sample_loop`` with the
    signatures below satisfies this protocol and can be used interchangeably via
    ``create_scheduler()``.

    **Sampler tokens by paradigm:**

    Gaussian diffusion — ``create_scheduler("diffusion", ...)``:
    ``sampler="ddpm"`` stochastic ancestral sampling (default),
    ``sampler="ddim"`` deterministic DDIM (pass ``eta=0.0`` for fully
    deterministic, or ``eta>0`` to re-introduce noise).

    Flow matching — ``create_scheduler("flow", ...)``:
    ``sampler="dopri5"`` adaptive Dormand-Prince ODE solver (default),
    ``sampler="euler"`` fixed-step Euler,
    ``sampler="heun"`` fixed-step Heun (2nd-order).
    """

    def training_losses(
        self,
        model: Any,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss for one batch.

        Returns a dict that always contains ``"loss"`` and ``"mse"``.
        Additional keys (e.g. ``"vb"``, ``"cos_loss"``) may be present
        depending on the scheduler configuration.
        """
        ...

    def p_sample_loop(
        self,
        model: Any,
        shape: Tuple,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        sampler: str = "default",
    ) -> torch.Tensor:
        """Generate a batch of samples from noise.

        Args:
            model: the generative model (nn.Module or callable).
            shape: desired output shape, e.g. ``(B, C, H, W)``.
            noise: optional starting noise tensor; drawn from N(0,I) if None.
            model_kwargs: extra kwargs forwarded to the model (e.g. conditioning).
            device: target device; inferred automatically if None.
            progress: show a tqdm progress bar where supported.
            sampler: sampler token (see class docstring for valid values per
                paradigm). Extra sampler-specific kwargs can be passed as
                ``**kwargs`` on concrete implementations.

        Returns:
            Tensor of shape ``shape`` containing the generated samples.
        """
        ...


class FirstStageModel(nn.Module, ABC):
    """Base class for all first-stage (tokenizer/autoencoder) models."""

    @property
    @abstractmethod
    def vae_stride(self) -> Tuple[int, ...]:
        """Spatial downsampling factor as a tuple, e.g. (8, 8) for f8."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the latent / token embedding."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> Any:
        """Encode input image to latent representation."""

    @abstractmethod
    def decode(self, z: Any) -> torch.Tensor:
        """Decode latent back to image space."""


class ContinuousFirstStage(FirstStageModel, ABC):
    """First-stage model with a continuous (e.g. Gaussian) latent space.

    ``encode(x)`` must return a 3-tuple ``(z, loss, extra)`` where ``z`` is the
    latent tensor of shape ``(B, C, H', W')``, ``loss`` is a scalar auxiliary
    regularisation loss (KL, perceptual, …), and ``extra`` is any
    implementation-specific metadata (may be ``None``).

    ``decode(z)`` must return a ``(B, C_in, H, W)`` float tensor — the reconstructed
    image.
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Number of latent channels (z_channels)."""


class DiscreteFirstStage(FirstStageModel, ABC):
    """First-stage model with a discrete (codebook) latent space.

    ``encode(x)`` must return a 3-tuple ``(quant, loss, indices)`` where
    ``quant`` is the quantized feature map ``(B, D, H', W')``, ``loss`` is a
    scalar codebook commitment loss, and ``indices`` is a ``Tensor`` of flat
    codebook indices ``(B, H'*W')`` for single-stage quantizers, or a
    ``List[Tensor]`` of per-stage indices for residual / multi-level
    quantizers.

    ``decode(quant)`` must return a ``(B, C_in, H, W)`` float tensor.

    ``decode_code(indices, out_shape)`` must return a ``(B, C_in, H, W)`` float tensor.

    ``encode_to_prequant(x)`` must return ``(h, None, None)`` where ``h`` is the
    pre-quantization feature map ``(B, D, H', W')``.

    ``decode_from_prequant(h)`` quantizes ``h`` and returns a ``(B, C_in, H, W)``
    tensor.
    """

    @property
    @abstractmethod
    def n_embed(self) -> int:
        """Codebook size."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension per codebook entry."""

    @abstractmethod
    def encode_to_prequant(self, x: torch.Tensor) -> Any:
        """Encode to pre-quantization features."""

    @abstractmethod
    def decode_from_prequant(self, h: torch.Tensor) -> torch.Tensor:
        """Quantize and decode from pre-quantization features."""

    @abstractmethod
    def decode_code(self, indices: torch.Tensor, out_shape: Optional[Tuple] = None) -> torch.Tensor:
        """Decode from codebook indices."""


class TokenFirstStage(FirstStageModel, ABC):
    """
    Learned tokenizer (TiTok, MAETok, VMAE, ViTA, etc.).
    Discrete codebook but different interface from VQ-VAE.
    """

    @property
    @abstractmethod
    def n_embed(self) -> int:
        """Codebook size."""

    @abstractmethod
    def decode_code(self, indices: torch.Tensor, out_shape: Optional[Tuple] = None) -> torch.Tensor:
        """Decode from codebook indices."""


class GeneratorModel(nn.Module, ABC):
    """Base class for all generator models."""

    @property
    def vae_stride(self) -> Optional[Tuple[int, ...]]:
        """Expected spatial downsampling of the first-stage model. None if unconstrained."""
        return None


class AutoregressiveGenerator(GeneratorModel, ABC):
    """Generator that produces outputs token-by-token or in masked fashion."""


class NonAutoregressiveGenerator(GeneratorModel, ABC):
    """Generator based on diffusion / score-matching over continuous latents."""
