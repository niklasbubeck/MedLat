# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
"""
VQ-family autoencoders implementing :class:`DiscreteFirstStage`.

Two concrete models share the same encode / quantize / decode plumbing; their
only real differences are (a) whether the pre/post-quant layer is a conv or a
``nn.Linear`` and (b) whether the encoder emits a transformer-style ``aux``
dict (``ids_restore``, …) that the decoder needs to invert masking.

The common plumbing lives on :class:`VQModelBase`. Subclasses override two
small hooks:

* :meth:`VQModelBase._build_pre_post_layers` — constructs the pre- and
  post-quant submodules.
* :meth:`VQModelBase._run_encoder` / :meth:`VQModelBase._run_decoder` —
  override only if the encoder emits auxiliary state that the decoder has
  to consume. :class:`VQModelTransformer` stashes that state on
  ``self._last_aux`` during encode and reads it in decode, so neither the
  public return shape nor the public decode signature has to carry
  transformer-specific arguments.
"""
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from medlat.base import DiscreteFirstStage
from medlat.first_stage.discrete.modules.ldm_modules import get_conv_layer
from medlat.first_stage.discrete.quantizer.quantize import (
    unflatten_spatial_to_channel_first,
)
from medlat.modules.alignments import AlignmentModule
from medlat.modules.metrics import MetricLoggerMixin
from medlat.utils import init_from_ckpt

__all__ = ["VQModel", "VQModelTransformer"]


class VQModelBase(MetricLoggerMixin, DiscreteFirstStage):
    """Shared plumbing for VQ-family autoencoders.

    Concrete subclasses supply the pre/post-quant layer construction via
    :meth:`_build_pre_post_layers` and, when the encoder returns auxiliary
    state (e.g. masking info), override :meth:`_run_encoder` and
    :meth:`_run_decoder` to route that state through without leaking it into
    the public interface.

    Observability
    -------------
    Inherits ``log_metric`` / ``get_metrics`` / ``reset_metrics`` from
    :class:`MetricLoggerMixin`. Each submodule owns its own logging: the
    quantizer publishes ``"perplexity"`` / ``"dead_code_ratio"`` / etc. from
    its post-forward hook, and the alignment module (when configured)
    publishes ``"alignment_loss"`` from its own forward. This class's
    :meth:`get_metrics` override aggregates all three levels (model-, quantizer-,
    alignment-) into a single flat snapshot so training loops can just
    ``wandb.log(model.get_metrics())``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        alignment: Optional[AlignmentModule] = None,
        ckpt_path: Optional[str] = None,
        **pre_post_kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.alignment = alignment

        # ── Codebook / spatial metadata (read from encoder + quantizer) ──
        self._embed_dim: int = quantizer.e_dim
        self._n_embed: int = quantizer.n_e
        self.z_channels: int = self._require_encoder_attr("z_channels")
        self.dims: int = self._require_encoder_attr("dims")
        self._vae_stride = getattr(encoder, "vae_stride", None)

        # ── Pre/post-quant layers are subclass-specific ──
        self.quant_conv, self.post_quant_conv = self._build_pre_post_layers(
            **pre_post_kwargs
        )

        # ── Optional checkpoint load (after all submodules exist) ──
        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_pre_post_layers(
        self, **kwargs: Any
    ) -> Tuple[nn.Module, nn.Module]:
        """Return ``(quant_conv, post_quant_conv)``. Must be overridden."""
        raise NotImplementedError

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Pass ``x`` through the encoder and return the feature map.

        Override in subclasses whose encoder returns auxiliary state the
        decoder needs — the convention is to stash such state on
        ``self._last_aux`` and return just the feature map here.
        """
        return self.encoder(x)

    def _run_decoder(self, quant: torch.Tensor) -> torch.Tensor:
        """Decode ``quant`` back to image space. Override for transformer-style
        decoders that need ``ids_restore`` or other encoder-produced state."""
        return self.decoder(quant)

    def _require_encoder_attr(self, name: str) -> Any:
        """Fail fast if the encoder doesn't declare a required attribute."""
        value = getattr(self.encoder, name, None)
        if value is None:
            raise ValueError(
                f"Encoder {type(self.encoder).__name__} must declare `{name}` "
                f"— {type(self).__name__} needs it at construction time."
            )
        return value

    # ------------------------------------------------------------------
    # DiscreteFirstStage contract
    # ------------------------------------------------------------------

    @property
    def vae_stride(self) -> Any:
        return self._vae_stride

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def n_embed(self) -> int:
        return self._n_embed

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Encode ``x`` and quantize. Returns ``(quant, loss, indices)``."""
        h = self._run_encoder(x)
        h = self.quant_conv(h)
        return self.quantizer(h)

    def quantize(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Run the quantizer on pre-quant features ``h``."""
        return self.quantizer(h)

    def encode_to_prequant(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """Encode ``x`` up to the pre-quant feature map ``h``.

        The two trailing ``None`` slots are there to match the first-stage
        contract shape (so callers can unpack uniformly). Any encoder-produced
        auxiliary state is stashed on ``self._last_aux`` by :meth:`_run_encoder`
        and consumed later by :meth:`decode`.
        """
        h = self._run_encoder(x)
        h = self.quant_conv(h)
        return h, None, None

    def decode_from_prequant(self, h: torch.Tensor) -> torch.Tensor:
        """Quantize pre-quant features ``h`` and decode back to image space."""
        quant, _, _ = self.quantizer(h)
        return self.decode(quant)

    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Run the post-quant layer and the decoder."""
        quant = self.post_quant_conv(quant)
        return self._run_decoder(quant)

    def decode_code(
        self,
        code_b: torch.Tensor,
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Decode a batch of codebook indices back to image space."""
        quant_b = self.quantizer.get_codebook_entry(code_b, shape=out_shape)
        # ``get_codebook_entry`` returns channel-last; shared helper inverts it.
        quant_b = unflatten_spatial_to_channel_first(quant_b, contiguous=True)
        return self.decode(quant_b)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input: torch.Tensor,
        return_pred_indices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Encode, decode, and optionally thread codebook indices out.

        ``alignment`` (if configured) is applied here, inside ``forward`` only.
        Callers that go through ``encode`` / ``decode`` separately will not see
        the alignment contribution — that is intentional since alignment
        depends on having both the encoded quant and the original input in
        scope at the same time.
        """
        quant, diff, ind = self.encode(input)
        dec = self.decode(quant)

        if self.alignment is not None:
            alignment_loss, _ = self.alignment(quant, input)
            diff = diff + alignment_loss

        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_metrics(self) -> Dict[str, Any]:
        """Merge model-, quantizer-, and alignment-level metrics into one dict.

        Each submodule owns its own logging: the quantizer publishes
        ``"perplexity"`` / ``"dead_code_ratio"`` / etc. via its post-forward
        hook, and the alignment module (when configured) publishes
        ``"alignment_loss"`` from its own forward. VQModelBase doesn't log
        anything itself by default; this method just aggregates so training
        loops can call ``wandb.log(model.get_metrics())`` without having to
        know which submodule each number came from.

        Precedence on key collisions: model-level (user ``log_metric`` calls)
        > quantizer > alignment.
        """
        merged = super().get_metrics()  # model-level via MetricLoggerMixin
        for submodule in (self.quantizer, self.alignment):
            if submodule is None or not hasattr(submodule, "get_metrics"):
                continue
            for k, v in submodule.get_metrics().items():
                merged.setdefault(k, v)
        return merged


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class VQModel(VQModelBase):
    """VQ-VAE / VQ-GAN autoencoder with conv pre/post-quant layers.

    Supports both 2D and 3D spatial inputs — the conv layer type is resolved
    from ``encoder.dims`` via :func:`get_conv_layer`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        alignment: Optional[AlignmentModule] = None,
        ckpt_path: Optional[str] = None,
        quant_conv_ks: int = 1,   # VQ-VAE / VQ-GAN use 1; VAR uses 3
        pre_post_layer: str = "conv",
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            alignment=alignment,
            ckpt_path=ckpt_path,
            quant_conv_ks=quant_conv_ks,
            pre_post_layer=pre_post_layer,
        )

    def _build_pre_post_layers(
        self, quant_conv_ks: int = 1, pre_post_layer: str = "conv"
    ) -> Tuple[nn.Module, nn.Module]:
        if pre_post_layer == "conv":
            conv_layer = get_conv_layer(self.dims)
            pad = quant_conv_ks // 2
            return (
                conv_layer(
                    self.z_channels, self.embed_dim,
                    quant_conv_ks, stride=1, padding=pad,
                ),
                conv_layer(
                    self.embed_dim, self.z_channels,
                    quant_conv_ks, stride=1, padding=pad,
                ),
            )
        if pre_post_layer == "none":
            return nn.Identity(), nn.Identity()
        raise ValueError(
            f"Invalid pre_post_layer={pre_post_layer!r}; expected 'conv' or 'none'."
        )


class VQModelTransformer(VQModelBase):
    """VQ autoencoder with a ViT-style encoder / decoder pair.

    The encoder is expected to return ``(features, aux)`` where ``aux`` is a
    dict containing at least ``ids_restore`` for masking inversion. We stash
    that aux state on ``self._last_aux`` during :meth:`_run_encoder` and
    consume it in :meth:`_run_decoder`, so the public interface matches
    :class:`VQModel` exactly — no transformer-specific kwargs leak out.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        alignment: Optional[AlignmentModule] = None,
        ckpt_path: Optional[str] = None,
        pre_post_layer: str = "linear",
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            alignment=alignment,
            ckpt_path=ckpt_path,
            pre_post_layer=pre_post_layer,
        )
        # Encoder-produced state consumed by the decoder; populated in
        # _run_encoder on every forward pass.
        self._last_aux: Optional[Dict[str, torch.Tensor]] = None

    def _build_pre_post_layers(
        self, pre_post_layer: str = "linear"
    ) -> Tuple[nn.Module, nn.Module]:
        if pre_post_layer == "linear":
            return (
                nn.Linear(self.z_channels, self._embed_dim),
                nn.Linear(self._embed_dim, self.z_channels),
            )
        if pre_post_layer == "none":
            return nn.Identity(), nn.Identity()
        raise ValueError(
            f"Invalid pre_post_layer={pre_post_layer!r}; expected 'linear' or 'none'."
        )

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        h, aux = self.encoder(x)
        self._last_aux = aux
        return h

    def _run_decoder(self, quant: torch.Tensor) -> torch.Tensor:
        ids_restore = None
        if self._last_aux is not None:
            ids_restore = self._last_aux.get("ids_restore")
        return self.decoder(quant, ids_restore=ids_restore)
