"""
KL-regularised continuous autoencoders implementing :class:`ContinuousFirstStage`.

Two concrete models share the same encode / posterior / decode plumbing; their
only real differences are (a) whether the pre/post-quant layer is a conv or a
``nn.Linear`` and (b) whether the encoder emits a transformer-style ``aux``
dict (``ids_restore``, …) that the decoder needs to invert masking.

The common plumbing lives on :class:`AutoencoderKLBase`. Subclasses override
two small hooks:

* :meth:`AutoencoderKLBase._build_pre_post_layers` — constructs the pre- and
  post-quant submodules.
* :meth:`AutoencoderKLBase._run_encoder` / :meth:`AutoencoderKLBase._run_decoder` —
  override only if the encoder emits auxiliary state that the decoder has to
  consume. :class:`AutoencoderKLTransformer` stashes that state on
  ``self._last_aux`` during encode and reads it in decode, so neither the
  public return shape nor the public decode signature has to carry
  transformer-specific arguments.
"""
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from medlat.base import ContinuousFirstStage
from medlat.first_stage.continuous.modules.ldm_modules import get_conv_layer
from medlat.first_stage.modules.gaussian_dist import (
    DiagonalGaussianDistribution,
    _DeterministicPosterior,
)
from medlat.modules.alignments import AlignmentModule
from medlat.modules.metrics import MetricLoggerMixin
from medlat.utils import init_from_ckpt

logger = logging.getLogger(__name__)

__all__ = ["AutoencoderKL", "AutoencoderKLTransformer"]


Posterior = Union[DiagonalGaussianDistribution, _DeterministicPosterior]


class AutoencoderKLBase(MetricLoggerMixin, ContinuousFirstStage):
    """Shared plumbing for KL-regularised continuous autoencoders.

    Concrete subclasses supply the pre/post-quant layer construction via
    :meth:`_build_pre_post_layers` and, when the encoder returns auxiliary
    state (e.g. masking info), override :meth:`_run_encoder` and
    :meth:`_run_decoder` to route that state through without leaking it into
    the public interface.

    Observability
    -------------
    Inherits ``log_metric`` / ``get_metrics`` / ``reset_metrics`` from
    :class:`MetricLoggerMixin`. Each submodule owns its own logging: the
    alignment module (when configured) publishes ``"alignment_loss"`` from
    its own forward. :meth:`get_metrics` merges that into the same flat dict
    so training loops can just ``wandb.log(model.get_metrics())``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        alignment: Optional[AlignmentModule] = None,
        embed_dim: Optional[int] = None,
        kl_weight: float = 1e-6,
        double_z: bool = True,
        channel_dim: int = 1,
        ckpt_path: Optional[str] = None,
        **pre_post_kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.alignment = alignment
        self.double_z = double_z
        self.channel_dim = channel_dim
        self.kl_weight = kl_weight

        # ── Encoder metadata ──
        self.encoder_z_channels: int = self._require_encoder_attr("z_channels")
        self._vae_stride = self._require_encoder_attr("vae_stride")

        # Embedding dim defaults to the encoder's z_channels if omitted.
        self._embed_dim: int = (
            embed_dim if embed_dim is not None else self.encoder_z_channels
        )

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

    def _run_decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Decode ``z`` back to image space. Override for transformer-style
        decoders that need ``ids_restore`` or other encoder-produced state."""
        return self.decoder(z)

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
    # ContinuousFirstStage contract
    # ------------------------------------------------------------------

    @property
    def vae_stride(self) -> Any:
        return self._vae_stride

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    # ------------------------------------------------------------------
    # Posterior + encode / decode
    # ------------------------------------------------------------------

    def get_posterior(self, x: torch.Tensor) -> Posterior:
        """Return the latent posterior distribution for ``x``.

        When ``double_z`` is ``False`` the encoder is expected to output the
        latent ``z`` directly (e.g. a pretrained DCAE) — we wrap it in a
        :class:`_DeterministicPosterior` so callers can keep a uniform
        ``posterior.sample() / posterior.mode()`` interface.
        """
        h = self._run_encoder(x)
        moments = self.quant_conv(h)
        if not self.double_z:
            return _DeterministicPosterior(moments)
        return DiagonalGaussianDistribution(moments, channel_dim=self.channel_dim)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Encode ``x`` and sample from the posterior.

        Returns ``(z, loss, None)`` as mandated by
        :class:`ContinuousFirstStage`. ``loss`` is the KL regularisation
        term weighted by :attr:`kl_weight`. The trailing ``None`` is the
        implementation-specific ``extra`` slot — not used here (any
        transformer aux state lives on :attr:`_last_aux` and is consumed by
        :meth:`decode`).
        """
        posterior = self.get_posterior(x)
        loss = self.p_loss(posterior, x.device)
        return posterior.sample(), loss, None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Run the post-quant layer and the decoder."""
        z = self.post_quant_conv(z)
        return self._run_decoder(z)

    # ------------------------------------------------------------------
    # KL loss
    # ------------------------------------------------------------------

    def p_loss(self, posterior: Posterior, device: torch.device) -> torch.Tensor:
        """Batch-mean KL divergence of ``posterior`` against ``N(0, I)``, scaled by :attr:`kl_weight`.

        Returns a scalar zero on the correct device when the posterior is
        deterministic (has no ``kl`` method), so callers can add it
        unconditionally.
        """
        kl_loss = torch.zeros((), device=device)
        if posterior is not None and hasattr(posterior, "kl"):
            kl = posterior.kl()
            kl_loss = kl.sum() / kl.shape[0]
        return self.kl_weight * kl_loss

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input: torch.Tensor,
        sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode, sample (or take mode), decode, and return ``(dec, loss)``.

        ``alignment`` (if configured) is applied here, inside ``forward`` only.
        Callers that go through ``encode`` / ``decode`` separately will not
        see the alignment contribution — that is intentional since alignment
        depends on having both the encoded ``z`` and the original input in
        scope at the same time.
        """
        posterior = self.get_posterior(input)
        loss = self.p_loss(posterior, input.device) if self.double_z else torch.zeros((), device=input.device)

        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)

        if self.alignment is not None:
            alignment_loss, _ = self.alignment(z, input)
            loss = loss + alignment_loss

        return dec, loss

    def get_metrics(self) -> Dict[str, Any]:
        """Merge model- and alignment-level metrics into a single flat snapshot.

        The alignment module (when configured) publishes ``"alignment_loss"``
        and any subclass-specific keys from its own forward. AutoencoderKL
        doesn't log anything itself by default; this override just
        aggregates so training loops can ``wandb.log(model.get_metrics())``
        without caring where each number came from.

        Precedence on key collisions: model-level (user ``log_metric`` calls)
        > alignment.
        """
        merged = super().get_metrics()  # model-level via MetricLoggerMixin
        if self.alignment is not None and hasattr(self.alignment, "get_metrics"):
            for k, v in self.alignment.get_metrics().items():
                merged.setdefault(k, v)
        return merged


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


class AutoencoderKL(AutoencoderKLBase):
    """KL-regularised continuous autoencoder with conv / linear pre/post-quant layers.

    Accepts three choices for ``pre_post_layer``:

    * ``"conv"`` — a 1×1 convolution (or 1×1×1 for 3D inputs). The conv rank is
      resolved from ``encoder.dims`` via :func:`get_conv_layer`.
    * ``"linear"`` — a plain ``nn.Linear``.
    * ``"none"`` — no projection (``nn.Identity``).

    Default ``channel_dim=1`` matches ``(B, C, H, W)`` layouts.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        alignment: Optional[AlignmentModule] = None,
        embed_dim: Optional[int] = None,
        kl_weight: float = 1e-6,
        pre_post_layer: str = "conv",
        double_z: bool = True,
        channel_dim: int = 1,
        ckpt_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            alignment=alignment,
            embed_dim=embed_dim,
            kl_weight=kl_weight,
            double_z=double_z,
            channel_dim=channel_dim,
            ckpt_path=ckpt_path,
            pre_post_layer=pre_post_layer,
        )

    def _build_pre_post_layers(
        self, pre_post_layer: str = "conv"
    ) -> Tuple[nn.Module, nn.Module]:
        if pre_post_layer == "conv":
            # get_conv_layer needs the encoder's spatial rank.
            dims = self._require_encoder_attr("dims")
            conv_layer = get_conv_layer(dims)
            return (
                conv_layer(2 * self.encoder_z_channels, 2 * self.embed_dim, 1),
                conv_layer(self.embed_dim, self.encoder_z_channels, 1),
            )
        if pre_post_layer == "linear":
            return (
                nn.Linear(2 * self.encoder_z_channels, 2 * self.embed_dim),
                nn.Linear(self.embed_dim, self.encoder_z_channels),
            )
        if pre_post_layer == "none":
            return nn.Identity(), nn.Identity()
        raise ValueError(
            f"Invalid pre_post_layer={pre_post_layer!r}; expected "
            f"'conv', 'linear', or 'none'."
        )


class AutoencoderKLTransformer(AutoencoderKLBase):
    """KL-regularised autoencoder with a ViT-style encoder / decoder pair.

    The encoder is expected to return ``(features, aux)`` where ``aux`` is a
    dict containing at least ``ids_restore`` for masking inversion. We stash
    that aux state on ``self._last_aux`` during :meth:`_run_encoder` and
    consume it in :meth:`_run_decoder`, so the public interface matches
    :class:`AutoencoderKL` exactly — no transformer-specific kwargs leak out
    of encode / decode / forward.

    Default ``channel_dim=2`` matches ``(B, N, C)`` token layouts.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        alignment: Optional[AlignmentModule] = None,
        embed_dim: Optional[int] = None,
        kl_weight: float = 1e-6,
        pre_post_layer: str = "linear",
        double_z: bool = True,
        channel_dim: int = 2,
        ckpt_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            alignment=alignment,
            embed_dim=embed_dim,
            kl_weight=kl_weight,
            double_z=double_z,
            channel_dim=channel_dim,
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
                nn.Linear(2 * self.encoder_z_channels, 2 * self.embed_dim),
                nn.Linear(self.embed_dim, self.encoder_z_channels),
            )
        if pre_post_layer == "none":
            return nn.Identity(), nn.Identity()
        raise ValueError(
            f"Invalid pre_post_layer={pre_post_layer!r}; expected "
            f"'linear' or 'none'."
        )

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        h, aux = self.encoder(x)
        self._last_aux = aux
        return h

    def _run_decoder(self, z: torch.Tensor) -> torch.Tensor:
        ids_restore = None
        if self._last_aux is not None:
            ids_restore = self._last_aux.get("ids_restore")
        return self.decoder(z, ids_restore=ids_restore)
