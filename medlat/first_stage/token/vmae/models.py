"""
VMAE — Variational Masked AutoEncoder.

Implementation of "Latent Diffusion Models with Masked AutoEncoders"
(Lee et al. 2025, https://arxiv.org/abs/2507.09984): a ViT-based MAE
tokenizer with a KL-regularized variational bottleneck.

Implemented as a thin specialization of
:class:`~medlat.first_stage.continuous.vae_models.AutoencoderKLTransformer`
— all the VAE machinery (posterior sampling, diagonal Gaussian, KL against
``N(0, I)``, pre/post-quant projections, alignment plumbing, metric
mixin) is inherited. This file only wires up:

* :class:`VMAEEncoder` — a :class:`~medlat.modules.vit_core.GenericViTEncoder`
  configured for MAE-style random masking during training and no-masking at
  eval. Returns ``(tokens, aux)`` with ``aux["ids_restore"]`` so the base
  class's ``_run_encoder`` can stash it for decoder consumption.
* :class:`VMAEDecoder` — a :class:`~medlat.modules.vit_core.GenericViTDecoder`
  that restores the full token grid from visible tokens via
  ``ids_restore`` and reconstructs pixels with a linear head.
* :class:`VariationalMaskedAutoencoder` — the top-level class. Overrides
  :meth:`forward` so it returns **only the reconstruction**; the KL
  regularizer is computed internally and published via
  :meth:`log_metric` so external training loops can pull it through
  :meth:`get_metrics` without the model having to own the total loss.

Dependencies: ``torch``, ``einops``, and MedLat's own VAE + ViT modules. No
``timm``, no ``diffusers``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from medlat.first_stage.continuous.vae_models import AutoencoderKLTransformer
from medlat.modules.alignments import AlignmentModule
from medlat.modules.vit_core import GenericViTDecoder, GenericViTEncoder
from medlat.utils import init_from_ckpt

logger = logging.getLogger(__name__)

__all__ = [
    "VariationalMaskedAutoencoder",
    "MaskedAutoencoderViT",   # backward-compat alias
    "VMAEEncoder",
    "VMAEDecoder",
]


# ---------------------------------------------------------------------------
# Encoder / decoder — thin specializations of the generic ViT scaffolding.
# ---------------------------------------------------------------------------


class VMAEEncoder(GenericViTEncoder):
    """ViT encoder with MAE masking during training, no-mask at eval.

    Configured so that :class:`AutoencoderKLTransformer` can pick it up
    unchanged: ``pos_type="sincos"``, ``use_rope=False``, no prefix/latent
    tokens, and ``masking="mae_random"`` with a fixed ratio. The forward
    signature matches ``GenericViTEncoder`` — returns ``(tokens, aux)`` so
    the base class's ``_run_encoder`` can stash ``aux["ids_restore"]`` on
    ``self._last_aux`` for the decoder.
    """

    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        mask_ratio: float,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pos_type="sincos",      # paper uses fixed sin-cos positions
            use_rope=False,
            num_prefix_tokens=0,    # no CLS (paper omits it)
            num_latent_tokens=0,    # MAE-style, not TiTok-style
            masking="mae_random",
            # Paper uses a fixed mask ratio. We give the generator a tiny
            # non-zero spread to keep scipy.truncnorm happy, but bypass the
            # generator entirely in :meth:`forward` — our override always
            # supplies a deterministic ratio so the generator is never used.
            mask_ratio_mu=mask_ratio,
            mask_ratio_std=0.05,
            mask_ratio_max=min(mask_ratio + 0.05, 1.0 - 1e-3),
            mask_ratio_min=max(mask_ratio - 0.05, 1e-3),
            # Required by AutoencoderKLTransformer: encoder output must
            # carry 2*z_channels channels so that quant_conv can split
            # them into (mean, logvar) for DiagonalGaussianDistribution.
            double_z=True,
        )
        self._mask_ratio = float(mask_ratio)

    def forward(
        self, x: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, dict]:
        # Use our fixed mask ratio during training, no masking at eval.
        # Bypass the stochastic ratio generator — the paper uses a fixed
        # ratio and we get byte-deterministic behavior by never sampling.
        if mask_ratio is None:
            mask_ratio = self._mask_ratio if self.training else 0.0
        return super().forward(x, mask_ratio=mask_ratio)


class VMAEDecoder(GenericViTDecoder):
    """ViT decoder that restores masked tokens and reconstructs pixels.

    Configured so that :class:`AutoencoderKLTransformer._run_decoder` can
    invoke it as ``self.decoder(z, ids_restore=...)`` without any
    transformer-specific surface leaking out of the public API.
    """

    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        out_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        token_dim: int,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pos_type="sincos",
            use_rope=False,
            num_prefix_tokens=0,
            to_pixel="linear",      # linear pixel head, faithful to paper
            token_dim=token_dim,     # matches AutoencoderKLTransformer's post_quant_conv output
        )


# ---------------------------------------------------------------------------
# Main VMAE class — AutoencoderKLTransformer specialized with our ViT pair.
# ---------------------------------------------------------------------------


class VariationalMaskedAutoencoder(AutoencoderKLTransformer):
    """Paper-faithful VMAE tokenizer built on :class:`AutoencoderKLTransformer`.

    All the VAE machinery (posterior sampling, diagonal Gaussian, KL, pre-
    and post-quant linear projections, alignment composition, metric mixin,
    checkpoint loading) is inherited. VMAE only adds:

    1. A :class:`VMAEEncoder` / :class:`VMAEDecoder` pair wired for MAE-style
       random masking during training (no masking at eval).
    2. Optional MLP pre/post-quant layers (``down_nonlinear=True``) that
       mirror the original VMAE codebase's non-linear latent projections.
    3. A :meth:`forward` override that returns **only the reconstruction**;
       the KL regularizer is published via :meth:`log_metric` under the key
       ``"loss_kl"`` so external training loops can retrieve it through
       :meth:`get_metrics` alongside any other module-level metrics.

    Legacy kwargs from the original VMAE codebase that never had any effect
    on the registered variants (``ldmae_mode``, ``gradual_resol``,
    ``smooth_output``, ``no_cls``, ``finetune_downsample_layer``,
    ``fixed_std``, ``pred_with_conv``, ``scaling_factor``,
    ``norm_pix_loss``, ``perceptual_loss_ratio``, ``perceptual_loss``) are
    absorbed via ``**_unused_kwargs`` and logged once at construction time.
    """

    def __init__(
        self,
        *,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 12,
        decoder_num_heads: int = 12,
        mlp_ratio: float = 4.0,
        latent_dim: int = 32,
        mask_ratio: float = 0.6,
        down_nonlinear: bool = False,
        kl_weight: float = 1e-6,
        alignment: Optional[AlignmentModule] = None,
        ckpt_path: Optional[str] = None,
        **_unused_kwargs: Any,
    ) -> None:
        if _unused_kwargs:
            logger.warning(
                "VariationalMaskedAutoencoder received unused legacy kwargs "
                "(ignored): %s. These parameters were silently no-ops in the "
                "previous implementation as well.",
                sorted(_unused_kwargs),
            )
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")

        encoder = VMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
        )
        decoder = VMAEDecoder(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=in_channels,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            # post_quant_conv produces encoder_z_channels = embed_dim;
            # the decoder's own decoder_embed then projects to
            # decoder_embed_dim if they differ.
            token_dim=embed_dim,
        )
        # ``pre_post_kwargs`` are forwarded to ``_build_pre_post_layers`` by
        # AutoencoderKLBase — we add our own ``down_nonlinear`` flag.
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            alignment=alignment,
            embed_dim=latent_dim,
            kl_weight=kl_weight,
            pre_post_layer="linear",
            double_z=True,
            channel_dim=2,           # (B, N, C) token layout
            ckpt_path=None,           # defer to end of our own __init__
        )
        self.mask_ratio = float(mask_ratio)
        self.latent_dim = int(latent_dim)

        # Optionally swap the linear pre/post-quant layers for 2-layer MLPs.
        if down_nonlinear:
            hidden = max(latent_dim, embed_dim) * 4
            self.quant_conv = nn.Sequential(
                nn.Linear(2 * embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * latent_dim),
            )
            self.post_quant_conv = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, embed_dim),
            )

        # Load the checkpoint only after every submodule (including the
        # optional MLP swap) exists, so weights land on the correct shapes.
        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

    # ------------------------------------------------------------------
    # Forward — reconstruction only; KL surfaces through get_metrics().
    # ------------------------------------------------------------------

    def forward(
        self,
        input: torch.Tensor,
        sample_posterior: bool = True,
    ) -> torch.Tensor:
        """Encode → sample → decode. Returns the reconstruction only.

        The KL regularizer (already weighted by :attr:`kl_weight`) is
        computed internally and published via
        :meth:`~medlat.modules.metrics.MetricLoggerMixin.log_metric` under
        the key ``"loss_kl"``. External training loops read it via
        :meth:`get_metrics` alongside any alignment-module or user-logged
        metrics.
        """
        posterior = self.get_posterior(input)
        kl_loss = self.p_loss(posterior, input.device)
        self.log_metric("loss_kl", kl_loss.detach())

        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)

        if self.alignment is not None:
            # Alignment losses are also purely logged; the training loop
            # picks them up from get_metrics().
            alignment_loss, _ = self.alignment(z, input)
            self.log_metric("loss_alignment", alignment_loss.detach())

        return dec


# ---------------------------------------------------------------------------
# Backward-compat alias — register.py imports MaskedAutoencoderViT by name.
# ---------------------------------------------------------------------------

MaskedAutoencoderViT = VariationalMaskedAutoencoder
