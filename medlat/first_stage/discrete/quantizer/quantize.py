"""
Quantizer module for MedLat.

Every concrete class shares a common hierarchy rooted at
:class:`AbstractQuantizer` (or :class:`ResidualQuantizerBase` for multi-level
variants). Alongside the contract that hierarchy formalises, the base also
provides four cross-cutting features that apply to every quantizer: automatic
``@autocast("cuda", enabled=False)`` on every ``forward``; a latest-only
metric sink (:meth:`log_metric` / :meth:`get_metrics`); lazy codebook-usage
tracking with dead-code statistics; a :meth:`revive_dead_codes` hook; and
opt-in MaskGIT-style entropy regularization via
:meth:`entropy_regularization`.

Four shared numerical helpers back the concrete forwards:
:func:`compute_perplexity`, :func:`straight_through_estimator`,
:func:`nearest_codebook_entry_l2`, and
:func:`flatten_spatial_to_channel_last` / inverse.
"""
import logging
import math
import random
import warnings
from abc import ABC, abstractmethod
from functools import partial, wraps
from itertools import zip_longest
from typing import Any, Dict, List, Mapping, Optional, Sequence, Text, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum
from torch.amp import autocast

from medlat.registry import register_model

from medlat.modules.metrics import MetricLoggerMixin

from .modules import *

logger = logging.getLogger(__name__)

__all__ = [
    # Abstract bases (new)
    "AbstractQuantizer",
    "ResidualQuantizerBase",
    # Concrete quantizers (unchanged names)
    "VectorQuantizer",
    "GumbelQuantize",
    "SimpleQINCo",
    "VectorQuantizer2",
    "SimVQ",
    "ResidualQuantizer",
    "MultiScaleResidualQuantizer",
    "MultiScaleResidualQuantizer3D",
    "LookupFreeQuantizer",
    "FiniteScalarQuantizer",
    "BinarySphericalQuantizer",
    "GroupedVQ",
    "QINCo",
    "QincoResidualQuantizer",
    "SoftVectorQuantizer",
    "DiVeQ",
    "SFDiVeQ",
]



_REGISTRY_PREFIX = "discrete.quantizer."

# ---------------------------------------------------------------------------
# Shared numerical helpers
#
# These functions replace formulas that were duplicated across almost every
# concrete quantizer. Each helper has a single call contract and is covered
# by the equivalence tests in ``tests/test_quantize_new_equivalence.py``.
# ---------------------------------------------------------------------------


def compute_perplexity(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Perplexity of a code-usage distribution, ``exp(-Σ p·log(p+ε))``.

    ``probs`` is a 1-D tensor of codebook-usage probabilities (or any
    probability distribution over discrete codes); typically obtained as
    ``one_hot.float().mean(dim=0)`` from a batch of hard assignments. The
    epsilon prevents log-of-zero on unused codes. Equivalent to the
    entropy-based perplexity
    :math:`\\exp(\\mathcal{H}(p))`.
    """
    return torch.exp(-torch.sum(probs * torch.log(probs + eps)))


def flatten_spatial_to_channel_last(
    z: torch.Tensor, contiguous: bool = False
) -> torch.Tensor:
    """Move the channel axis from front to back with 2D/3D dispatch.

    * 4D input ``(B, C, H, W)`` → ``(B, H, W, C)``
    * 5D input ``(B, C, D, H, W)`` → ``(B, D, H, W, C)``
    * Other ranks are returned unchanged (the caller is assumed to have
      already flattened or provided a channel-last tensor).

    Set ``contiguous=True`` to force a memory copy after the permutation —
    some downstream ops (``.view``, some einsums) require contiguous input.
    """
    if z.ndim == 4:
        out = rearrange(z, "b c h w -> b h w c")
    elif z.ndim == 5:
        out = rearrange(z, "b c d h w -> b d h w c")
    else:
        return z
    return out.contiguous() if contiguous else out


def unflatten_spatial_to_channel_first(
    z: torch.Tensor, contiguous: bool = False
) -> torch.Tensor:
    """Inverse of :func:`flatten_spatial_to_channel_last`.

    * 4D input ``(B, H, W, C)`` → ``(B, C, H, W)``
    * 5D input ``(B, D, H, W, C)`` → ``(B, C, D, H, W)``
    * Other ranks returned unchanged.
    """
    if z.ndim == 4:
        out = rearrange(z, "b h w c -> b c h w")
    elif z.ndim == 5:
        out = rearrange(z, "b d h w c -> b c d h w")
    else:
        return z
    return out.contiguous() if contiguous else out


def nearest_codebook_entry_l2(
    z_flat: torch.Tensor,
    codebook: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest codebook entry for each row of ``z_flat`` by L2 distance.

    Expands the squared L2 norm
    :math:`\\|z - e\\|^2 = \\|z\\|^2 + \\|e\\|^2 - 2 z \\cdot e^\\top`
    to avoid materializing the ``(N, K, D)`` difference tensor. Returned
    distances are useful for callers that go on to compute softmax logits,
    entropy losses, or distance-weighted auxiliaries.

    Args:
        z_flat: ``(N, D)`` query vectors.
        codebook: ``(K, D)`` codebook embeddings.

    Returns:
        ``(indices, distances)`` where ``indices`` is a ``(N,)`` int64 tensor
        of nearest-codebook indices and ``distances`` is the ``(N, K)`` matrix
        of squared L2 distances.
    """
    d = (
        z_flat.pow(2).sum(dim=1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * (z_flat @ codebook.t())
    )
    indices = torch.argmin(d, dim=1)
    return indices, d


def straight_through_estimator(
    z: torch.Tensor,
    z_q: torch.Tensor,
    use_rotation_trick: bool = False,
) -> torch.Tensor:
    """Preserve encoder gradients through the quantization step.

    Returns a tensor whose forward value equals ``z_q`` but whose backward
    pass routes gradients to ``z``:

    * ``use_rotation_trick=False`` (default) — the classic VQ-VAE
      straight-through estimator, ``z + (z_q - z).detach()``, yielding an
      identity gradient with respect to ``z``.
    * ``use_rotation_trick=True`` — the orthogonal rotation from
      https://arxiv.org/abs/2410.06424, which rotates ``z``'s direction onto
      ``z_q``'s direction. Produces a non-identity but better-conditioned
      gradient; falls back to ``rotate_to`` defined in ``.modules``.

    ``z`` and ``z_q`` must be broadcast-compatible; the output shape matches
    ``z_q``.

    .. note::
       This is NOT the right helper for multi-scale residual quantizers
       (:class:`MultiScaleResidualQuantizer` and its 3D variant), which use
       a different residual-injection formula rather than a plain STE.
    """
    if use_rotation_trick:
        return rotate_to(z, z_q)
    return z + (z_q - z).detach()


# ---------------------------------------------------------------------------
# Abstract base classes
#
# The quantizer family in this file follows a strong but previously-implicit
# contract. These classes make that contract explicit, without moving any
# numerical code. Existing concrete implementations need only inherit from
# the appropriate base.
# ---------------------------------------------------------------------------


class AbstractQuantizer(nn.Module, MetricLoggerMixin, ABC):
    """Shared contract for every quantizer in MedLat.

    The family includes classical VQ-VAE / VQ-GAN (:class:`VectorQuantizer`,
    :class:`VectorQuantizer2`, :class:`SimVQ`), codebook-free lookup
    quantizers (:class:`LookupFreeQuantizer`, :class:`BinarySphericalQuantizer`,
    :class:`FiniteScalarQuantizer`), soft relaxations
    (:class:`GumbelQuantize`, :class:`SoftVectorQuantizer`), and the multi-level
    variants which extend :class:`ResidualQuantizerBase` instead.

    **Core contract**

    Subclasses expose two integer attributes that describe the codebook:

    * ``n_e`` — codebook cardinality (number of discrete codes). For
      codebook-free quantizers (LFQ/BSQ/FSQ) this is the *effective* number of
      reachable codes, i.e. ``2 ** embed_dim`` or the product of per-axis
      level counts.
    * ``e_dim`` — embedding dimension per code.

    Subclasses implement one required method:

    * :meth:`forward` — quantize a batch and return a 3-tuple
      ``(z_q, loss, indices)``. ``z_q`` is the quantized feature map with the
      same spatial shape as the input; ``loss`` is a scalar commitment /
      entropy loss; ``indices`` is an integer ``Tensor`` of codebook indices
      for single-level quantizers, or a ``List[Tensor]`` of per-level indices
      for residual variants. Diagnostics (``perplexity``, one-hot encodings,
      per-level components) are surfaced via :meth:`log_metric` /
      :meth:`get_metrics` rather than the return value.

    And optionally override:

    * :meth:`get_codebook_entry` — decode codebook indices back to embeddings.
      The default raises :class:`NotImplementedError` so that codebook-free
      quantizers which *don't* support this operation fail loudly rather than
      silently returning nonsense.

    **Automatic autocast-disable**

    Every subclass's :meth:`forward` is automatically wrapped with
    ``torch.amp.autocast('cuda', enabled=False)`` via :meth:`__init_subclass__`.
    Quantizer forward passes compute squared-L2 distances and argmins that
    need FP32 precision; running them under an outer mixed-precision context
    can produce spurious NaNs or miscoded tokens. Subclasses therefore do NOT
    need to (and should not) redeclare the ``@autocast`` decorator — doing so
    just double-wraps the call. Outside an active autocast region the wrapper
    is a no-op, so CPU and eager-FP32 use is unaffected.

    **Built-in instrumentation**

    Three cross-cutting observability features are provided for free, all
    driven from a post-forward hook installed by :meth:`__init_subclass__`:

    * :meth:`log_metric` / :meth:`get_metrics` — a latest-value scalar sink.
      Training loops read ``model.get_metrics()`` each step and route the
      dict to whatever logger they use (wandb, tensorboard, …). No buffering;
      each new forward overwrites the previous values. Subclasses can call
      :meth:`log_metric` from inside ``forward`` to publish custom metrics.
    * **Codebook usage tracking** — on every forward the hook reads the
      ``indices`` from the returned info tuple and ``scatter_add``-accumulates
      them into a lazily-allocated buffer ``_usage_buffer`` of shape
      ``(n_e,)``. From this the base exposes ``active_code_count``,
      ``dead_code_ratio``, and ``codebook_utilization`` through
      :meth:`get_metrics`. Set class- or instance-attribute ``track_usage =
      False`` to disable (e.g. for codebooks with millions of entries).
    * :meth:`revive_dead_codes` — pass a batch of encoder activations and
      any codebook rows whose hit count is below ``dead_code_threshold`` are
      overwritten with random samples from that batch. No-op for
      codebook-free classes (LFQ / BSQ / FSQ) that have no learnable
      ``nn.Embedding``. Set :attr:`revive_dead_codes_after` at construction
      to have the quantizer auto-call this every N forward passes during
      training — no need to wire it into your own loop.

    **Instrumentation kwargs at construction**

    ``track_usage``, ``dead_code_threshold``, and ``revive_dead_codes_after``
    are implicit constructor kwargs on every subclass — :meth:`__init_subclass__`
    pops them out of the call before the subclass's own ``__init__`` sees
    them and sets them on the instance once construction finishes. Pass them
    directly to :func:`~medlat.get_model` or the factory::

        m = get_model("discrete.quantizer.vector_quantizer2",
                      n_e=8192, e_dim=16,
                      revive_dead_codes_after=1000,
                      dead_code_threshold=5)
    * :meth:`entropy_regularization` — opt-in SoftVQ-style entropy loss term,
      the formula shared by every quantizer in the family. Configured via
      :attr:`entropy_loss_weight`, :attr:`entropy_loss_temperature`, and
      :attr:`entropy_gamma`. Gated on ``self.training``; returns a zero scalar
      otherwise. :class:`VectorQuantizer2` historically exposed different
      entropy kwargs (``entropy_loss_ratio``, ``entropy_temperature``,
      ``entropy_loss_type``) — those are gone; callers must pass the new
      names.
    """

    #: codebook cardinality; subclasses assign in ``__init__``.
    n_e: int
    #: embedding dimension; subclasses assign in ``__init__``.
    e_dim: int

    #: whether to accumulate per-code usage counts on every forward.
    #: Override at class- or instance-level to disable for very large codebooks.
    track_usage: bool = True
    #: minimum number of hits for a code to count as "alive".
    dead_code_threshold: int = 1

    #: weight on the MaskGIT-style entropy regularization term. ``0.0`` (the
    #: default) disables it entirely and :meth:`entropy_regularization` returns
    #: a zero scalar. Subclasses that want the regularizer call the method
    #: from their :meth:`forward`; LookupFreeQuantizer, BinarySphericalQuantizer,
    #: and SoftVectorQuantizer already do.
    entropy_loss_weight: float = 0.0
    #: softmax temperature used when deriving a probability distribution from
    #: the affinity matrix supplied to :meth:`entropy_regularization`.
    entropy_loss_temperature: float = 1.0
    #: weighting applied to the batch-averaged ``avg_entropy`` component inside
    #: :func:`entropy_loss_fn`. Higher ``gamma`` pushes harder against codebook
    #: collapse. Attribute name matches the constructor kwarg on the concrete
    #: classes (LFQ / BSQ / SoftVQ) that historically owned this config.
    entropy_gamma: float = 1.0

    #: if > 0, :meth:`revive_dead_codes` is called automatically every N
    #: forward passes during training, using the first positional argument of
    #: ``forward`` as the encoder activation pool. ``0`` (default) disables
    #: auto-revival entirely — call :meth:`revive_dead_codes` yourself from
    #: the training loop. Requires a learnable ``nn.Embedding`` codebook and
    #: a populated usage buffer (i.e. at least one prior forward pass).
    #:
    #: Buffer-reset semantics: revival no longer resets ``_usage_buffer``.
    #: The buffer is owned exclusively by the windowed-snapshot machinery
    #: (see :attr:`usage_window_steps`), so revival simply reads whichever
    #: rolling-window state is current when it fires.
    revive_dead_codes_after: int = 0

    #: number of forward passes per usage-snapshot window. At every Nth
    #: forward the codebook-usage distribution metrics (``info/quantizer/
    #: usage_entropy_norm``, ``..._gini``, ``..._dead_code_ratio``, etc.) are
    #: computed from :attr:`_usage_buffer`, written via :meth:`log_metric`,
    #: and the buffer is then zeroed so the next window starts fresh. The
    #: previous snapshot's values stay in :attr:`_metrics` for the entire
    #: next window — :class:`~medlat.modules.metrics.MetricLoggerMixin` keeps
    #: latest values until something overwrites them, so loggers see a flat
    #: held value between window closes rather than ramping numbers from a
    #: partially-filled buffer.
    usage_window_steps: int = 1000

    #: Instrumentation kwargs that every subclass ``__init__`` implicitly
    #: accepts — :meth:`__init_subclass__` wraps each concrete ``__init__`` to
    #: pop these from the call, forward the remaining kwargs through to the
    #: subclass body, and ``setattr`` them on the instance once construction
    #: finishes. Users can therefore pass any of them at instantiation time::
    #:
    #:     m = get_model("discrete.quantizer.vector_quantizer",
    #:                   n_e=1024, e_dim=8, revive_dead_codes_after=1000,
    #:                   usage_window_steps=500)
    #:
    #: without the underlying factory's signature having to declare them.
    _INSTRUMENTATION_KWARGS = (
        "track_usage",
        "dead_code_threshold",
        "revive_dead_codes_after",
        "usage_window_steps",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only wrap classes that define their own ``forward`` (a concrete
        # override). Intermediate abstract bases like ``ResidualQuantizerBase``
        # that inherit ``forward`` unchanged are skipped; the wrap then
        # propagates automatically once a concrete leaf overrides it.
        if "forward" in cls.__dict__:
            original = cls.forward

            @autocast("cuda", enabled=False)
            @wraps(original)
            def forward_with_instrumentation(self, *args, **kwargs):
                output = original(self, *args, **kwargs)
                # Never let instrumentation failures bubble up — they must be
                # invisible to the training pipeline.
                try:
                    self._post_forward(output, args, kwargs)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.debug(
                        "Quantizer instrumentation hook raised %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                return output

            cls.forward = forward_with_instrumentation

        # Wrap ``__init__`` so every subclass implicitly accepts the
        # instrumentation kwargs without having to declare them. Only wrap
        # classes that define their own ``__init__``; classes inheriting a
        # parent's ``__init__`` (e.g. BinarySphericalQuantizer inheriting
        # LookupFreeQuantizer's) pick up the parent's already-wrapped init.
        if "__init__" in cls.__dict__:
            original_init = cls.__init__
            instrumentation_keys = cls._INSTRUMENTATION_KWARGS

            @wraps(original_init)
            def init_with_instrumentation(self, *args, **kwargs):
                overrides = {
                    k: kwargs.pop(k)
                    for k in instrumentation_keys
                    if k in kwargs
                }
                original_init(self, *args, **kwargs)
                for k, v in overrides.items():
                    setattr(self, k, v)

            cls.__init__ = init_with_instrumentation

    @abstractmethod
    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
        """Quantize ``z`` and return ``(z_q, loss, indices)``.

        * ``z_q`` preserves the input's batch and spatial shape.
        * ``loss`` is a scalar tensor suitable for ``.backward()``.
        * ``indices`` is an integer ``Tensor`` of codebook indices for
          single-level quantizers, or a ``List[Tensor]`` of per-level indices
          for residual / multi-level quantizers.

        Diagnostics that used to ride along in the info tuple (``perplexity``,
        one-hot encodings, per-level components) are now surfaced via
        :meth:`log_metric` and :meth:`get_metrics`; the post-forward hook
        computes :math:`H` automatically from ``indices`` for every
        single-level quantizer, so subclasses do not need to re-expose it.
        """

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Reconstruct features from codebook indices.

        Optional: only classical and residual quantizers override this. Default
        raises so callers can check support via ``hasattr``/``try`` if needed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_codebook_entry(); "
            "this is expected for codebook-free quantizers (LFQ/BSQ/FSQ)."
        )

    # ------------------------------------------------------------------
    # Metric logger — ``log_metric``, ``get_metrics``, ``reset_metrics``
    # inherited from :class:`MetricLoggerMixin`. All quantizer-side metrics
    # (Tier-A per-batch fidelity, Tier-B codebook-state geometry, Tier-C
    # windowed usage-distribution snapshots) flow through ``log_metric``
    # under the ``info/quantizer/`` prefix. Tier-C snapshots persist in
    # ``_metrics`` between window closes thanks to latest-value semantics.
    # ------------------------------------------------------------------

    def reset_usage(self) -> None:
        """Zero the codebook usage buffer (single-level + per-level)."""
        if hasattr(self, "_usage_buffer"):
            self._usage_buffer.zero_()
        for buf in self._iter_level_usage_buffers():
            buf.zero_()

    def _iter_level_usage_buffers(self) -> List[torch.Tensor]:
        """Return per-level usage buffers in level order, or ``[]`` if none."""
        n = getattr(self, "_n_level_usage_buffers", 0)
        return [getattr(self, f"_level_usage_buffer_{i}") for i in range(n)]

    # ------------------------------------------------------------------
    # Post-forward hook — extracts indices for auto-logging.
    # ------------------------------------------------------------------

    def _post_forward(
        self,
        output: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called automatically after every ``forward`` with its return value.

        Unpacks ``(z_q, loss, indices)`` defensively: ``indices`` may be a
        single ``Tensor`` (single-level quantizers), a ``list`` / ``tuple`` of
        per-level ``Tensor``s (residual variants), or ``None``. An optional
        trailing element (Gumbel's ``return_logits=True`` 4-tuple) is ignored.

        Three tiers of metrics are populated under the ``info/quantizer/``
        prefix:

        * **Tier A — per-batch, exact.** Quantization-error (mean / p99 /
          cosine), encoder-norm summary, encoder-to-codebook norm ratio,
          per-sample unique-code diversity, single-step perplexity. Computed
          on every forward.
        * **Tier B — codebook-state geometry.** Codebook entry-norm summary,
          stable rank, and step-to-step update norm. Only emitted for
          quantizers that expose a tensor codebook via ``self.embedding``;
          codebook-free LFQ/BSQ/FSQ are skipped.
        * **Tier C — windowed usage distribution.** Every
          :attr:`usage_window_steps` forwards (default 1000) the cumulative
          ``_usage_buffer`` is summarised into ``usage_entropy_norm``,
          ``usage_kl_to_uniform``, ``usage_gini``, ``top1_share``,
          ``top1pct_share``, ``active_code_count``, ``dead_code_ratio``,
          ``codebook_utilization``, ``total_tokens_seen_window``. The buffer
          is then zeroed; the snapshot persists in :attr:`_metrics` for the
          full next window via the mixin's latest-value semantics.

        Auto-revival (when :attr:`revive_dead_codes_after` > 0) fires on its
        own counter and reads whatever rolling-window usage state the buffer
        currently holds — it no longer resets the buffer itself.

        ``args`` is the positional arg tuple that the wrapped forward was
        called with — we need ``args[0]`` (typically ``z``) for auto-revival
        and Tier-A encoder-side metrics.
        """
        if not isinstance(output, tuple) or len(output) < 3:
            return
        z_q, loss, indices = output[0], output[1], output[2]

        if isinstance(loss, torch.Tensor) and loss.dim() == 0:
            self.log_metric("info/quantizer/total_loss", loss.detach())

        # ── Tier A: per-batch fidelity / encoder-side metrics ─────────────
        z = args[0] if args else None
        if isinstance(z, torch.Tensor) and isinstance(z_q, torch.Tensor):
            self._log_tier_a_per_batch(z, z_q)

        if isinstance(indices, torch.Tensor):
            # Single-level quantizer — log `perplexity` + update usage buffer.
            self._log_perplexity_from_indices(
                indices, self._n_e_safe(), "info/quantizer/perplexity"
            )
            self._log_unique_codes_per_sample(indices, "info/quantizer/")
            if self.track_usage:
                self._update_usage(indices)
        elif isinstance(indices, (list, tuple)):
            # Residual / multi-level — per-level perplexity, per-sample
            # diversity, and per-level usage buffers (so Tier-C snapshots
            # have a per-level histogram to summarise at window close).
            levels = getattr(self, "levels", None)
            fallback_n_e = self._n_e_safe()
            level_n_es: List[int] = []
            for i, level_idx in enumerate(indices):
                if not isinstance(level_idx, torch.Tensor):
                    level_n_es.append(0)
                    continue
                level_n_e = fallback_n_e
                if levels is not None:
                    try:
                        level_n_e = int(levels[i].n_e)
                    except (AttributeError, IndexError, TypeError):
                        pass
                level_n_es.append(level_n_e)
                self._log_perplexity_from_indices(
                    level_idx, level_n_e, f"info/quantizer/perplexity_level_{i}"
                )
                self._log_unique_codes_per_sample(
                    level_idx, f"info/quantizer/", suffix=f"_level_{i}"
                )
            if self.track_usage:
                self._update_level_usage(indices, level_n_es)

        # ── Tier B: codebook-state geometry (skipped for LFQ/BSQ/FSQ) ─────
        self._log_tier_b_codebook_state()

        # ── Tier C: windowed usage-distribution snapshot ──────────────────
        # Counter advances on every forward, regardless of train/eval, so
        # the snapshot cadence is predictable. ``track_usage = False`` opts
        # out completely (some very-large-codebook configs don't allocate
        # the buffer).
        if self.track_usage:
            count = getattr(self, "_window_step_count", 0) + 1
            self._window_step_count = count
            if count >= max(1, int(self.usage_window_steps)):
                self._snapshot_window_metrics()
                self.reset_usage()
                self._window_step_count = 0

        # ── Auto-revive dead codes on a fixed cadence ─────────────────────
        # Independent counter from the window cadence. Reads whichever
        # rolling-window usage state the buffer currently holds.
        if (
            self.revive_dead_codes_after > 0
            and self.training
            and args
            and isinstance(args[0], torch.Tensor)
        ):
            count = getattr(self, "_forward_count", 0) + 1
            self._forward_count = count
            if count % self.revive_dead_codes_after == 0:
                n_revived = self.revive_dead_codes(args[0])
                if n_revived > 0:
                    self.log_metric("info/quantizer/codes_revived", n_revived)

    def _n_e_safe(self) -> int:
        """Best-effort int read of ``self.n_e``; returns 0 if unavailable."""
        try:
            return int(self.n_e)
        except (AttributeError, TypeError):
            return 0

    def _log_perplexity_from_indices(
        self, indices: torch.Tensor, n_e: int, key: str
    ) -> None:
        """Compute perplexity from a ``(..., )`` long-tensor and log it under ``key``.

        Silently skips when ``n_e <= 0`` or ``indices`` is empty, so residual
        variants with missing/invalid per-level configs don't crash the hook.
        """
        if n_e <= 0:
            return
        flat = indices.detach().flatten().long()
        if flat.numel() == 0:
            return
        flat = flat.clamp(0, n_e - 1)
        counts = torch.bincount(flat, minlength=n_e).to(torch.float32)
        probs = counts / counts.sum().clamp_min(1.0)
        self.log_metric(key, compute_perplexity(probs).detach())

    # ------------------------------------------------------------------
    # Usage tracking — lazily-allocated buffer.
    # ------------------------------------------------------------------

    def _update_usage(self, indices: torch.Tensor) -> None:
        """Increment the per-code hit counter for every index in ``indices``."""
        try:
            n_e = int(self.n_e)
        except (AttributeError, TypeError):
            return  # subclass doesn't expose n_e in an int-castable form
        if n_e <= 0:
            return

        flat = indices.detach().flatten().long()
        if flat.numel() == 0:
            return

        if not hasattr(self, "_usage_buffer"):
            # Register on first call so we pick up the right device.
            # persistent=False keeps training stats out of the model checkpoint.
            self.register_buffer(
                "_usage_buffer",
                torch.zeros(n_e, dtype=torch.int64, device=flat.device),
                persistent=False,
            )

        # Defensive clamp — indices outside [0, n_e) would scatter into OOB.
        flat = flat.clamp(0, n_e - 1)
        self._usage_buffer.scatter_add_(
            0, flat, torch.ones_like(flat, dtype=self._usage_buffer.dtype)
        )

    def _update_level_usage(
        self, indices: Sequence[Any], level_n_es: List[int]
    ) -> None:
        """Increment per-level usage buffers for residual / multi-level quantizers.

        Lazily registers ``_level_usage_buffer_{i}`` on first call (one per
        level) and remembers how many it created in
        ``_n_level_usage_buffers`` so :meth:`_iter_level_usage_buffers` can
        find them later. Levels that report ``n_e <= 0`` or whose entry
        isn't a tensor (e.g. dropped levels in residual training) are
        skipped without disturbing the others.
        """
        if not hasattr(self, "_n_level_usage_buffers"):
            self._n_level_usage_buffers = 0

        for i, level_idx in enumerate(indices):
            if not isinstance(level_idx, torch.Tensor):
                continue
            n_e = level_n_es[i] if i < len(level_n_es) else 0
            if n_e <= 0:
                continue
            flat = level_idx.detach().flatten().long()
            if flat.numel() == 0:
                continue

            buf_name = f"_level_usage_buffer_{i}"
            if not hasattr(self, buf_name):
                self.register_buffer(
                    buf_name,
                    torch.zeros(n_e, dtype=torch.int64, device=flat.device),
                    persistent=False,
                )
                # Track the highest-index level we've allocated for so the
                # iterator helper picks them up in level order regardless of
                # which level happened to fire first.
                self._n_level_usage_buffers = max(
                    self._n_level_usage_buffers, i + 1
                )

            buf = getattr(self, buf_name)
            flat = flat.clamp(0, n_e - 1)
            buf.scatter_add_(
                0, flat, torch.ones_like(flat, dtype=buf.dtype)
            )

    def _log_unique_codes_per_sample(
        self,
        indices: torch.Tensor,
        prefix: str,
        suffix: str = "",
    ) -> None:
        """Mean number of distinct code indices used inside one sample.

        ``indices[0]`` is taken as the batch dim. A 0/1-D tensor falls back
        to a single global count so callers don't need to special-case it.
        """
        idx = indices.detach()
        if idx.ndim == 0:
            return
        if idx.ndim == 1:
            uniq = float(torch.unique(idx).numel())
        else:
            B = idx.shape[0]
            flat = idx.reshape(B, -1)
            counts = [torch.unique(row).numel() for row in flat]
            uniq = sum(counts) / max(1, len(counts))
        self.log_metric(f"{prefix}unique_codes_per_sample{suffix}", float(uniq))

    # ------------------------------------------------------------------
    # Direction-only spread on the unit sphere (used by Tier A and B).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_directional_spread(
        self, X: torch.Tensor, prefix: str
    ) -> None:
        """Log how rows of ``X`` are spread on the unit sphere.

        Direction-only diagnostics that complement the magnitude-only
        norms in Tier A/B. All operations are O(n·d): the mean
        off-diagonal cosine uses the sum trick
        ``Σᵢⱼ <xᵢ,xⱼ> = ‖Σᵢ xᵢ‖²``, and the singular spectrum of the
        row-normalized matrix is cheap because ``d ≪ n`` everywhere
        this is called (codebook ``d ≤ 32`` and encoder activations
        flattened to ``(N, d)``).

        Uniform points on S^{d-1} → ``mean_cos ≈ 0``, ``sv_ratio ≈ 1``,
        ``effective_dim ≈ d``. Concentration in a cone pushes
        ``mean_cos`` toward 1, ``sv_ratio`` upward, and
        ``effective_dim`` toward 1.

        Metric keys (under ``prefix``):
          * ``mean_cos``       — average off-diagonal cosine similarity
          * ``sv_ratio``       — σ_max / σ_min of the row-normalized matrix
          * ``effective_dim``  — exp of the entropy of normalized σ_i²
                                 (a smooth analogue of stable rank)
        """
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
            return
        Xn = F.normalize(X.detach().float(), dim=-1)
        n, d = Xn.shape

        total = Xn.sum(dim=0).pow(2).sum()
        denom = float(n * (n - 1))
        mean_cos = (total - n) / max(denom, 1.0)
        self.log_metric(f"{prefix}mean_cos", mean_cos)

        try:
            sv = torch.linalg.svdvals(Xn).clamp_min(0.0)
            sv_max = sv.max().clamp_min(1e-12)
            sv_min = sv.min().clamp_min(1e-12)
            self.log_metric(f"{prefix}sv_ratio", sv_max / sv_min)
            p = sv.pow(2)
            p = p / p.sum().clamp_min(1e-12)
            ent = -(p * p.clamp_min(1e-12).log()).sum()
            self.log_metric(f"{prefix}effective_dim", ent.exp())
        except Exception:
            # SVD can fail on degenerate matrices; treat as
            # "metric unavailable this step" rather than crashing.
            pass

    # ------------------------------------------------------------------
    # Tier-A: per-batch fidelity / encoder-side metrics.
    # ------------------------------------------------------------------

    def _log_tier_a_per_batch(
        self, z: torch.Tensor, z_q: torch.Tensor
    ) -> None:
        """Quantization-error and encoder-norm summaries for the current batch.

        Computed in FP32 inside an autocast-disabled wrapper (see
        :meth:`__init_subclass__`), so the small extra arithmetic is exact
        regardless of the outer mixed-precision context. Errors and norms
        are taken along the channel dim — the last axis once the tensor is
        in channel-last layout for both rank-4 ``(B,C,H,W)`` and rank-5
        ``(B,C,D,H,W)`` inputs, and the first axis for ``(B, N, C)`` token
        layouts. ``flatten_spatial_to_channel_last`` returns the token-style
        tensor unchanged so this works in both layouts.
        """
        if z.shape != z_q.shape:
            # Some quantizers reshape; if shapes differ we can't pair tokens
            # element-wise. Skip rather than emit garbage.
            return
        z_cl = flatten_spatial_to_channel_last(z.detach()).reshape(-1, z.shape[1] if z.ndim >= 4 else z.shape[-1])
        z_q_cl = flatten_spatial_to_channel_last(z_q.detach()).reshape(-1, z_q.shape[1] if z_q.ndim >= 4 else z_q.shape[-1])

        diff = (z_cl - z_q_cl).float()
        err = diff.norm(dim=-1)  # (N,)
        self.log_metric("info/quantizer/quant_error_mean", err.mean())
        if err.numel() > 1:
            self.log_metric(
                "info/quantizer/quant_error_p99",
                torch.quantile(err, 0.99),
            )

        cos = F.cosine_similarity(z_cl.float(), z_q_cl.float(), dim=-1)
        self.log_metric("info/quantizer/quant_error_cosine", cos.mean())

        z_norm = z_cl.float().norm(dim=-1)
        self.log_metric("info/quantizer/encoder_z_norm_mean", z_norm.mean())
        self.log_metric("info/quantizer/encoder_z_norm_std", z_norm.std())

        # Direction-only spread of the encoder activations: tells us
        # whether the encoder uses the unit sphere or sits in a cone,
        # independently of the magnitude statistics above.
        self._log_directional_spread(z_cl, "info/quantizer/encoder_z_dir_")

        # Encoder-to-codebook norm ratio (only meaningful when the codebook
        # is a tensor we can reduce). SimVQ exposes ``embedding`` as a
        # property returning a projected codebook tensor; classical VQ uses
        # ``nn.Embedding``; LFQ/BSQ/FSQ have no learnable codebook → skip.
        cb = self._resolve_codebook_tensor()
        if cb is not None:
            cb_norm_mean = cb.float().norm(dim=-1).mean().clamp_min(1e-8)
            self.log_metric(
                "info/quantizer/encoder_to_codebook_norm_ratio",
                z_norm.mean() / cb_norm_mean,
            )

    def _resolve_codebook_tensor(self) -> Optional[torch.Tensor]:
        """Return the (n_e, d) codebook tensor or ``None`` for codebook-free quantizers.

        Handles the three storage flavours used in this file:

        * ``nn.Embedding`` (classical VQ family) — read ``.weight``.
        * ``nn.Parameter`` (SoftVQ) — the parameter itself is the codebook.
        * Property returning a tensor (SimVQ — ``code_transform(frozen)``).

        Anything else (None, modules without a tensor, codebook-free LFQ /
        BSQ / FSQ) returns ``None``.
        """
        emb = getattr(self, "embedding", None)
        if emb is None:
            return None
        if isinstance(emb, nn.Embedding):
            return emb.weight
        if isinstance(emb, torch.Tensor):
            return emb
        weight = getattr(emb, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
        return None

    # ------------------------------------------------------------------
    # Tier-B: codebook-state geometry.
    # ------------------------------------------------------------------

    def _log_tier_b_codebook_state(self) -> None:
        """Emit codebook-norm summary, stable rank, and step-to-step update norm.

        Pair-distance metrics (mean / min) are deliberately omitted — they
        cost O(n_e²·d) which is prohibitive for an 8k+ codebook on every
        forward. Stable rank uses ``torch.linalg.matrix_norm(..., ord=2)``
        which is cheap when ``d ≪ n_e`` (the common case in this codebase).
        """
        cb = self._resolve_codebook_tensor()
        if cb is None or cb.ndim != 2 or cb.numel() == 0:
            return
        E = cb.detach().float()

        norms = E.norm(dim=-1)
        self.log_metric("info/quantizer/codebook_norm_mean", norms.mean())
        self.log_metric("info/quantizer/codebook_norm_std", norms.std())

        try:
            fro_sq = (E * E).sum()
            spec = torch.linalg.matrix_norm(E, ord=2)
            stable_rank = fro_sq / spec.pow(2).clamp_min(1e-12)
            self.log_metric(
                "info/quantizer/codebook_stable_rank", stable_rank
            )
        except Exception:
            # SVD can fail on degenerate matrices; we treat it as
            # "metric unavailable this step" rather than crashing the hook.
            pass

        # Direction-only spread of the codebook: answers "do the codes
        # cover the sphere or live in a cone?" — complements the
        # magnitude-only ``codebook_norm_*`` and ``codebook_stable_rank``.
        self._log_directional_spread(E, "info/quantizer/codebook_dir_")

        prev = getattr(self, "_prev_codebook", None)
        if isinstance(prev, torch.Tensor) and prev.shape == E.shape:
            update = (E - prev).norm()
            self.log_metric("info/quantizer/codebook_update_norm", update)
        # Cache without registering as a buffer — we don't want the
        # snapshot moving with .to() unexpectedly or appearing in
        # state_dict; it's a transient instrumentation cache.
        self._prev_codebook = E.clone()

    # ------------------------------------------------------------------
    # Tier-C: windowed usage-distribution snapshot.
    # ------------------------------------------------------------------

    def _snapshot_window_metrics(self) -> None:
        """Summarise the cumulative usage buffer(s) at the close of a window.

        Writes the full Tier-C bundle via :meth:`log_metric` so the values
        persist through the next window without being recomputed each step.
        Single-level quantizers emit one bundle; residual / multi-level
        variants additionally emit per-level bundles keyed with a
        ``_level_{i}`` suffix.
        """
        usage = getattr(self, "_usage_buffer", None)
        if isinstance(usage, torch.Tensor):
            self._log_window_metrics_from_usage(usage, "info/quantizer/", "")
        for i, buf in enumerate(self._iter_level_usage_buffers()):
            self._log_window_metrics_from_usage(
                buf, "info/quantizer/", f"_level_{i}"
            )

    def _log_window_metrics_from_usage(
        self, usage: torch.Tensor, prefix: str, suffix: str
    ) -> None:
        """Compute the Tier-C metric bundle from one usage histogram."""
        n_e = int(usage.numel())
        if n_e == 0:
            return
        total = float(usage.sum().item())
        if total <= 0.0:
            # Empty window (track_usage off, or no forwards counted) —
            # don't emit misleading "all dead" numbers.
            return
        usage_f = usage.detach().float()
        probs = usage_f / usage_f.sum().clamp_min(1.0)

        alive = int((usage >= self.dead_code_threshold).sum().item())
        self.log_metric(f"{prefix}active_code_count{suffix}", alive)
        self.log_metric(
            f"{prefix}dead_code_ratio{suffix}",
            1.0 - alive / n_e,
        )
        self.log_metric(
            f"{prefix}codebook_utilization{suffix}",
            alive / n_e,
        )
        self.log_metric(
            f"{prefix}total_tokens_seen_window{suffix}", int(total)
        )

        # Distribution-shape metrics. ``log(n_e)`` is the entropy of a
        # uniform distribution over the codebook; normalising by it yields
        # a 0..1 number that's directly readable.
        log_n_e = math.log(n_e) if n_e > 1 else 1.0
        # H(p) over the support of `probs`. zeros contribute 0·log0 = 0.
        nonzero = probs[probs > 0]
        H = -(nonzero * nonzero.log()).sum().item() if nonzero.numel() > 0 else 0.0
        self.log_metric(f"{prefix}usage_entropy_norm{suffix}", H / log_n_e)
        # KL(p || U) = log(n_e) - H(p) ≥ 0; rises to log(n_e) at full collapse.
        self.log_metric(
            f"{prefix}usage_kl_to_uniform{suffix}", log_n_e - H
        )

        # Sorted shares for tail-mass / dominance metrics.
        sorted_probs, _ = torch.sort(probs, descending=True)
        self.log_metric(
            f"{prefix}top1_share{suffix}", sorted_probs[0].item()
        )
        top1pct = max(1, math.ceil(n_e / 100))
        self.log_metric(
            f"{prefix}top1pct_share{suffix}",
            sorted_probs[:top1pct].sum().item(),
        )

        # Gini coefficient on the usage histogram. Standard formula:
        #   G = (Σ_i (2i − n − 1) · x_(i)) / (n · Σ x)
        # where x_(i) is the sorted (ascending) sequence. Implemented from
        # the sorted-descending probs we already have.
        ascending, _ = torch.sort(probs)  # ascending
        idx = torch.arange(1, n_e + 1, device=ascending.device, dtype=ascending.dtype)
        weighted = ((2 * idx - n_e - 1) * ascending).sum()
        denom = (n_e * ascending.sum()).clamp_min(1e-12)
        self.log_metric(f"{prefix}usage_gini{suffix}", (weighted / denom).item())

    # ------------------------------------------------------------------
    # Entropy regularization (opt-in, cross-cutting).
    # ------------------------------------------------------------------

    def entropy_regularization(self, affinity: torch.Tensor) -> torch.Tensor:
        """Canonical entropy regularization term, shared by every quantizer.

        This is the formula previously duplicated inside
        :class:`LookupFreeQuantizer`, :class:`BinarySphericalQuantizer`, and
        :class:`SoftVectorQuantizer`. It is now lifted into the base class so
        any quantizer can pick it up with a single call in its ``forward``::

            affinity = -distances               # or raw logits for soft variants
            loss = loss + self.entropy_regularization(affinity)

        The math matches :func:`medlat.first_stage.discrete.quantizer.modules.entropy_loss_fn`:

        .. math::
           L_{\\text{ent}} = w \\cdot \\bigl(H_{\\text{per-sample}}
           - \\gamma \\cdot H_{\\text{avg}}\\bigr)

        where :math:`w` is :attr:`entropy_loss_weight`, :math:`\\gamma` is
        :attr:`entropy_gamma`, and the two entropies are computed over
        softmax(``affinity`` / :attr:`entropy_loss_temperature``).

        The helper is **gated on** ``self.training`` — returning a zero scalar
        in eval — to match SoftVQ / BSQ semantics. Also short-circuits to zero
        when :attr:`entropy_loss_weight` is zero, so callers can always chain
        ``loss + self.entropy_regularization(aff)`` unconditionally.

        Args:
            affinity: ``(..., n_e)`` tensor whose last dim is the codebook
                dimension. For hard-argmin quantizers pass ``-d`` (negative
                squared-L2 distances); for soft quantizers pass raw logits.

        Returns:
            Scalar tensor; zero when disabled or in eval.

        .. note::
           :class:`VectorQuantizer2` historically used a different formula
           (``-ratio * mean(per_row_entropy)``, with an optional
           ``entropy_loss_type="gumbel"`` path). Both are gone; VQ2 now uses
           this helper. Legacy kwargs (``entropy_loss_ratio``,
           ``entropy_temperature``, ``entropy_loss_type``) were removed —
           callers must pass the new names :attr:`entropy_loss_weight`,
           :attr:`entropy_loss_temperature`, :attr:`entropy_gamma`.
        """
        if self.entropy_loss_weight == 0.0 or not self.training:
            zero = affinity.new_zeros(())
            # Also populate the components cache so callers that include
            # per_sample / avg in their info tuple don't need a separate
            # code path for the disabled case.
            self._last_entropy_info = (zero, zero, zero)
            return zero

        # ``entropy_loss_fn`` mutates its ``affinity`` argument in place
        # (``flat_affinity /= temperature``). Clone so callers can reuse the
        # tensor — e.g. for distance-weighted aux losses — without surprise.
        per_sample, avg = entropy_loss_fn(
            affinity.clone(),
            temperature=self.entropy_loss_temperature,
            entropy_gamma=self.entropy_gamma,
        )
        self.log_metric("info/quantizer/entropy_per_sample", per_sample.detach())
        self.log_metric("info/quantizer/entropy_avg", avg.detach())
        entropy_loss = self.entropy_loss_weight * (per_sample - avg)
        self.log_metric("info/quantizer/entropy_loss", entropy_loss.detach())
        # Stash non-detached components for callers (LFQ / BSQ) that include
        # them in their forward return tuple. Access via ``self._last_entropy_info``.
        self._last_entropy_info = (entropy_loss, per_sample, avg)
        return entropy_loss

    # ------------------------------------------------------------------
    # Dead-code revival.
    # ------------------------------------------------------------------

    def revive_dead_codes(self, encoder_output: torch.Tensor) -> int:
        """Re-initialise unused codebook entries with encoder-activation samples.

        Finds codes whose hit count is below :attr:`dead_code_threshold` and
        overwrites the corresponding rows of ``self.embedding.weight`` with
        random samples from ``encoder_output``. Returns the number of codes
        revived (0 if none or if the subclass has no learnable
        ``nn.Embedding``).

        The usage counter for revived codes is set to ``dead_code_threshold``
        so they don't trigger revival again on the very next call.

        Typical usage from a training loop::

            if step % 1000 == 0:
                model.revive_dead_codes(encoder_output)
                model.reset_usage()        # optional — restart the "recency"

        Args:
            encoder_output: any tensor whose *last dim* matches
                ``self.embedding.embedding_dim``. Intermediate shape is
                flattened to ``(N, D)``.
        """
        embedding = getattr(self, "embedding", None)
        if not isinstance(embedding, nn.Embedding):
            return 0
        if not hasattr(self, "_usage_buffer"):
            return 0

        dead_mask = self._usage_buffer < self.dead_code_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return 0

        d = embedding.embedding_dim
        flat = encoder_output.detach().reshape(-1, d)
        n_samples = flat.shape[0]
        if n_samples == 0:
            return 0

        # Random draw with replacement; this matches standard practice.
        sample_ids = torch.randint(0, n_samples, (n_dead,), device=flat.device)
        new_codes = flat[sample_ids].to(embedding.weight.dtype)

        with torch.no_grad():
            embedding.weight.data[dead_mask] = new_codes
            # Prevent immediate re-revival on the next step.
            self._usage_buffer[dead_mask] = self.dead_code_threshold

        return n_dead


class ResidualQuantizerBase(AbstractQuantizer, ABC):
    """Base class for quantizers that stack multiple quantization levels.

    Covers :class:`ResidualQuantizer`, :class:`QincoResidualQuantizer`, and
    :class:`MultiScaleResidualQuantizer` (plus its 3D variant). Their
    ``forward`` return shape differs from single-stage quantizers only in the
    last slot: ``indices`` is a ``List[Tensor]`` (one entry per residual
    level) rather than a single ``Tensor``. The ``z_q`` tensor is still the
    summed/composed result ready for decoding, and the ``loss`` is still a
    scalar.

    Subclasses additionally expose:

    * ``n_levels`` — number of stacked quantization stages.
    """

    #: number of residual / multi-scale levels; subclasses assign in ``__init__``.
    n_levels: int


@register_model(f"{_REGISTRY_PREFIX}vector_quantizer",
code_url="https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py",
paper_url="https://arxiv.org/abs/1711.00937",)
class VectorQuantizer(AbstractQuantizer):
    """
    Standard VQ-VAE/VQ-GAN quantizer

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        rotation_trick: Whether to apply rotation trick
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta, rotation_trick: bool = False, 
                 use_ema: bool = False, ema_decay: float = 0.99, ema_eps: float = 1e-5):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema
        
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Ensure quantization is performed using f32
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j  (z - e)^2 = z^2 + e^2 - 2 e·z
        min_encoding_indices, d = nearest_codebook_entry_l2(z_flattened, self.embedding.weight)
        min_encoding_indices = min_encoding_indices.unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # Perform EMA update if enabled
        if self.use_ema:
            z_flattened = z_flattened  # Already defined above
            self.embedding.perform_ema_update(min_encodings, z_flattened, self.n_e)
        
        # compute loss for embedding
        commitment_loss = torch.mean((z_q.detach()-z)**2)
        codebook_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)
        loss = commitment_loss + codebook_loss
        self.log_metric("info/quantizer/commitment_loss", commitment_loss.detach())
        self.log_metric("info/quantizer/codebook_loss", codebook_loss.detach())

        # Assignment margin: gap between nearest and second-nearest code.
        # Low values flag unstable, near-tie assignments. Computed without
        # gradients — purely diagnostic.
        if d.size(1) >= 2:
            with torch.no_grad():
                top2 = torch.topk(d, 2, dim=1, largest=False).values
                self.log_metric(
                    "info/quantizer/assignment_margin",
                    (top2[:, 1] - top2[:, 0]).mean(),
                )

        # Rotation trick (https://arxiv.org/abs/2410.06424) or classic STE
        z_q = straight_through_estimator(z, z_q, use_rotation_trick=self.rotation_trick)

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


@register_model(f"{_REGISTRY_PREFIX}gumbel_quantizer",
code_url="https://github.com/karpathy/deep-vector-quantization/blob/main/model.py",
paper_url="https://arxiv.org/abs/1611.01144",)
class GumbelQuantize(AbstractQuantizer):
    """
    Gumbel Softmax trick quantizer

    Args:
        num_hiddens: Number of hidden dimensions
        embedding_dim: Dimension of embedding
        n_embed: Number of embeddings
        straight_through: Whether to use straight through estimator
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1

        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)
        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        # Soft-assignment diagnostics (cheap; computed from ``qy`` either way).
        with torch.no_grad():
            # Per-token entropy: H(qy) per spatial location, then mean.
            log_q = torch.log(qy.clamp_min(1e-10))
            per_token_H = -(qy * log_q).sum(dim=1)
            self.log_metric(
                "info/quantizer/assignment_entropy_per_token",
                per_token_H.mean(),
            )
            self.log_metric(
                "info/quantizer/mean_top1_prob",
                qy.max(dim=1).values.mean(),
            )

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, ind, logits
            return z_q, diff, ind
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


@register_model(f"{_REGISTRY_PREFIX}vector_quantizer2",
code_url="https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py",
paper_url="https://arxiv.org/abs/1711.00937",)
class VectorQuantizer2(AbstractQuantizer):
    """
    Improved VectorQuantizer with optional EMA, rotation trick, and cosine
    normalization. Entropy regularization is handled by the unified
    :meth:`AbstractQuantizer.entropy_regularization` helper — pass
    ``entropy_loss_weight``, ``entropy_loss_temperature``, ``entropy_gamma``
    to activate it.
    """
    def __init__(
        self,
        n_e,
        e_dim,
        beta=0.25,
        legacy=True,
        rotation_trick=False,
        use_norm=False,
        use_ema=False,
        ema_decay=0.99,
        ema_eps=1e-5,
        entropy_loss_weight: float = 0.0,
        entropy_loss_temperature: float = 1.0,
        entropy_gamma: float = 1.0,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        self.beta = beta
        self.legacy = legacy
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema

        # Entropy regularization hyperparameters (consumed by the base-class
        # helper self.entropy_regularization; zero weight = disabled).
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


    ## Ensure quantization is performed using fp32
    def forward(self, z):
        z = z.float()

        # Put channel last (2D or 3D)
        z = flatten_spatial_to_channel_last(z)

        z_flat = z.reshape(-1, self.e_dim)
        z_flat = self.norm(z_flat)

        embedding = self.norm(self.embedding.weight)

        # Nearest codebook entry via L2 distance (MaskGIT/VQGAN style)
        min_indices, d = nearest_codebook_entry_l2(z_flat, embedding)
        z_q = self.embedding(min_indices).view_as(z)
        z_q = self.norm(z_q)

        # EMA update
        if self.use_ema:
            onehot = F.one_hot(min_indices, self.n_e).type(z.dtype)
            self.embedding.perform_ema_update(onehot, z_flat, self.n_e)

        # Standard VQ loss — split into named components for observability.
        # In legacy mode the unweighted term is the commitment side and the
        # ``beta``-weighted term is the codebook side; non-legacy swaps the
        # two (matching the fix discussed in :class:`VectorQuantizer2`'s
        # docstring). Logging the two pieces separately lets ``beta`` be
        # tuned without inferring contributions from a single combined loss.
        if self.legacy:
            commitment_loss = torch.mean((z_q.detach() - z) ** 2)
            codebook_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)
        else:
            commitment_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            codebook_loss = torch.mean((z_q - z.detach()) ** 2)
        loss = commitment_loss + codebook_loss
        self.log_metric("info/quantizer/commitment_loss", commitment_loss.detach())
        self.log_metric("info/quantizer/codebook_loss", codebook_loss.detach())

        if d.size(1) >= 2:
            with torch.no_grad():
                top2 = torch.topk(d, 2, dim=1, largest=False).values
                self.log_metric(
                    "info/quantizer/assignment_margin",
                    (top2[:, 1] - top2[:, 0]).mean(),
                )

        loss = loss + self.entropy_regularization(-d)

        # Rotation trick or STE
        z_q = straight_through_estimator(z, z_q, use_rotation_trick=self.rotation_trick)

        # Restore shape (channel-first)
        z_q = unflatten_spatial_to_channel_first(z_q)

        return z_q, loss, min_indices

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return self.norm(z_q)



class SimpleQINCo(VectorQuantizer2):
    def __init__(self, n_e, e_dim, beta=0.25,
                 hidden_dim=256, num_layers=3,
                 **kwargs):

        super().__init__(n_e, e_dim, beta=beta, **kwargs)

        # Replace table with implicit MLP
        self.embedding = ImplicitEmbedding(
            n_e=n_e,
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

@register_model(f"{_REGISTRY_PREFIX}simple_qinco",
code_url="https://github.com/facebookresearch/Qinco",
paper_url="https://arxiv.org/abs/2401.14732",
)
class QINCo(nn.Module):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        top_a: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        commitment_weight: float = 1.0,
    ):
        """
        n_e: number of codes
        e_dim: embedding dimension
        beta: commitment loss factor (like VQ-VAE)
        top_a: number of top candidates per vector
        hidden_dim, num_layers: for QincoSubstep
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.commitment_weight = commitment_weight
        self.top_a = top_a
        # implicit base codebook
        self.embedding = ImplicitEmbedding(
            n_e=n_e,
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # QINCo-style transform
        self.transform = QincoSubstep(
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    @property
    def e_dim_prop(self):
        return self.e_dim

    @property
    def n_e_prop(self):
        return self.n_e

    def forward(self, residual: torch.Tensor, x_prev: torch.Tensor):
        """
        residual: (B, D, ...)  flattened to (B_flat, D)
        x_prev:  same shape as residual
        """
        orig_shape = residual.shape
        B_flat = residual.numel() // self.e_dim
        residual_flat = residual.view(B_flat, self.e_dim)
        x_prev_flat = x_prev.view(B_flat, self.e_dim)

        # ----- 1) cheap distance using base codes (no transform) -----
        base_codes = self.embedding.weight            # (K, D)
        K = base_codes.size(0)

        # (B_flat, 1, D) - (1, K, D) -> (B_flat, K, D)
        diff_base = residual_flat.unsqueeze(1) - base_codes.unsqueeze(0)
        dist_base = (diff_base ** 2).sum(-1)          # (B_flat, K)

        A = min(self.top_a, K)
        # top-A smallest distances
        topk_dist, topk_idx = dist_base.topk(A, dim=-1, largest=False, sorted=False)  # (B_flat, A)

        # ----- 2) run transform only on these A candidates -----
        # gather base codes for selected indices
        codes_sel = base_codes[topk_idx]              # (B_flat, A, D)
        x_prev_sel = x_prev_flat.unsqueeze(1).expand(-1, A, -1)   # (B_flat, A, D)

        # flatten for QincoSubstep
        codes_in = codes_sel.reshape(-1, self.e_dim)  # (B_flat*A, D)
        x_in = x_prev_sel.reshape(-1, self.e_dim)     # (B_flat*A, D)

        deltas_sel = self.transform(codes_in, x_in)   # (B_flat*A, D)
        deltas_sel = deltas_sel.view(B_flat, A, self.e_dim)  # (B_flat, A, D)

        # ----- 3) pick best among the A candidates -----
        diff = residual_flat.unsqueeze(1) - deltas_sel      # (B_flat, A, D)
        dist = (diff ** 2).sum(-1)                         # (B_flat, A)
        best_in_A = dist.argmin(-1)                        # (B_flat,)

        # map back to global code indices
        indices = topk_idx[torch.arange(B_flat, device=residual.device), best_in_A]  # (B_flat,)

        # final quantized vector: transform(selected_base_code, x_prev)
        chosen_base = base_codes[indices]                  # (B_flat, D)
        z_q_flat = self.transform(chosen_base, x_prev_flat)  # (B_flat, D)
        z_q = z_q_flat.view(orig_shape)

        # ----- 4) losses & perplexity -----
        loss_commit = F.mse_loss(z_q_flat.detach(), residual_flat)
        loss_embed = F.mse_loss(z_q_flat, residual_flat.detach())
        loss = loss_embed + self.beta * loss_commit

        indices = indices.view(-1)

        return z_q, loss, indices


class SimVQ(AbstractQuantizer):
    """
    A VQ module using a frozen / implicit codebook with optional linear projection.
    Designed to be compatible with ResidualQuantizer / GroupedResidualVQ wrappers.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        in_channels: int = None, # usually not used as we have quant conv
        codebook_transform: nn.Module | None = None,
        rotation_trick: bool = True,
        beta: float = 0.25,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.n_e = n_e
        self.in_channels = in_channels if in_channels is not None else e_dim
        self.e_dim = e_dim
        self.rotation_trick = rotation_trick
        self.beta = beta
        self.commitment_weight = commitment_weight

        # frozen codebook buffer
        codebook = torch.randn(n_e, self.e_dim) * (self.e_dim ** -0.5) # scaling 
        self.register_buffer("frozen_codebook", codebook)

        # linear projection from frozen codebook to actual quantized space
        if codebook_transform is None:
            self.code_transform = nn.Linear(self.e_dim, self.in_channels, bias=False)
        else:
            self.code_transform = codebook_transform

    @property
    def embedding(self):
        """For compatibility with ResidualQuantizer wrappers"""
        return self.code_transform(self.frozen_codebook)

    def forward(self, z: torch.Tensor):
        """
        VectorQuantizer2-style forward for SimVQ.
        Supports 2D or 3D feature maps with channel-first format.
        Returns: z_q, loss, indices
        """
        z = z.float()  # ensure FP32 for distance computation

        # Reshape input to (B, H, W, C) or (B, D, H, W, C) style for distance computation
        z = flatten_spatial_to_channel_last(z, contiguous=True)

        # Flatten for distance computation
        z_flat = z.view(-1, self.in_channels)
        codebook = self.embedding  # projected codebook

        # Nearest codebook entry via L2 distance. Keep ``d`` so we can log
        # an assignment-margin diagnostic without recomputing distances.
        with torch.no_grad():
            indices, d = nearest_codebook_entry_l2(z_flat, codebook)


        # Get quantized vectors
        z_q_flat = codebook[indices]

        # Commitment loss with STE trick — split into named components.
        commitment_loss = F.mse_loss(z_flat.detach(), z_q_flat) * self.commitment_weight
        codebook_loss = F.mse_loss(z_flat, z_q_flat.detach()) * self.beta * self.commitment_weight
        loss = commitment_loss + codebook_loss
        self.log_metric("info/quantizer/commitment_loss", commitment_loss.detach())
        self.log_metric("info/quantizer/codebook_loss", codebook_loss.detach())

        if d.size(1) >= 2:
            with torch.no_grad():
                top2 = torch.topk(d, 2, dim=1, largest=False).values
                self.log_metric(
                    "info/quantizer/assignment_margin",
                    (top2[:, 1] - top2[:, 0]).mean(),
                )

        # Rotation trick or straight-through
        z_q_flat = straight_through_estimator(z_flat, z_q_flat, use_rotation_trick=self.rotation_trick)

        # Reshape back to original spatial dimensions
        z_q = z_q_flat.view(z.shape)
        z_q = unflatten_spatial_to_channel_first(z_q, contiguous=True)

        return z_q, loss, indices


    def get_codebook_entry(self, indices, shape=None):
        codebook = self.embedding  # (n_e, in_channels)
        # lookup
        z_q = codebook[indices]
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

@register_model(f"{_REGISTRY_PREFIX}residual_quantizer",
paper_url="https://arxiv.org/abs/2107.03312",
description="Acts as wrapper for all the other quantizers")
class ResidualQuantizer(ResidualQuantizerBase):
    def __init__(
        self,
        quantizer_class: nn.Module,
        num_quantizers: int,
        quantizer_kwargs_list: List[Dict],
        shared_codebook: bool = False,
        quantize_dropout: bool = False,   ### as in the EnCodec paper
        dropout_start_level: int = 0,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.quantize_dropout = quantize_dropout
        self.dropout_start_level = dropout_start_level
        self.shared_codebook = shared_codebook

        # Build levels
        self.levels = nn.ModuleList([
            quantizer_class(**quantizer_kwargs_list[i])
            for i in range(num_quantizers)
        ])

        # ---- Shared Codebook Mode ------------------------------------------------------
        # All quantizers share the codebook of the first quantizer
        if shared_codebook:
            first = self.levels[0]
            shared = first.embedding
            # link all quantizers to same object
            for q in self.levels[1:]:
                q.embedding = shared

    # VQ Model needs to know for quant_conv
    @property
    def e_dim(self):
        return self.levels[0].e_dim

    @property
    def n_e(self):
        return self.levels[0].n_e

    # ----------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        residual = x
        quantized_outputs = []
        losses = []
        all_indices = []

        # -------- Determine dropout level ----------------------------------------------
        # During training, randomly skip fine quantizers
        if self.training and self.quantize_dropout and self.num_quantizers > 1:
            # choose a dropout boundary: deeper ones are removed
            dropout_level = torch.randint(
                self.dropout_start_level,
                self.num_quantizers,
                (1,)
            ).item()
        else:
            dropout_level = self.num_quantizers

        # -------- Iterate through levels -----------------------------------------------
        for i, q in enumerate(self.levels):

            # ---------------------------------------------------------
            # DROPOUT: skip quantization for deeper levels
            # ---------------------------------------------------------
            if i >= dropout_level:
                # output placeholder
                quantized_outputs.append(torch.zeros_like(residual))
                losses.append(torch.tensor(0.0, device=x.device))
                all_indices.append(torch.full_like(residual[..., 0], -1, dtype=torch.long))
                continue

            # ---------------------------------------------------------
            # ACTIVE quantizer
            # ---------------------------------------------------------
            z_q, loss, indices = q(residual)

            # Per-level diagnostics — residual norm before this level
            # quantises it (catches "deep levels see vanishing residual"),
            # and the level's own loss (so the user can see which level is
            # carrying the commitment cost).
            with torch.no_grad():
                self.log_metric(
                    f"info/quantizer/residual_norm_level_{i}",
                    residual.detach().norm(),
                )
            if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                self.log_metric(
                    f"info/quantizer/commitment_loss_level_{i}",
                    loss.detach(),
                )

            quantized_outputs.append(z_q)
            losses.append(loss)
            all_indices.append(indices)

            # Residual refinement (correct for STE quantizers)
            residual = residual - z_q.detach()

        # -------- Aggregate outputs -----------------------------------------------------
        final_quantized = sum(quantized_outputs)
        total_loss = sum(losses)

        # Level contribution share — what fraction of the final quantized
        # tensor's energy comes from each level. Cheap; helps spot deep
        # levels that have stopped contributing.
        with torch.no_grad():
            denom = final_quantized.detach().pow(2).sum().clamp_min(1e-12)
            for i, q_i in enumerate(quantized_outputs):
                share = q_i.detach().pow(2).sum() / denom
                self.log_metric(
                    f"info/quantizer/level_contribution_share_{i}", share
                )

        return final_quantized, total_loss, all_indices

    def get_codebook_entry(self, indices, shape=None):
        """
        indices: Tensor of shape (B, X) or list of tensors
                assumed ordered coarse → fine if Tensor
        """

        # ------------------------------------------------------------
        # Normalize to list of per-level indices
        # ------------------------------------------------------------
        if isinstance(indices, torch.Tensor):
            B, X = indices.shape
            Q = self.num_quantizers

            if X % Q != 0:
                raise ValueError(
                    f"Total indices {X} not divisible by num_quantizers {Q}"
                )

            chunk = X // Q
            indices_list = [
                indices[:, i * chunk : (i + 1) * chunk]
                for i in range(Q)
            ]

        elif isinstance(indices, (list, tuple)):
            if len(indices) != self.num_quantizers:
                raise ValueError(
                    f"Expected {self.num_quantizers} levels, got {len(indices)}"
                )
            indices_list = list(indices)

        else:
            raise TypeError("indices must be Tensor or list/tuple")

        # ------------------------------------------------------------
        # Lookup & sum residual codebooks
        # ------------------------------------------------------------
        z_q = None

        for q, idx in zip(self.levels, indices_list):
            idx = idx.long()

            # Handle quantizer dropout
            if torch.all(idx < 0):
                continue

            z_q_i = q.get_codebook_entry(idx)

            z_q = z_q_i if z_q is None else z_q + z_q_i

        if z_q is None:
            raise RuntimeError("All quantizer levels were dropped.")

        # ------------------------------------------------------------
        # Reshape if needed
        # ------------------------------------------------------------
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

class QincoResidualQuantizer(ResidualQuantizerBase):
    def __init__(
        self,
        quantizer_class: nn.Module,
        num_quantizers: int,
        quantizer_kwargs_list: List[Dict],
        shared_codebook: bool = False,
        quantize_dropout: bool = False,
        dropout_start_level: int = 0,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.quantize_dropout = quantize_dropout
        self.dropout_start_level = dropout_start_level
        self.shared_codebook = shared_codebook

        self.levels = nn.ModuleList([
            quantizer_class(**quantizer_kwargs_list[i])
            for i in range(num_quantizers)
        ])

        if shared_codebook:
            first = self.levels[0]
            shared = first.embedding
            for q in self.levels[1:]:
                q.embedding = shared

    @property
    def e_dim(self):
        return self.levels[0].e_dim_prop

    @property
    def n_e(self):
        return self.levels[0].n_e_prop

    def forward(self, x: torch.Tensor):
        """
        x: (B, D, ...) – same as your original
        """
        residual = x
        x_prev = torch.zeros_like(x)

        quantized_outputs = []
        losses = []
        all_indices = []

        # dropout level
        if self.training and self.quantize_dropout and self.num_quantizers > 1:
            dropout_level = torch.randint(
                self.dropout_start_level,
                self.num_quantizers,
                (1,)
            ).item()
        else:
            dropout_level = self.num_quantizers

        for i, q in enumerate(self.levels):

            if i >= dropout_level:
                quantized_outputs.append(torch.zeros_like(residual))
                losses.append(torch.tensor(0.0, device=x.device))
                all_indices.append(torch.full_like(residual[..., 0], -1, dtype=torch.long))
                continue

            # QINCo: quantizer sees residual and x_prev (partial reconstruction)
            z_q, loss, indices = q(residual, x_prev=x_prev)

            with torch.no_grad():
                self.log_metric(
                    f"info/quantizer/residual_norm_level_{i}",
                    residual.detach().norm(),
                )
            if isinstance(loss, torch.Tensor) and loss.dim() == 0:
                self.log_metric(
                    f"info/quantizer/commitment_loss_level_{i}",
                    loss.detach(),
                )

            quantized_outputs.append(z_q)
            losses.append(loss)
            all_indices.append(indices)

            # update partial reconstruction
            x_prev = x_prev + z_q

            # update residual (detach like RQ init)
            residual = (x - x_prev).detach()

        final_quantized = sum(quantized_outputs)
        total_loss = sum(losses)

        with torch.no_grad():
            denom = final_quantized.detach().pow(2).sum().clamp_min(1e-12)
            for i, q_i in enumerate(quantized_outputs):
                share = q_i.detach().pow(2).sum() / denom
                self.log_metric(
                    f"info/quantizer/level_contribution_share_{i}", share
                )

        return final_quantized, total_loss, all_indices

@register_model(f"{_REGISTRY_PREFIX}grouped_residual_quantizer",
    code_url="https://github.com/yangdongchao/AcademiCodec",
    paper_url="https://arxiv.org/pdf/2305.02765",
    description="Grouped VQ for improved efficiency original uses ResidualQuantizers!")
class GroupedVQ(AbstractQuantizer):
    """
    Applies a quantizer independently on channel groups.
    Each group gets its own quantizer instance (usually ResidualQuantizer).
    """
    def __init__(
        self,
        quantizer_class: nn.Module,
        quantizer_kwargs_list: List[Dict],
        groups: int = 4,
        split_dim: int = 1,
    ):
        super().__init__()
        self.groups = groups
        self.split_dim = split_dim

        assert len(quantizer_kwargs_list) == groups, \
            "One quantizer config per group required"

        assert split_dim in [1, 2], \
            "Split dimension must be either 1 for grouping for resolution or 2 for channels"

        # Build one quantizer per group (usually ResidualQuantizer)
        self.vqs = nn.ModuleList([
            quantizer_class(**quantizer_kwargs_list[i])
            for i in range(groups)
        ])

        self.e_dim = self.vqs[0].e_dim
        self.n_e = self.vqs[0].n_e
    
    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(self, z: torch.Tensor):

        z = z.float()
        # Put channel last (2D or 3D)
        z = flatten_spatial_to_channel_last(z)

        B, C = z.shape[0], z.shape[-1]

        z_flat = z.reshape(B, -1, C)
        S = z_flat.shape[1] # sequence length
        # 1) Split channels into groups
        if self.split_dim == 1:
            assert S % self.groups == 0, f"S={S} must be divisible by groups={self.groups}"
            dim_per_group = S // self.groups
        elif self.split_dim == 2:
            assert C % self.groups == 0, f"C={C} must be divisible by groups={self.groups}"
            dim_per_group = C // self.groups
        else:
            raise ValueError(f"Invalid split dimension: {self.split_dim}, has to be either 1 for resolution or 2 for channels")

        x_groups = z_flat.split(dim_per_group, dim=self.split_dim)

        # 2) Apply VQ to each group independently
        group_results = []
        for group_x, vq in zip(x_groups, self.vqs):
            q, loss, indices = vq(group_x)
            group_results.append((q, loss, indices))

        # 3) Unpack results
        quantized_list = [r[0] for r in group_results]
        losses_list    = [r[1] for r in group_results]
        all_indices    = [r[2] for r in group_results]

        # 4) Concatenate quantized outputs across groups
        quantized = torch.cat(quantized_list, dim=self.split_dim)

        # 5) Combine losses
        total_loss = sum(losses_list)

        # Restore shape (channel-first)
        quantized = unflatten_spatial_to_channel_first(quantized)

        return quantized, total_loss, all_indices


@register_model(f"{_REGISTRY_PREFIX}msrq_vector_quantizer2",
code_url="https://github.com/FoundationVision/VAR/blob/main/models/quant.py",
paper_url="https://arxiv.org/pdf/2404.02905",)
class MultiScaleResidualQuantizer(ResidualQuantizerBase):
    """
    Multi-Scale Residual Quantizer 
    As presented in VAR: Visual Autoregressive Models
    https://arxiv.org/pdf/2404.02905

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        using_znorm: Whether to use z-normalization
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
        default_qresi_counts: Number of quantizers to use
        v_patch_nums: List of patch sizes
        quant_resi: Quantization residual ratio
        share_quant_resi: Number of quantizers to share
    """
    def __init__(
        self, 
        n_e: int,
        e_dim: int,
        using_znorm: bool = True,
        beta: float = 0.25,
        rotation_trick: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        default_qresi_counts: int = 0,
        v_patch_nums: Tuple[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        quant_resi: float = 0.5, 
        share_quant_resi: int = 4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.using_znorm = using_znorm
        self.use_ema = use_ema
        self.v_patch_nums = v_patch_nums
        self.rotation_trick = rotation_trick
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.n_e), fill_value=0.0))
        self.record_hit = 0
        
        self.beta = beta
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, decay=ema_decay, eps=ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.n_e, abs(eini) / self.n_e)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw):
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.n_e, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            encoding_indices_list = []
            level_contributions: List[torch.Tensor] = []
            for si, pn in enumerate(self.v_patch_nums):
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                # Residual norm before this scale acts on it — falls
                # monotonically with depth in a healthy decomposition.
                self.log_metric(
                    f"info/quantizer/residual_norm_level_{si}",
                    f_rest.detach().norm(),
                )

                hit_V = idx_N.bincount(minlength=self.n_e).float()
                encoding_indices_list.append(idx_N)

                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)  # This will be identity if no quant_resi was provided
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                level_contributions.append(h_BChw.detach().pow(2).sum())

                if self.training:
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1. / SN

            denom = f_hat.detach().pow(2).sum().clamp_min(1e-12)
            for i, energy in enumerate(level_contributions):
                self.log_metric(
                    f"info/quantizer/level_contribution_share_{i}",
                    energy / denom,
                )
            if self.rotation_trick:
                f_hat = rotate_to(f_hat, f_BChw)
            else:
                f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        return f_hat, mean_vq_loss, encoding_indices_list
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, n_e)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    
    def idxBl_to_msrq_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """Convert indices to MSRQ input"""
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.e_dim
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_BChw)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Get next autoregressive input"""
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')))
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_BChw)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            return f_hat, f_hat

@register_model(f"{_REGISTRY_PREFIX}msrq_vector_quantizer3d",
code_url="https://github.com/FoundationVision/VAR/blob/main/models/quant.py",
paper_url="https://arxiv.org/pdf/2404.02905",)
class MultiScaleResidualQuantizer3D(ResidualQuantizerBase):
    """
    Multi-Scale Residual Quantizer supporting both 2D and 3D inputs
    As presented in VAR: Visual Autoregressive Models
    https://arxiv.org/pdf/2404.02905

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        dims: Number of spatial dimensions (2 for 2D, 3 for 3D)
        using_znorm: Whether to use z-normalization
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
        default_qresi_counts: Number of quantizers to use
        v_patch_nums: List of patch sizes (int for cubic patches, or tuple for non-cubic)
        quant_resi: Quantization residual ratio
        share_quant_resi: Number of quantizers to share
    """
    def __init__(
        self, 
        n_e: int,
        e_dim: int,
        dims: int = 2,
        using_znorm: bool = True,
        beta: float = 0.25,
        rotation_trick: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        default_qresi_counts: int = 0,
        v_patch_nums: Tuple[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        quant_resi: float = 0.5, 
        share_quant_resi: int = 4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        assert dims in [2, 3], f"dims must be 2 or 3, got {dims}"
        self.n_e = n_e
        self.e_dim = e_dim
        self.dims = dims
        self.using_znorm = using_znorm
        self.use_ema = use_ema
        self.rotation_trick = rotation_trick
        self.quant_resi_ratio = quant_resi
        
        # Parse and normalize patch sizes once
        self.patch_sizes = self._parse_patch_sizes(v_patch_nums)
        self.v_patch_nums = v_patch_nums  # Keep original for compatibility
        
        # Set interpolation modes based on dims
        self.interp_mode_down = 'area' if dims == 2 else 'trilinear'
        self.interp_mode_up = 'bicubic' if dims == 2 else 'trilinear'
        
        # Set permute patterns based on dims
        if dims == 2:
            self.permute_to_channel_last = lambda x: x.permute(0, 2, 3, 1)
            self.permute_to_channel_first = lambda x: x.permute(0, 3, 1, 2)
        else:  # dims == 3
            self.permute_to_channel_last = lambda x: x.permute(0, 2, 3, 4, 1)
            self.permute_to_channel_first = lambda x: x.permute(0, 4, 1, 2, 3)
        
        # Create Phi or Phi3D based on dims
        from .modules import Phi, Phi3D, PhiNonShared, PhiShared, PhiPartiallyShared
        
        PhiClass = Phi if dims == 2 else Phi3D
            
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(PhiClass(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.patch_sizes))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(PhiClass(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(PhiClass(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.patch_sizes), self.n_e), fill_value=0.0))
        self.record_hit = 0
        
        self.beta = beta
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, decay=ema_decay, eps=ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def _parse_patch_sizes(self, v_patch_nums):
        """Parse patch sizes to standardized tuple format"""
        patch_sizes = []
        for pn in v_patch_nums:
            if isinstance(pn, (tuple, list)):
                if self.dims == 2:
                    patch_sizes.append((pn[0], pn[1]) if len(pn) >= 2 else (pn[0], pn[0]))
                else:
                    patch_sizes.append((pn[0], pn[1], pn[2]) if len(pn) >= 3 else (pn[0], pn[0], pn[0]))
            else:
                patch_sizes.append((pn, pn) if self.dims == 2 else (pn, pn, pn))
        return patch_sizes
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.n_e, abs(eini) / self.n_e)
    
    def extra_repr(self) -> str:
        return f'dims={self.dims}, {self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.patch_sizes)}, quant_resi={self.quant_resi_ratio}'
    
    def _get_spatial_shape(self, tensor):
        """Extract spatial dimensions from input tensor"""
        if self.dims == 2:
            return tensor.shape[2:]  # (H, W)
        else:
            return tensor.shape[2:]  # (D, H, W)
    
    def _reshape_indices(self, idx_N, B, patch_size):
        """Reshape indices to spatial grid"""
        if self.dims == 2:
            return idx_N.view(B, patch_size[0], patch_size[1])
        else:
            return idx_N.view(B, patch_size[0], patch_size[1], patch_size[2])
    
    def _compute_quantization(self, f_rest, patch_size, C, si, SN):
        """Compute quantization for a given scale"""
        if si != SN-1:
            rest_NC = F.interpolate(f_rest, size=patch_size, mode=self.interp_mode_down)
            rest_NC = self.permute_to_channel_last(rest_NC).reshape(-1, C)
        else:
            rest_NC = self.permute_to_channel_last(f_rest).reshape(-1, C)
        
        if self.using_znorm:
            rest_NC = F.normalize(rest_NC, dim=-1)
            idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
            idx_N = torch.argmin(d_no_grad, dim=1)
        
        return idx_N
    
    def _reconstruct_from_indices(self, idx_spatial, target_size, si, SN):
        """Reconstruct quantized features from indices"""
        h = self.embedding(idx_spatial)
        h = self.permute_to_channel_first(h)
        if si != SN-1:
            h = F.interpolate(h, size=target_size, mode=self.interp_mode_up).contiguous()
        else:
            h = h.contiguous()
        return self.quant_resi[si/(SN-1)](h)
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_input):
        tokenized_input = False
        if f_input.ndim == 3: 
            tokenized_input = True
            if self.dims == 2:
                f_input = rearrange(f_input, 'b (h w) c -> b c h w', h=self.patch_sizes[-1][0], w=self.patch_sizes[-1][1])
            else:
                f_input = rearrange(f_input, 'b (d h w) c -> b c d h w', d=self.patch_sizes[-1][0], h=self.patch_sizes[-1][1], w=self.patch_sizes[-1][2])
        
        dtype = f_input.dtype
        if dtype != torch.float32: f_input = f_input.float()
        
        B, C = f_input.shape[:2]
        spatial_shape = self._get_spatial_shape(f_input)
        
        f_no_grad = f_input.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.n_e, dtype=torch.float, device=f_input.device)
            SN = len(self.patch_sizes)
            encoding_indices_list = []
            level_contributions: List[torch.Tensor] = []

            for si, patch_size in enumerate(self.patch_sizes):
                idx_N = self._compute_quantization(f_rest, patch_size, C, si, SN)

                self.log_metric(
                    f"info/quantizer/residual_norm_level_{si}",
                    f_rest.detach().norm(),
                )

                hit_V = idx_N.bincount(minlength=self.n_e).float()
                encoding_indices_list.append(idx_N)

                idx_spatial = self._reshape_indices(idx_N, B, patch_size)
                h = self._reconstruct_from_indices(idx_spatial, spatial_shape, si, SN)

                f_hat = f_hat + h
                f_rest -= h
                level_contributions.append(h.detach().pow(2).sum())

                if self.training:
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_input).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1. / SN

            denom = f_hat.detach().pow(2).sum().clamp_min(1e-12)
            for i, energy in enumerate(level_contributions):
                self.log_metric(
                    f"info/quantizer/level_contribution_share_{i}",
                    energy / denom,
                )
            if self.rotation_trick:
                f_hat = rotate_to(f_hat, f_input)
            else:
                f_hat = (f_hat.data - f_no_grad).add_(f_input)

        if tokenized_input:
            if self.dims == 2:
                f_hat = rearrange(f_hat, 'b c h w -> b (h w) c', h=self.patch_sizes[-1][0], w=self.patch_sizes[-1][1])
            else:
                f_hat = rearrange(f_hat, 'b c d h w -> b (d h w) c', d=self.patch_sizes[-1][0], h=self.patch_sizes[-1][1], w=self.patch_sizes[-1][2])
        # Return in the same format as other quantizers
        return f_hat, mean_vq_loss, encoding_indices_list
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        max_size = self.patch_sizes[-1]
        min_size = self.patch_sizes[0]
        SN = len(self.patch_sizes)
        
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, *max_size, dtype=torch.float32)
            for si, h_input in enumerate(ms_h_BChw):
                h = h_input
                if si < SN - 1:
                    h = F.interpolate(h, size=max_size, mode=self.interp_mode_up)
                h = self.quant_resi[si/(SN-1)](h)
                f_hat.add_(h)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, *min_size, dtype=torch.float32)
            for si, (patch_size, h_input) in enumerate(zip(self.patch_sizes, ms_h_BChw)):
                f_hat = F.interpolate(f_hat, size=patch_size, mode=self.interp_mode_up)
                h = self.quant_resi[si/(SN-1)](h_input)
                f_hat.add_(h)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_input: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int], Tuple[int, int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:
        B, C = f_input.shape[:2]
        spatial_shape = self._get_spatial_shape(f_input)
        
        f_no_grad = f_input.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        # Use provided patch sizes or default to self.patch_sizes
        patch_sizes = self._parse_patch_sizes(v_patch_nums) if v_patch_nums is not None else self.patch_sizes
        
        # Verify final patch size matches input spatial shape
        assert patch_sizes[-1] == spatial_shape, f'{patch_sizes[-1]=} != {spatial_shape=}'
        
        SN = len(patch_sizes)
        for si, patch_size in enumerate(patch_sizes):
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            
            idx_N = self._compute_quantization(f_rest, patch_size, C, si, SN)
            idx_spatial = self._reshape_indices(idx_N, B, patch_size)
            h = self._reconstruct_from_indices(idx_spatial, spatial_shape, si, SN)
            
            f_hat.add_(h)
            f_rest.sub_(h)
            
            if to_fhat:
                f_hat_or_idx_Bl.append(f_hat.clone())
            else:
                # Flatten indices for output
                num_patches = 1
                for dim in patch_size:
                    num_patches *= dim
                f_hat_or_idx_Bl.append(idx_N.reshape(B, num_patches))
        
        return f_hat_or_idx_Bl
    
    def idxBl_to_msrq_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """Convert indices to MSRQ input"""
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.e_dim
        max_size = self.patch_sizes[-1]
        SN = len(self.patch_sizes)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, *max_size, dtype=torch.float32)
        
        for si in range(SN-1):
            patch_size_curr = self.patch_sizes[si]
            patch_size_next = self.patch_sizes[si+1]
            
            # gt_ms_idx_Bl[si] has shape (B, num_patches) - flattened indices
            # Get embeddings: (B, num_patches, C)
            h_flat = self.embedding(gt_ms_idx_Bl[si])
            # Transpose to (B, C, num_patches) and reshape to spatial
            h = h_flat.transpose(1, 2).view(B, C, *patch_size_curr)
            # Interpolate to max size
            h = F.interpolate(h, size=max_size, mode=self.interp_mode_up)
            
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h))
            
            # Downsample for next scale input
            h_down = F.interpolate(f_hat, size=patch_size_next, mode=self.interp_mode_down)
            # Flatten and transpose: (B, C, *patch_size_next) -> (B, C, num_patches) -> (B, num_patches, C)
            num_patches_next = 1
            for dim in patch_size_next:
                num_patches_next *= dim
            h_flat_next = h_down.view(B, C, num_patches_next).transpose(1, 2)
            next_scales.append(h_flat_next)
        
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Get next autoregressive input"""
        max_size = self.patch_sizes[-1]
        
        if si != SN-1:
            next_size = self.patch_sizes[si+1]
            
            # Handle both Identity and Phi cases
            h_up = F.interpolate(h_input, size=max_size, mode=self.interp_mode_up)
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_up)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_up))
            
            f_hat_down = F.interpolate(f_hat, size=next_size, mode=self.interp_mode_down)
            return f_hat, f_hat_down
        else:
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_input)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_input))
            return f_hat, f_hat

@register_model(f"{_REGISTRY_PREFIX}lookup_free_quantizer",)
class LookupFreeQuantizer(AbstractQuantizer):
    def __init__(
        self,
        token_bits: int = 10,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.02,    # from MaskBIT  https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/maskbit_tokenizer_10bit.yaml
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
    ):
        """ 
        Args:
            token_bits -> int: The number of bits per token.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature for the entropy loss.
            entropy_gamma -> float: The gamma for the entropy loss.
            dims -> int: The number of dimensions of the input.
        """
        super().__init__()
        self.token_size = token_bits
        self.codebook_size = 2 ** token_bits

        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        bits_to_indices = torch.pow(2.0, torch.arange(0, self.token_size, dtype=torch.float32))
        self.register_buffer('bits_to_indices', bits_to_indices.int())

        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.bits_to_indices) != 0).float()
        self.register_buffer('codebook', bits * 2.0 - 1.0)

    @property
    def e_dim(self):
        return self.token_size

    @property
    def n_e(self):
        return self.codebook_size

        # Ensure quantization is performed using f32
    def forward(self, z: torch.Tensor):
        z=z.float()
        # Reshape input to (B, H, W, C) or (B, D, H, W, C) style for distance computation
        z = flatten_spatial_to_channel_last(z, contiguous=True)

        ones = torch.ones_like(z)
        sign_mask = (z > 0.0)
        z_quantized = torch.where(sign_mask, ones, -ones)

        min_encoding_indices = self.convert_bits_to_indices(z_quantized)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        self.log_metric("info/quantizer/commitment_loss", commitment_loss.detach())

        # Entropy regularization via the shared base-class helper (opt-in via
        # ``entropy_loss_weight``; gated on ``self.training``). Skip the
        # affinity computation entirely when the feature is disabled.
        if self.entropy_loss_weight != 0.0 and self.training:
            d = -2 * torch.einsum('... c, n c -> ... n', z, self.codebook)
            entropy_loss = self.entropy_regularization(-1 * d)
        else:
            entropy_loss = self.entropy_regularization(z)   # short-circuits; populates cache
        _, per_sample_entropy, avg_entropy = self._last_entropy_info

        loss = commitment_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # Reshape back to original spatial dimensions
        z_q = z_quantized
        z_q = unflatten_spatial_to_channel_first(z_q, contiguous=True)

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices
        )

        # return z_quantized, result_dict # Old return
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices: torch.Tensor, shape=None) -> torch.Tensor:
        """
        indices: Tensor of shape (B, N) or (B, H, W)
        shape:   target shape, e.g. (B, C, H, W)
        """

        indices = indices.long()
        logger.debug(f"indices (before reshape): {indices.shape}")
        if shape is not None:
            indices = indices.reshape(-1, shape[-3], shape[-2])
        logger.debug(f"indices (after reshape): {indices.shape}")
        bits = ((indices[..., None] & self.bits_to_indices) != 0).float()
        tokens = bits * 2.0 - 1.0  # (..., token_bits)

        logger.debug(f"tokens: {tokens.shape}")
        return tokens

    def convert_bits_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        sign_mask = (tokens > 0.0)
        return reduce(sign_mask.int() * self.bits_to_indices, '... c -> ...', 'sum')

    def convert_indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.long()
        return self.get_codebook_entry(indices)

@register_model(f"{_REGISTRY_PREFIX}binary_spherical_quantizer",
paper_url="https://arxiv.org/pdf/2406.07548",
code_url="https://github.com/zhaoyue-zephyrus/bsq-vit",)
class BinarySphericalQuantizer(LookupFreeQuantizer):
    """BSQ by inheriting LFQ - only overrides forward with L2 normalization"""
    
    def forward(self, z: torch.Tensor):
        z = z.float()
        orig_ndim = z.ndim
        
        # Reshape to channel-last for norm/sign
        z = flatten_spatial_to_channel_last(z, contiguous=True)

        # *** BSQ CORE: L2 normalize to unit sphere ***
        z_norm = torch.norm(z, dim=-1, keepdim=True) + 1e-8
        z_unit = z / z_norm  # u = v / ||v|| [file:1]
        
        # Binary quantize on sphere
        ones = torch.ones_like(z_unit)
        sign_mask = (z_unit > 0.0)
        sign_u = torch.where(sign_mask, ones, -ones)
        sign_u = torch.where(z_unit == 0, ones, sign_u)  # sign(0) -> 1
        
        # Scale to unit sphere: hat{u} = sign(u) / sqrt(L)
        sqrt_L = 1.0 / math.sqrt(self.token_size)
        z_quantized = sqrt_L * sign_u

        # Indices from unscaled signs (LFQ compatible)
        min_encoding_indices = self.convert_bits_to_indices(sign_u)

        # Losses (use unit sphere reference)
        commitment_loss = self.commitment_cost * F.mse_loss(z_quantized.detach(), z_unit)
        self.log_metric("info/quantizer/commitment_loss", commitment_loss.detach())

        # Entropy regularization via the shared base-class helper. BSQ uses the
        # normalized input (``z_unit``) for a better soft-quantization signal.
        if self.entropy_loss_weight != 0.0 and self.training:
            d = -2 * torch.einsum('... c, n c -> ... n', z_unit, self.codebook)
            entropy_loss = self.entropy_regularization(-d)
        else:
            entropy_loss = self.entropy_regularization(z_unit)   # short-circuits; populates cache
        _, per_sample_entropy, avg_entropy = self._last_entropy_info

        loss = commitment_loss + entropy_loss

        # Straight-Through Estimator on unit sphere
        z_q = z_unit + (z_quantized - z_unit).detach()

        # Reshape back (z_q.ndim matches orig_ndim, so helper dispatches correctly)
        z_q = unflatten_spatial_to_channel_first(z_q, contiguous=True)

        return z_q, loss, min_encoding_indices


@register_model(f"{_REGISTRY_PREFIX}finite_scalar_quantizer",
                paper_url="https://arxiv.org/pdf/2309.15505",)
class FiniteScalarQuantizer(AbstractQuantizer):
    """
    Minimal Finite Scalar Quantizer compatible with your VQ wrappers.

    Args:
        levels: list of ints, number of quantization levels per scalar channel (length = code_dim)
        dim: input feature dimension (channels). If different from code_dim, projections are used.
        commitment_cost: optional float, weight for commitment loss (default 0.0).
    """
    def __init__(self, levels: List[int], dim: Optional[int] = None, commitment_cost: float = 0.0):
        super().__init__()
        assert isinstance(levels, (list, tuple)) and len(levels) > 0
        self._levels = torch.tensor(list(levels), dtype=torch.int64)           # (d,)
        self.code_dim = len(self._levels)                                     # d
        self.codebook_size = int(int(torch.prod(self._levels).item()))        # n_e (product)
        self.commitment_cost = float(commitment_cost)

        # basis for mixed-radix (digits -> index)
        basis = torch.cumprod(torch.cat((torch.tensor([1], dtype=torch.int64), self._levels[:-1].to(torch.int64))), dim=0)
        self.register_buffer("_basis", basis)     # (d,)

        # half widths and offsets (use simple formula)
        # half_width = (L - 1) / 2
        half_widths = (self._levels - 1).to(torch.float32) / 2.0   # (d,)
        offsets = torch.where((self._levels % 2) == 0, 0.5, 0.0).to(torch.float32)
        self.register_buffer("_half_widths", half_widths)
        self.register_buffer("_offsets", offsets)

        # projections if input dim != code_dim
        self.in_dim = self.code_dim if dim is None else int(dim)
        self.has_projections = (self.in_dim != self.code_dim)
        if self.has_projections:
            self.project_in = nn.Linear(self.in_dim, self.code_dim)
            self.project_out = nn.Linear(self.code_dim, self.in_dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

    @property
    def e_dim(self):
        return self.code_dim

    @property
    def n_e(self):
        return self.codebook_size

    # -------- helpers: mixed-radix index conversions ---------------------------
    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (..., d) integer-like or floats near integers representing digits in [0..L-1]
        returns indices: (...) int64
        """
        device = codes.device
        # ensure digits as integer long
        digits = torch.round(codes).long()  # (..., d)
        basis = self._basis.to(device)      # (d,)
        idx = torch.sum(digits * basis.to(digits.device), dim=-1)
        return idx

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: (...) int
        returns codes: (..., d) in integer digit space [0..L-1] (float)
        """
        device = indices.device
        indices = indices.long().unsqueeze(-1)   # (...,1)
        basis = self._basis.to(device).unsqueeze(0)  # (1,d)
        levels = self._levels.to(device).unsqueeze(0) # (1,d)
        digits = (indices // basis) % levels      # (..., d)
        return digits.to(torch.float32)

    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[tuple] = None) -> torch.Tensor:
        """
        Return code vectors for given indices in normalized [-1, 1] per code-dim space.
        shape: optional shape to view the returned codes (expects last dim = d)
        """
        digits = self.indices_to_codes(indices)  # (..., d)
        # map digits to normalized [-1,1]: digit -> centered value then / half_width
        half = self._half_widths.to(digits.device)
        centered = digits - torch.floor(self._levels.to(digits.device).to(torch.float32) / 2.0)
        normalized = centered / (half + 1e-12)
        if shape is not None:
            normalized = normalized.view(shape)
        return normalized

    # -------- bounding & quantization primitives --------------------------------
    def _bound_and_round(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple bounding: map x -> roughly [-half_width..half_width] via tanh,
        then round to integers. Return integer-like rounded digits in [-half..half].
        x expected shape (..., d)
        """
        half = self._half_widths.to(x.device)        # (d,)
        offsets = self._offsets.to(x.device)         # (d,)
        # apply tanh-based soft bound then shift
        bounded = torch.tanh(x) * half - offsets    # (..., d)
        rounded = torch.round(bounded)              # integer-like
        return rounded

    # -------- forward ----------------------------------------------------------
    def forward(self, z: torch.Tensor):
        """
        z: (B, C, H, W) or (B, C, D, H, W) or (..., in_dim)
        returns: (z_q, loss, indices)
        """
        orig_ndim = z.ndim
        z = z.float()

        # bring to channel-last layout if image/video (returns z unchanged otherwise)
        z_cl = flatten_spatial_to_channel_last(z, contiguous=True)

        shape_cl = z_cl.shape   # (..., in_dim)
        assert shape_cl[-1] == self.in_dim, f"expected last dim {self.in_dim}, got {shape_cl[-1]}"

        # project into code_dim
        flat = z_cl.view(-1, shape_cl[-1])           # (N, in_dim)
        proj = self.project_in(flat)                  # (N, d)

        # bound & round in projection space
        rounded = self._bound_and_round(proj)         # (N, d) integer-like in approx [-half..half]
        # convert to digits in [0..L-1]
        half = torch.floor(self._levels.to(rounded.device).to(torch.float32) / 2.0).to(rounded.dtype)
        digits = (rounded + half)                     # (N, d)
        levels = self._levels.to(digits.device).to(torch.float32)
        mins = torch.zeros_like(levels)
        maxs = levels - 1.0
        digits_clamped = torch.clamp(digits, min=mins, max=maxs)
        # compute mixed-radix indices
        indices_flat = torch.sum(digits_clamped.long() * self._basis.to(digits_clamped.device), dim=-1)  # (N,)

        # commitment loss (pull proj towards quantized normalized representation)
        # normalized quantized in [-1,1]
        normalized = (rounded / (half + 1e-12)).to(proj.dtype)   # (N, d)
        commitment_loss = torch.tensor(0.0, device=z.device)
        if self.commitment_cost != 0.0:
            commitment_loss = self.commitment_cost * F.mse_loss(proj.detach(), normalized)

        # reconstruct quantized in input space
        q_proj = normalized                               # (N, d)
        q_out_flat = self.project_out(q_proj)             # (N, in_dim)
        # shape back
        q_out_cl = q_out_flat.view(*shape_cl)
        z_q = unflatten_spatial_to_channel_first(q_out_cl, contiguous=True)
        if orig_ndim == 4:
            indices = indices_flat.view(z_cl.shape[0], z_cl.shape[1], z_cl.shape[2])  # (B,H,W)
        elif orig_ndim == 5:
            indices = indices_flat.view(z_cl.shape[0], z_cl.shape[1], z_cl.shape[2], z_cl.shape[3])  # (B,D,H,W)
        else:
            indices = indices_flat.view(*z_cl.shape[:-1])

        # Straight-through estimator for gradients: preserve encoder gradients
        z_q = straight_through_estimator(z, z_q)

        loss = commitment_loss

        return z_q, loss, indices


@register_model(f"{_REGISTRY_PREFIX}soft_vector_quantizer",
                paper_url="https://arxiv.org/pdf/2412.10958v1",
                code_url="https://github.com/Hhhhhhao/continuous_tokenizer/blob/f4d60a0fefe2ef94253d78333a769cb8d35de477/modelling/quantizers/softvq.py")
class SoftVectorQuantizer(AbstractQuantizer):
    def __init__(
        self,
        n_e,
        e_dim,
        entropy_loss_weight=0.01,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        tau=0.07,
        use_norm=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma
        self.use_norm = use_norm
        self.tau = tau
        
        # Single embedding layer for all codebooks
        self.embedding = nn.Parameter(torch.randn(n_e, e_dim))
        self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        
    def forward(self, z):
        # Handle different input shapes
        z = z.float()

        # Track original ndim for restoration
        orig_ndim = z.ndim

        # Put channel last (2D or 3D), same as your VQ
        z = flatten_spatial_to_channel_last(z)

        # Flatten to (N, D)
        z_flat = z.reshape(-1, self.e_dim)
        z_flat = self.norm(z_flat)  # optional L2
        embedding = self.norm(self.embedding) # optional L2

        # ------------------------------------------------------------------
        # SoftVQ: similarities, softmax over codewords, weighted sum
        # ------------------------------------------------------------------
        # Similarity logits: (N, n_e)
        logits = torch.einsum('bd,nd->bn', z_flat, embedding)  # dot product

        # Softmax over codewords with temperature
        probs = F.softmax(logits / self.tau, dim=-1)

        # Continuous quantized vector (weighted sum of codewords)
        # z_q_flat: (N, D)
        z_q_flat = torch.matmul(probs, embedding)

        # Reshape back to original channel-last shape
        z_q = z_q_flat.view_as(z)
        z_q = self.norm(z_q)  # keep same normalization behavior
        
        # Calculate cosine similarity
        # with torch.no_grad():
        #     zq_z_cos = F.cosine_similarity(
        #         z.view(-1, self.e_dim),
        #         z_q.view(-1, self.e_dim),
        #         dim=-1
        #     ).mean()
        
        # Get indices for usage tracking
        indices = torch.argmax(probs, dim=-1)  # (N,)

        # Soft-assignment diagnostics (cheap; ``probs`` is already on hand).
        with torch.no_grad():
            log_p = torch.log(probs.clamp_min(1e-10))
            per_token_H = -(probs * log_p).sum(dim=-1)
            self.log_metric(
                "info/quantizer/assignment_entropy_per_token",
                per_token_H.mean(),
            )
            self.log_metric(
                "info/quantizer/mean_top1_prob",
                probs.max(dim=-1).values.mean(),
            )

        # Entropy regularization via the shared base-class helper (opt-in via
        # ``entropy_loss_weight``; gated on ``self.training``; returns zero
        # otherwise). Canonical formula for the whole quantizer family.
        entropy_loss = self.entropy_regularization(logits)

        # Restore shape (channel-first)
        z_q = unflatten_spatial_to_channel_first(z_q)

        return z_q, entropy_loss, indices


@register_model(f"{_REGISTRY_PREFIX}diveq",
                paper_url="https://arxiv.org/pdf/2509.26469")
class DiVeQ(AbstractQuantizer):
    """DiVeQ — Differentiable Vector Quantization without auxiliary losses.

    End-to-end-trainable VQ that replaces the straight-through estimator with
    a directional-noise-based differentiable surrogate, removing the need for
    commitment / codebook losses or temperature schedules. A built-in
    codebook-replacement step periodically discards rarely-used entries by
    perturbing frequently-used ones.

    Args:
        n_e: codebook cardinality.
        e_dim: embedding dimension per code.
        noise_var: std-dev of the directional noise injected during training
            (the original paper names this ``noise_var`` even though it is the
            std passed to ``Normal``). Recommended < 1e-2.
        replacement_iters: codebook-replacement interval in training steps.
            Recommended 50–300.
        discard_threshold: usage-ratio below which a codeword is replaced
            (``0.01`` → entries used less than 1 % of the window).
        perturb_eps: magnitude of the random perturbation added to a used
            codeword when overwriting an unused entry.
        uniform_init: initialise codebook from a scaled uniform (paper
            default) instead of a scaled normal.
        allow_warning: emit ``UserWarning``s when hyperparameters are outside
            the recommended ranges.
        verbose: log codebook-replacement events.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        noise_var: float = 0.001,
        replacement_iters: int = 100,
        discard_threshold: float = 0.01,
        perturb_eps: float = 1e-9,
        uniform_init: bool = True,
        allow_warning: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.noise_var = noise_var
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.verbose = verbose

        self._check_constraints()
        if allow_warning:
            self._emit_constructor_warnings()

        # Codebook initialisation matches the paper: small-magnitude samples
        # scaled by ``1 / n_e`` so the quantizer starts near the encoder's
        # output regime.
        if uniform_init:
            codebook = torch.rand(n_e, e_dim) * (1.0 / n_e)
        else:
            codebook = torch.randn(n_e, e_dim) * (1.0 / n_e)
        # Stored as ``embedding`` (not ``codebook``) so Tier-B codebook-state
        # metrics in :class:`AbstractQuantizer` pick it up automatically.
        self.embedding = nn.Parameter(codebook)

        # Replacement bookkeeping — persistent so resumed runs keep their
        # window state. Separate from the framework's ``_usage_buffer`` which
        # is used by Tier-C snapshot metrics.
        self.register_buffer(
            "codebook_usage", torch.zeros(n_e, dtype=torch.int64)
        )
        self.register_buffer(
            "iter_counter", torch.zeros(1, dtype=torch.int64)
        )

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor):
        """Quantize ``z`` and return ``(z_q, loss, indices)``.

        ``loss`` is a zero scalar — DiVeQ has no auxiliary loss by design.
        During training the differentiable surrogate (directional noise) is
        applied; in eval mode the output is the hard nearest-codeword.
        """
        z = z.float()

        # Channel-last + flatten so the core math operates on (N, D).
        z = flatten_spatial_to_channel_last(z, contiguous=True)
        flat_shape = z.shape
        z_flat = z.reshape(-1, self.e_dim)

        codebook = self.embedding

        indices, d = nearest_codebook_entry_l2(z_flat, codebook)
        z_hard_flat = codebook[indices]

        if self.training:
            # Directional noise around (z_hard - z); the magnitude is fixed
            # to the true quantization error and only the *direction* is
            # randomised, then detached so the gradient through ``z_q``
            # reduces to identity w.r.t. ``z``.
            direction = z_hard_flat - z_flat
            noise = torch.randn_like(z_flat) * self.noise_var
            random_vectors = noise + direction
            normalized = random_vectors / torch.linalg.norm(
                random_vectors, dim=1, keepdim=True
            ).clamp_min(1e-12)
            error_magnitude = torch.linalg.norm(
                direction, dim=1, keepdim=True
            )
            vq_error = error_magnitude * normalized.detach()
            z_q_flat = z_flat + vq_error
        else:
            z_q_flat = z_hard_flat

        # Assignment-margin diagnostic (cheap, shared with VQ2/SimVQ).
        if d.size(1) >= 2:
            with torch.no_grad():
                top2 = torch.topk(d, 2, dim=1, largest=False).values
                self.log_metric(
                    "info/quantizer/assignment_margin",
                    (top2[:, 1] - top2[:, 0]).mean(),
                )

        # Codebook replacement bookkeeping — training-time only so eval
        # passes don't tick the counter.
        if self.training:
            with torch.no_grad():
                self.codebook_usage.scatter_add_(
                    0, indices, torch.ones_like(indices, dtype=self.codebook_usage.dtype)
                )
                self.iter_counter += 1
                if int(self.iter_counter.item()) % self.replacement_iters == 0:
                    self._replace_unused_entries()

        # Restore spatial shape.
        z_q = z_q_flat.view(flat_shape)
        z_q = unflatten_spatial_to_channel_first(z_q, contiguous=True)

        loss = torch.zeros((), device=z_q.device, dtype=z_q.dtype)
        return z_q, loss, indices

    # ------------------------------------------------------------------
    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        z_q = self.embedding[indices]
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    # ---------------- Utility / validation ----------------
    def _check_constraints(self) -> None:
        if self.noise_var <= 0.0:
            raise ValueError(
                "`noise_var` must be a positive float; recommended < 1e-2."
            )
        if (self.replacement_iters <= 0) or (not isinstance(self.replacement_iters, int)):
            raise ValueError(
                "`replacement_iters` must be a positive integer; recommended 50–300."
            )
        if (self.discard_threshold < 0.0) or (self.discard_threshold > 1.0):
            raise ValueError(
                "`discard_threshold` must be in [0, 1]; recommended 0.01–0.05."
            )

    def _emit_constructor_warnings(self) -> None:
        if self.noise_var > 0.01:
            warnings.warn(
                f"`noise_var`={self.noise_var} is large; values > 0.01 may"
                f" overshoot nearest-neighbor mapping.",
                UserWarning,
            )
        if self.replacement_iters < 50:
            warnings.warn(
                f"`replacement_iters`={self.replacement_iters} is small;"
                f" values < 50 may cause too-frequent codebook replacements.",
                UserWarning,
            )
        elif self.replacement_iters > 300:
            warnings.warn(
                f"`replacement_iters`={self.replacement_iters} is large;"
                f" values > 300 may cause too-sporadic codebook replacements.",
                UserWarning,
            )
        if self.discard_threshold > 0.05:
            warnings.warn(
                f"`discard_threshold`={self.discard_threshold} is large;"
                f" values > 0.05 may discard rarely-but-usefully-used codewords.",
                UserWarning,
            )
        if self.perturb_eps > 1e-6:
            warnings.warn(
                f"`perturb_eps`={self.perturb_eps} is large; values > 1e-6"
                f" may cause big perturbations from used codewords.",
                UserWarning,
            )

    @torch.no_grad()
    def _replace_unused_entries(self) -> None:
        usage_ratio = self.codebook_usage.float() / float(self.replacement_iters)
        unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
        used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

        if unused_indices.numel() == 0 or used_indices.numel() == 0:
            self.codebook_usage.zero_()
            return

        unused_count = unused_indices.numel()
        used_probs = self.codebook_usage[used_indices].float()
        used_probs = used_probs / used_probs.sum()
        sampled = used_probs.multinomial(num_samples=unused_count, replacement=True)
        sampled_indices = used_indices[sampled]

        used_codebooks = self.embedding[sampled_indices].clone()
        self.embedding.data[unused_indices] = (
            used_codebooks + self.perturb_eps * torch.randn_like(used_codebooks)
        )
        self.codebook_usage.zero_()

        self.log_metric("info/quantizer/diveq_codes_replaced", float(unused_count))
        if self.verbose:
            logger.info("DiVeQ replaced %d codewords", unused_count)

    def extra_repr(self) -> str:
        return (
            f"n_e={self.n_e}, e_dim={self.e_dim}, "
            f"noise_var={self.noise_var}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"uniform_init={self.uniform_init}"
        )


@register_model(f"{_REGISTRY_PREFIX}sf_diveq",
                paper_url="https://arxiv.org/pdf/2509.26469")
class SFDiVeQ(AbstractQuantizer):
    """SF-DiVeQ — Space-Filling Differentiable Vector Quantization.

    Variant of :class:`DiVeQ` that replaces codebook replacement with a
    space-filling curve construction. The codewords are ordered: index ``i``
    and ``i+1`` are treated as adjacent vertices of a polyline, and during
    training the quantizer interpolates along the segment between the
    nearest pair (a "dithered" codebook). Training proceeds in three phases:

    1. **Warmup pass-through** (``iter < skip_iters - avg_iters``) — quantizer
       is a no-op; encoder/decoder learn an unquantized representation.
    2. **Latent collection** (``skip_iters - avg_iters <= iter < skip_iters``) —
       still pass-through, but encoder activations are pooled. At
       ``iter == skip_iters - 1`` the pooled latents are sliced into ``n_e``
       groups (along the batch axis) and each group's mean becomes the
       initial codeword for that index, giving the codebook a built-in
       ordering along the data manifold.
    3. **Quantization** (``iter >= skip_iters``) — dithered interpolated
       quantization with DiVeQ-style directional noise routing.

    Eval mode uses segment-projection hard quantization (see the ``inference``
    method in the original paper).

    Args:
        n_e: codebook cardinality (number of polyline vertices).
        e_dim: embedding dimension.
        noise_var: std-dev of the directional noise (paper convention).
        skip_iters: training-step count at which quantization activates.
            Recommended > 1000.
        avg_iters: number of warmup steps over which to pool latents for
            codebook initialisation. Recommended 50–100.
        uniform_init: uniform vs. normal pre-init scaled by ``1/n_e``.
        allow_warning: emit ``UserWarning``s for out-of-range hyperparameters.
        latents_on_cpu: pool latents on CPU during collection to save GPU
            memory (recommended for large feature maps / batch sizes).
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        noise_var: float = 0.001,
        skip_iters: int = 1000,
        avg_iters: int = 50,
        uniform_init: bool = True,
        allow_warning: bool = True,
        latents_on_cpu: bool = False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.noise_var = noise_var
        self.skip_iters = skip_iters
        self.avg_iters = avg_iters
        self.uniform_init = uniform_init
        self.allow_warning = allow_warning
        self.latents_on_cpu = latents_on_cpu

        self._check_constraints()
        if allow_warning:
            self._emit_constructor_warnings()

        if uniform_init:
            codebook = torch.rand(n_e, e_dim) * (1.0 / n_e)
        else:
            codebook = torch.randn(n_e, e_dim) * (1.0 / n_e)
        self.embedding = nn.Parameter(codebook)

        # Persistent training-step counter so phase transitions survive
        # checkpointing. Plain Python list for the transient latent pool —
        # not registered (cleared after init) so it never touches state_dict.
        self.register_buffer(
            "iter_counter", torch.zeros(1, dtype=torch.int64)
        )
        self.latent_list: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor):
        """Quantize ``z`` and return ``(z_q, loss, indices)``.

        ``loss`` is a zero scalar — SF-DiVeQ has no auxiliary loss.
        Training drives the three-phase pipeline; eval uses
        segment-projection hard quantization.
        """
        z = z.float()
        z = flatten_spatial_to_channel_last(z, contiguous=True)
        flat_shape = z.shape
        z_flat = z.reshape(-1, self.e_dim)

        if not self.training:
            # Eval / inference path — segment-projection hard quantization.
            z_q_flat, indices = self._segment_project(z_flat)
            z_q = z_q_flat.view(flat_shape)
            z_q = unflatten_spatial_to_channel_first(z_q, contiguous=True)
            loss = torch.zeros((), device=z_q.device, dtype=z_q.dtype)
            return z_q, loss, indices

        # ── Training path ────────────────────────────────────────────────
        with torch.no_grad():
            self.iter_counter += 1
        step = int(self.iter_counter.item())

        loss = torch.zeros((), device=z_flat.device, dtype=z_flat.dtype)

        if step < self.skip_iters:
            # Phases 1 & 2: pass-through. Optionally pool latents for the
            # init step. The placeholder index tensor keeps the framework's
            # post-forward hook well-typed (and the perplexity it derives
            # will be ~1.0, a clear "warmup" signal in the logs).
            if step >= (self.skip_iters - self.avg_iters):
                pooled = z_flat.detach()
                if self.latents_on_cpu:
                    pooled = pooled.cpu()
                self.latent_list.append(pooled)
                if step == (self.skip_iters - 1):
                    self._init_codebook_from_pool(target_device=z_flat.device)

            z_q = z_flat.view(flat_shape)
            z_q = unflatten_spatial_to_channel_first(z_q, contiguous=True)
            indices = torch.zeros(
                z_flat.shape[0], device=z_flat.device, dtype=torch.int64
            )
            self.log_metric("info/quantizer/sf_diveq_active", 0.0)
            return z_q, loss, indices

        # Phase 3: dithered interpolated quantization.
        dithered, dither = self._generate_dithered_codebook(z_flat.device)
        # Distances against the dithered codebook (n_e - 1 segment points).
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + dithered.pow(2).sum(dim=1)
            - 2 * (z_flat @ dithered.t())
        )
        indices = torch.argmin(distances, dim=1)

        cb = self.embedding
        c_first = cb[indices]
        c_second = cb[indices + 1]
        interp_lambda = dither[indices]  # (N, 1)

        # Directional-noise surrogate routed independently along the two
        # segment endpoints, then summed and detached so gradient through
        # ``z_q`` is identity w.r.t. ``z_flat``.
        direction_first = c_first - z_flat
        direction_second = c_second - z_flat
        noise1 = torch.randn_like(z_flat) * self.noise_var + direction_first
        noise2 = torch.randn_like(z_flat) * self.noise_var + direction_second
        norm1 = noise1 / torch.linalg.norm(noise1, dim=1, keepdim=True).clamp_min(1e-12)
        norm2 = noise2 / torch.linalg.norm(noise2, dim=1, keepdim=True).clamp_min(1e-12)

        mag1 = torch.linalg.norm(direction_first, dim=1, keepdim=True)
        mag2 = torch.linalg.norm(direction_second, dim=1, keepdim=True)
        vq_error1 = mag1 * ((1.0 - interp_lambda) * norm1).detach()
        vq_error2 = mag2 * (interp_lambda * norm2).detach()
        z_q_flat = z_flat + vq_error1 + vq_error2

        self.log_metric("info/quantizer/sf_diveq_active", 1.0)
        if distances.size(1) >= 2:
            with torch.no_grad():
                top2 = torch.topk(distances, 2, dim=1, largest=False).values
                self.log_metric(
                    "info/quantizer/assignment_margin",
                    (top2[:, 1] - top2[:, 0]).mean(),
                )

        z_q = z_q_flat.view(flat_shape)
        z_q = unflatten_spatial_to_channel_first(z_q, contiguous=True)
        return z_q, loss, indices

    # ------------------------------------------------------------------
    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        z_q = self.embedding[indices]
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    # ---------------- Internals ----------------
    def _check_constraints(self) -> None:
        if self.noise_var <= 0.0:
            raise ValueError(
                "`noise_var` must be a positive float; recommended < 1e-2."
            )
        if self.skip_iters < 0:
            raise ValueError(
                "`skip_iters` must be a non-negative integer; recommended > 1000."
            )
        if self.avg_iters < 0:
            raise ValueError(
                "`avg_iters` must be a non-negative integer; recommended 50–100."
            )
        if self.avg_iters > self.skip_iters:
            raise ValueError("`avg_iters` must be ≤ `skip_iters`.")

    def _emit_constructor_warnings(self) -> None:
        if self.noise_var > 0.01:
            warnings.warn(
                f"`noise_var`={self.noise_var} is large; values > 0.01 may"
                f" overshoot nearest-neighbor mapping.",
                UserWarning,
            )
        if self.skip_iters < 1000:
            warnings.warn(
                f"`skip_iters`={self.skip_iters} is small; set it large"
                f" enough for the latent space to settle before quantising.",
                UserWarning,
            )
        if self.avg_iters < 50:
            warnings.warn(
                f"`avg_iters`={self.avg_iters} is small; values < 50 yield a"
                f" small latent pool for codebook init.",
                UserWarning,
            )
        elif self.avg_iters > 100:
            warnings.warn(
                f"`avg_iters`={self.avg_iters} is large; values > 100 risk"
                f" pooling out-of-date latents for codebook init.",
                UserWarning,
            )

    def _generate_dithered_codebook(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a per-step interpolated codebook of size ``n_e - 1``.

        Each segment endpoint pair ``(c_i, c_{i+1})`` is randomly dithered
        with ``λ ∈ U(0, 1)``; ``λ`` is also returned for use in the
        directional-noise routing during quantization.
        """
        dither = torch.rand((self.n_e - 1, 1), device=device)
        idx = torch.arange(self.n_e - 1, device=device, dtype=torch.long)
        c1 = self.embedding[idx]
        c2 = self.embedding[idx + 1]
        dithered = (1.0 - dither) * c1 + dither * c2
        return dithered, dither

    @torch.no_grad()
    def _init_codebook_from_pool(self, target_device: torch.device) -> None:
        """Set the codebook by averaging the pooled latents into ``n_e`` bins.

        Called exactly once, on the final warmup step (``iter == skip_iters
        - 1``). Frees the latent pool afterwards.
        """
        if not self.latent_list:
            return
        stacked = torch.cat(self.latent_list, dim=0)
        n_total = stacked.shape[0]
        hop = max(int(math.floor(n_total / self.n_e)), 1)
        new_codebook = torch.zeros(
            self.n_e, self.e_dim,
            device=stacked.device, dtype=self.embedding.dtype,
        )
        for j in range(self.n_e):
            start = j * hop
            end = start + hop
            chunk = stacked[start:end]
            if chunk.numel() == 0:
                # Pool too small to fill every bin — fall back to the last
                # available slice rather than leaving zeros.
                chunk = stacked[-hop:]
            new_codebook[j] = chunk.mean(dim=0)
        self.embedding.data = new_codebook.to(target_device).clone()
        # Free pool memory — it's transient warmup state.
        self.latent_list = []
        self.log_metric(
            "info/quantizer/sf_diveq_codebook_initialized", 1.0
        )

    def _segment_project(
        self, z_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard quantization: project ``z`` onto the polyline through the codebook.

        Picks the nearest vertex, evaluates the two adjacent segments
        (``[v-1, v]`` and ``[v, v+1]``), and returns the closer projection
        clamped to the segment endpoints.
        """
        cb = self.embedding
        n_e = self.n_e

        with torch.no_grad():
            distances = (
                z_flat.pow(2).sum(dim=1, keepdim=True)
                + cb.pow(2).sum(dim=1)
                - 2.0 * (z_flat @ cb.t())
            )
            indices = torch.argmin(distances, dim=1)

            # Two adjacent segments need ``idx-1`` and ``idx+1`` to exist.
            idx_c = indices.clamp(min=1, max=n_e - 2)
            cm = cb[idx_c - 1]
            cc = cb[idx_c]
            cp = cb[idx_c + 1]

            t_m = (((cc - cm) * (z_flat - cm)).sum(dim=1)
                   / (cc - cm).pow(2).sum(dim=1).clamp_min(1e-12)).unsqueeze(-1).clamp(0.0, 1.0)
            t_p = (((cp - cc) * (z_flat - cc)).sum(dim=1)
                   / (cp - cc).pow(2).sum(dim=1).clamp_min(1e-12)).unsqueeze(-1).clamp(0.0, 1.0)

            z_m = (1.0 - t_m) * cm + t_m * cc
            z_p = (1.0 - t_p) * cc + t_p * cp
            d_m = (z_flat - z_m).pow(2).sum(dim=1)
            d_p = (z_flat - z_p).pow(2).sum(dim=1)
            offset = (d_p < d_m).to(torch.int64) - 1   # -1 or 0

            c1 = cb[idx_c + offset]
            c2 = cb[idx_c + offset + 1]
            t = (((c2 - c1) * (z_flat - c1)).sum(dim=1)
                 / (c2 - c1).pow(2).sum(dim=1).clamp_min(1e-12)).clamp(0.0, 1.0)
            z_q = c1 + t.unsqueeze(-1) * (c2 - c1)
            # Use the boundary-clamped index for downstream logging so the
            # framework's perplexity / usage stats see a valid bin id.
            return z_q, idx_c

    def extra_repr(self) -> str:
        return (
            f"n_e={self.n_e}, e_dim={self.e_dim}, "
            f"noise_var={self.noise_var}, "
            f"skip_iters={self.skip_iters}, "
            f"avg_iters={self.avg_iters}, "
            f"uniform_init={self.uniform_init}, "
            f"latents_on_cpu={self.latents_on_cpu}"
        )

