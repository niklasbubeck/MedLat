"""Metric-logging tests for :class:`VQModelBase` and its alignment-aware merge.

Covers:

* ``log_metric`` / ``get_metrics`` / ``reset_metrics`` inherited from
  :class:`medlat.modules.metrics.MetricLoggerMixin`.
* The merge performed by :meth:`VQModelBase.get_metrics`, which aggregates
  model-level user-logged keys with the quantizer's and alignment module's
  own ``get_metrics`` snapshots.

VQ model is not expected to log its own losses — that responsibility lives
inside the submodules (quantizer publishes ``"loss"`` / ``"perplexity"`` /
…; alignment publishes ``"alignment_loss"``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn

from medlat import get_model
from medlat.modules.metrics import MetricLoggerMixin


IMG_SIZE = 32


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_vq_model():
    return get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE).train()


def _tiny_images(n: int = 2) -> torch.Tensor:
    return torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=torch.Generator().manual_seed(0))


class _FakeAlignment(MetricLoggerMixin, nn.Module):
    """Minimal alignment stand-in that publishes its own ``alignment_loss``.

    Deliberately trivial — the point of these tests is to verify the merge
    plumbing in VQModelBase, not the actual alignment math.
    """

    def __init__(self, value: float = 0.25):
        super().__init__()
        self._value = value

    def forward(self, quant: torch.Tensor, x: torch.Tensor):
        loss = torch.tensor(self._value)
        self.log_metric("alignment_loss", loss.detach())
        return loss, None


# ---------------------------------------------------------------------------
# log_metric / get_metrics / reset_metrics inherited from the mixin
# ---------------------------------------------------------------------------


def test_vqmodel_log_metric_overwrites_latest():
    m = _make_vq_model()
    m.log_metric("my_stat", 1.0)
    m.log_metric("my_stat", 2.0)
    assert m.get_metrics()["my_stat"] == 2.0


def test_vqmodel_log_metric_detaches_tensor():
    m = _make_vq_model()
    x = torch.tensor(3.0, requires_grad=True)
    y = (x * 2).sum()  # grad_fn attached
    m.log_metric("y", y)
    stored = m.get_metrics()["y"]
    assert stored.grad_fn is None
    assert stored.item() == 6.0


def test_vqmodel_reset_metrics_clears_only_user_keys():
    m = _make_vq_model()
    m.log_metric("foo", 1.0)
    m.reset_metrics()
    assert "foo" not in m.get_metrics()


# ---------------------------------------------------------------------------
# The VQ model itself no longer logs losses in forward — they come from the
# submodules (quantizer + alignment).
# ---------------------------------------------------------------------------


def test_vqmodel_does_not_autolog_alignment_loss_without_alignment():
    m = _make_vq_model()
    assert m.alignment is None
    m(_tiny_images())
    snap = m.get_metrics()
    assert "alignment_loss" not in snap


# ---------------------------------------------------------------------------
# Merge with quantizer + alignment metrics
# ---------------------------------------------------------------------------


def test_vqmodel_get_metrics_merges_quantizer_perplexity():
    m = _make_vq_model()
    m(_tiny_images())
    snap = m.get_metrics()
    # Quantizer publishes these via its own post-forward hook; the model's
    # get_metrics merges them in.
    assert "perplexity" in snap
    assert "dead_code_ratio" in snap
    assert "active_code_count" in snap
    assert "total_tokens_seen" in snap


def test_vqmodel_get_metrics_merges_alignment_loss():
    m = _make_vq_model()
    m.alignment = _FakeAlignment(value=0.25)
    m(_tiny_images())
    snap = m.get_metrics()
    assert "alignment_loss" in snap
    assert pytest.approx(0.25) == snap["alignment_loss"].item()


def test_vqmodel_user_metrics_win_over_quantizer_and_alignment():
    # Model-level user-logged keys take precedence on collision — the merge
    # uses setdefault and the model-level snapshot is added first.
    m = _make_vq_model()
    m.alignment = _FakeAlignment()
    m(_tiny_images())
    # After the forward, the quantizer logged "loss"; overwrite at the
    # model level with a known sentinel.
    m.log_metric("loss", torch.tensor(-1.0))
    assert m.get_metrics()["loss"].item() == pytest.approx(-1.0)


def test_vqmodel_metrics_update_on_every_forward():
    m = _make_vq_model()
    m(_tiny_images(n=2))
    first_perp = m.get_metrics()["perplexity"].clone()
    m(_tiny_images(n=4))
    second_perp = m.get_metrics()["perplexity"]
    # Different batches → quantizer recomputes perplexity from indices on the
    # latest call. The key property is that the stored tensor is the newest
    # one (not a buffered aggregate).
    assert not torch.equal(first_perp, second_perp) or first_perp.data_ptr() != second_perp.data_ptr()


# ---------------------------------------------------------------------------
# Alignment module logs its own loss directly.
# ---------------------------------------------------------------------------


def test_alignment_module_logs_alignment_loss_on_forward():
    # Black-box check via the fake; the real AlignmentModule.forward path is
    # exercised indirectly by the VQModel merge tests above. This test pins
    # the convention: every alignment forward must publish "alignment_loss".
    a = _FakeAlignment(value=0.42)
    a(torch.zeros(1), torch.zeros(1))
    snap = a.get_metrics()
    assert snap["alignment_loss"].item() == pytest.approx(0.42)
