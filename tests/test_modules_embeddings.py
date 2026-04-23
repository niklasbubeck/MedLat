"""Unit tests for medlat.modules.embeddings — AdaLN helpers + label embedders."""
from __future__ import annotations

import pytest
import torch

from medlat.modules.embeddings import (
    DatasetEmbedder,
    LabelEmbedder,
    TimestepEmbedder,
    modulate,
)


# ---------------------------------------------------------------------------
# modulate
# ---------------------------------------------------------------------------


def test_modulate_identity_shift_and_scale():
    # shift=0, scale=0 → output == x (since x * (1+0) + 0 = x)
    x = torch.randn(2, 5, 8)
    shift = torch.zeros(2, 8)
    scale = torch.zeros(2, 8)
    torch.testing.assert_close(modulate(x, shift, scale), x)


def test_modulate_constant_shift():
    x = torch.zeros(2, 3, 4)
    shift = torch.full((2, 4), 7.0)
    scale = torch.zeros(2, 4)
    out = modulate(x, shift, scale)
    assert torch.all(out == 7.0)


def test_modulate_broadcasts_across_sequence_dim():
    B, L, D = 2, 10, 6
    x = torch.randn(B, L, D)
    shift = torch.randn(B, D)
    scale = torch.randn(B, D)
    out = modulate(x, shift, scale)
    # Shape preserved
    assert out.shape == (B, L, D)
    # Each sequence position receives the same shift/scale
    manual = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    torch.testing.assert_close(out, manual)


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------


def test_timestep_embedder_output_shape():
    emb = TimestepEmbedder(hidden_size=32, frequency_embedding_size=64)
    t = torch.arange(4, dtype=torch.float32)
    out = emb(t)
    assert out.shape == (4, 32)


def test_timestep_embedder_static_embedding_shape():
    t = torch.arange(5, dtype=torch.float32)
    emb = TimestepEmbedder.timestep_embedding(t, dim=16)
    assert emb.shape == (5, 16)


def test_timestep_embedder_static_embedding_odd_dim_pads():
    t = torch.arange(3, dtype=torch.float32)
    emb = TimestepEmbedder.timestep_embedding(t, dim=11)
    assert emb.shape == (3, 11)
    assert torch.all(emb[:, -1] == 0)


# ---------------------------------------------------------------------------
# LabelEmbedder
# ---------------------------------------------------------------------------


def test_label_embedder_shape_no_cfg():
    emb = LabelEmbedder(num_classes=10, hidden_size=8, dropout_prob=0.0)
    labels = torch.tensor([0, 3, 9])
    # In eval mode, no dropout is applied.
    emb.eval()
    out = emb(labels, train=False)
    assert out.shape == (3, 8)


def test_label_embedder_adds_cfg_slot_when_dropout_positive():
    emb_no_cfg = LabelEmbedder(num_classes=10, hidden_size=8, dropout_prob=0.0)
    emb_cfg = LabelEmbedder(num_classes=10, hidden_size=8, dropout_prob=0.1)
    # With dropout, one extra class (the CFG unconditional token) is added.
    assert emb_cfg.embedding_table.num_embeddings == 11
    assert emb_no_cfg.embedding_table.num_embeddings == 10


def test_label_embedder_force_drop_ids():
    emb = LabelEmbedder(num_classes=4, hidden_size=8, dropout_prob=0.5)
    labels = torch.tensor([0, 1, 2, 3])
    # Force all labels to be dropped → all rows use the CFG (num_classes) token.
    force = torch.ones(4, dtype=torch.long)
    dropped = emb.token_drop(labels, force_drop_ids=force)
    assert torch.all(dropped == emb.num_classes)


# ---------------------------------------------------------------------------
# DatasetEmbedder (parallel to LabelEmbedder)
# ---------------------------------------------------------------------------


def test_dataset_embedder_shape():
    emb = DatasetEmbedder(num_datasets=5, hidden_size=12, dropout_prob=0.0)
    ds = torch.tensor([0, 1, 2])
    emb.eval()
    out = emb(ds, train=False)
    assert out.shape == (3, 12)
