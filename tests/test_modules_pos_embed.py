"""Unit tests for medlat.modules.pos_embed — pure-math positional embeddings.

No GPU, no trained weights. Tests assert shape invariants and a few sanity
properties (zeros for cls tokens, determinism, dtype).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medlat.modules.pos_embed import (
    apply_rotary_emb,
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
    get_4d_sincos_pos_embed,
    get_rope_tensor_2d,
    get_rope_tensor_3d,
    get_sincos_pos_embed,
    to_ntuple,
)


# ---------------------------------------------------------------------------
# to_ntuple
# ---------------------------------------------------------------------------


def test_to_ntuple_from_int():
    assert to_ntuple(4, 3) == (4, 4, 4)


def test_to_ntuple_from_sequence():
    assert to_ntuple([1, 2, 3], 3) == (1, 2, 3)
    assert to_ntuple((8, 16), 2) == (8, 16)


def test_to_ntuple_wrong_length_raises():
    with pytest.raises(ValueError, match="Expected"):
        to_ntuple([1, 2], 3)


# ---------------------------------------------------------------------------
# 1D sincos
# ---------------------------------------------------------------------------


def test_1d_sincos_from_grid_shape():
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim=16, pos=np.arange(5))
    assert emb.shape == (5, 16)


def test_1d_sincos_from_grid_requires_even_dim():
    with pytest.raises(AssertionError):
        get_1d_sincos_pos_embed_from_grid(embed_dim=15, pos=np.arange(3))


def test_1d_sincos_seq_shape():
    assert get_1d_sincos_pos_embed(8, seq_len=10).shape == (10, 8)


def test_1d_sincos_cls_token_prepends_zeros():
    emb = get_1d_sincos_pos_embed(8, seq_len=5, cls_token=True)
    assert emb.shape == (6, 8)
    assert np.all(emb[0] == 0)


def test_1d_sincos_is_deterministic():
    a = get_1d_sincos_pos_embed(16, 10)
    b = get_1d_sincos_pos_embed(16, 10)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 2D sincos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [4, (4, 6), [8, 8]])
def test_2d_sincos_shape_from_int_or_tuple(grid):
    emb = get_2d_sincos_pos_embed(embed_dim=16, grid_size=grid)
    gh, gw = (grid, grid) if isinstance(grid, int) else grid
    assert emb.shape == (gh * gw, 16)


def test_2d_sincos_cls_token_prepends_single_row():
    emb = get_2d_sincos_pos_embed(16, grid_size=4, cls_token=True)
    assert emb.shape == (4 * 4 + 1, 16)
    assert np.all(emb[0] == 0)


def test_2d_sincos_extra_tokens_prepends_rows():
    emb = get_2d_sincos_pos_embed(16, grid_size=4, extra_tokens=3)
    assert emb.shape == (4 * 4 + 3, 16)
    assert np.all(emb[:3] == 0)


# ---------------------------------------------------------------------------
# 3D sincos
# ---------------------------------------------------------------------------


def test_3d_sincos_positional_via_tuple():
    emb = get_3d_sincos_pos_embed(embed_dim=12, grid_size=(2, 3, 4))
    assert emb.shape == (2 * 3 * 4, 12)


def test_3d_sincos_via_kwargs():
    emb = get_3d_sincos_pos_embed(
        embed_dim=12, grid_depth=2, grid_height=3, grid_width=4
    )
    assert emb.shape == (2 * 3 * 4, 12)


def test_3d_sincos_without_args_raises():
    with pytest.raises(ValueError):
        get_3d_sincos_pos_embed(embed_dim=12)


def test_3d_sincos_requires_dim_divisible_by_3():
    with pytest.raises(AssertionError):
        get_3d_sincos_pos_embed(embed_dim=10, grid_size=(2, 2, 2))


# ---------------------------------------------------------------------------
# 4D sincos
# ---------------------------------------------------------------------------


def test_4d_sincos_shape():
    emb = get_4d_sincos_pos_embed(
        embed_dim=16, grid_time=2, grid_depth=3, grid_height=4, grid_width=5
    )
    assert emb.shape == (2 * 3 * 4 * 5, 16)


def test_4d_sincos_cls_token_prepends_rows():
    emb = get_4d_sincos_pos_embed(
        embed_dim=16, grid_time=1, grid_depth=2, grid_height=2, grid_width=2,
        cls_token=2,
    )
    assert emb.shape == (2 + 1 * 2 * 2 * 2, 16)
    assert np.all(emb[:2] == 0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_sincos_dispatch_2d():
    emb = get_sincos_pos_embed(16, 4, dims=2)
    assert emb.shape == (16, 16)


def test_sincos_dispatch_3d():
    emb = get_sincos_pos_embed(12, 2, dims=3)
    assert emb.shape == (8, 12)


def test_sincos_dispatch_bad_dims():
    with pytest.raises(ValueError):
        get_sincos_pos_embed(12, 2, dims=5)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def test_rope_2d_shape():
    rope = get_rope_tensor_2d(head_dim=8, height=4, width=5)
    # rope packs cos+sin along last dim → head_dim*2 total → 16
    assert rope.shape == (4 * 5, 16)
    assert rope.dtype == torch.float32


def test_rope_3d_shape():
    rope = get_rope_tensor_3d(head_dim=8, depth=2, height=3, width=4)
    assert rope.shape == (2 * 3 * 4, 16)


def test_apply_rotary_emb_preserves_shape():
    # get_rope_tensor_2d(head_dim=D, h, w) returns (h*w, 2*D) packing cos+sin;
    # chunked in apply_rotary_emb, cos/sin each have last-dim=D, which must
    # match the last dim of the query/key tensor x.
    head_dim = 8
    rope = get_rope_tensor_2d(head_dim=head_dim, height=4, width=4)
    x = torch.randn(2, 4, 16, head_dim)  # (B, H, N, D) — last dim = head_dim
    out = apply_rotary_emb(x, rope)
    assert out.shape == x.shape


def test_apply_rotary_emb_rejects_bad_rope_dim():
    x = torch.randn(2, 4, 16, 16)
    bad_rope = torch.randn(1, 2, 3, 4)  # 4D
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        apply_rotary_emb(x, bad_rope)
