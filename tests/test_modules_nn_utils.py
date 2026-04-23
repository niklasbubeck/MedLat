"""Unit tests for medlat.modules.nn_utils — pure-math utility layer."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from medlat.modules.nn_utils import (
    GroupNorm32,
    SiLU,
    avg_pool_nd,
    conv_nd,
    linear,
    mean_flat,
    normalization,
    scale_module,
    timestep_embedding,
    update_ema,
    zero_module,
)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dims,cls", [(1, nn.Conv1d), (2, nn.Conv2d), (3, nn.Conv3d)])
def test_conv_nd_dispatch(dims, cls):
    m = conv_nd(dims, 3, 8, kernel_size=3, padding=1)
    assert isinstance(m, cls)


def test_conv_nd_unsupported_dim():
    with pytest.raises(ValueError, match="unsupported"):
        conv_nd(4, 3, 8, kernel_size=3)


@pytest.mark.parametrize(
    "dims,cls", [(1, nn.AvgPool1d), (2, nn.AvgPool2d), (3, nn.AvgPool3d)]
)
def test_avg_pool_nd_dispatch(dims, cls):
    assert isinstance(avg_pool_nd(dims, 2), cls)


def test_avg_pool_nd_unsupported_dim():
    with pytest.raises(ValueError, match="unsupported"):
        avg_pool_nd(4, 2)


def test_linear_factory_returns_nn_linear():
    m = linear(3, 5)
    assert isinstance(m, nn.Linear)
    assert m.in_features == 3 and m.out_features == 5


def test_normalization_is_groupnorm32():
    layer = normalization(64)
    assert isinstance(layer, GroupNorm32)
    assert layer.num_groups == 32
    assert layer.num_channels == 64


# ---------------------------------------------------------------------------
# Activation & norm
# ---------------------------------------------------------------------------


def test_silu_matches_torch_silu():
    x = torch.randn(4, 8)
    ref = x * torch.sigmoid(x)
    out = SiLU()(x)
    torch.testing.assert_close(out, ref)


def test_groupnorm32_preserves_dtype():
    layer = GroupNorm32(num_groups=2, num_channels=8)
    x = torch.randn(1, 8, 4, 4, dtype=torch.float16)
    out = layer(x)
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# Module parameter mutators
# ---------------------------------------------------------------------------


def test_zero_module_zeroes_all_params():
    m = nn.Linear(4, 4)
    m.weight.data.fill_(3.0)
    m.bias.data.fill_(1.0)
    out = zero_module(m)
    assert out is m, "must return the same module"
    assert torch.all(m.weight == 0)
    assert torch.all(m.bias == 0)


def test_scale_module_scales_all_params():
    m = nn.Linear(4, 4)
    m.weight.data.fill_(2.0)
    m.bias.data.fill_(3.0)
    out = scale_module(m, 0.5)
    assert out is m
    assert torch.all(m.weight == 1.0)
    assert torch.all(m.bias == 1.5)


def test_update_ema_shrinks_towards_source():
    # rate=0 → target = source
    tgt = [torch.zeros(3)]
    src = [torch.ones(3)]
    update_ema(tgt, src, rate=0.0)
    torch.testing.assert_close(tgt[0], torch.ones(3))


def test_update_ema_no_change_at_rate_one():
    # rate=1 → target unchanged
    tgt = [torch.zeros(3)]
    src = [torch.ones(3)]
    update_ema(tgt, src, rate=1.0)
    torch.testing.assert_close(tgt[0], torch.zeros(3))


def test_update_ema_intermediate():
    tgt = [torch.zeros(3)]
    src = [torch.ones(3)]
    update_ema(tgt, src, rate=0.5)
    torch.testing.assert_close(tgt[0], torch.full((3,), 0.5))


# ---------------------------------------------------------------------------
# mean_flat
# ---------------------------------------------------------------------------


def test_mean_flat_collapses_non_batch_dims():
    x = torch.ones(3, 4, 5, 6)
    out = mean_flat(x)
    assert out.shape == (3,)
    torch.testing.assert_close(out, torch.ones(3))


def test_mean_flat_2d_tensor():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # row means: [1.5, 5.5, 9.5]
    torch.testing.assert_close(mean_flat(x), torch.tensor([1.5, 5.5, 9.5]))


# ---------------------------------------------------------------------------
# timestep_embedding
# ---------------------------------------------------------------------------


def test_timestep_embedding_shape_even_dim():
    t = torch.arange(4, dtype=torch.float32)
    emb = timestep_embedding(t, dim=16)
    assert emb.shape == (4, 16)


def test_timestep_embedding_shape_odd_dim_pads():
    t = torch.arange(4, dtype=torch.float32)
    emb = timestep_embedding(t, dim=17)
    # odd dims get a zero column appended
    assert emb.shape == (4, 17)
    assert torch.all(emb[:, -1] == 0)


def test_timestep_embedding_is_deterministic():
    t = torch.arange(4, dtype=torch.float32)
    a = timestep_embedding(t, dim=32)
    b = timestep_embedding(t, dim=32)
    torch.testing.assert_close(a, b)


def test_timestep_embedding_repeat_only():
    t = torch.arange(4, dtype=torch.float32)
    emb = timestep_embedding(t, dim=8, repeat_only=True)
    assert emb.shape == (4, 8)
    # Each row must be constant (value = t)
    for i in range(4):
        assert torch.all(emb[i] == float(i))


def test_timestep_embedding_value_at_t_zero():
    # At t=0, cos(0)=1 and sin(0)=0 → first half all ones, second half all zeros.
    t = torch.zeros(1)
    emb = timestep_embedding(t, dim=8)
    torch.testing.assert_close(emb[0, :4], torch.ones(4))
    torch.testing.assert_close(emb[0, 4:], torch.zeros(4))
