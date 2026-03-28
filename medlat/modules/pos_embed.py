from __future__ import annotations

import logging
from typing import Sequence, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def to_ntuple(value: int | Sequence[int], dims: int) -> Tuple[int, ...]:
    """Normalize an int or sequence to a tuple of length *dims*."""
    if isinstance(value, Sequence):
        if len(value) != dims:
            raise ValueError(f"Expected {dims} values, got {len(value)}")
        return tuple(int(v) for v in value)
    return (int(value),) * dims


def _to_tuple(value: int | Sequence[int], dims: int) -> Tuple[int, ...]:
    if isinstance(value, Sequence):
        if len(value) != dims:
            raise ValueError(f"Expected {dims} values, got {len(value)}")
        return tuple(int(v) for v in value)
    return (int(value),) * dims


def _build_grid(size: Tuple[int, ...]) -> torch.Tensor:
    axes = [torch.arange(length, dtype=torch.float32) for length in size]
    mesh = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack(mesh, dim=-1).reshape(-1, len(size))
    return coords


# --------------------------------------------------------
# 1-D sine-cosine helpers
# --------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: array of positions to encode, shape (M,)
    returns: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, seq_len: int, cls_token: bool = False) -> np.ndarray:
    """
    Returns sincos pos embed for a 1-D sequence.
    Shape: [seq_len, embed_dim]  (or [1+seq_len, embed_dim] if cls_token=True).
    """
    pos = np.arange(seq_len, dtype=np.float32)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


# --------------------------------------------------------
# 2-D sine-cosine position embedding
# References: MAE, MoCo v3, Transformer (Google)
# --------------------------------------------------------

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | Sequence[int],
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    """
    grid_size: int (square grid) or (H, W) tuple.
    Returns pos_embed of shape [H*W, embed_dim],
    or [(extra_tokens + H*W), embed_dim] when cls_token=True and extra_tokens>0.
    """
    if isinstance(grid_size, (int, np.integer)):
        gh, gw = int(grid_size), int(grid_size)
    else:
        gh, gw = int(grid_size[0]), int(grid_size[1])

    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, gh, gw])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token or extra_tokens > 0:
        n_extra = extra_tokens if extra_tokens > 0 else 1
        pos_embed = np.concatenate([np.zeros([n_extra, embed_dim]), pos_embed], axis=0)
    return pos_embed


# --------------------------------------------------------
# 3-D sine-cosine position embedding
# --------------------------------------------------------

def get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 3 == 0
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Optional[Sequence[int]] = None,
    *,
    grid_depth: Optional[int] = None,
    grid_height: Optional[int] = None,
    grid_width: Optional[int] = None,
    cls_token: bool = False,
) -> np.ndarray:
    """
    Accepts either a (D, H, W) tuple via grid_size or explicit keyword arguments
    grid_depth / grid_height / grid_width.

    Returns shape [D*H*W, embed_dim], or [1+D*H*W, embed_dim] if cls_token=True.
    """
    if grid_size is not None:
        gd, gh, gw = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    elif grid_depth is not None and grid_height is not None and grid_width is not None:
        gd, gh, gw = int(grid_depth), int(grid_height), int(grid_width)
    else:
        raise ValueError("Provide either grid_size tuple or grid_depth/grid_height/grid_width kwargs.")

    grid_d = np.arange(gd, dtype=np.float32)
    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0).reshape([3, 1, gd, gh, gw])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# --------------------------------------------------------
# 4-D sine-cosine position embedding  (T, D, H, W)
# --------------------------------------------------------

def get_4d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 4 == 0
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[0])
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[1])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[2])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[3])
    return np.concatenate([emb_d, emb_t, emb_h, emb_w], axis=1)


def get_4d_sincos_pos_embed(
    embed_dim: int,
    grid_time: int,
    grid_depth: int,
    grid_height: int,
    grid_width: int,
    cls_token: Optional[int] = None,
) -> np.ndarray:
    """
    Returns shape [T*D*H*W, embed_dim], or [cls_token+T*D*H*W, embed_dim] if cls_token is not None.
    """
    grid_t = np.arange(grid_time, dtype=float)
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_height, dtype=float)
    grid_w = np.arange(grid_width, dtype=float)
    grid = np.meshgrid(grid_t, grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0).reshape([4, 1, grid_time, grid_depth, grid_height, grid_width])
    pos_embed = get_4d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


# convenience dispatcher
def get_sincos_pos_embed(embed_dim: int, grid_size: int | Sequence[int], dims: int) -> np.ndarray:
    """
    grid_size: int or tuple of grid dimensions.
    dims: 2 for 2D, 3 for 3D.
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size,) * dims
    if dims == 2:
        return get_2d_sincos_pos_embed(embed_dim, grid_size)
    elif dims == 3:
        return get_3d_sincos_pos_embed(embed_dim, grid_size)
    else:
        raise ValueError("dims must be 2 or 3.")


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References: DeiT
# --------------------------------------------------------

def interpolate_pos_embed(model: nn.Module, checkpoint_model: dict) -> None:
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


# --------------------------------------------------------
# RoPE (Rotary Position Embedding)
# --------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(-2)


def _build_rope_tensor_1d(head_dim: int, seq_len: int) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for rotary embeddings.")
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    cos = angles.cos()
    sin = angles.sin()
    return torch.cat([cos, sin], dim=-1)


def get_rope_tensor_2d(head_dim: int, height: int, width: int) -> torch.Tensor:
    return _build_rope_tensor_1d(head_dim * 2, height * width)


def get_rope_tensor_3d(head_dim: int, depth: int, height: int, width: int) -> torch.Tensor:
    return _build_rope_tensor_1d(head_dim * 2, depth * height * width)


def apply_rotary_emb(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    if rope.dim() == 2:
        cos, sin = rope.chunk(2, dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif rope.dim() == 3:
        cos, sin = rope.chunk(2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError("RoPE tensor must have 2 or 3 dimensions.")

    cos = cos.to(dtype=x.dtype, device=x.device)
    sin = sin.to(dtype=x.dtype, device=x.device)
    return (x * cos) + (_rotate_half(x) * sin)
