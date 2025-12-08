from __future__ import annotations

from typing import Sequence, Tuple

import torch


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


def get_sincos_pos_embed(
    dim: int,
    grid_size: Sequence[int] | int,
    dims: int = 2,
    cls_token: bool = False,
) -> torch.Tensor:
    if dim % dims != 0:
        raise ValueError(f"Embedding dimension {dim} must be divisible by {dims}.")
    grid = _build_grid(_to_tuple(grid_size, dims))
    per_axis = dim // dims
    inv_freq = torch.arange(per_axis // 2, dtype=torch.float32) / (per_axis // 2)
    inv_freq = 1.0 / (10000 ** inv_freq)

    embeddings = []
    for axis in range(dims):
        angles = torch.einsum("i,j->ij", grid[:, axis], inv_freq)
        embeddings.append(torch.sin(angles))
        embeddings.append(torch.cos(angles))

    pos_embed = torch.cat(embeddings, dim=1)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, pos_embed.shape[1]), pos_embed], dim=0)
    return pos_embed


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

