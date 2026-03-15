from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np
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


def get_sincos_pos_embed(embed_dim, grid_size, dims):
    """
    grid_size: int or tuple of the grid dimensions (depth, height, width) for 3D or (height, width) for 2D
    dims: int, 2 for 2D or 3 for 3D
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size,) * dims
    if dims == 2:
        return get_2d_sincos_pos_embed(embed_dim, grid_size)
    elif dims == 3:
        return get_3d_sincos_pos_embed(embed_dim, grid_size)
    else:
        raise ValueError("dims must be 2 or 3.")

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)
    grid = np.stack(grid, axis=0).reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


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

# """"2D and 3D sine-cosine position embedding"""

# def get_sincos_pos_embed(embed_dim, grid_size, dims):
#     """
#     grid_size: int or tuple of the grid dimensions (depth, height, width) for 3D or (height, width) for 2D
#     dims: int, 2 for 2D or 3 for 3D
#     return:
#     pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#     """
#     if isinstance(grid_size, int):
#         grid_size = (grid_size,) * dims
#     if dims == 2:
#         return get_2d_sincos_pos_embed(embed_dim, grid_size)
#     elif dims == 3:
#         return get_3d_sincos_pos_embed(embed_dim, grid_size)
#     else:
#         raise ValueError("dims must be 2 or 3.")

# def get_2d_sincos_pos_embed(embed_dim, grid_size):
#     grid_h = np.arange(grid_size[0], dtype=np.float32)
#     grid_w = np.arange(grid_size[1], dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h)
#     grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
#     return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

# def get_3d_sincos_pos_embed(embed_dim, grid_size):
#     grid_d = np.arange(grid_size[0], dtype=np.float32)
#     grid_h = np.arange(grid_size[1], dtype=np.float32)
#     grid_w = np.arange(grid_size[2], dtype=np.float32)
#     grid = np.meshgrid(grid_w, grid_h, grid_d)
#     grid = np.stack(grid, axis=0).reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
#     return get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

# def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 2 == 0
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
#     return np.concatenate([emb_h, emb_w], axis=1)

# def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
#     assert embed_dim % 3 == 0
#     emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
#     emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
#     emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
#     return np.concatenate([emb_d, emb_h, emb_w], axis=1)

# def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
#     assert embed_dim % 2 == 0
#     omega = np.arange(embed_dim // 2, dtype=np.float32)
#     omega /= embed_dim / 2.
#     omega = 1. / 10000**omega
#     pos = pos.reshape(-1)
#     out = np.einsum('m,d->md', pos, omega)
#     emb_sin = np.sin(out)
#     emb_cos = np.cos(out)
#     return np.concatenate([emb_sin, emb_cos], axis=1)

# """RoPE position embedding"""


# def rotate_half(x: Tensor) -> Tensor:
#     """rotate half of the input tensor for rotary position embedding."""
#     x = rearrange(x, "... (d r) -> ... d r", r=2)
#     x1, x2 = x.unbind(dim=-1)
#     x = torch.stack((-x2, x1), dim=-1)
#     return rearrange(x, "... d r -> ... (d r)")


# def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
#     """apply rotary position embedding to input tensor."""
#     freqs_cos, freqs_sin = freqs_cis.unsqueeze(1).chunk(2, dim=-1)
#     return x * freqs_cos + rotate_half(x) * freqs_sin


# def get_rope_tensor_3d(
#     dim: int, seq_d: int, seq_h: int, seq_w: int,
#     max_freq: float = 7.0, min_freq: float = 7e-4
# ) -> Tensor:
#     """
#     Generate rotary position embedding tensor for 3D sequences.
#     dim must be divisible by 6 (cos+sin for each of D, H, W).
#     """
#     if dim % 6 != 0:
#         raise ValueError(f"dim per attn head must be divisible by 6 for 3D RoPE. Got {dim}")

#     dim_each = dim // 3  # split equally for d,h,w
#     # freqs_1d = max_freq * (max_freq / min_freq) ** torch.linspace(0, -1, dim_each // 2)
#     freqs_1d = torch.logspace(
#         start=np.log10(min_freq),
#         end=np.log10(max_freq),
#         steps=dim_each // 2,
#         base=10.0
#     )

#     # repeat for cos/sin
#     freqs_1d = torch.cat([freqs_1d, freqs_1d])  # [dim_each]

#     # frequency matrices per axis
#     freqs_3d = torch.zeros(3, dim)
#     freqs_3d[0, :dim_each] = freqs_1d  # depth axis
#     freqs_3d[1, dim_each:2 * dim_each] = freqs_1d  # height axis
#     freqs_3d[2, 2 * dim_each:] = freqs_1d  # width axis
#     freqs_3d = freqs_3d * 2 * torch.pi

#     # coordinate grid [seq_d * seq_h * seq_w, 3]
#     coord_d = torch.linspace(0, 1, seq_d)
#     coord_h = torch.linspace(0, 1, seq_h)
#     coord_w = torch.linspace(0, 1, seq_w)
#     coords_all = torch.cartesian_prod(coord_d, coord_h, coord_w)

#     # angles: [N, dim]
#     angle = coords_all @ freqs_3d
#     rope_tensor = torch.cat([angle.cos(), angle.sin()], dim=-1)  # [N, 2*dim]
#     return rope_tensor


# def get_rope_tensor_2d(
#     dim: int, seq_h: int, seq_w: int, max_freq: float = 7.0, min_freq: float = 7e-4
# ) -> Tensor:
#     """generate rotary position embedding tensor for 2D sequences."""
#     freqs_1d = max_freq * (max_freq / min_freq) ** torch.linspace(0, -1, dim // 4)
#     freqs_1d = torch.cat([freqs_1d, freqs_1d])
#     freqs_2d = torch.zeros(2, dim)
#     freqs_2d[0, : dim // 2] = freqs_1d
#     freqs_2d[1, -dim // 2 :] = freqs_1d
#     freqs_2d = freqs_2d * 2 * torch.pi
#     coord_x = torch.linspace(0, 1, seq_h)
#     coord_y = torch.linspace(0, 1, seq_w)
#     coords_all = torch.cartesian_prod(coord_x, coord_y)
#     angle = coords_all @ freqs_2d
#     rope_tensor = torch.cat([angle.cos(), angle.sin()], dim=-1)
#     return rope_tensor
