"""
Integrated Vision Transformer Models for MAE-Tok

This module contains self-contained implementations of TimmViTEncoder and TimmViTDecoder
that include all the Vision Transformer logic directly, without relying on external timm models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import scipy.stats as stats
from typing import Optional, Tuple
from torch import Tensor

from medtok.modules.in_and_out import PatchEmbed, ToPixel
from medtok.modules.pos_embed import (
    get_sincos_pos_embed, 
    apply_rotary_emb, 
    get_rope_tensor_2d,
    get_rope_tensor_3d
)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, rope_tensor=None, num_prefix_tokens=1, num_latent_tokens=32):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope_tensor is not None:
            # Apply RoPE to the image tokens (skip prefix and latent tokens)
            # rope_tensor shape: (num_positions, head_dim * 2) where num_positions = grid_h * grid_w
            # x shape: (B, num_heads, seq_len, head_dim)
            # We need to apply RoPE to image tokens only
            if num_latent_tokens == 0:
                # Apply RoPE to all tokens except prefix
                num_img_tokens = q.shape[2] - num_prefix_tokens
                q_img = q[:, :, num_prefix_tokens:]  # (B, num_heads, num_img_tokens, head_dim)
                k_img = k[:, :, num_prefix_tokens:]
            else:
                # Apply RoPE only to image tokens (between prefix and latent)
                num_img_tokens = q.shape[2] - num_prefix_tokens - num_latent_tokens
                q_img = q[:, :, num_prefix_tokens:-num_latent_tokens]  # (B, num_heads, num_img_tokens, head_dim)
                k_img = k[:, :, num_prefix_tokens:-num_latent_tokens]
            
            # Slice rope_tensor to match number of image tokens
            # rope_tensor: (num_positions, head_dim * 2)
            # We need: (num_img_tokens, head_dim * 2)
            if rope_tensor.shape[0] < num_img_tokens:
                raise ValueError(
                    f"rope_tensor has {rope_tensor.shape[0]} positions but need {num_img_tokens} "
                    f"for image tokens. seq_len={q.shape[2]}, num_prefix_tokens={num_prefix_tokens}, "
                    f"num_latent_tokens={num_latent_tokens}"
                )
            rope_tensor_sliced = rope_tensor[:num_img_tokens]
            
            # Apply RoPE
            q_rot = apply_rotary_emb(q_img, rope_tensor_sliced)
            k_rot = apply_rotary_emb(k_img, rope_tensor_sliced)
            
            # Concatenate back
            if num_latent_tokens == 0:
                q = torch.cat([q[:, :, :num_prefix_tokens], q_rot], dim=2)
                k = torch.cat([k[:, :, :num_prefix_tokens], k_rot], dim=2)
            else:
                q = torch.cat([q[:, :, :num_prefix_tokens], q_rot, q[:, :, -num_latent_tokens:]], dim=2)
                k = torch.cat([k[:, :, :num_prefix_tokens], k_rot, k[:, :, -num_latent_tokens:]], dim=2)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLUFFN(nn.Module):
    """Swish-Gated Linear Unit Feed-Forward Network."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
                 drop_path=0., mlp_layer=Mlp, attn_layer=Attention, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, 
                             attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rope_tensor=None, num_prefix_tokens=1, num_latent_tokens=32):
        x = x + self.drop_path(self.attn(self.norm1(x), rope_tensor=rope_tensor, 
                                        num_prefix_tokens=num_prefix_tokens, num_latent_tokens=num_latent_tokens))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def modulate(x, shift, scale):
    return x * scale + shift


class MoVQNorm(nn.Module):
    def __init__(self, latent_dim, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * dim, bias=True)
        )

    def forward(self, x, interpolate_zq, num_prefix_tokens=1, num_latent_tokens=32):
        shift, scale = self.adaLN_modulation(interpolate_zq).chunk(2, dim=-1)
        norm_out = self.norm(x)
        
        norm_out = torch.cat([
            norm_out[:, :-num_latent_tokens],
            modulate(norm_out[:, -num_latent_tokens:], shift, scale)
        ], dim=1)
            
        return norm_out 


class MoVQBlockv2(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            attn_layer: nn.Module = Attention,
            latent_dim: int = None,
    ) -> None:
        assert latent_dim is not None
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=False, eps=1e-06)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=False, eps=1e-06)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 4 * dim, bias=True)
        )

    def forward(self, x: torch.Tensor, interpolate_zq: torch.Tensor,
                attn_mask: torch.Tensor = None, 
                rope_tensor=None, num_prefix_tokens=1, num_latent_tokens=32) -> torch.Tensor:
        
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(interpolate_zq).chunk(4, dim=-1)
        
        # attention
        attn_out = self.norm1(x)
        
        attn_out = torch.cat([
            attn_out[:, :-num_latent_tokens],
            modulate(attn_out[:, -num_latent_tokens:], shift_msa, scale_msa)
        ], dim=1)
            
        attn_out = self.attn(attn_out, attn_mask, rope_tensor=rope_tensor, num_prefix_tokens=num_prefix_tokens, num_latent_tokens=num_latent_tokens)
        x = x + self.drop_path1(attn_out)
        
        # mlp
        mlp_out = self.norm2(x)
        

        mlp_out = torch.cat([
            mlp_out[:, :-num_latent_tokens],
            modulate(mlp_out[:, -num_latent_tokens:], shift_mlp, scale_mlp)
        ], dim=1)
            
        mlp_out = self.mlp(mlp_out)
        x = x + self.drop_path2(mlp_out)
        
        return x

class MAETokViTEncoder(nn.Module):
    """
    Integrated Vision Transformer Encoder that includes all logic directly.
    """
    def __init__(self, in_channels=3, num_latent_tokens=32,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 img_size=224, patch_size=16, drop_path_rate=0.0,
                 qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 token_drop=0.4, token_drop_max=0.6, base_img_size=224):
        super().__init__()

        self.num_latent_tokens = num_latent_tokens
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 1  # CLS token
        
        # Patch embedding
        self.patch_embed = PatchEmbed(to_embed='conv', img_size=img_size, patch_size=patch_size, 
                                     in_chans=in_channels, embed_dim=embed_dim)
        self.num_img_tokens = self.patch_embed.num_patches
        
        # Position embedding - use sinusoidal from pos_embed.py
        self.use_ape = use_ape
        if self.use_ape:
            # Calculate grid size for position embedding
            if isinstance(img_size, int):
                grid_size = img_size // patch_size
            elif len(img_size) == 2:
                grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
            else:  # 3D
                grid_size = tuple(s // p for s, p in zip(img_size, patch_size))
            
            pos_embed = get_sincos_pos_embed(
                dim=embed_dim,
                grid_size=grid_size,
                dims=self.patch_embed.dims,
                cls_token=True
            )
            self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)
        else:
            self.pos_embed = None
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Dropout
        self.pos_drop = nn.Dropout(p=proj_drop)
        
        # Latent tokens
        if self.num_latent_tokens:
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, embed_dim))
            nn.init.normal_(self.latent_tokens, std=0.02)
            # Latent position embeddings remain learnable
            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, embed_dim))
            nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

        # Token dropout
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            self.mask_ratio_generator = stats.truncnorm(
                (token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25
            )
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)

        # RoPE
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        
        # Determine if we're using 3D based on patch_embed dimensions
        self.dims = self.patch_embed.dims
        # Force absolute positional embeddings for 3D data
        if self.dims == 3:
            self.use_rope = False
            self.use_ape = True
        
        # Initialize RoPE tensors using pos_embed.py functions
        if self.use_rope:
            # Calculate grid dimensions for RoPE
            if isinstance(img_size, int):
                grid_h = grid_w = img_size // patch_size
            elif len(img_size) == 2:
                grid_h = img_size[0] // patch_size
                grid_w = img_size[1] // patch_size
            else:
                # 3D - RoPE not supported, should have been disabled above
                grid_h = grid_w = 1
            
            # Use get_rope_tensor_2d from pos_embed.py
            # head_dim is embed_dim // num_heads
            head_dim = embed_dim // num_heads
            rope_tensor = get_rope_tensor_2d(head_dim, grid_h, grid_w)
            self.register_buffer('rope_tensor', rope_tensor)
            
            # Store grid dimensions for dynamic resizing
            self.grid_h = grid_h
            self.grid_w = grid_w
            
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, drop=proj_drop, attn_drop=attn_drop, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'latent_tokens', 'latent_pos_embed'}

    def sample_orders(self, bsz, seq_len):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                           src=torch.ones(bsz, num_masked_tokens, device=x.device))
        return mask

    def forward(self, x, return_mask=False):
        if x.ndim == 4:
            B, _, H, W = x.shape
        elif x.ndim == 5:
            B, _, H, W, D = x.shape
        else:
            raise ValueError(f"Input tensor must be 4D or 5D, got {x.ndim}D")
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # Token dropout
        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_masking(x, orders).unsqueeze(-1)
            x = torch.where(mask.bool(), self.mask_token, x)
        else:
            mask = None
        
        # Position embedding
        if self.use_ape and self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)  # (B, 1 + num_patches, embed_dim)
        

        # Add latent tokens
        if self.num_latent_tokens:
            z = self.latent_tokens.expand(x.size(0), -1, -1)  # (B, num_latent_tokens, embed_dim)
            x = torch.cat([x, z + self.latent_pos_embed], dim=1)  # (B, 1 + num_patches + num_latent_tokens, embed_dim)
        # Forward through transformer blocks
        if self.use_ape:
            for blk in self.blocks:
                x = blk(x)
        elif self.use_rope:
            # Get or compute RoPE tensor for current spatial dimensions
            # patch_size is stored as int in encoder
            patch_h = self.patch_size if isinstance(self.patch_size, int) else self.patch_size[0]
            patch_w = self.patch_size if isinstance(self.patch_size, int) else self.patch_size[1] if len(self.patch_size) > 1 else self.patch_size[0]
            
            current_grid_h = H // patch_h
            current_grid_w = W // patch_w
            
            # Recompute RoPE tensor if spatial dimensions changed
            if hasattr(self, 'grid_h') and (current_grid_h != self.grid_h or current_grid_w != self.grid_w):
                head_dim = self.embed_dim // self.blocks[0].attn.num_heads
                rope_tensor = get_rope_tensor_2d(head_dim, current_grid_h, current_grid_w)
                rope_tensor = rope_tensor.to(x.device)
            else:
                rope_tensor = self.rope_tensor
            
            for blk in self.blocks:
                x = blk(x, rope_tensor=rope_tensor, num_prefix_tokens=self.num_prefix_tokens, 
                       num_latent_tokens=self.num_latent_tokens)
        else:
            for blk in self.blocks:
                x = blk(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Extract output
        if self.num_latent_tokens:
            out = x[:, -self.num_latent_tokens:]  # (B, num_latent_tokens, embed_dim)
        else:
            out = x[:, self.num_prefix_tokens:]  # (B, num_patches, embed_dim)
        
        if return_mask:
            return out, mask
        else:
            return out


class MAETokViTDecoder(nn.Module):
    """
    Integrated Vision Transformer Decoder that includes all logic directly.
    """
    def __init__(self, in_channels=3, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 img_size=224, patch_size=16, drop_path_rate=0.0,
                 qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 num_latent_tokens=32, to_pixel='linear', codebook_embed_dim=32,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 cls_token=True, base_img_size=224, use_movq=False):
        super().__init__()

        # Handle img_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.dims = 2
        elif len(img_size) == 2:
            self.img_size = img_size
            self.dims = 2
        elif len(img_size) == 3:
            self.img_size = img_size
            self.dims = 3
        else:
            raise ValueError(f"img_size must be int or tuple of length 2 or 3, got {img_size}")
        
        # Handle patch_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(patch_size, int):
            if self.dims == 2:
                self.patch_size = (patch_size, patch_size)
            else:  # 3D
                self.patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            if self.dims == 2:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 2 not compatible with 3D img_size {img_size}")
        elif len(patch_size) == 3:
            if self.dims == 3:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 3 not compatible with 2D img_size {img_size}")
        else:
            raise ValueError(f"patch_size must be int or tuple of length 2 or 3, got {patch_size}")
        
        # Calculate number of image tokens
        if self.dims == 2:
            self.num_img_tokens = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        else:  # 3D
            self.num_img_tokens = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1]) * (self.img_size[2] // self.patch_size[2])
        
        self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if cls_token else 0
        self.num_latent_tokens = num_latent_tokens
        self.use_movq = use_movq

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Latent position embedding
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, embed_dim))
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

        # Position embedding - use sinusoidal from pos_embed.py
        self.use_ape = use_ape
        if self.use_ape:
            # Calculate grid size for position embedding
            if isinstance(img_size, int):
                grid_size = img_size // patch_size
            elif len(img_size) == 2:
                grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
            else:  # 3D
                grid_size = tuple(s // p for s, p in zip(img_size, patch_size))
            
            pos_embed = get_sincos_pos_embed(
                dim=embed_dim,
                grid_size=grid_size,
                dims=self.dims,
                cls_token=cls_token
            )
            self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)
        else:
            self.pos_embed = None

        # CLS token
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # Dropout
        self.pos_drop = nn.Dropout(p=proj_drop)

        # To pixel head
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=self.img_size, out_channels=in_channels,
                               in_dim=embed_dim, patch_size=self.patch_size)

        # RoPE
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        
        # Force absolute positional embeddings for 3D data
        if self.dims == 3:
            self.use_rope = False
            self.use_ape = True
        
        # Initialize RoPE tensors using pos_embed.py functions
        if self.use_rope:
            # Calculate grid dimensions for RoPE
            if isinstance(img_size, int):
                grid_h = grid_w = img_size // patch_size
            elif len(img_size) == 2:
                grid_h = img_size[0] // patch_size
                grid_w = img_size[1] // patch_size
            else:
                # 3D - RoPE not supported, should have been disabled above
                grid_h = grid_w = 1
            
            # Use get_rope_tensor_2d from pos_embed.py
            # head_dim is embed_dim // num_heads
            head_dim = embed_dim // num_heads
            rope_tensor = get_rope_tensor_2d(head_dim, grid_h, grid_w)
            self.register_buffer('rope_tensor', rope_tensor)
            
            # Store grid dimensions for dynamic resizing
            self.grid_h = grid_h
            self.grid_w = grid_w
           
        

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if use_movq:
            self.blocks = nn.ModuleList([
                MoVQBlockv2(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                           qk_norm=qk_norm, drop=proj_drop, attn_drop=attn_drop, drop_path=dpr[i],
                           latent_dim=codebook_embed_dim)
                for i in range(depth)
            ])
            self.norm = MoVQNorm(codebook_embed_dim, embed_dim)
            
            # Zero-out adaLN modulation layers
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
            # Zero-out output layers
            nn.init.constant_(self.norm.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.norm.adaLN_modulation[-1].bias, 0)
        else:
            self.blocks = nn.ModuleList([
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                      qk_norm=qk_norm, drop=proj_drop, attn_drop=attn_drop, drop_path=dpr[i])
                for i in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token', 'latent_pos_embed'}

    @property
    def last_layer(self):
        return self.to_pixel.get_last_layer()

    def forward(self, z, interpolate_zq=None, H=None, W=None, D=None):
        B = z.shape[0]

        num_img_tokens = self.num_img_tokens

        # Create mask tokens
        if self.num_latent_tokens:
            x = self.mask_token.expand(B, num_img_tokens, -1)
        else:
            x = z

        # Add position embedding
        if self.use_ape and self.pos_embed is not None:
            x = x + self.pos_embed[:, :num_img_tokens]
        x = self.pos_drop(x)

        # Add latent tokens with position embedding
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        # Forward through transformer blocks
        if self.use_ape:
            for i, blk in enumerate(self.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, 
                           num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)
        elif self.use_rope:
            # Get or compute RoPE tensor for current spatial dimensions
            if H is not None and W is not None:
                current_grid_h = H // (self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size)
                current_grid_w = W // (self.patch_size[1] if isinstance(self.patch_size, tuple) else self.patch_size)
            else:
                # Use default grid dimensions
                current_grid_h = self.grid_h
                current_grid_w = self.grid_w
            
            # Recompute RoPE tensor if spatial dimensions changed
            if hasattr(self, 'grid_h') and (current_grid_h != self.grid_h or current_grid_w != self.grid_w):
                head_dim = self.embed_dim // self.blocks[0].attn.num_heads
                rope_tensor = get_rope_tensor_2d(head_dim, current_grid_h, current_grid_w)
                rope_tensor = rope_tensor.to(x.device)
            else:
                rope_tensor = self.rope_tensor
            
            for i, blk in enumerate(self.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq, rope_tensor=rope_tensor, 
                           num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, rope_tensor=rope_tensor, num_prefix_tokens=self.num_prefix_tokens, 
                           num_latent_tokens=self.num_latent_tokens)
        else:
            for i, blk in enumerate(self.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, 
                           num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)

        # Final normalization
        if self.use_movq:
            x = self.norm(x, interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, 
                         num_latent_tokens=self.num_latent_tokens)
        else:
            x = self.norm(x)

        # Extract image tokens
        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        # Convert to pixel space
        out = self.to_pixel(x)
        return out


if __name__ == '__main__':
    data = torch.randn(1, 3, 224, 384)
    encoder = MAETokViTEncoder(in_channels=3, num_latent_tokens=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 img_size=[224, 384], patch_size=[16, 16], drop_path_rate=0.0,
                 qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 base_img_size=[224, 384])
    out = encoder(data)
    ### Usually here we have the quant layers to align like in MaeTok which go enc_embed_dim -> codebook_embed_dim -> decoder_embed_dim
    decoder = MAETokViTDecoder(in_channels=3, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 img_size=[224, 384], patch_size=[16, 16], drop_path_rate=0.0,
                 qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 num_latent_tokens=32, to_pixel='linear', codebook_embed_dim=32,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 cls_token=True, base_img_size=[224, 384], use_movq=False)
    out = decoder(out)


    data = torch.randn(1, 3, 224, 224, 384)
    encoder = MAETokViTEncoder(in_channels=3, num_latent_tokens=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 img_size=[224, 224, 384], patch_size=[16, 16, 16], drop_path_rate=0.0,
                 qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 base_img_size=[224, 224, 384])
    out = encoder(data)
    ### Usually here we have the "quant" layers to align like in MaeTok which go enc_embed_dim -> codebook_embed_dim -> decoder_embed_dim
    decoder = MAETokViTDecoder(in_channels=3, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                 img_size=[224, 224, 384], patch_size=[16, 16, 16], drop_path_rate=0.0,
                 qkv_bias=True, qk_norm=False, attn_drop=0., proj_drop=0.,
                 num_latent_tokens=32, to_pixel='linear', codebook_embed_dim=32,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 cls_token=True, base_img_size=[224, 224, 384], use_movq=False)
    out = decoder(out)