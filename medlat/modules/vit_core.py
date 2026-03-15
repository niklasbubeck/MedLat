from __future__ import annotations

from typing import Optional, Tuple, Literal

import torch
from scipy import stats as scipy_stats
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from .in_and_out import PatchEmbed, ToPixel
from .pos_embed import (
    get_sincos_pos_embed,
    get_rope_tensor_2d,
    get_rope_tensor_3d,
    apply_rotary_emb,
)


PosType = Literal["none", "learned", "sincos"]
MaskType = Literal["none", "mae_random"]


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer=nn.GELU, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Attention(nn.Module):
    """
    Multi-head attention with optional 2D/3D RoPE applied to a contiguous
    range of "image tokens" in the sequence.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads != 0, got {dim} and {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        rope_tensor: Optional[Tensor] = None,
        img_token_start: int = 0,
        img_token_end: Optional[int] = None,
    ) -> Tensor:
        bsz, n_ctx, ch = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads).unbind(0)

        # Apply RoPE only to image tokens if provided
        if rope_tensor is not None:
            if img_token_end is None:
                img_token_end = n_ctx
            q_img = q[:, :, img_token_start:img_token_end]
            k_img = k[:, :, img_token_start:img_token_end]

            # rope_tensor shape: (num_img_tokens, head_dim * 2)
            num_img_tokens = img_token_end - img_token_start
            if rope_tensor.shape[0] < num_img_tokens:
                raise ValueError(
                    f"RoPE tensor has {rope_tensor.shape[0]} positions but need {num_img_tokens} "
                    f"for image tokens (seq_len={n_ctx}, img_token_start={img_token_start}, img_token_end={img_token_end})."
                )
            rope_slice = rope_tensor[:num_img_tokens]
            q_rot = apply_rotary_emb(q_img, rope_slice)
            k_rot = apply_rotary_emb(k_img, rope_slice)

            q = torch.cat([q[:, :, :img_token_start], q_rot, q[:, :, img_token_end:]], dim=2)
            k = torch.cat([k[:, :, :img_token_start], k_rot, k[:, :, img_token_end:]], dim=2)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(bsz, n_ctx, ch)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: Tensor,
        rope_tensor: Optional[Tensor] = None,
        img_token_start: int = 0,
        img_token_end: Optional[int] = None,
    ) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rope_tensor=rope_tensor,
                                         img_token_start=img_token_start, img_token_end=img_token_end))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GenericViTEncoder(nn.Module):
    """
    General-purpose ViT-style encoder:

    - 2D or 3D PatchEmbed
    - optional CLS / prefix tokens
    - optional latent tokens
    - absolute pos (none / learned / sincos)
    - optional 2D/3D RoPE
    - optional MAE-style random masking

    Returns token sequence; caller can decide how to interpret it.
    """

    def __init__(
        self,
        img_size: int | Tuple[int, ...] = 224,
        patch_size: int | Tuple[int, ...] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        pos_type: PosType = "learned",
        use_rope: bool = True,
        dims: int = 2,
        num_prefix_tokens: int = 0,
        num_latent_tokens: int = 0,
        drop_path_rate: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        masking: MaskType = "none",
        mask_ratio_mu: float = 0.0,
        mask_ratio_std: float = 0.15,
        mask_ratio_max: float = 0.75,
        mask_ratio_min: float = 0.05,
        double_z: bool = False,
    ) -> None:
        super().__init__()

        self.mask_ratio_mu = mask_ratio_mu
        self.mask_ratio_std = mask_ratio_std
        self.mask_ratio_max = mask_ratio_max
        self.mask_ratio_min = mask_ratio_min

        self.patch_embed = PatchEmbed(
            to_embed="conv",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        self.embed_dim = embed_dim
        # For VQModel compatibility (used as z_channels)
        self.z_channels = embed_dim
        self.vae_stride = patch_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        self.dims = self.patch_embed.dims
        self.num_img_tokens = self.patch_embed.num_patches

        # Prefix tokens (e.g. CLS)
        if self.num_prefix_tokens > 0:
            self.prefix_tokens = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.prefix_tokens, std=0.02)
        else:
            self.prefix_tokens = None

        # Latent tokens (e.g. MAETok latent tokens)
        if self.num_latent_tokens > 0:
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.latent_tokens, std=0.02)
            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)
        else:
            self.latent_tokens = None
            self.latent_pos_embed = None

        # Absolute positional embeddings
        if pos_type == "none":
            self.pos_embed = None
        elif pos_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_prefix_tokens + self.num_img_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_type == "sincos":
            # build fixed sin-cos positions for image tokens and prepend zeros for prefix
            grid_size = self.patch_embed.grid_size
            if self.dims == 2:
                sincos = get_sincos_pos_embed(self.embed_dim, grid_size, dims=2)
            else:
                if self.embed_dim % 3 != 0:
                    raise ValueError(
                        f"3D sin-cos positional embedding requires embed_dim divisible by 3, got {self.embed_dim}"
                    )
                sincos = get_sincos_pos_embed(self.embed_dim, grid_size, dims=3)
            pos = torch.from_numpy(sincos).float()  # (num_img_tokens, dim)
            if self.num_prefix_tokens > 0:
                prefix_pos = torch.zeros(self.num_prefix_tokens, self.embed_dim)
                pos = torch.cat([prefix_pos, pos], dim=0)
            self.pos_embed = nn.Parameter(pos.unsqueeze(0), requires_grad=False)
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

        # Masking config (MAE-style)
        self.masking = masking
        if self.masking == "mae_random":
            # Avoid division by zero when std is 0; use uniform [min, max] if std too small
            std = max(self.mask_ratio_std, 1e-8)
            a = (self.mask_ratio_min - self.mask_ratio_mu) / std
            b = (self.mask_ratio_max - self.mask_ratio_mu) / std
            self.mask_ratio_generator = scipy_stats.truncnorm(a, b, loc=self.mask_ratio_mu, scale=std)
            # simple mask token
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = None

        # RoPE buffers (2D or 3D)
        self.use_rope = use_rope
        if self.use_rope:
            head_dim = self.embed_dim // num_heads
            grid = self.patch_embed.grid_size
            if self.dims == 2:
                rope = get_rope_tensor_2d(head_dim, grid[0], grid[1])
                self.grid_h, self.grid_w = grid
                self.register_buffer("rope_tensor", rope, persistent=False)
            else:
                rope = get_rope_tensor_3d(head_dim, grid[0], grid[1], grid[2])
                self.grid_d, self.grid_h, self.grid_w = grid
                self.register_buffer("rope_tensor", rope, persistent=False)
        else:
            self.register_buffer("rope_tensor", None, persistent=False)

        # Transformer blocks
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim if not double_z else 2 * self.embed_dim)
        self.variational_z_doubler = nn.Linear(self.embed_dim, 2 * self.embed_dim) if double_z else nn.Identity()

    # --- Masking ---
    def _mae_random_mask(self, x: Tensor, mask_ratio: Optional[float] = None):
        """
        x: (B, L, C) full sequence including prefix + image + latent tokens.
        We only mask *image* tokens (between prefix and before latent).
        Returns only visible tokens for efficient processing.
        """
        bsz, seq_len, _ = x.shape
        img_start = self.num_prefix_tokens
        img_end = self.num_prefix_tokens + self.num_img_tokens
        img_tokens = x[:, img_start:img_end]

        if mask_ratio is None:
            mask_ratio = float(self.mask_ratio_max)

        if mask_ratio <= 0.0:
            mask = torch.zeros(bsz, self.num_img_tokens, device=x.device)
            return x, mask, None, None

        len_keep = int(self.num_img_tokens * (1.0 - mask_ratio))

        noise = torch.rand(bsz, self.num_img_tokens, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]  # Original positions of visible tokens (in shuffled order)

        # Extract only visible tokens (not mask tokens)
        visible_img_tokens = torch.gather(img_tokens, 1, ids_keep[..., None].repeat(1, 1, img_tokens.shape[-1]))

        # Create mask for full sequence (for decoder reconstruction)
        mask = torch.ones(bsz, self.num_img_tokens, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        # Build sequence with only visible tokens: prefix + visible_img + latent
        tokens_list = []
        if self.num_prefix_tokens > 0:
            tokens_list.append(x[:, :img_start])  # prefix tokens
        tokens_list.append(visible_img_tokens)  # only visible image tokens
        if self.num_latent_tokens > 0:
            tokens_list.append(x[:, img_end:])  # latent tokens
        
        x_visible = torch.cat(tokens_list, dim=1) if len(tokens_list) > 1 else tokens_list[0]
        
        return x_visible, mask, ids_restore, ids_keep

    def forward(self, x: Tensor, mask_ratio: Optional[float] = None):
        """
        x: (B, C, H, W) or (B, C, D, H, W)
        Returns:
          tokens: (B, L, C)
          optional dict with mask / ids_restore for MAE-style usage.
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        tokens = []
        if self.prefix_tokens is not None:
            tokens.append(self.prefix_tokens.expand(B, -1, -1))
        tokens.append(x)
        seq = torch.cat(tokens, dim=1) if tokens else x  # prefix + image

        # absolute pos for prefix+image tokens
        if self.pos_embed is not None:
            seq = seq + self.pos_embed

        # append latent tokens (with their pos)
        if self.latent_tokens is not None:
            latent = self.latent_tokens.expand(B, -1, -1)
            if self.latent_pos_embed is not None:
                latent = latent + self.latent_pos_embed
            seq = torch.cat([seq, latent], dim=1)

        # masking (on image tokens only)
        mask = None
        ids_restore = None
        ids_keep = None
        num_visible_tokens = None
        if self.masking == "mae_random":
            mask_ratio = self.mask_ratio_generator.rvs(1)[0] if mask_ratio is None else mask_ratio
            print(f"mask_ratio: {mask_ratio}")
            seq, mask, ids_restore, ids_keep = self._mae_random_mask(seq, mask_ratio=mask_ratio)
            # After masking, seq contains only visible tokens
            # Calculate number of visible image tokens
            num_visible_tokens = ids_keep.shape[1] if ids_keep is not None else int(self.num_img_tokens * (1.0 - (mask_ratio if mask_ratio is not None else self.mask_ratio_max)))

        # RoPE tensor (for image tokens only)
        rope_tensor = None
        img_token_start = self.num_prefix_tokens
        # When masking, img_token_end is adjusted to only visible tokens
        if num_visible_tokens is not None:
            img_token_end = self.num_prefix_tokens + num_visible_tokens
        else:
            img_token_end = self.num_prefix_tokens + self.num_img_tokens

        if self.use_rope and self.rope_tensor is not None:
            rope_tensor_full = self.rope_tensor.to(seq.device, dtype=seq.dtype)
            if num_visible_tokens is not None and ids_keep is not None:
                # Extract RoPE for visible tokens: ids_keep contains original positions of visible tokens
                # Use first batch's ids_keep to extract RoPE (same for all batches)
                rope_tensor = rope_tensor_full[ids_keep[0]]
            else:
                rope_tensor = rope_tensor_full

        for blk in self.blocks:
            seq = blk(seq, rope_tensor=rope_tensor, img_token_start=img_token_start, img_token_end=img_token_end)

        seq = self.variational_z_doubler(seq)
            
        seq = self.norm(seq)

        # Extract tokens for output: exclude prefix tokens (CLS) from latent space
        # When masking, return only visible tokens; otherwise return all image tokens
        img_start = self.num_prefix_tokens
        
        if self.num_latent_tokens > 0:
            # Return only latent tokens (B, num_latent_tokens, C)
            tokens_out = seq[:, -self.num_latent_tokens:]
        else:
            # Return only visible image tokens (when masking) or all image tokens (when not masking)
            # Exclude prefix tokens
            tokens_out = seq[:, img_start:img_token_end]

        aux = {"mask": mask, "ids_restore": ids_restore}
        return tokens_out, aux


class GenericViTDecoder(nn.Module):
    """
    General-purpose ViT-style decoder:

    - accepts latent tokens (optionally plus mask restoration indices)
    - can insert mask tokens and restore full image token sequence (MAE-style)
    - absolute pos + optional 2D/3D RoPE
    - generic ToPixel head for reconstruction
    """

    def __init__(
        self,
        img_size: int | Tuple[int, ...] = 224,
        patch_size: int | Tuple[int, ...] = 16,
        out_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        pos_type: PosType = "learned",
        use_rope: bool = True,
        dims: int = 2,
        num_prefix_tokens: int = 0,
        num_latent_tokens: int = 0,
        drop_path_rate: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        to_pixel: str = "conv",
        token_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # number of image tokens from img_size / patch_size
        self.to_pixel = ToPixel(
            to_pixel=to_pixel,
            img_size=img_size,
            out_channels=out_channels,
            in_dim=embed_dim,
            patch_size=patch_size,
        )
        self.num_img_tokens = self.to_pixel.num_patches

        self.num_prefix_tokens = num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens

        # Prefix tokens (independent from encoder)
        if self.num_prefix_tokens > 0:
            self.prefix_tokens = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.prefix_tokens, std=0.02)
        else:
            self.prefix_tokens = None

        # mask token for MAE-style reconstruction
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Absolute positional embeddings
        if pos_type == "none":
            self.pos_embed = None
        elif pos_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_prefix_tokens + self.num_img_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_type == "sincos":
            grid_size = self.to_pixel.grid_size
            if self.to_pixel.dims == 2:
                sincos = get_sincos_pos_embed(self.embed_dim, grid_size, dims=2)
            else:
                sincos = get_sincos_pos_embed(self.embed_dim, grid_size, dims=3)
            pos = torch.from_numpy(sincos).float()
            if self.num_prefix_tokens > 0:
                prefix_pos = torch.zeros(self.num_prefix_tokens, self.embed_dim)
                pos = torch.cat([prefix_pos, pos], dim=0)
            self.pos_embed = nn.Parameter(pos.unsqueeze(0), requires_grad=False)
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

        # RoPE buffers
        self.use_rope = use_rope
        if self.use_rope:
            head_dim = self.embed_dim // num_heads
            grid = self.to_pixel.grid_size
            if self.to_pixel.dims == 2:
                rope = get_rope_tensor_2d(head_dim, grid[0], grid[1])
                self.grid_h, self.grid_w = grid
                self.register_buffer("rope_tensor", rope, persistent=False)
            else:
                rope = get_rope_tensor_3d(head_dim, grid[0], grid[1], grid[2])
                self.grid_d, self.grid_h, self.grid_w = grid
                self.register_buffer("rope_tensor", rope, persistent=False)
        else:
            self.register_buffer("rope_tensor", None, persistent=False)

        # Input projection: tokens -> embed_dim
        in_dim = token_dim or self.embed_dim
        if in_dim != self.embed_dim:
            self.decoder_embed = nn.Linear(in_dim, self.embed_dim)
        else:
            self.decoder_embed = nn.Identity()

        # Transformer blocks
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def get_last_layer(self):
        """Return the last layer weights for adaptive GAN loss weighting."""
        return self.to_pixel.get_last_layer()

    def forward(self, tokens: Tensor, ids_restore: Optional[Tensor] = None) -> Tensor:
        """
        tokens: (B, L, C_token) latent tokens or visible tokens.
        ids_restore: if provided, MAE-style unshuffling and mask-token insertion.
        """
        x = self.decoder_embed(tokens)
        B, L, _ = x.shape

        # If ids_restore provided, create masked sequence of image tokens
        if ids_restore is not None:
            num_img_tokens = ids_restore.shape[1]
            num_mask_tokens = num_img_tokens - L
            if num_mask_tokens < 0:
                raise ValueError(f"ids_restore expects at least {L} tokens, got {num_img_tokens}")
            mask_tokens = self.mask_token.expand(B, num_mask_tokens, -1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # (B, L + num_mask_tokens, C)
            x = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]))

        # Prepend decoder's own prefix tokens (independent from encoder)
        seq = x
        if self.prefix_tokens is not None:
            prefix = self.prefix_tokens.expand(B, -1, -1)
            seq = torch.cat([prefix, seq], dim=1)  # (B, num_prefix_tokens + L, C)

        # Add absolute pos on prefix+image tokens
        if self.pos_embed is not None:
            seq = seq + self.pos_embed

        img_token_start = self.num_prefix_tokens
        img_token_end = self.num_prefix_tokens + self.num_img_tokens

        rope_tensor = None
        if self.use_rope and self.rope_tensor is not None:
            rope_tensor = self.rope_tensor.to(seq.device, dtype=seq.dtype)

        for blk in self.blocks:
            seq = blk(seq, rope_tensor=rope_tensor, img_token_start=img_token_start, img_token_end=img_token_end)

        seq = self.norm(seq)

        # Drop prefix tokens before ToPixel
        seq_img = seq[:, img_token_start:img_token_end]
        out = self.to_pixel(seq_img)
        return out

