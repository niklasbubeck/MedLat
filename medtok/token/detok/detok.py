import logging
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from medtok.modules.in_and_out import PatchEmbed, ToPixel
from medtok.modules.pos_embed import get_rope_tensor_2d, get_rope_tensor_3d, apply_rotary_emb

from medtok.modules.gaussian_dist import DiagonalGaussianDistribution
from medtok.utils import init_from_ckpt
from medtok.registry import register_model

logger = logging.getLogger("DeTok")



# ================================
# Utility Functions
# ================================

def _to_tensor(x):
    return x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x)


# ================================
# Neural Network Components
# ================================


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


class Attention(nn.Module):
    """multi-head attention with optional rotary position embedding."""

    def __init__(self, dim: int, num_heads: int = 8, use_rope: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: Tensor = None) -> Tensor:
        bsz, n_ctx, ch = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads).unbind(0)
        
        if self.use_rope and rope is not None:
            q, k = apply_rotary_emb(q, rope), apply_rotary_emb(k, rope)
        
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(bsz, n_ctx, ch))


class Block(nn.Module):
    """transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.RMSNorm, eps=1e-6),
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)
        self.attn = Attention(dim, num_heads, use_rope=use_rope)
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))

    def forward(self, x: Tensor, rope: Tensor = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


# ================================
# Encoder and Decoder
# ================================


class Encoder(nn.Module):
    """vision Transformer encoder with masked autoencoding capability."""

    def __init__(
        self,
        img_size: int | tuple[int, ...] = 256,
        patch_size: int | tuple[int, ...] = 16,
        in_channels: int = 3,
        width: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        token_channels: int = 16,
        mask_ratio: float = 0.75,
        dims: int = 2,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dims = dims
        # needs to split into mean and std
        self.token_channels = token_channels * 2
        self.mask_ratio = mask_ratio
        self.in_channels = in_channels
        
        # Handle different input formats
        if isinstance(img_size, int):
            if dims == 2:
                self.img_size = (img_size, img_size)
            elif dims == 3:
                self.img_size = (img_size, img_size, img_size)
        else:
            self.img_size = img_size
            
        if isinstance(patch_size, int):
            if dims == 2:
                self.patch_size = (patch_size, patch_size)
            elif dims == 3:
                self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = patch_size
            
        # Calculate grid sizes
        if dims == 2:
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.seq_len = self.grid_size[0] * self.grid_size[1]
        elif dims == 3:
            self.grid_size = (self.img_size[0] // self.patch_size[0], 
                            self.img_size[1] // self.patch_size[1], 
                            self.img_size[2] // self.patch_size[2])
            self.seq_len = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            raise ValueError(f"Unsupported dims: {dims}")

        num_layers, num_heads, width = depth, num_heads, width
        
        self.width = width

        # Patch embedding
        self.patch_embed = PatchEmbed(to_embed='conv', img_size=img_size, patch_size=patch_size, 
                                     in_chans=in_channels, embed_dim=width)

        # learnable embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))

        # transformer layers
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer, use_rope=use_rope) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)
        self.latent_head = nn.Linear(width, self.token_channels)

        # rotary position embedding (only for 2D)
        if dims == 2 and use_rope:
            head_dim = self.transformer[0].attn.head_dim
            rope_tensor = get_rope_tensor_2d(head_dim, self.grid_size[0], self.grid_size[1]).unsqueeze(0)
            self.register_buffer("rope_tensor", rope_tensor, persistent=False)
        elif dims == 3 and use_rope:
            head_dim = self.transformer[0].attn.head_dim
            rope_tensor = get_rope_tensor_3d(head_dim, self.grid_size[0], self.grid_size[1], self.grid_size[2]).unsqueeze(0)
            self.register_buffer("rope_tensor", rope_tensor, persistent=False)
        else:
            self.register_buffer("rope_tensor", None, persistent=False)



    def mae_random_masking(self, x: Tensor, mask_ratio: float = -1):
        """apply masked autoencoding random masking."""
        bsz, seq_len, chans = x.shape
        # mask: 0 for visible, 1 for masked
        if mask_ratio == 0:
            # no masking
            rope = self.rope_tensor.expand(bsz, -1, -1)
            return x, torch.zeros(bsz, seq_len, device=x.device), None, rope

        if mask_ratio < 0:
            mask_ratio = max(0.0, random.uniform(-0.1, self.mask_ratio))
        len_keep = int(np.ceil(seq_len * (1 - mask_ratio)))
        noise = torch.rand(bsz, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep[..., None].repeat(1, 1, chans))
        rope = self.rope_tensor.expand(bsz, -1, -1)
        rope_visible = torch.gather(rope, 1, ids_keep[..., None].repeat(1, 1, rope.shape[-1]))

        mask = torch.ones(bsz, seq_len, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_visible, mask, ids_restore, rope_visible

    def forward(self, x: Tensor, mask_ratio: float = -1):
        """forward pass through encoder."""
        x = self.patch_embed(x)

        x = x + self.positional_embedding
        x, _, ids_restore, rope = self.mae_random_masking(x, mask_ratio=mask_ratio)

        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x, rope)
        x = self.ln_post(x)

        tokens = self.latent_head(x)

        return tokens, ids_restore


class Decoder(nn.Module):
    """vision Transformer decoder with mask tokens for image reconstruction."""

    def __init__(
        self,
        img_size: int | tuple[int, ...] = 256,
        patch_size: int | tuple[int, ...] = 16,
        out_channels: int = 3,
        width: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        token_channels: int = 16,
        dims: int = 2,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dims = dims
        self.width = width
        self.depth = depth
        self.num_heads = num_heads
        self.token_channels = token_channels
        self.out_channels = out_channels
        # Handle different input formats
        if isinstance(img_size, int):
            if dims == 2:
                self.img_size = (img_size, img_size)
            elif dims == 3:
                self.img_size = (img_size, img_size, img_size)
        else:
            self.img_size = img_size
            
        if isinstance(patch_size, int):
            if dims == 2:
                self.patch_size = (patch_size, patch_size)
            elif dims == 3:
                self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = patch_size
        
        # Calculate grid sizes
        if dims == 2:
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.seq_len = self.grid_size[0] * self.grid_size[1]
            self.output_channels = 3
        elif dims == 3:
            self.grid_size = (self.img_size[0] // self.patch_size[0], 
                            self.img_size[1] // self.patch_size[1], 
                            self.img_size[2] // self.patch_size[2])
            self.seq_len = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
            self.output_channels = 1  # For 3D, typically single channel
        else:
            raise ValueError(f"Unsupported dims: {dims}")

        num_layers, num_heads, width = depth, num_heads, width

        # learnable embeddings
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, width))

        # decoder layers
        self.decoder_embed = nn.Linear(self.token_channels, width)
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer, use_rope=use_rope) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)

        # output layers
        # To pixel head
        self.to_pixel = ToPixel(to_pixel="conv", img_size=self.img_size, out_channels=out_channels,
                               in_dim=width, patch_size=self.patch_size)
        # rotary position embedding
        if dims == 2 and use_rope:
            head_dim = self.transformer[0].attn.head_dim
            rope_tensor = get_rope_tensor_2d(head_dim, self.grid_size[0], self.grid_size[1]).unsqueeze(0)
            self.register_buffer("rope_tensor", rope_tensor, persistent=False)
        elif dims == 3 and use_rope:
            head_dim = self.transformer[0].attn.head_dim
            rope_tensor = get_rope_tensor_3d(head_dim, self.grid_size[0], self.grid_size[1], self.grid_size[2]).unsqueeze(0)
            self.register_buffer("rope_tensor", rope_tensor, persistent=False)
        else:
            self.register_buffer("rope_tensor", None, persistent=False)

    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None) -> Tensor:
        """forward pass through decoder."""
        z = self.decoder_embed(z_latents)
        bsz, seq_len, _ = z.shape

        if ids_restore is not None:
            num_mask_tokens = ids_restore.shape[1] + 1 - seq_len
            mask_tokens = self.mask_token.repeat(bsz, num_mask_tokens, 1)
            z_ = torch.cat([z, mask_tokens], dim=1)
            expanded_ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, z_.shape[-1])
            z = torch.gather(z_, dim=1, index=expanded_ids_restore)

        z = z + self.positional_embedding

        z = self.ln_pre(z)
        if self.rope_tensor is not None:
            rope = self.rope_tensor.expand(bsz, -1, -1)
        else:
            rope = None
        for block in self.transformer:
            z = block(z, rope)
        z = self.ln_post(z)

        z = self.to_pixel(z)

        return z


# ================================
# Main DeTok Model
# ================================


class DeTok(nn.Module): 
    """
    l-DeTok: latent denoising makes good visual tokenizers.
    Supports both 2D and 3D inputs with arbitrary dimss.
    """
    def __init__(
        self,
        image_size: int | tuple[int, ...] = 256,
        patch_size: int | tuple[int, ...] = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        enc_width: int = 768,
        dec_width: int = 768,
        enc_depth: int = 12,
        dec_depth: int = 12,
        enc_num_heads: int = 12,
        dec_num_heads: int = 12,
        token_channels: int = 16,
        mask_ratio: float = 0.75,
        gamma: float = 3.0,
        use_additive_noise: bool = False,
        dims: int = 2,
        # normalization parameters used for generative model training
        mean=0.0,
        std=1.0,
        scale_factor: float = 1.0,
        ckpt_path: str = None,
        use_rope: bool = True,
        kl_weight: float = 1e-6,
    ) -> None:
        super().__init__()

        # initialize encoder and decoder
        self.encoder = Encoder(
            img_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            width=enc_width,
            depth=enc_depth,
            num_heads=enc_num_heads,
            token_channels=token_channels,
            mask_ratio=mask_ratio,
            dims=dims,
            use_rope=use_rope,
        )
        self.decoder = Decoder(
            img_size=image_size,
            patch_size=patch_size,
            out_channels=out_channels,
            width=dec_width,
            depth=dec_depth,
            num_heads=dec_num_heads,
            token_channels=token_channels,
            dims=dims,
            use_rope=use_rope,
        )

        # model configuration
        self.dims = dims
        self.image_size = image_size
        self.patch_size = patch_size
        self.kl_weight = kl_weight
        
        # Calculate grid sizes for tokenization
        if isinstance(image_size, int):
            if dims == 2:
                self.img_size = (image_size, image_size)
            elif dims == 3:
                self.img_size = (image_size, image_size, image_size)
        else:
            self.img_size = image_size
            
        if isinstance(patch_size, int):
            if dims == 2:
                self.patch_size_tuple = (patch_size, patch_size)
            elif dims == 3:
                self.patch_size_tuple = (patch_size, patch_size, patch_size)
        else:
            self.patch_size_tuple = patch_size
            
        if dims == 2:
            self.grid_size = (self.img_size[0] // self.patch_size_tuple[0], self.img_size[1] // self.patch_size_tuple[1])
        elif dims == 3:
            self.grid_size = (self.img_size[0] // self.patch_size_tuple[0], 
                            self.img_size[1] // self.patch_size_tuple[1], 
                            self.img_size[2] // self.patch_size_tuple[2])
            
        self.width = enc_width
        self.use_additive_noise = use_additive_noise
        self.gamma = gamma

        self.scale_factor = scale_factor

        # initialize weights
        self.apply(self._init_weights)

        # setup to-posteriors function
        self.to_posteriors = partial(DiagonalGaussianDistribution, channel_dim=-1)

        # setup normalization parameters
        if isinstance(mean, np.ndarray) or isinstance(mean, list):
            mean = np.array(mean).reshape(1, -1, 1, 1)
            std = np.array(std).reshape(1, -1, 1, 1)
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("std", torch.tensor(std), persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] params: {params_M:.2f}M, {dims}D, size: {self.img_size}, patch: {self.patch_size_tuple}")
        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

    def _init_weights(self, module: nn.Module) -> None:
        """initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def freeze_everything_but_decoder(self) -> None:
        """freeze all parameters except the decoder, used for decoder fine-tuning"""
        for param in self.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = True

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] trainable params: {params_M:.2f}M (after freezing all but decoder)")

    def reset_stats(self, mean: Tensor | np.ndarray | float, std: Tensor | np.ndarray | float) -> None:
        if isinstance(mean, float) and isinstance(std, float) or (mean.ndim == 0 and std.ndim == 0):
            # a single digit global mean and global std
            self.register_buffer("mean", _to_tensor(mean), persistent=False)
            self.register_buffer("std", _to_tensor(std), persistent=False)
        else:
            n_chans = mean.shape[-1]
            self.register_buffer("mean", _to_tensor(mean).reshape(1, 1, n_chans), persistent=False)
            self.register_buffer("std", _to_tensor(std).reshape(1, 1, n_chans), persistent=False)
        logger.info(f"Resetting mean and std ({mean.shape=}, {std.shape=})")
        logger.info(f"Mean: {self.mean}")
        logger.info(f"Std: {self.std}")

    def denormalize_z(self, z: Tensor) -> Tensor:
        """denormalize latent tokens."""
        return z * self.std.to(z) / self.scale_factor + self.mean.to(z)

    def normalize_z(self, z: Tensor) -> Tensor:
        """normalize latent tokens."""
        return (z - self.mean.to(z)) * self.scale_factor / self.std.to(z)

    def encode_into_posteriors(self, x: Tensor):
        """encode image into posterior distributions."""
        z = self.encoder(x, mask_ratio=0.0)[0]
        return self.to_posteriors(z)

    def encode(self, x: Tensor, sampling: bool = False, mask_ratio: float = -1, noise_level: float = -1.0):
        """encode image into latent tokens."""
        z, ids_restore = self.encoder(x, mask_ratio=mask_ratio)

        posteriors = self.to_posteriors(z)
        z_latents = posteriors.sample() if sampling else posteriors.mean

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            bsz, n_tokens, chans = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((bsz, 1, 1), noise_level, device=device)
            else:
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
            noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise

        return z_latents, posteriors, ids_restore

    def p_loss(self, posterior, device): 
        kl_loss = torch.zeros((), device=device)
        if posterior is not None:
            # assume extra_result_dict contains posteriors with kl method
            if hasattr(posterior, "kl"):
                kl_loss = posterior.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = self.kl_weight * kl_loss
        return kl_loss

    def forward(self, x: Tensor):
        """forward pass through the entire model."""
        z_latents, posteriors, ids_restore = self.encode(x, sampling=self.training)
        decoded = self.decoder(z_latents, ids_restore=ids_restore)
        return decoded, self.p_loss(posteriors, x.device)

    def tokenize(self, x: Tensor, sampling: bool = False) -> Tensor:
        """tokenize input image and normalize the latent tokens."""
        z = self.encode(x, sampling=sampling, mask_ratio=0.0)[0]
        z = self.normalize_z(z)
        if self.dims == 2:
            z = rearrange(z, "b (h w) c -> b c h w", h=self.grid_size[0], w=self.grid_size[1])
        elif self.dims == 3:
            z = rearrange(z, "b (d h w) c -> b c d h w", d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2])
        return z

    def detokenize(self, z: Tensor) -> Tensor:
        """detokenize latent representation back to image."""
        if self.dims == 2:
            z = rearrange(z, "b c h w -> b (h w) c")
        elif self.dims == 3:
            z = rearrange(z, "b c d h w -> b (d h w) c")
        z = self.denormalize_z(z)
        decoded_images = self.decoder(z)
        return decoded_images

    def sample_from_moments(self, moments: Tensor) -> Tensor:
        """sample from latent moments."""
        z = DiagonalGaussianDistribution(moments, channel_dim=-1).sample()
        z = self.normalize_z(z)
        if self.dims == 2:
            z = rearrange(z, "b (h w) c -> b c h w", h=self.grid_size[0], w=self.grid_size[1])
        elif self.dims == 3:
            z = rearrange(z, "b (d h w) c -> b c d h w", d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2])
        return z

    @torch.inference_mode()
    def reconstruct(self, x: Tensor) -> Tensor:
        """reconstruct input image."""
        return self.detokenize(self.tokenize(x))


# ================================
# Model Factory Functions
# ================================

@register_model("token.detok.ss")
def detok_SS(**kwargs) -> DeTok:
    return DeTok(enc_width=512, dec_width=512, enc_depth=8, dec_depth=8, enc_num_heads=8, dec_num_heads=8, **kwargs)


@register_model("token.detok.sb")
def detok_SB(**kwargs) -> DeTok:
    return DeTok(enc_width=512, dec_width=768, enc_depth=8, dec_depth=12, enc_num_heads=8, dec_num_heads=12, **kwargs)


@register_model("token.detok.sl")
def detok_SL(**kwargs) -> DeTok:
    return DeTok(enc_width=512, dec_width=1024, enc_depth=8, dec_depth=24, enc_num_heads=8, dec_num_heads=16, **kwargs)


@register_model("token.detok.bs")
def detok_BS(**kwargs) -> DeTok:
    return DeTok(enc_width=768, dec_width=512, enc_depth=12, dec_depth=8, enc_num_heads=12, dec_num_heads=8, **kwargs)


@register_model("token.detok.bb")
def detok_BB(**kwargs) -> DeTok:
    return DeTok(enc_width=768, dec_width=768, enc_depth=12, dec_depth=12, enc_num_heads=12, dec_num_heads=12, **kwargs)


@register_model("token.detok.bl")
def detok_BL(**kwargs) -> DeTok:
    return DeTok(enc_width=768, dec_width=1024, enc_depth=12, dec_depth=24, enc_num_heads=12, dec_num_heads=16, **kwargs)


@register_model("token.detok.ls")
def detok_LS(**kwargs) -> DeTok:
    return DeTok(enc_width=1024, dec_width=512, enc_depth=24, dec_depth=8, enc_num_heads=16, dec_num_heads=8, **kwargs)


@register_model("token.detok.lb")
def detok_LB(**kwargs) -> DeTok:
    return DeTok(enc_width=1024, dec_width=768, enc_depth=24, dec_depth=12, enc_num_heads=16, dec_num_heads=12, **kwargs)


@register_model("token.detok.ll")
def detok_LL(**kwargs) -> DeTok:
    return DeTok(enc_width=1024, dec_width=1024, enc_depth=24, dec_depth=24, enc_num_heads=16, dec_num_heads=16, **kwargs)


@register_model("token.detok.xlxl")
def detok_XLXL(**kwargs) -> DeTok:
    return DeTok(enc_width=1152, dec_width=1152, enc_depth=28, dec_depth=28, enc_num_heads=16, dec_num_heads=16, **kwargs)


