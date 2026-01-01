
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Attention, Mlp
from medtok.modules.in_and_out import PatchEmbed, ToPixel
from medtok.modules.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed

__all__ = ["DiT"]

####################################################################



class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


#####################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DatasetEmbedder(nn.Module):
    """
    Embeds dataset IDs into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_datasets, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_datasets + use_cfg_embedding, hidden_size)
        self.num_datasets = num_datasets
        self.dropout_prob = dropout_prob

    def token_drop(self, dataset_ids, force_drop_ids=None):
        """
        Drops dataset IDs to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(dataset_ids.shape[0], device=dataset_ids.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        dataset_ids = torch.where(drop_ids, self.num_datasets, dataset_ids)
        return dataset_ids

    def forward(self, dataset_ids, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            dataset_ids = self.token_drop(dataset_ids, force_drop_ids)
        embeddings = self.embedding_table(dataset_ids)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        img_size: int | tuple[int, ...] = 256,
        vae_stride: int | tuple[int, ...] = 16,
        patch_size: int | tuple[int, ...] = 2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        label_drop_prob=0.1,
        num_classes=1000,
        dataset_num=None,
        learn_sigma=True,
        dims=2,
    ):
        super().__init__()
        self.dims = dims
        self.img_size = img_size
        self.patch_size = patch_size
        if isinstance(img_size, int):
            self.img_size = (img_size,) * self.dims
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * self.dims

        print(f"img_size: {self.img_size}, vae_stride: {vae_stride}")
        self.img_size = tuple(i // v for i, v in zip(self.img_size, (vae_stride,) * self.dims))

        print(f"img_size: {self.img_size}, patch_size: {self.patch_size}")

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(to_embed='conv', img_size=self.img_size, patch_size=self.patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.to_pixel = ToPixel(to_pixel='none', img_size=self.img_size, out_channels=self.out_channels, in_dim=hidden_size, patch_size=self.patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, label_drop_prob)
        
        # Dataset ID Embedding (optional)
        self.use_dataset_conditioning = dataset_num is not None
        if self.use_dataset_conditioning:
            self.dataset_embedder = DatasetEmbedder(dataset_num, hidden_size, label_drop_prob)
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.dims == 2:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        elif self.dims == 3:
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        else:
            raise ValueError(f"dims must be 2 or 3, got {self.dims}")
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        # Initialize dataset embedding table if used:
        if self.use_dataset_conditioning:
            nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, dataset_id=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        dataset_id: (N,) tensor of dataset IDs, optional. Only used if dataset_num was provided in __init__
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        
        # Add dataset embedding if dataset conditioning is enabled
        if self.use_dataset_conditioning and dataset_id is not None:
            dataset_emb = self.dataset_embedder(dataset_id, self.training)  # (N, D)
            c = c + dataset_emb
        
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.to_pixel(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, dataset_id=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        # Prepare dataset_id for CFG: first half gets real dataset_id, second half gets fake (unconditional)
        if self.use_dataset_conditioning:
            if dataset_id is not None:
                # Concatenate real dataset_id for conditional pass and fake for unconditional pass
                dataset_id_cfg = torch.cat([
                    dataset_id,
                    torch.full_like(dataset_id, self.dataset_embedder.num_datasets)  # Use fake dataset ID for unconditional
                ], dim=0)
            else:
                # If dataset_id is None, use fake for both (fully unconditional)
                dataset_id_cfg = torch.full((combined.shape[0],), self.dataset_embedder.num_datasets, 
                                           device=combined.device, dtype=torch.long)
        else:
            dataset_id_cfg = None
        
        model_out = self.forward(combined, t, y, dataset_id_cfg)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


if __name__ == "__main__":
    model = DiT(img_size=[32, 32, 32], patch_size=[2, 2, 2])
    x = torch.randn(1, 4, 32, 32, 32) # (N, C, H, W, D)
    t = torch.randint(0, 1000, (1,))
    y = torch.randint(0)
    out = model(x, t, y)
    print(out.shape)