
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
from medlat.modules.in_and_out import PatchEmbed, ToPixel
from medlat.modules.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from medlat.modules.embeddings import (
    modulate,
    TimestepEmbedder,
    LabelEmbedder,
    DatasetEmbedder,
    FinalLayer,
)

__all__ = ["DiT"]


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond_dim=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # FIXED: Accept cond_dim input (2x or 3x hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim or hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """
    Improved Diffusion Transformer with strong dataset+label conditioning.
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
        class_label_drop_prob=0.1,
        dataset_label_drop_prob=0.05,
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

        self.img_size = tuple(i // v for i, v in zip(self.img_size, (vae_stride,) * self.dims))

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # FIXED: Support 3 conditions (t, y, dataset) -> 3*hidden_size input
        self.cond_dim = 3 * hidden_size if dataset_num is not None else 2 * hidden_size

        self.x_embedder = PatchEmbed(to_embed='conv', img_size=self.img_size, patch_size=self.patch_size,
                                   in_chans=in_channels, embed_dim=hidden_size)
        self.to_pixel = ToPixel(to_pixel='none', img_size=self.img_size, out_channels=self.out_channels,
                              in_dim=hidden_size, patch_size=self.patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_label_drop_prob)

        self.use_dataset_conditioning = dataset_num is not None
        if self.use_dataset_conditioning:
            self.dataset_embedder = DatasetEmbedder(dataset_num, hidden_size, dataset_label_drop_prob)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cond_dim=self.cond_dim)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels, cond_dim=self.cond_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.dims == 2:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        elif self.dims == 3:
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        else:
            raise ValueError(f"dims must be 2 or 3, got {self.dims}")
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if self.use_dataset_conditioning:
            nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.05)  # Stronger dataset signal

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers (now takes cond_dim input)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, dataset_id=None):
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)

        # FIXED: Concatenate instead of add to avoid signal interference
        c_list = [t_emb, y_emb]
        if self.use_dataset_conditioning and dataset_id is not None:
            ds_emb = self.dataset_embedder(dataset_id, self.training)
            c_list.append(ds_emb)

        c = torch.cat(c_list, dim=1)  # (N, cond_dim)

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.to_pixel(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, dataset_id=None, cfg_dataset=True):
        """
        Flexible CFG: control whether dataset_id gets CFG or stays fixed.

        Args:
            cfg_dataset (bool): If True, apply CFG to both labels + dataset_id
                            If False, apply CFG only to labels (dataset_id fixed)

        Perfect for:
        - cfg_dataset=True:  Full control (dataset switching)
        - cfg_dataset=False: Fixed dataset + label control only (RetinaMNIST classes)
        """
        batch_size = x.shape[0]

        # DOUBLE x and t
        x_doubled = torch.cat([x, x], dim=0)
        t_doubled = torch.cat([t, t], dim=0)

        # CLASS-LABEL CFG (always applied)
        y_cond = y
        y_uncond = torch.full_like(y, self.y_embedder.num_classes)
        y_doubled = torch.cat([y_cond, y_uncond], dim=0)

        # DATASET CFG (user controlled)
        if self.use_dataset_conditioning and dataset_id is not None:
            if cfg_dataset:
                # Full CFG: conditional + unconditional dataset
                ds_cond = dataset_id
                ds_uncond = torch.full_like(dataset_id, self.dataset_embedder.num_datasets)
                dataset_id_doubled = torch.cat([ds_cond, ds_uncond], dim=0)
            else:
                # FIXED dataset: same for both passes
                dataset_id_doubled = torch.cat([dataset_id, dataset_id], dim=0)
        else:
            dataset_id_doubled = None

        # Forward pass
        model_out_doubled = self.forward(x_doubled, t_doubled, y_doubled, dataset_id_doubled)

        # Apply CFG
        cond_out, uncond_out = torch.split(model_out_doubled, batch_size, dim=0)
        guided_out = uncond_out + cfg_scale * (cond_out - uncond_out)

        return guided_out[:batch_size]




if __name__ == "__main__":
    model = DiT(img_size=[32, 32, 32], patch_size=[2, 2, 2])
    x = torch.randn(1, 4, 32, 32, 32) # (N, C, H, W, D)
    t = torch.randint(0, 1000, (1,))
    y = torch.randint(0)
    out = model(x, t, y)
    print(out.shape)
