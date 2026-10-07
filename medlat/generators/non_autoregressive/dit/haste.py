"""HASTE: Holistic Alignment for Accelerating Diffusion Transformer Training.

HASTE-specific DiT components:
- :class:`AttentionWithLogits` — attention with optional pre-softmax logit return
- :class:`HASTEDiTBlock` — DiT block using AttentionWithLogits
- :class:`HASTEDiT` — DiT subclass with attention distillation layer support

Reference: https://arxiv.org/abs/2505.16792
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from medlat.modules.embeddings import modulate
from .models import DiT

__all__ = ["HASTEDiT"]


class AttentionWithLogits(nn.Module):
    """Multi-head self-attention with optional pre-softmax logit return.

    When ``return_attn=False`` (default), uses ``F.scaled_dot_product_attention``
    for speed.  When ``return_attn=True``, computes ``q @ k^T * scale``
    manually and returns the logits alongside the output.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if return_attn:
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn_logits.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj_drop(self.proj(x))
            return x, attn_logits

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class HASTEDiTBlock(nn.Module):
    """DiT block with attention logit capture support."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond_dim=None, **block_kwargs):
        super().__init__()
        from timm.models.vision_transformer import Mlp
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithLogits(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu, drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim or hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, return_attn=False):
        mod = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        if return_attn:
            attn_out, attn_logits = self.attn(normed, return_attn=True)
            x = x + gate_msa.unsqueeze(1) * attn_out
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x, attn_logits
        x = x + gate_msa.unsqueeze(1) * self.attn(normed)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class HASTEDiT(DiT):
    """DiT with HASTE attention distillation support.

    Replaces standard DiT blocks with :class:`HASTEDiTBlock` so that
    attention logits can be captured from selected layers and passed to
    the alignment module for attention map distillation.
    """

    def __init__(self, attn_distill_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.attn_distill_layers = set(attn_distill_layers or [])
        depth = len(self.blocks)
        self.blocks = nn.ModuleList([
            HASTEDiTBlock(
                self.hidden_size, self.num_heads,
                cond_dim=self.cond_dim,
            )
            for _ in range(depth)
        ])
        self.initialize_weights()

    def forward(self, x, t, y, dataset_id=None, input_image=None):
        x_noisy = x
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)

        c_list = [t_emb, y_emb]
        if self.use_dataset_conditioning and dataset_id is not None:
            ds_emb = self.dataset_embedder(dataset_id, self.training)
            c_list.append(ds_emb)
        c = torch.cat(c_list, dim=1)

        self._auxiliary_losses = {}
        student_attn_logits = []

        for i, block in enumerate(self.blocks):
            need_attn = (i + 1) in self.attn_distill_layers
            if need_attn:
                x, attn_logits = block(x, c, return_attn=True)
                student_attn_logits.append(attn_logits)
            else:
                x = block(x, c)

            if (self.generator_alignment is not None
                    and (i + 1) == self.alignment_layer
                    and input_image is not None):
                align_loss, _ = self.generator_alignment(
                    x, input_image=input_image,
                    noisy_latent=x_noisy, timestep=t,
                    class_labels=y, dataset_id=dataset_id,
                    student_attn_logits=student_attn_logits,
                )
                self._auxiliary_losses["alignment_loss"] = align_loss

            if (self.dispersive_loss is not None
                    and (i + 1) == self.dispersive_layer
                    and self.training):
                self._auxiliary_losses["dispersive_loss"] = self.dispersive_loss(x)

        x = self.final_layer(x, c)
        x = self.to_pixel(x)
        return x
