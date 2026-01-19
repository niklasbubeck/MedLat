import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, reduce
from functools import partial
from itertools import zip_longest
from typing import Any, List, Union, Optional, Tuple, Sequence, Text, Mapping, Dict
import random
from .modules import *
from torch.amp import autocast
from medtok.registry import register_model


__all__ = ["VectorQuantizer", "GumbelQuantize", "SimpleQINCo", "VectorQuantizer2", "SimVQ", "ResidualQuantizer", "GroupedVQ", "MultiScaleResidualQuantizer", "LookupFreeQuantizer", "FiniteScalarQuantizer", "BinarySphericalQuantizer", "QINCo", "QincoResidualQuantizer"]

_REGISTRY_PREFIX = "discrete.quantizer."

@register_model(f"{_REGISTRY_PREFIX}vector_quantizer",
code_url="https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py",
paper_url="https://arxiv.org/abs/1711.00937",)
class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE/VQ-GAN quantizer

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        rotation_trick: Whether to apply rotation trick
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta, rotation_trick: bool = False, 
                 use_ema: bool = False, ema_decay: float = 0.99, ema_eps: float = 1e-5):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema
        
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # Perform EMA update if enabled
        if self.use_ema:
            z_flattened = z_flattened  # Already defined above
            self.embedding.perform_ema_update(min_encodings, z_flattened, self.n_e)
        
        # compute loss for embedding
        commitment_loss = torch.mean((z_q.detach()-z)**2)
        codebook_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)
        loss = commitment_loss + codebook_loss

        if self.rotation_trick:
            # apply rotation trick -> https://arxiv.org/abs/2410.06424
            z_q = rotate_to(z, z_q)
        else:     
            # preserve gradients -> STE
            z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


@register_model(f"{_REGISTRY_PREFIX}gumbel_quantizer",
code_url="https://github.com/karpathy/deep-vector-quantization/blob/main/model.py",
paper_url="https://arxiv.org/abs/1611.01144",)
class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer

    Args:
        num_hiddens: Number of hidden dimensions
        embedding_dim: Dimension of embedding
        n_embed: Number of embeddings
        straight_through: Whether to use straight through estimator
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1

        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)
        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, in_channels=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, in_channels=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(in_channels=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


@register_model(f"{_REGISTRY_PREFIX}vector_quantizer2",
code_url="https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py",
paper_url="https://arxiv.org/abs/1711.00937",)
class VectorQuantizer2(nn.Module):
    """
    Improved VectorQuantizer with optional EMA, rotation trick,
    cosine normalization, and MaskGIT-style entropy loss.
    """
    def __init__(
        self, n_e, e_dim, beta=0.25,
        legacy=True, rotation_trick=False,
        use_norm=False, use_ema=False, ema_decay=0.99, ema_eps=1e-5,
        # ---- NEW ENTROPY OPTIONS as in MaskGITs ----
        entropy_loss_ratio=0.0,
        entropy_loss_type="softmax",   # ["softmax", "gumbel"]
        entropy_temperature=1.0
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x
        self.beta = beta
        self.legacy = legacy
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema

        # Entropy hyperparameters
        self.entropy_loss_ratio = entropy_loss_ratio
        self.entropy_loss_type = entropy_loss_type
        self.entropy_temperature = entropy_temperature
        
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


    ## Ensure quantization is performed using fp32
    @autocast('cuda', enabled=False)
    def forward(self, z):
        z = z.float()

        # Put channel last (2D or 3D)
        if z.ndim == 4:  
            z = rearrange(z, 'b c h w -> b h w c')
        elif z.ndim == 5: 
            z = rearrange(z, 'b c d h w -> b d h w c')

        z_flat = z.reshape(-1, self.e_dim)
        z_flat = self.norm(z_flat)

        embedding = self.norm(self.embedding.weight)

        # Compute distances (efficient MaskGIT/VQGAN style)
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum('bd,nd->bn', z_flat, embedding)
        )

        # Nearest neighbour lookup
        min_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_indices).view_as(z)
        z_q = self.norm(z_q)

        perplexity = None
        min_encodings = None

        # EMA update
        if self.use_ema:
            onehot = F.one_hot(min_indices, self.n_e).type(z.dtype)
            self.embedding.perform_ema_update(onehot, z_flat, self.n_e)

            avg_probs = onehot.float().mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Standard VQ loss
        if self.legacy:
            loss = torch.mean((z_q.detach() - z) ** 2) + \
                   self.beta * torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        # ---------------------------
        #       ENTROPY LOSS  
        # ---------------------------
        entropy_loss = torch.tensor(0.0, device=z.device)

        if self.entropy_loss_ratio > 0:
            logits = -d  # MaskGIT uses negative distances as logits

            if self.entropy_loss_type == "softmax":
                probs = F.softmax(logits / self.entropy_temperature, dim=-1)
            elif self.entropy_loss_type == "gumbel":
                probs = F.gumbel_softmax(logits, tau=self.entropy_temperature, hard=False)
            else:
                raise ValueError(f"Invalid entropy_loss_type: {self.entropy_loss_type}")

            # Entropy = -Σ p log p
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            # MaskGIT penalizes *low entropy* (i.e., encourages diversity)
            entropy_loss = -entropy.mean() * self.entropy_loss_ratio

            loss = loss + entropy_loss

        # Rotation trick or STE
        if self.rotation_trick:
            z_q = rotate_to(z, z_q)
        else:
            z_q = z + (z_q - z).detach()

        # Restore shape (channel-first)
        if z.ndim == 4:
            z_q = rearrange(z_q, 'b h w c -> b c h w')
        elif z.ndim == 5:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w')

        return z_q, loss, (perplexity, None, min_indices)

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return self.norm(z_q)



class SimpleQINCo(VectorQuantizer2):
    def __init__(self, n_e, e_dim, beta=0.25,
                 hidden_dim=256, num_layers=3,
                 **kwargs):

        super().__init__(n_e, e_dim, beta=beta, **kwargs)

        # Replace table with implicit MLP
        self.embedding = ImplicitEmbedding(
            n_e=n_e,
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

@register_model(f"{_REGISTRY_PREFIX}simple_qinco",
code_url="https://github.com/facebookresearch/Qinco",
paper_url="https://arxiv.org/abs/2401.14732",
)
class QINCo(nn.Module):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        top_a: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        commitment_weight: float = 1.0,
    ):
        """
        n_e: number of codes
        e_dim: embedding dimension
        beta: commitment loss factor (like VQ-VAE)
        top_a: number of top candidates per vector
        hidden_dim, num_layers: for QincoSubstep
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.commitment_weight = commitment_weight
        self.top_a = top_a
        # implicit base codebook
        self.embedding = ImplicitEmbedding(
            n_e=n_e,
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # QINCo-style transform
        self.transform = QincoSubstep(
            e_dim=e_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    @property
    def e_dim_prop(self):
        return self.e_dim

    @property
    def n_e_prop(self):
        return self.n_e

    def forward(self, residual: torch.Tensor, x_prev: torch.Tensor):
        """
        residual: (B, D, ...)  flattened to (B_flat, D)
        x_prev:  same shape as residual
        """
        orig_shape = residual.shape
        B_flat = residual.numel() // self.e_dim
        residual_flat = residual.view(B_flat, self.e_dim)
        x_prev_flat = x_prev.view(B_flat, self.e_dim)

        # ----- 1) cheap distance using base codes (no transform) -----
        base_codes = self.embedding.weight            # (K, D)
        K = base_codes.size(0)

        # (B_flat, 1, D) - (1, K, D) -> (B_flat, K, D)
        diff_base = residual_flat.unsqueeze(1) - base_codes.unsqueeze(0)
        dist_base = (diff_base ** 2).sum(-1)          # (B_flat, K)

        A = min(self.top_a, K)
        # top-A smallest distances
        topk_dist, topk_idx = dist_base.topk(A, dim=-1, largest=False, sorted=False)  # (B_flat, A)

        # ----- 2) run transform only on these A candidates -----
        # gather base codes for selected indices
        codes_sel = base_codes[topk_idx]              # (B_flat, A, D)
        x_prev_sel = x_prev_flat.unsqueeze(1).expand(-1, A, -1)   # (B_flat, A, D)

        # flatten for QincoSubstep
        codes_in = codes_sel.reshape(-1, self.e_dim)  # (B_flat*A, D)
        x_in = x_prev_sel.reshape(-1, self.e_dim)     # (B_flat*A, D)

        deltas_sel = self.transform(codes_in, x_in)   # (B_flat*A, D)
        deltas_sel = deltas_sel.view(B_flat, A, self.e_dim)  # (B_flat, A, D)

        # ----- 3) pick best among the A candidates -----
        diff = residual_flat.unsqueeze(1) - deltas_sel      # (B_flat, A, D)
        dist = (diff ** 2).sum(-1)                         # (B_flat, A)
        best_in_A = dist.argmin(-1)                        # (B_flat,)

        # map back to global code indices
        indices = topk_idx[torch.arange(B_flat, device=residual.device), best_in_A]  # (B_flat,)

        # final quantized vector: transform(selected_base_code, x_prev)
        chosen_base = base_codes[indices]                  # (B_flat, D)
        z_q_flat = self.transform(chosen_base, x_prev_flat)  # (B_flat, D)
        z_q = z_q_flat.view(orig_shape)

        # ----- 4) losses & perplexity -----
        loss_commit = F.mse_loss(z_q_flat.detach(), residual_flat)
        loss_embed = F.mse_loss(z_q_flat, residual_flat.detach())
        loss = loss_embed + self.beta * loss_commit

        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=self.n_e).float()
            avg_probs = one_hot.mean(0)
            perplexity = torch.exp(- (avg_probs * (avg_probs + 1e-10).log()).sum())

        indices = indices.view(-1)

        return z_q, loss, (perplexity, None, indices)


class SimVQ(nn.Module):
    """
    A VQ module using a frozen / implicit codebook with optional linear projection.
    Designed to be compatible with ResidualQuantizer / GroupedResidualVQ wrappers.
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        in_channels: int = None, # usually not used as we have quant conv
        codebook_transform: nn.Module | None = None,
        rotation_trick: bool = True,
        beta: float = 0.25,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.n_e = n_e
        self.in_channels = in_channels if in_channels is not None else e_dim
        self.e_dim = e_dim
        self.rotation_trick = rotation_trick
        self.beta = beta
        self.commitment_weight = commitment_weight

        # frozen codebook buffer
        codebook = torch.randn(n_e, self.e_dim) * (self.e_dim ** -0.5) # scaling 
        self.register_buffer("frozen_codebook", codebook)

        # linear projection from frozen codebook to actual quantized space
        if codebook_transform is None:
            self.code_transform = nn.Linear(self.e_dim, self.in_channels, bias=False)
        else:
            self.code_transform = codebook_transform

    @property
    def embedding(self):
        """For compatibility with ResidualQuantizer wrappers"""
        return self.code_transform(self.frozen_codebook)

    @autocast('cuda', enabled=False)
    def forward(self, z: torch.Tensor):
        """
        VectorQuantizer2-style forward for SimVQ.
        Supports 2D or 3D feature maps with channel-first format.
        Returns: z_q, loss, (perplexity=None, _, indices)
        """
        z = z.float()  # ensure FP32 for distance computation

        # Reshape input to (B, H, W, C) or (B, D, H, W, C) style for distance computation
        if z.ndim == 4:  # 2D
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        elif z.ndim == 5:  # 3D
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        else:  # already flattened or channel-last
            pass

        # Flatten for distance computation
        z_flat = z.view(-1, self.in_channels)
        codebook = self.embedding  # projected codebook

        # Compute distances: (z - e)^2 = z^2 + e^2 - 2 z.e
        with torch.no_grad():
            d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook**2, dim=1) - 2 * torch.einsum('bd,nd->bn', z_flat, codebook)
            indices = torch.argmin(d, dim=1)

        
        # Get quantized vectors
        z_q_flat = codebook[indices]

        # Commitment loss with STE trick
        loss = (
            F.mse_loss(z_flat.detach(), z_q_flat)
            + F.mse_loss(z_flat, z_q_flat.detach()) * self.beta
        ) * self.commitment_weight

        # Rotation trick or straight-through
        if self.rotation_trick:
            z_q_flat = rotate_to(z_flat, z_q_flat)
        else:
            z_q_flat = (z_q_flat - z_flat).detach() + z_flat

        # Reshape back to original spatial dimensions
        z_q = z_q_flat.view(z.shape)
        if z.ndim == 4:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        elif z.ndim == 5:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        return z_q, loss, (None, None, indices)


    def get_codebook_entry(self, indices, shape=None):
        codebook = self.embedding  # (n_e, in_channels)
        # lookup
        z_q = codebook[indices]
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

@register_model(f"{_REGISTRY_PREFIX}residual_quantizer",
paper_url="https://arxiv.org/abs/2107.03312",
description="Acts as wrapper for all the other quantizers")
class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        quantizer_class: nn.Module,
        num_quantizers: int,
        quantizer_kwargs_list: List[Dict],
        shared_codebook: bool = False,
        quantize_dropout: bool = False,   ### as in the EnCodec paper
        dropout_start_level: int = 0,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.quantize_dropout = quantize_dropout
        self.dropout_start_level = dropout_start_level
        self.shared_codebook = shared_codebook

        # Build levels
        self.levels = nn.ModuleList([
            quantizer_class(**quantizer_kwargs_list[i])
            for i in range(num_quantizers)
        ])

        # ---- Shared Codebook Mode ------------------------------------------------------
        # All quantizers share the codebook of the first quantizer
        if shared_codebook:
            first = self.levels[0]
            shared = first.embedding
            # link all quantizers to same object
            for q in self.levels[1:]:
                q.embedding = shared

    # VQ Model needs to know for quant_conv
    @property
    def e_dim(self):
        return self.levels[0].e_dim

    @property
    def n_e(self):
        return self.levels[0].n_e

    # ----------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        residual = x
        quantized_outputs = []
        losses = []
        all_indices = []
        all_perplexities = []

        # -------- Determine dropout level ----------------------------------------------
        # During training, randomly skip fine quantizers
        if self.training and self.quantize_dropout and self.num_quantizers > 1:
            # choose a dropout boundary: deeper ones are removed
            dropout_level = torch.randint(
                self.dropout_start_level,
                self.num_quantizers,
                (1,)
            ).item()
        else:
            dropout_level = self.num_quantizers

        # -------- Iterate through levels -----------------------------------------------
        for i, q in enumerate(self.levels):

            # ---------------------------------------------------------
            # DROPOUT: skip quantization for deeper levels
            # ---------------------------------------------------------
            if i >= dropout_level:
                # output placeholder
                quantized_outputs.append(torch.zeros_like(residual))
                losses.append(torch.tensor(0.0, device=x.device))
                all_perplexities.append(None)
                all_indices.append(torch.full_like(residual[..., 0], -1, dtype=torch.long))
                continue

            # ---------------------------------------------------------
            # ACTIVE quantizer
            # ---------------------------------------------------------
            z_q, loss, (perplexity, _, indices) = q(residual)

            quantized_outputs.append(z_q)
            losses.append(loss)
            all_indices.append(indices)
            all_perplexities.append(perplexity)

            # Residual refinement (correct for STE quantizers)
            residual = residual - z_q.detach()

        # -------- Aggregate outputs -----------------------------------------------------
        final_quantized = sum(quantized_outputs)
        total_loss = sum(losses)

        return final_quantized, total_loss, (all_perplexities, quantized_outputs, all_indices)

    def get_codebook_entry(self, indices, shape=None):
        """
        indices: Tensor of shape (B, X) or list of tensors
                assumed ordered coarse → fine if Tensor
        """

        # ------------------------------------------------------------
        # Normalize to list of per-level indices
        # ------------------------------------------------------------
        if isinstance(indices, torch.Tensor):
            B, X = indices.shape
            Q = self.num_quantizers

            if X % Q != 0:
                raise ValueError(
                    f"Total indices {X} not divisible by num_quantizers {Q}"
                )

            chunk = X // Q
            indices_list = [
                indices[:, i * chunk : (i + 1) * chunk]
                for i in range(Q)
            ]

        elif isinstance(indices, (list, tuple)):
            if len(indices) != self.num_quantizers:
                raise ValueError(
                    f"Expected {self.num_quantizers} levels, got {len(indices)}"
                )
            indices_list = list(indices)

        else:
            raise TypeError("indices must be Tensor or list/tuple")

        # ------------------------------------------------------------
        # Lookup & sum residual codebooks
        # ------------------------------------------------------------
        z_q = None

        for q, idx in zip(self.levels, indices_list):
            idx = idx.long()

            # Handle quantizer dropout
            if torch.all(idx < 0):
                continue

            z_q_i = q.get_codebook_entry(idx)

            z_q = z_q_i if z_q is None else z_q + z_q_i

        if z_q is None:
            raise RuntimeError("All quantizer levels were dropped.")

        # ------------------------------------------------------------
        # Reshape if needed
        # ------------------------------------------------------------
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

class QincoResidualQuantizer(nn.Module):
    def __init__(
        self,
        quantizer_class: nn.Module,
        num_quantizers: int,
        quantizer_kwargs_list: List[Dict],
        shared_codebook: bool = False,
        quantize_dropout: bool = False,
        dropout_start_level: int = 0,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.quantize_dropout = quantize_dropout
        self.dropout_start_level = dropout_start_level
        self.shared_codebook = shared_codebook

        self.levels = nn.ModuleList([
            quantizer_class(**quantizer_kwargs_list[i])
            for i in range(num_quantizers)
        ])

        if shared_codebook:
            first = self.levels[0]
            shared = first.embedding
            for q in self.levels[1:]:
                q.embedding = shared

    @property
    def e_dim(self):
        return self.levels[0].e_dim_prop

    @property
    def n_e(self):
        return self.levels[0].n_e_prop

    def forward(self, x: torch.Tensor):
        """
        x: (B, D, ...) – same as your original
        """
        residual = x
        x_prev = torch.zeros_like(x)

        quantized_outputs = []
        losses = []
        all_indices = []
        all_perplexities = []

        # dropout level
        if self.training and self.quantize_dropout and self.num_quantizers > 1:
            dropout_level = torch.randint(
                self.dropout_start_level,
                self.num_quantizers,
                (1,)
            ).item()
        else:
            dropout_level = self.num_quantizers

        for i, q in enumerate(self.levels):

            if i >= dropout_level:
                quantized_outputs.append(torch.zeros_like(residual))
                losses.append(torch.tensor(0.0, device=x.device))
                all_perplexities.append(None)
                all_indices.append(torch.full_like(residual[..., 0], -1, dtype=torch.long))
                continue

            # QINCo: quantizer sees residual and x_prev (partial reconstruction)
            z_q, loss, (perplexity, _, indices) = q(residual, x_prev=x_prev)

            quantized_outputs.append(z_q)
            losses.append(loss)
            all_indices.append(indices)
            all_perplexities.append(perplexity)

            # update partial reconstruction
            x_prev = x_prev + z_q

            # update residual (detach like RQ init)
            residual = (x - x_prev).detach()

        final_quantized = sum(quantized_outputs)
        total_loss = sum(losses)

        return final_quantized, total_loss, (all_perplexities, quantized_outputs, all_indices)

@register_model(f"{_REGISTRY_PREFIX}grouped_residual_quantizer",
    code_url="https://github.com/yangdongchao/AcademiCodec",
    paper_url="https://arxiv.org/pdf/2305.02765",
    description="Grouped VQ for improved efficiency original uses ResidualQuantizers!")
class GroupedVQ(nn.Module):
    """
    Applies a quantizer independently on channel groups.
    Each group gets its own quantizer instance (usually ResidualQuantizer).
    """
    def __init__(
        self,
        quantizer_class: nn.Module,
        in_channels: int,
        quantizer_kwargs_list: List[Dict],
        groups: int = 1,
        split_dim: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.split_dim = split_dim

        assert in_channels % groups == 0, \
            f"in_channels {in_channels} must be divisible by groups {groups}"
        assert len(quantizer_kwargs_list) == groups, \
            "One quantizer config per group required"

        # Build one quantizer per group (usually ResidualQuantizer)
        self.vqs = nn.ModuleList([
            quantizer_class(**quantizer_kwargs_list[i])
            for i in range(groups)
        ])

        self.dim_per_group = in_channels // groups

    # Optional helpers
    @property
    def e_dim(self):
        return self.vqs[0].levels[0].e_dim

    @property
    def n_e(self):
        return self.vqs[0].levels[0].n_e

    @property
    def codebooks(self):
        """Return codebooks from all groups (first level only)."""
        return torch.stack([vq.levels[0].embedding.weight for vq in self.vqs])


    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    @autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor):
        """
        Returns:
            quantized:     (B, C, ...)
            total_loss:    scalar loss
            all_outputs:   tuple of:
                - all_perplexities:  [groups][levels] or None
                - all_quantized:     [groups][levels]
                - all_indices:       [groups][levels][..]
        """

        # 1) Split channels into groups
        x_groups = x.split(self.dim_per_group, dim=self.split_dim)

        # 2) Apply VQ to each group independently
        group_results = []
        for group_x, vq in zip(x_groups, self.vqs):
            q, loss, extras = vq(group_x)
            group_results.append((q, loss, extras))

        # 3) Unpack results
        quantized_list   = [r[0] for r in group_results]
        losses_list      = [r[1] for r in group_results]
        extras_list      = [r[2] for r in group_results]

        # 4) Concatenate quantized outputs across groups
        quantized = torch.cat(quantized_list, dim=self.split_dim)

        # 5) Combine losses
        total_loss = sum(losses_list)

        # 6) Stack metadata cleanly
        all_perplexities  = [e[0] for e in extras_list]
        all_quantized_lvls = [e[1] for e in extras_list]
        all_indices       = [e[2] for e in extras_list]

        return quantized, total_loss, (all_perplexities, all_quantized_lvls, all_indices)


@register_model(f"{_REGISTRY_PREFIX}msrq_vector_quantizer2",
code_url="https://github.com/FoundationVision/VAR/blob/main/models/quant.py",
paper_url="https://arxiv.org/pdf/2404.02905",)
class MultiScaleResidualQuantizer(nn.Module):
    """
    Multi-Scale Residual Quantizer 
    As presented in VAR: Visual Autoregressive Models
    https://arxiv.org/pdf/2404.02905

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        using_znorm: Whether to use z-normalization
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
        default_qresi_counts: Number of quantizers to use
        v_patch_nums: List of patch sizes
        quant_resi: Quantization residual ratio
        share_quant_resi: Number of quantizers to share
    """
    def __init__(
        self, n_e: int, e_dim: int, using_znorm: bool = True, beta: float = 0.25, rotation_trick: bool = False, use_ema: bool = False, ema_decay: float = 0.99, ema_eps: float = 1e-5,
        default_qresi_counts: int = 0, v_patch_nums: Tuple[int] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16), quant_resi: float = 0.5, share_quant_resi: int = 4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.using_znorm = using_znorm
        self.use_ema = use_ema
        self.v_patch_nums = v_patch_nums
        self.rotation_trick = rotation_trick
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.n_e), fill_value=0.0))
        self.record_hit = 0
        
        self.beta = beta
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, decay=ema_decay, eps=ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.n_e, abs(eini) / self.n_e)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw):
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.n_e, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            encoding_indices_list = []
            for si, pn in enumerate(self.v_patch_nums):
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                hit_V = idx_N.bincount(minlength=self.n_e).float()
                encoding_indices_list.append(idx_N)
                
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)  # This will be identity if no quant_resi was provided
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                
                if self.training:
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            if self.rotation_trick:
                f_hat = rotate_to(f_hat, f_BChw)
            else:
                f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        # Calculate perplexity
        encodings = F.one_hot(encoding_indices_list[-1], self.n_e).type(f_BChw.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Return in the same format as other quantizers
        return f_hat, mean_vq_loss, (perplexity, encodings, encoding_indices_list[-1])
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.e_dim, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, n_e)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    
    def idxBl_to_msrq_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """Convert indices to MSRQ input"""
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.e_dim
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_BChw)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Get next autoregressive input"""
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')))
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            # Handle both Identity and Phi cases
            if isinstance(self.quant_resi, nn.Identity):
                f_hat.add_(h_BChw)
            else:
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            return f_hat, f_hat



@register_model(f"{_REGISTRY_PREFIX}lookup_free_quantizer",)
class LookupFreeQuantizer(torch.nn.Module):
    def __init__(
        self,
        token_bits: int = 10,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.02,    # from MaskBIT  https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/maskbit_tokenizer_10bit.yaml
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
    ):
        """ 
        Args:
            token_bits -> int: The number of bits per token.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature for the entropy loss.
            entropy_gamma -> float: The gamma for the entropy loss.
            dims -> int: The number of dimensions of the input.
        """
        super().__init__()
        self.token_size = token_bits
        self.codebook_size = 2 ** token_bits

        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        bits_to_indices = torch.pow(2.0, torch.arange(0, self.token_size, dtype=torch.float32))
        self.register_buffer('bits_to_indices', bits_to_indices.int())

        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.bits_to_indices) != 0).float()
        self.register_buffer('codebook', bits * 2.0 - 1.0)

    @property
    def e_dim(self):
        return self.token_size

    @property
    def n_e(self):
        return self.codebook_size

        # Ensure quantization is performed using f32
    @autocast('cuda',enabled=False)
    def forward(self, z: torch.Tensor):
        z=z.float()
        # Reshape input to (B, H, W, C) or (B, D, H, W, C) style for distance computation
        if z.ndim == 4:  # 2D
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        elif z.ndim == 5:  # 3D
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        else:  # already flattened or channel-last
            pass

        ones = torch.ones_like(z)
        sign_mask = (z > 0.0)
        z_quantized = torch.where(sign_mask, ones, -ones)

        min_encoding_indices = self.convert_bits_to_indices(z_quantized)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            d = -2 * torch.einsum('... c, n c -> ... n', z, self.codebook)

            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # Reshape back to original spatial dimensions
        z_q = z_quantized
        if z.ndim == 4:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        elif z.ndim == 5:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices
        )

        # return z_quantized, result_dict # Old return
        return z_q, loss, (per_sample_entropy, None, min_encoding_indices) # We don't have one_hot encodings here

    def get_codebook_entry(self, indices: torch.Tensor, shape=None) -> torch.Tensor:
        """
        indices: Tensor of shape (B, N) or (B, H, W)
        shape:   target shape, e.g. (B, C, H, W)
        """

        indices = indices.long()
        print(f"indices: {indices.shape}")
        if shape is not None:
            indices = indices.reshape(-1, shape[-3], shape[-2])
        print(f"indices: {indices.shape}")
        bits = ((indices[..., None] & self.bits_to_indices) != 0).float()
        tokens = bits * 2.0 - 1.0  # (..., token_bits)

        print(f"tokens: {tokens.shape}")
        return tokens

    def convert_bits_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        sign_mask = (tokens > 0.0)
        return reduce(sign_mask.int() * self.bits_to_indices, '... c -> ...', 'sum')

    def convert_indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.long()
        return self.get_codebook_entry(indices)

@register_model(f"{_REGISTRY_PREFIX}binary_spherical_quantizer",
paper_url="https://arxiv.org/pdf/2406.07548",
code_url="https://github.com/zhaoyue-zephyrus/bsq-vit",)
class BinarySphericalQuantizer(LookupFreeQuantizer):
    """BSQ by inheriting LFQ - only overrides forward with L2 normalization"""
    
    @autocast('cuda', enabled=False)
    def forward(self, z: torch.Tensor):
        z = z.float()
        orig_ndim = z.ndim
        
        # Reshape to channel-last for norm/sign
        if z.ndim == 4:  # (B,C,H,W) -> (B,H,W,C)
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        elif z.ndim == 5:  # (B,C,D,H,W) -> (B,D,H,W,C)
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()

        # *** BSQ CORE: L2 normalize to unit sphere ***
        z_norm = torch.norm(z, dim=-1, keepdim=True) + 1e-8
        z_unit = z / z_norm  # u = v / ||v|| [file:1]
        
        # Binary quantize on sphere
        ones = torch.ones_like(z_unit)
        sign_mask = (z_unit > 0.0)
        sign_u = torch.where(sign_mask, ones, -ones)
        sign_u = torch.where(z_unit == 0, ones, sign_u)  # sign(0) -> 1
        
        # Scale to unit sphere: hat{u} = sign(u) / sqrt(L)
        sqrt_L = 1.0 / math.sqrt(self.token_size)
        z_quantized = sqrt_L * sign_u

        # Indices from unscaled signs (LFQ compatible)
        min_encoding_indices = self.convert_bits_to_indices(sign_u)

        # Losses (use unit sphere reference)
        commitment_loss = self.commitment_cost * F.mse_loss(z_quantized.detach(), z_unit)
        
        # Entropy on normalized input (better soft quantization)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)
        
        if self.entropy_loss_weight != 0.0 and self.training:
            d = -2 * torch.einsum('... c, n c -> ... n', z_unit, self.codebook)
            per_sample_entropy, avg_entropy = entropy_loss_fn(-d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + entropy_loss

        # Straight-Through Estimator on unit sphere
        z_q = z_unit + (z_quantized - z_unit).detach()

        # Reshape back
        if orig_ndim == 4:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        elif orig_ndim == 5:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        return z_q, loss, (per_sample_entropy, None, min_encoding_indices)


@register_model(f"{_REGISTRY_PREFIX}finite_scalar_quantizer",
                paper_url="https://arxiv.org/pdf/2309.15505",)
class FiniteScalarQuantizer(nn.Module):
    """
    Minimal Finite Scalar Quantizer compatible with your VQ wrappers.

    Args:
        levels: list of ints, number of quantization levels per scalar channel (length = code_dim)
        dim: input feature dimension (channels). If different from code_dim, projections are used.
        commitment_cost: optional float, weight for commitment loss (default 0.0).
    """
    def __init__(self, levels: List[int], dim: Optional[int] = None, commitment_cost: float = 0.0):
        super().__init__()
        assert isinstance(levels, (list, tuple)) and len(levels) > 0
        self._levels = torch.tensor(list(levels), dtype=torch.int64)           # (d,)
        self.code_dim = len(self._levels)                                     # d
        self.codebook_size = int(int(torch.prod(self._levels).item()))        # n_e (product)
        self.commitment_cost = float(commitment_cost)

        # basis for mixed-radix (digits -> index)
        basis = torch.cumprod(torch.cat((torch.tensor([1], dtype=torch.int64), self._levels[:-1].to(torch.int64))), dim=0)
        self.register_buffer("_basis", basis)     # (d,)

        # half widths and offsets (use simple formula)
        # half_width = (L - 1) / 2
        half_widths = (self._levels - 1).to(torch.float32) / 2.0   # (d,)
        offsets = torch.where((self._levels % 2) == 0, 0.5, 0.0).to(torch.float32)
        self.register_buffer("_half_widths", half_widths)
        self.register_buffer("_offsets", offsets)

        # projections if input dim != code_dim
        self.in_dim = self.code_dim if dim is None else int(dim)
        self.has_projections = (self.in_dim != self.code_dim)
        if self.has_projections:
            self.project_in = nn.Linear(self.in_dim, self.code_dim)
            self.project_out = nn.Linear(self.code_dim, self.in_dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

    @property
    def e_dim(self):
        return self.code_dim

    @property
    def n_e(self):
        return self.codebook_size

    # -------- helpers: mixed-radix index conversions ---------------------------
    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: (..., d) integer-like or floats near integers representing digits in [0..L-1]
        returns indices: (...) int64
        """
        device = codes.device
        # ensure digits as integer long
        digits = torch.round(codes).long()  # (..., d)
        basis = self._basis.to(device)      # (d,)
        idx = torch.sum(digits * basis.to(digits.device), dim=-1)
        return idx

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: (...) int
        returns codes: (..., d) in integer digit space [0..L-1] (float)
        """
        device = indices.device
        indices = indices.long().unsqueeze(-1)   # (...,1)
        basis = self._basis.to(device).unsqueeze(0)  # (1,d)
        levels = self._levels.to(device).unsqueeze(0) # (1,d)
        digits = (indices // basis) % levels      # (..., d)
        return digits.to(torch.float32)

    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[tuple] = None) -> torch.Tensor:
        """
        Return code vectors for given indices in normalized [-1, 1] per code-dim space.
        shape: optional shape to view the returned codes (expects last dim = d)
        """
        digits = self.indices_to_codes(indices)  # (..., d)
        # map digits to normalized [-1,1]: digit -> centered value then / half_width
        half = self._half_widths.to(digits.device)
        centered = digits - torch.floor(self._levels.to(digits.device).to(torch.float32) / 2.0)
        normalized = centered / (half + 1e-12)
        if shape is not None:
            normalized = normalized.view(shape)
        return normalized

    # -------- bounding & quantization primitives --------------------------------
    def _bound_and_round(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple bounding: map x -> roughly [-half_width..half_width] via tanh,
        then round to integers. Return integer-like rounded digits in [-half..half].
        x expected shape (..., d)
        """
        half = self._half_widths.to(x.device)        # (d,)
        offsets = self._offsets.to(x.device)         # (d,)
        # apply tanh-based soft bound then shift
        bounded = torch.tanh(x) * half - offsets    # (..., d)
        rounded = torch.round(bounded)              # integer-like
        return rounded

    # -------- forward ----------------------------------------------------------
    def forward(self, z: torch.Tensor):
        """
        z: (B, C, H, W) or (B, C, D, H, W) or (..., in_dim)
        returns: (z_q, loss, (perplexity, None, indices))
        """
        orig_ndim = z.ndim
        z = z.float()

        # bring to channel-last layout if image/video
        if z.ndim == 4:   # (B, C, H, W)
            z_cl = rearrange(z, 'b c h w -> b h w c').contiguous()
        elif z.ndim == 5: # (B, C, D, H, W)
            z_cl = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        else:
            z_cl = z

        shape_cl = z_cl.shape   # (..., in_dim)
        assert shape_cl[-1] == self.in_dim, f"expected last dim {self.in_dim}, got {shape_cl[-1]}"

        # project into code_dim
        flat = z_cl.view(-1, shape_cl[-1])           # (N, in_dim)
        proj = self.project_in(flat)                  # (N, d)

        # bound & round in projection space
        rounded = self._bound_and_round(proj)         # (N, d) integer-like in approx [-half..half]
        # convert to digits in [0..L-1]
        half = torch.floor(self._levels.to(rounded.device).to(torch.float32) / 2.0).to(rounded.dtype)
        digits = (rounded + half)                     # (N, d)
        levels = self._levels.to(digits.device).to(torch.float32)
        mins = torch.zeros_like(levels)
        maxs = levels - 1.0
        digits_clamped = torch.clamp(digits, min=mins, max=maxs)
        # compute mixed-radix indices
        indices_flat = torch.sum(digits_clamped.long() * self._basis.to(digits_clamped.device), dim=-1)  # (N,)

        # commitment loss (pull proj towards quantized normalized representation)
        # normalized quantized in [-1,1]
        normalized = (rounded / (half + 1e-12)).to(proj.dtype)   # (N, d)
        commitment_loss = torch.tensor(0.0, device=z.device)
        if self.commitment_cost != 0.0:
            commitment_loss = self.commitment_cost * F.mse_loss(proj.detach(), normalized)

        # reconstruct quantized in input space
        q_proj = normalized                               # (N, d)
        q_out_flat = self.project_out(q_proj)             # (N, in_dim)
        # shape back
        q_out_cl = q_out_flat.view(*shape_cl)
        if orig_ndim == 4:
            z_q = rearrange(q_out_cl, 'b h w c -> b c h w').contiguous()
            indices = indices_flat.view(z_cl.shape[0], z_cl.shape[1], z_cl.shape[2])  # (B,H,W)
        elif orig_ndim == 5:
            z_q = rearrange(q_out_cl, 'b d h w c -> b c d h w').contiguous()
            indices = indices_flat.view(z_cl.shape[0], z_cl.shape[1], z_cl.shape[2], z_cl.shape[3])  # (B,D,H,W)
        else:
            z_q = q_out_cl
            indices = indices_flat.view(*z_cl.shape[:-1])

        # Straight-through estimator for gradients: preserve encoder gradients
        z_q = z + (z_q - z).detach()

        # compute perplexity over indices
        with torch.no_grad():
            flat_inds = indices_flat
            if self.codebook_size <= 2_000_000:
                counts = torch.bincount(flat_inds, minlength=self.codebook_size).float()
                probs = counts / counts.sum().clamp_min(1.0)
                perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            else:
                unique = torch.unique(flat_inds)
                usage = unique.numel() / float(self.codebook_size)
                perplexity = torch.tensor(usage * float(self.codebook_size), device=z.device)

        loss = commitment_loss

        return z_q, loss, (perplexity, None, indices)