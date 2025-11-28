import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, reduce
from functools import partial
from typing import List, Union, Optional, Tuple, Sequence, Text, Mapping, Dict
import random
from .modules import *
from torch.amp import autocast
from src.registry import register_model


__all__ = ["VectorQuantizer", "GumbelQuantize", "VectorQuantizer2", "SimVQ", "ResidualQuantizer", "GroupedResidualVQ", "MultiScaleResidualQuantizer", "LookupFreeQuantizer"]

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

        d = torch.sum(z_flattened ** 2, in_channels=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, in_channels=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, in_channels=1).unsqueeze(1)

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
        e_mean = torch.mean(min_encodings, in_channels=0)
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
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), in_channels=1).mean()

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
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    Supports optional EMA updates via use_ema parameter.

    Args:
        n_e: Number of embeddings
        e_dim: Dimension of embedding
        beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        remap: Whether to remap indices
        unknown_index: Index to use for unknown values
        sane_index_shape: Whether to keep index shape sane
        legacy: Whether to use legacy mode
        rotation_trick: Whether to apply rotation trick, -> https://arxiv.org/pdf/2410.06424
        use_norm: Whether to use normalization -> basically then measures distance by cosine similarity
        use_ema: Whether to use EMA updates for embeddings
        ema_decay: EMA decay rate
        ema_eps: Epsilon value for numerical stability
    """
    def __init__(self, n_e, e_dim, beta,
                 legacy=True, rotation_trick: bool = False, 
                 use_norm=False, use_ema=False, ema_decay=0.99, ema_eps=1e-5):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.norm = lambda x: F.normalize(x, in_channels=-1) if use_norm else x
        self.beta = beta
        self.legacy = legacy
        self.rotation_trick = rotation_trick
        self.use_ema = use_ema
        
        if use_ema:
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim, ema_decay, ema_eps)
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)


    ## Ensure quantization is performed using fp32
    @autocast('cuda', enabled=False)
    def forward(self, z):
        z=z.float()
        # Reshape input to (B, H, W, C) or (B, D, H, W, C) style for distance computation
        if z.ndim == 4:  # 2D
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        elif z.ndim == 5:  # 3D
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        else:  # already flattened or channel-last
            pass
            
        z_flattened = z.view(-1, self.e_dim)

        z_flattened = self.norm(z_flattened)
        embedding = self.norm(self.embedding.weight)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, in_channels=1, keepdim=True) + \
            torch.sum(embedding**2, in_channels=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, in_channels=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q, z = self.norm(z_q), self.norm(z)
        perplexity = None
        min_encodings = None

        # Perform EMA update if enabled
        if self.use_ema:
            encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
            self.embedding.perform_ema_update(encodings, z_flattened, self.n_e)
            avg_probs = torch.mean(encodings, in_channels=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        if self.rotation_trick:
            # apply rotation trick -> https://arxiv.org/abs/2410.06424
            z_q = rotate_to(z, z_q)
        else:     
            # preserve gradients -> STE
            z_q = z + (z_q - z).detach()

        # Reshape back to original spatial dimensions
        z_q = z_q_flat.view(z.shape)
        if z.ndim == 4:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        elif z.ndim == 5:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)

        z_q = self.norm(z_q)
        return z_q

@register_model(f"{_REGISTRY_PREFIX}qinco_vector_quantizer",
code_url="https://github.com/facebookresearch/Qinco",
paper_url="https://arxiv.org/abs/2401.14732",
)
class QINCoVectorQuantizer2(VectorQuantizer2):
    def __init__(self, n_e, e_dim,
                 hidden_dim=256, num_layers=3,
                 **kwargs):

        super().__init__(n_e, e_dim, **kwargs)

        # Replace table with implicit MLP
        self.embedding = ImplicitEmbedding(
            num_codes=n_e,
            in_channels=e_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

class SimVQ(nn.Module):
    """
    A VQ module using a frozen / implicit codebook with optional linear projection.
    Designed to be compatible with ResidualQuantizer / GroupedResidualVQ wrappers.
    """

    def __init__(
        self,
        in_channels: int,
        n_e: int,
        e_dim: int | None = None,
        codebook_transform: nn.Module | None = None,
        rotation_trick: bool = True,
        beta: float = 0.25,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        self.n_e = n_e
        self.in_channels = in_channels
        self.e_dim = e_dim or in_channels
        self.rotation_trick = rotation_trick
        self.beta = beta
        self.commitment_weight = commitment_weight

        # frozen codebook buffer
        codebook = torch.randn(n_e, self.e_dim) * (self.e_dim ** -0.5) # scaling 
        self.register_buffer("frozen_codebook", codebook)

        # linear projection from frozen codebook to actual quantized space
        if codebook_transform is None:
            self.code_transform = nn.Linear(self.e_dim, in_channels, bias=False)
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
            d = torch.sum(z_flat ** 2, in_channels=1, keepdim=True) + \
                torch.sum(codebook**2, in_channels=1) - 2 * torch.einsum('bd,nd->bn', z_flat, codebook)
            indices = torch.argmin(d, in_channels=1)

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


@register_model(f"{_REGISTRY_PREFIX}grouped_residual_quantizer",
    code_url="https://github.com/yangdongchao/AcademiCodec",
    paper_url="https://arxiv.org/pdf/2305.02765",)
class GroupedResidualVQ(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        groups: int = 1,
        accept_image_fmap: bool = False,
        **residual_quantizer_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        assert (in_channels % groups) == 0, f"in_channels {in_channels} must be divisible by groups {groups}"
        self.dim_per_group = in_channels // groups
        self.accept_image_fmap = accept_image_fmap

        # Build one ResidualQuantizer per group
        self.rvqs = ModuleList([
            ResidualQuantizer(
                **residual_quantizer_kwargs,
                quantizer_kwargs_list=residual_quantizer_kwargs['quantizer_kwargs_list']
            )
            for _ in range(groups)
        ])

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    @property
    def codebooks(self):
        # Stack codebooks from all groups
        return torch.stack([rvq.levels[0].embedding.weight for rvq in self.rvqs])

    def get_codes_from_indices(self, indices: List[torch.Tensor]):
        """
        Retrieve all codes from each group given a list of indices per group.
        """
        codes = tuple(rvq.get_codes_from_indices(idx) for rvq, idx in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices: List[torch.Tensor]):
        """
        Reconstruct full feature vector from grouped indices.
        """
        outputs = tuple(rvq.get_output_from_indices(idx) for rvq, idx in zip(self.rvqs, indices))
        return torch.cat(outputs, in_channels=self.split_dim)

    def forward(
        self,
        x: torch.Tensor,
        indices: List[torch.Tensor] = None,
        return_all_codes: bool = False,
        sample_codebook_temp = None,
        freeze_codebook: bool = False,
        mask = None,
    ):
        split_dim = self.split_dim
        assert x.shape[split_dim] == self.in_channels, f"Input in_channels {x.shape[split_dim]} != {self.in_channels}"

        # Split input along feature dimension
        x_chunks = x.chunk(self.groups, in_channels=split_dim)
        indices = indices or [None] * self.groups
        return_ce_loss = any(idx is not None for idx in indices)

        forward_kwargs = dict(
            return_all_codes=return_all_codes,
            sample_codebook_temp=sample_codebook_temp,
            mask=mask,
            freeze_codebook=freeze_codebook
        )

        # Forward pass through each group
        out = tuple(
            rvq(chunk, indices=chunk_idx, **forward_kwargs)
            for rvq, chunk, chunk_idx in zip_longest(self.rvqs, x_chunks, indices)
        )
        out = tuple(zip(*out))

        if return_ce_loss:
            quantized, ce_losses = out[:2]
            return torch.cat(quantized, in_channels=split_dim), sum(ce_losses)

        # Otherwise, combine all outputs
        quantized, all_indices, commit_losses, *maybe_all_codes = out
        quantized = torch.cat(quantized, in_channels=split_dim)
        all_indices = torch.stack(all_indices)
        commit_losses = torch.stack(commit_losses)

        return (quantized, all_indices, commit_losses, *maybe_all_codes)


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
        self, n_e: int, e_dim: int, using_znorm: bool = True, beta: float = 0.25, use_ema: bool = False, ema_decay: float = 0.99, ema_eps: float = 1e-5,
        default_qresi_counts: int = 0, v_patch_nums: Tuple[int] = None, quant_resi: float = 0.5, share_quant_resi: int = 4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.using_znorm = using_znorm
        self.use_ema = use_ema
        self.v_patch_nums = v_patch_nums
        
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
            vocab_hit_V = torch.zeros(self.n_embeddings, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            encoding_indices_list = []
            for si, pn in enumerate(self.v_patch_nums):
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, in_channels=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, in_channels=0), in_channels=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), in_channels=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), in_channels=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, in_channels=1)
                
                hit_V = idx_N.bincount(minlength=self.n_embeddings).float()
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
        encodings = F.one_hot(encoding_indices_list[-1], self.n_embeddings).type(f_BChw.dtype)
        avg_probs = torch.mean(encodings, in_channels=0)
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
                z_NC = F.normalize(z_NC, in_channels=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, in_channels=0), in_channels=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), in_channels=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), in_channels=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, n_e)
                idx_N = torch.argmin(d_no_grad, in_channels=1)
            
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
        return torch.cat(next_scales, in_channels=1) if len(next_scales) else None

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
        entropy_loss_weight: float = 0.1,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
        dims: int = 2
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
        self.is_3d = dims == 3

        bits_to_indices = torch.pow(2.0, torch.arange(0, self.token_size, dtype=torch.float32))
        self.register_buffer('bits_to_indices', bits_to_indices.int())

        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.bits_to_indices) != 0).float()
        self.register_buffer('codebook', bits * 2.0 - 1.0)

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
        z_q = z_q_flat.view(z.shape)
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
        return z_quantized, loss, (per_sample_entropy, None, min_encoding_indices) # We don't have one_hot encodings here

    def get_codebook_entry(self, indices: torch.Tensor, shape=None) -> torch.Tensor:
        indices = indices.long()
        bits = ((indices[..., None].int() & self.bits_to_indices) != 0).float()
        tokens = bits * 2.0 - 1.0  # scale to -1..1
        return tokens

    def convert_bits_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        sign_mask = (tokens > 0.0)
        return reduce(sign_mask.int() * self.bits_to_indices, '... c -> ...', 'sum')

    def convert_indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.long()
        return self.get_codebook_entry(indices)
    
