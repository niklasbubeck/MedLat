import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, pack, unpack
from typing import List, Tuple, Optional, Sequence, Union
from math import ceil

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


class EmbeddingEMA(nn.Module):
    """
    EMA-based embedding that can be used as a drop-in replacement for nn.Embedding.
    Provides exponential moving average updates for codebook embeddings during training.

    Args:
            num_tokens: Number of tokens in codebook
            codebook_dim: Dimension of embedding
            decay: EMA decay rate
            eps: Epsilon value for numerical stability
    """
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):

        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        """Drop-in replacement for nn.Embedding.forward()"""
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        """Update cluster size with EMA."""
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        """Update embedding average with EMA."""
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        """Update embedding weights based on EMA statistics."""
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)
    
    def perform_ema_update(self, encodings, z_flattened, num_tokens):
        """
        Perform EMA update given encodings and flattened input features.
        
        Args:
            encodings: One-hot encodings of shape (N, num_tokens) where N is flattened spatial dims
            z_flattened: Flattened input features of shape (N, codebook_dim)
            num_tokens: Total number of tokens in codebook
        """
        if self.training and self.update:
            # EMA cluster size
            encodings_sum = encodings.sum(0)
            self.cluster_size_ema_update(encodings_sum)
            # EMA embedding average
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            self.embed_avg_ema_update(embed_sum)
            # Normalize embed_avg and update weight
            self.weight_update(num_tokens)   


class ImplicitEmbedding(nn.Module):
    """
    Implicit embedding that can be used as a drop-in replacement for nn.Embedding.
    Provides a small MLP to map coordinates to codebook vectors.
    QINCode: https://arxiv.org/abs/2401.14732

    Args:
        n_e: Number of codes in codebook
        dim: Dimension of embedding
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of layers of MLP
    """
    def __init__(self, n_e, e_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        # coords or latent code parameters
        self.coords = nn.Parameter(torch.randn(n_e, e_dim))

        # small MLP: coords → code vectors
        layers = []
        in_dim = e_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, e_dim))
        self.mlp = nn.Sequential(*layers)

    @property
    def weight(self):
        """Acts like nn.Embedding.weight: returns full codebook matrix."""
        return self.mlp(self.coords)   # (n_e, e_dim)

    def forward(self, indices):
        """Acts like nn.Embedding(indices)."""
        codes = self.weight                 # (n_e, e_dim)
        return codes[indices]               # indexing works identically


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
    
def safe_div(num, den, eps = 1e-6):
    return num / den.clamp(min = eps)

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = l2norm(u + q, dim = 1).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )

def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, '* d')
    tgt, _ = pack_one(tgt, '* d')

    norm_src = src.norm(dim = -1, keepdim = True)
    norm_tgt = tgt.norm(dim = -1, keepdim = True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src),
        safe_div(tgt, norm_tgt),
        src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)

def clamp_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """ Clamps the input tensor and computes the log.

    Args:
        x -> torch.Tensor: The input tensor.
        eps -> float: The epsilon value serving as the lower bound.

    Returns:
        torch.Tensor: The log of the clamped input tensor.
    """
    return torch.log(torch.clamp(x, eps))

def entropy_loss_fn(
    affinity: torch.Tensor,
    temperature: float,
    entropy_gamma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Computes the entropy loss.

    Args:
        affinity -> torch.Tensor: The affinity matrix.
        temperature -> float: The temperature.
        entropy_gamma -> float: The entropy gamma.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The per-sample and average entropy.
    """
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature

    probability = flat_affinity.softmax(dim=-1)
    average_probability = torch.mean(probability, dim=0)

    per_sample_entropy = -1 * torch.mean(torch.sum(probability * clamp_log(probability), dim=-1))
    avg_entropy = torch.sum(-1 * average_probability * clamp_log(average_probability))

    return (per_sample_entropy, avg_entropy * entropy_gamma)
