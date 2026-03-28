import logging
import torch
import torch.nn as nn
import numpy as np
from medlat.utils import init_from_ckpt

logger = logging.getLogger(__name__)
from medlat.modules.alignments import AlignmentModule
from medlat.first_stage.continuous.modules.ldm_modules import get_conv_layer
from medlat.first_stage.modules.gaussian_dist import DiagonalGaussianDistribution, _DeterministicPosterior

__all__ = [
    "AutoencoderKL",
    "AutoencoderKLTransformer",
]


class AutoencoderKL(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 alignment: AlignmentModule = None,
                 embed_dim: int = None,
                 kl_weight: float = 1e-6,
                 pre_post_layer: str = "conv",
                 double_z: bool = True,
                 channel_dim: int = 1,      # B, C, H, W --> 1 | B, N, C --> 2
                 ckpt_path: str = None):
        super().__init__()
        
        if pre_post_layer == "conv":
            self.dims = getattr(encoder, "dims", 2)
            conv_layer = get_conv_layer(self.dims)

        self.encoder = encoder
        self.decoder = decoder
        self.alignment = alignment
        self.double_z = double_z
        self.channel_dim = channel_dim

        self.encoder_z_channels = getattr(encoder, "z_channels", None)
        if self.encoder_z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        self.vae_stride = getattr(encoder, "vae_stride", None)
        if self.vae_stride is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define vae_stride.")

        self.kl_weight = kl_weight
        if embed_dim is None:
            embed_dim = self.encoder_z_channels
        self.embed_dim = embed_dim

        if pre_post_layer == "conv":
            self.quant_conv = conv_layer(2 * self.encoder_z_channels, 2 * embed_dim, 1)
            self.post_quant_conv = conv_layer(embed_dim, self.encoder_z_channels, 1)
        elif pre_post_layer == "linear":
            self.quant_conv = nn.Linear(2 * self.encoder_z_channels, 2 * embed_dim) 
            self.post_quant_conv = nn.Linear(embed_dim, self.encoder_z_channels)
        elif pre_post_layer == "none":
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        else:
            raise ValueError(f"Invalid pre_post_layer: {pre_post_layer}")

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)


    def get_posterior(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.double_z:
            # Encoder outputs z directly (e.g. pretrained DCAE); no Gaussian
            return _DeterministicPosterior(moments)
        posterior = DiagonalGaussianDistribution(moments, channel_dim=self.channel_dim)
        return posterior

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments, channel_dim=self.channel_dim)
        return posterior.sample(), self.p_loss(posterior, x.device), None


    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


    def p_loss(self, posterior, device): 
        kl_loss = torch.zeros((), device=device)
        if posterior is not None:
            # assume extra_result_dict contains posteriors with kl method
            if hasattr(posterior, "kl"):
                kl_loss = posterior.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = self.kl_weight * kl_loss
        return kl_loss

    def forward(self, input, sample_posterior=True):
        posterior = self.get_posterior(input)
        if self.double_z:
            p_loss = self.p_loss(posterior, input.device)
        else:
            p_loss = torch.zeros((), device=input.device)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        
        if self.alignment is not None:
            alignment_loss, _ = self.alignment(z, input)
            return dec, p_loss + alignment_loss
        
        return dec, p_loss



class AutoencoderKLTransformer(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 alignment: AlignmentModule = None,
                 embed_dim: int = None,
                 kl_weight: float = 1e-6,
                 pre_post_layer: str = "linear",
                 double_z: bool = True,
                 channel_dim: int = 2,      # B, C, H, W --> 1 | B, N, C --> 2
                 ckpt_path: str = None):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.alignment = alignment
        self.double_z = double_z
        self.channel_dim = channel_dim

        self.encoder_z_channels = getattr(encoder, "z_channels", None)
        if self.encoder_z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        self.vae_stride = getattr(encoder, "vae_stride", None)
        if self.vae_stride is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define vae_stride.")

        self.kl_weight = kl_weight
        if embed_dim is None:
            embed_dim = self.encoder_z_channels
        self.embed_dim = embed_dim

        if pre_post_layer == "linear":
            self.quant_conv = nn.Linear(2 * self.encoder_z_channels, 2 * embed_dim) 
            self.post_quant_conv = nn.Linear(embed_dim, self.encoder_z_channels)
        elif pre_post_layer == "none":
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        else:
            raise ValueError(f"Invalid pre_post_layer: {pre_post_layer}")

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)


    def get_posterior(self, x):
        h, aux = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.double_z:
            # Encoder outputs z directly (e.g. pretrained DCAE); no Gaussian
            return _DeterministicPosterior(moments), aux
        posterior = DiagonalGaussianDistribution(moments, channel_dim=self.channel_dim)
        return posterior, aux

    def encode(self, x):
        h, aux = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments, channel_dim=self.channel_dim)
        return posterior.sample(), self.p_loss(posterior, x.device), None, aux


    def decode(self, z, aux=None):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, ids_restore=aux["ids_restore"] if aux is not None else None)
        return dec


    def p_loss(self, posterior, device): 
        kl_loss = torch.zeros((), device=device)
        if posterior is not None:
            # assume extra_result_dict contains posteriors with kl method
            if hasattr(posterior, "kl"):
                kl_loss = posterior.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = self.kl_weight * kl_loss
        return kl_loss

    def forward(self, input, sample_posterior=True):
        posterior, aux = self.get_posterior(input)
        if self.double_z:
            p_loss = self.p_loss(posterior, input.device)
        else:
            p_loss = torch.zeros((), device=input.device)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        logger.debug(f"z: {z.shape}")
        dec = self.decode(z, aux=aux)
        
        if self.alignment is not None:
            alignment_loss, _ = self.alignment(z, input)
            return dec, p_loss + alignment_loss
        
        return dec, p_loss



