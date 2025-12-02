# modelling/alignments.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use MAETokViTDecoder from modules
from .modules import MAETokViTDecoder, HOGGenerator
# For external models (DINO, CLIP) - try importing timm but make it optional
try:
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    create_model = None

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class AlignmentModule(ABC, nn.Module):
    """
    Base class for auxiliary alignment modules.
    Each module:
      - contains a decoder (MAETokViTDecoder)
      - contains projection heads (post_quant_conv and to_pixel)
      - contains a target model (frozen or external callable)
    Subclasses must implement `compute_target` to obtain the target representation for an input image.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the target features from the input image x.
        Should not require gradients (freeze target model).
        Returns tensor shaped (B, L, D_target)
        """
        raise NotImplementedError

    @abstractmethod
    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized tokens to predicted features in target space.
        Returns tensor shaped (B, L, D_target)
        """
        raise NotImplementedError

    def forward(self, quant: torch.Tensor, input_image: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute alignment loss between decoder(quant) and target features from input_image.
        Returns: (loss, predicted_features) — predicted_features optional (for logging)
        If mask is provided (same mask used in MaskAEModel), apply mask-aware reduction as in original code.
        """
        if input_image is None:
            raise ValueError("AlignmentModule requires input_image to compute target features")

        # predicted features from decoder/projection
        pred = self.decode_projection(quant)  # (B, L, D)
        # target features (usually frozen model)
        with torch.no_grad():
            target = self.compute_target(input_image)  # (B, L, D_target)

        # normalize both (original code normalized for dino/clip)
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)

        # compute per-token negative cosine (like original)
        per_token = -(pred_n * target_n).sum(dim=-1, keepdim=True)  # (B, L, 1)

        if mask is not None:
            # mask shape expected (B, L, 1) or (B, L)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            masked_sum = (per_token * mask).sum()
            denom = mask.sum().clamp(min=1.0)
            loss = masked_sum / denom
        else:
            loss = mean_flat(per_token.squeeze(-1))
            loss = loss.mean()

        return loss, pred

########################################################################
# HOG alignment module
########################################################################
class HOGAlignment(AlignmentModule):
    def __init__(
        self,
        image_size: int,
        num_latent_tokens: int,
        codebook_embed_dim: int,
        base_img_size: int,
        use_ape: bool,
        use_rope: bool,
        rope_mixed: bool,
        rope_theta: float,
        aux_dec_cls_token: bool = True,
        dec_embed_dim: int = 384,
        dec_depth: int = 12,
        dec_num_heads: int = 6,
        dec_mlp_ratio: float = 4.0,
        dec_patch_size: int = 16,
        use_movq: bool = False,
    ):
        super().__init__('hog')

        if HOGGenerator is None:
            raise RuntimeError("HOGGenerator not available; ensure modules.hog exists or pass an alternative.")

        # build decoder using MAETokViTDecoder
        self.decoder = MAETokViTDecoder(
            in_channels=3,
            embed_dim=dec_embed_dim,
            depth=dec_depth,
            num_heads=dec_num_heads,
            mlp_ratio=dec_mlp_ratio,
            img_size=image_size,
            patch_size=dec_patch_size,
            drop_path_rate=0.0,
            num_latent_tokens=num_latent_tokens,
            to_pixel='identity',
            codebook_embed_dim=codebook_embed_dim,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            use_rope=use_rope,
            use_ape=use_ape,
            cls_token=aux_dec_cls_token,
            base_img_size=base_img_size,
            use_movq=use_movq,
        )
        self.post_quant_conv = nn.Linear(codebook_embed_dim, self.decoder.embed_dim)
        # final pixel projection in original produced 108-d HOG channels
        self.to_pixel = nn.Linear(self.decoder.embed_dim, 108)
        self.hog_generator = HOGGenerator()

        self.hog_use_movq = use_movq

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        # HOG generator returns (B, L, 108) presumably
        z_hog = self.hog_generator(x)
        return z_hog

    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        tmp = quant
        x = self.post_quant_conv(quant)
        # decoder signature: decoder(z, interpolate_zq, H, W, D) 
        if self.hog_use_movq:
            dec = self.decoder(x, interpolate_zq=tmp, H=None, W=None, D=None)
        else:
            dec = self.decoder(x, interpolate_zq=None, H=None, W=None, D=None)
        # dec should be (B, L, embed_dim) when to_pixel='identity'
        # Apply to_pixel Linear layer to project to HOG feature dimension
        dec = self.to_pixel(dec)
        return dec

########################################################################
# Dino alignment module
########################################################################
class DinoAlignment(AlignmentModule):
    def __init__(
        self,
        image_size: int,
        num_latent_tokens: int,
        codebook_embed_dim: int,
        base_img_size: int,
        use_ape: bool,
        use_rope: bool,
        rope_mixed: bool,
        rope_theta: float,
        aux_dec_cls_token: bool = True,
        repa_model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        repa_patch_size: int = 14,
        dec_embed_dim: int = 384,
        dec_depth: int = 12,
        dec_num_heads: int = 6,
        dec_mlp_ratio: float = 4.0,
        use_movq: bool = False,
    ):
        super().__init__('dino')
        
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for DinoAlignment. Please install timm.")
        
        # Instantiate the repa/dinov2 model
        self.repa_model = create_model(repa_model_name, pretrained=True, img_size=image_size, patch_size=repa_patch_size)
        for p in self.repa_model.parameters():
            p.requires_grad = False
        self.repa_model.eval()
        
        # Normalization for DINO (ImageNet normalization)
        self.normalize = self._create_normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.denormalize = self._create_denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.decoder = MAETokViTDecoder(
            in_channels=3,
            embed_dim=dec_embed_dim,
            depth=dec_depth,
            num_heads=dec_num_heads,
            mlp_ratio=dec_mlp_ratio,
            img_size=image_size,
            patch_size=repa_patch_size,
            drop_path_rate=0.0,
            num_latent_tokens=num_latent_tokens,
            to_pixel='identity',
            codebook_embed_dim=codebook_embed_dim,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            use_rope=use_rope,
            use_ape=use_ape,
            cls_token=aux_dec_cls_token,
            base_img_size=base_img_size,
            use_movq=use_movq,
        )
        self.post_quant_conv = nn.Linear(codebook_embed_dim, self.decoder.embed_dim)
        # final projection to repa_model.embed_dim
        self.to_pixel = nn.Linear(self.decoder.embed_dim, self.repa_model.embed_dim)
        self.dino_use_movq = use_movq
    
    def _create_normalize(self, mean, std):
        class Normalize(nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
                self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
            def forward(self, x):
                return (x - self.mean) / self.std
        return Normalize(mean, std)
    
    def _create_denormalize(self, mean, std):
        class Denormalize(nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
                self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
            def forward(self, x):
                return x * self.std + self.mean
        return Denormalize(mean, std)

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess for repa_model: normalize using ImageNet stats
        # First denormalize from [0, 1] to ImageNet range, then normalize
        x_normalized = self.normalize(self.denormalize(x))
        z = self.repa_model.forward_features(x_normalized)[:, self.repa_model.num_prefix_tokens:]
        return z

    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        tmp = quant
        x = self.post_quant_conv(quant)
        if self.dino_use_movq:
            dec = self.decoder(x, interpolate_zq=tmp, H=None, W=None, D=None)
        else:
            dec = self.decoder(x, interpolate_zq=None, H=None, W=None, D=None)
        dec = self.to_pixel(dec)
        return dec

########################################################################
# CLIP alignment module
########################################################################
class ClipAlignment(AlignmentModule):
    def __init__(
        self,
        image_size: int,
        num_latent_tokens: int,
        codebook_embed_dim: int,
        base_img_size: int,
        use_ape: bool,
        use_rope: bool,
        rope_mixed: bool,
        rope_theta: float,
        aux_dec_cls_token: bool = True,
        clip_model_name: str = 'vit_so400m_patch14_siglip_gap_224',
        clip_patch_size: int = 14,
        dec_embed_dim: int = 384,
        dec_depth: int = 12,
        dec_num_heads: int = 6,
        dec_mlp_ratio: float = 4.0,
        use_movq: bool = False,
    ):
        super().__init__('clip')
        
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for ClipAlignment. Please install timm.")
        
        # Instantiate the CLIP model
        self.clip_model = create_model(clip_model_name, pretrained=True, img_size=image_size, patch_size=clip_patch_size)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        # Don't set dynamic_img_size=True as it expects spatial format from patch_embed
        # but the model returns flattened tokens (B, L, C)
        self.clip_model.eval()
        
        # Normalization for CLIP
        self.normalize = self._create_normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.denormalize = self._create_denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        self.decoder = MAETokViTDecoder(
            in_channels=3,
            embed_dim=dec_embed_dim,
            depth=dec_depth,
            num_heads=dec_num_heads,
            mlp_ratio=dec_mlp_ratio,
            img_size=image_size,
            patch_size=clip_patch_size,
            drop_path_rate=0.0,
            num_latent_tokens=num_latent_tokens,
            to_pixel='identity',
            codebook_embed_dim=codebook_embed_dim,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            use_rope=use_rope,
            use_ape=use_ape,
            cls_token=aux_dec_cls_token,
            base_img_size=base_img_size,
            use_movq=use_movq,
        )
        self.post_quant_conv = nn.Linear(codebook_embed_dim, self.decoder.embed_dim)
        self.to_pixel = nn.Linear(self.decoder.embed_dim, self.clip_model.embed_dim)
        self.clip_use_movq = use_movq
    
    def _create_normalize(self, mean, std):
        class Normalize(nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
                self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
            def forward(self, x):
                return (x - self.mean) / self.std
        return Normalize(mean, std)
    
    def _create_denormalize(self, mean, std):
        class Denormalize(nn.Module):
            def __init__(self, mean, std):
                super().__init__()
                self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
                self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
            def forward(self, x):
                return x * self.std + self.mean
        return Denormalize(mean, std)

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess for clip_model: normalize using ImageNet stats
        # First denormalize from [0, 1] to ImageNet range, then normalize
        x_normalized = self.normalize(self.denormalize(x))
        z = self.clip_model.forward_features(x_normalized)[:, self.clip_model.num_prefix_tokens:]
        return z

    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        tmp = quant
        x = self.post_quant_conv(quant)
        if self.clip_use_movq:
            dec = self.decoder(x, interpolate_zq=tmp, H=None, W=None, D=None)
        else:
            dec = self.decoder(x, interpolate_zq=None, H=None, W=None, D=None)
        dec = self.to_pixel(dec)
        return dec
