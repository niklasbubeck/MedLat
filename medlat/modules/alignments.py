# modelling/alignments.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchvision.transforms as T

from medlat.modules.metrics import MetricLoggerMixin

# For external models (DINO, CLIP) - try importing timm but make it optional
try:
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    create_model = None




class HOGGenerator(nn.Module):
    """Generate HOG feature for images.

    This module is used in MaskFeat to generate HOG feature. The code is
    modified from file `slowfast/models/operators.py
    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/operators.py>`_.
    Here is the link of `HOG wikipedia
    <https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients>`_.

    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16) -> None:
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).contiguous()
        weight_y = weight_x.transpose(2, 3).contiguous()
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gaussian_kernel = self.get_gaussian_kernel(gaussian_window,
                                                       gaussian_window // 2)
            self.register_buffer('gaussian_kernel', gaussian_kernel)

    def get_gaussian_kernel(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        kernel_1d = _gaussian_fn(kernlen, std)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d / kernel_2d.sum()

    def _reshape(self, hog_feat: torch.Tensor) -> torch.Tensor:
        """Reshape HOG Features for output."""
        hog_feat = hog_feat.flatten(1, 2)
        self.unfold_size = hog_feat.shape[-1] // 16
        hog_feat = hog_feat.permute(0, 2, 3, 1)
        # print(hog_feat.shape)
        hog_feat = hog_feat.unfold(1, self.unfold_size,
                                   self.unfold_size).unfold(
                                       2, self.unfold_size, self.unfold_size)
        hog_feat = hog_feat.flatten(1, 2).flatten(2)
        return hog_feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.

        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Hog features.
        """
        # input is RGB image with shape [B 3 H W]
        self.h, self.w = x.size(-2), x.size(-1)
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gaussian_kernel = self.gaussian_kernel.repeat(
                    [repeat_rate, repeat_rate])
            else:
                temp_gaussian_kernel = self.gaussian_kernel
            norm_rgb *= temp_gaussian_kernel

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        self.out = F.normalize(out, p=2, dim=2)

        return self._reshape(self.out)

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


class _Normalize(nn.Module):
    """Channel-wise normalisation: (x - mean) / std. Buffers stay on the correct device."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _Denormalize(nn.Module):
    """Inverse of _Normalize: x * std + mean."""
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class AlignmentModule(MetricLoggerMixin, ABC, nn.Module):
    """Base class for auxiliary alignment modules.

    Each module contains a decoder (``MAETokViTDecoder``), projection heads
    (``post_quant_conv`` and ``to_pixel``), and a target model (frozen or
    external callable). Subclasses must implement ``compute_target`` to obtain
    the target representation for an input image.

    **Decoder interface contract** — subclasses that call a decoder must conform
    to the following signature::

        decoder(x, interpolate_zq, H, W, D) -> Tensor  # (B, L, embed_dim)

    where ``x`` is the post-projected token sequence ``(B, L, codebook_embed_dim)``,
    ``interpolate_zq`` is the raw quant tokens for skip connections (or ``None``),
    and ``H, W, D`` are the spatial dimensions (pass ``None`` for 2-D inputs).
    The decoder must return a token sequence shaped ``(B, L, embed_dim)``.

    **Observability** — inherits ``log_metric`` / ``get_metrics`` /
    ``reset_metrics`` from :class:`MetricLoggerMixin`. Every :meth:`forward`
    call auto-logs ``"alignment_loss"``; the owning first-stage autoencoder
    (:class:`VQModelBase` / :class:`AutoencoderKLBase`) merges this into
    ``model.get_metrics()``.
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

        # target features (usually frozen model)
        with torch.no_grad():
            target = self.compute_target(input_image)  # (B, L, D_target)

        # Allow subclasses to adapt projection heads if target dim changes with image size
        self.ensure_projection_dim(target.size(-1))

        # predicted features from decoder/projection
        pred = self.decode_projection(quant)  # (B, L_pred, D)

        # Align token grids if lengths differ (e.g., different patch sizes)
        if pred.shape[1] != target.shape[1]:
            pred = self._interpolate_tokens_to_match(pred, target)
            if mask is not None:
                mask = self._interpolate_mask_to_match(mask, target)
        elif mask is not None and mask.shape[1] != pred.shape[1]:
            # If pred and target already match but mask length differs, align mask to pred
            mask = self._interpolate_mask_to_match(mask, pred)

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

        self.log_metric("alignment_loss", loss.detach())
        return loss, pred

    # ------------------------------------------------------------------ #
    # Helpers for dynamic projection and token/mask alignment
    # ------------------------------------------------------------------ #
    def ensure_projection_dim(self, target_dim: int):
        """Subclasses can override if their projection depends on target dim."""
        return

    def _infer_grid_hw(self, seq_len: int) -> Tuple[int, int]:
        """Infer a square grid (h, w) from sequence length.

        Assumes the token sequence folds into a square spatial grid, i.e.
        seq_len must be a perfect square (e.g. 256 → 16×16).

        Non-square inputs are not supported here.  If your model produces
        non-square token grids (e.g. 192×256 image with patch_size=16 → 12×16
        tokens), override ``_interpolate_tokens_to_match`` in the subclass and
        track H/W explicitly.

        Raises:
            ValueError: if seq_len is not a perfect square.
        """
        side = int(math.sqrt(seq_len))
        if side * side != seq_len:
            raise ValueError(
                f"Cannot infer square grid from seq_len={seq_len}. "
                "AlignmentModule assumes square token grids. For non-square grids "
                "override _interpolate_tokens_to_match in the subclass."
            )
        return side, side

    def _interpolate_tokens_to_match(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Reshape pred tokens to grid, interpolate to target grid, then flatten back.
        Assumes 2D tokens.
        """
        b, lp, c = pred.shape
        lt = target.shape[1]
        ph, pw = self._infer_grid_hw(lp)
        th, tw = self._infer_grid_hw(lt)
        pred_map = pred.view(b, ph, pw, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        pred_map = F.interpolate(pred_map, size=(th, tw), mode='bilinear', align_corners=False)
        pred = pred_map.permute(0, 2, 3, 1).reshape(b, lt, c)
        return pred

    def _interpolate_mask_to_match(self, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Reshape mask tokens to grid, interpolate to target grid, then flatten back.
        Keeps mask float-valued.
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        b, lp, c = mask.shape
        lt = target.shape[1]
        ph, pw = self._infer_grid_hw(lp)
        th, tw = self._infer_grid_hw(lt)
        mask_map = mask.view(b, ph, pw, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        mask_map = F.interpolate(mask_map, size=(th, tw), mode='nearest')
        mask = mask_map.permute(0, 2, 3, 1).reshape(b, lt, c)
        return mask

########################################################################
# HOG alignment module
########################################################################
class HOGAlignment(AlignmentModule):
    def __init__(
        self,
        decoder: nn.Module,
        codebook_embed_dim: int,
        use_movq: bool = False,
    ):
        super().__init__('hog')

        if HOGGenerator is None:
            raise RuntimeError("HOGGenerator not available; ensure modules.hog exists or pass an alternative.")

        # Use provided decoder
        self.decoder = decoder
        self.post_quant_conv = nn.Linear(codebook_embed_dim, self.decoder.embed_dim)
        # final pixel projection in original produced 108-d HOG channels
        self.to_pixel = nn.Linear(self.decoder.embed_dim, 108)
        self.hog_generator = HOGGenerator()

        self.hog_use_movq = use_movq

    def ensure_projection_dim(self, target_dim: int):
        # Rebuild projection if target channels change with image size
        if self.to_pixel.out_features != target_dim:
            requires_grad = self.to_pixel.weight.requires_grad
            self.to_pixel = nn.Linear(self.decoder.embed_dim, target_dim).to(self.to_pixel.weight.device)
            self.to_pixel.requires_grad_(requires_grad)

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
        decoder: nn.Module,
        codebook_embed_dim: int,
        img_size: int,
        repa_model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        repa_patch_size: int = 14,
        use_movq: bool = False,
    ):
        super().__init__('dino')
        
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for DinoAlignment. Please install timm.")
        
        # Instantiate the repa/dinov2 model
        self.repa_model = create_model(repa_model_name, pretrained=True, img_size=img_size, patch_size=repa_patch_size)
        for p in self.repa_model.parameters():
            p.requires_grad = False
        self.repa_model.eval()
        
        # Normalize inputs to ImageNet statistics expected by DINOv2.
        # Assumes input images are in [0, 1] range (standard for VAE training).
        self.normalize = _Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Use provided decoder
        self.decoder = decoder
        self.post_quant_conv = nn.Linear(codebook_embed_dim, self.decoder.embed_dim)
        # final projection to repa_model.embed_dim
        self.to_pixel = nn.Linear(self.decoder.embed_dim, self.repa_model.embed_dim)
        self.dino_use_movq = use_movq
    
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected in [0, 1]; normalize to ImageNet mean/std for DINOv2.
        x_normalized = self.normalize(x)
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
        decoder: nn.Module,
        codebook_embed_dim: int,
        img_size: int,
        clip_model_name: str = 'vit_so400m_patch14_siglip_gap_224',
        clip_patch_size: int = 14,
        use_movq: bool = False,
    ):
        super().__init__('clip')
        
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for ClipAlignment. Please install timm.")
        
        # Instantiate the CLIP model
        self.clip_model = create_model(clip_model_name, pretrained=True, img_size=img_size, patch_size=clip_patch_size)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        # Don't set dynamic_img_size=True as it expects spatial format from patch_embed
        # but the model returns flattened tokens (B, L, C)
        self.clip_model.eval()
        
        # Normalization for CLIP: input expected in [-1, 1] (mean=0.5, std=0.5).
        # denormalize maps [-1,1] → [0,1], normalize applies ImageNet stats.
        self.normalize = _Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.denormalize = _Denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # Use provided decoder
        self.decoder = decoder
        self.post_quant_conv = nn.Linear(codebook_embed_dim, self.decoder.embed_dim)
        self.to_pixel = nn.Linear(self.decoder.embed_dim, self.clip_model.embed_dim)
        self.clip_use_movq = use_movq

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected in [-1, 1] (mean=0.5, std=0.5 normalization).
        # denormalize maps [-1,1] → [0,1], then normalize applies ImageNet stats.
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


########################################################################
# BiomedCLIP alignment module (mirrors original BiomedClipLoss behavior)
########################################################################
# class BiomedClipAlignment(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         decoder: nn.Module = None,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.decoder = decoder
#         if self.decoder is None: # then use original default
#             self.decoder = torch.nn.Sequential(
#                 torch.nn.Conv2d(self.embed_dim, 64, 1),
#                 torch.nn.ReLU(),
#                 torch.nn.Conv2d(64, 64, 3, padding="same"),
#                 torch.nn.ReLU(),
#                 torch.nn.Conv2d(64, self.embed_dim, 1),
#             )
#         self.channel_proj = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1)


#         # Load BiomedCLIP (OpenCLIP) and set to eval/frozen
#         self.clip_model, _, _ = open_clip.create_model_and_transforms(
#             model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
#         )
#         for p in self.clip_model.parameters():
#             p.requires_grad_(False)
#         self.clip_model.eval()

#         # Preprocessing to mirror the original loss
#         self.transform = T.Compose(
#             [
#                 T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
#                 T.CenterCrop(size=(224, 224)),
#                 T.Normalize(
#                     mean=[0.48145466, 0.4578275, 0.40821073],
#                     std=[0.26862954, 0.26130258, 0.27577711],
#                 ),
#             ]
#         )

#     def forward(self, quant: torch.Tensor, input_image: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
#         if input_image is None:
#             raise ValueError("BiomedClipAlignment requires input_image to compute target features")

#         latent = self.channel_proj(self.decoder(quant) + quant)  # (B, L, D)
#         img = self.transform(input_image)
#         img_features = self.clip_model.encode_image(img)  # (B, D)

#         latent = latent / 4.6
#         latent = latent.mean(1, keepdim=True)
#         latent = self.transform(latent.expand(-1, 3, -1, -1))
#         latent_features = self.clip_model.encode_image(latent)

#         img_lat_loss = ((img_features - latent_features) ** 2).sum(1).mean()
#         return img_lat_loss, None


########################################################################
# Vision-Foundation alignment module (VA-VAE style)
########################################################################


class FoundationFeatureExtractor(nn.Module):
    """
    Lightweight wrapper to fetch frozen vision-foundation features.

    Supports MAE and DINOv2-L. Produces spatial feature maps shaped (B, C, H', W').
    """

    def __init__(self, model_type: str):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for FoundationFeatureExtractor.")

        self.model_type = model_type.lower()
        if self.model_type == "mae":
            model_name = "hf-hub:timm/vit_large_patch16_224.mae"
            self.model = create_model(model_name, pretrained=True, dynamic_img_size=True)
            self.patch_size = 16
            self.base_size = 224
            self.feature_dim = 1024
        elif self.model_type == "dinov2":
            model_name = "hf-hub:timm/vit_large_patch14_dinov2.lvd142m"
            self.model = create_model(model_name, pretrained=True, dynamic_img_size=True)
            self.patch_size = 14
            self.base_size = 224
            self.feature_dim = 1024
        elif self.model_type == "biomedclip":
            # OpenCLIP BiomedCLIP ViT-B/16; use encode_image for vision tower
            import open_clip
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            )
            self.patch_size = 16
            self.base_size = 224
            # ViT-B hidden dim
            self.feature_dim = 512
            # Normalize inputs like OpenCLIP preprocess
            mean = torch.tensor(self.model.visual.image_mean).view(1, -1, 1, 1)
            std = torch.tensor(self.model.visual.image_std).view(1, -1, 1, 1)
            self.register_buffer("biomed_mean", mean)
            self.register_buffer("biomed_std", std)
        else:
            raise ValueError(f"Unsupported foundation model type: {model_type}")

        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.model_type == "dinov2":
            # DINOv2 expects 224x224 crops; resize then reshape tokens back to spatial grid.
            if x.shape[-2:] != (224, 224):
                x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            tokens = self.model.forward_features(x)[:, 1:]
            # DINOv2 with patch_size=14 on 224x224 gives 16x16 patches
            feat_h = 224 // 14  # = 16
            feat_w = 224 // 14  # = 16
            return tokens.reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2)
        if self.model_type == "biomedclip":
            # BiomedCLIP ViT-B/16: use encode_image, returns (B, D)
            target_size = (self.base_size, self.base_size)
            if x.shape[-2:] != target_size:
                x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x_norm = (x - self.biomed_mean) / self.biomed_std
            emb = self.model.encode_image(x_norm)  # (B, D)
            return emb.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)

        # MAE supports dynamic image size; reshape patch tokens (drop cls).
        tokens = self.model.forward_features(x)[:, 1:]
        feat_h = h // self.patch_size
        feat_w = w // self.patch_size
        return tokens.reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2)



class VFFoundationAlignment(AlignmentModule):
    """Align latent feature maps with frozen vision foundation model features.

    Uses a two-part VF loss: ``vf_loss_1`` computes similarity-matrix distance
    with a margin, and ``vf_loss_2`` computes a per-location cosine margin loss.
    """

    def __init__(
        self,
        latent_channels: int,
        foundation_type: str = "dinov2",
        reverse_proj: bool = True,
        distmat_margin: float = 0.25,
        cos_margin: float = 0.5,
        distmat_weight: float = 1.0,
        cos_weight: float = 1.0,
    ):
        super().__init__('vf')
        self.foundation_model = FoundationFeatureExtractor(foundation_type)
        self.reverse_proj = reverse_proj
        self.distmat_margin = distmat_margin
        self.cos_margin = cos_margin
        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight

        aux_dim = self.foundation_model.feature_dim
        if reverse_proj:
            # Map latent -> foundation space
            self.linear_proj = nn.Conv2d(latent_channels, aux_dim, kernel_size=1)
        else:
            # Map foundation -> latent space
            self.linear_proj = nn.Conv2d(aux_dim, latent_channels, kernel_size=1)

    def _ensure_4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x
        if x.dim() == 3:
            b, l, c = x.shape
            side = int(math.sqrt(l))
            if side * side != l:
                raise ValueError("Latent tokens length is not a perfect square; provide 4D feature maps.")
            return x.transpose(1, 2).reshape(b, c, side, side)
        raise ValueError("Expected latent as (B,C,H,W) or (B,L,C)")

    def _match_spatial(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resize tensors so they share spatial resolution (H, W).
        """
        if a.shape[-2:] == b.shape[-2:]:
            return a, b
        return (
            nn.functional.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False),
            b,
        )

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.foundation_model(x)

    def decode_projection(self, quant: torch.Tensor) -> torch.Tensor:
        # For VF alignment, quant is treated as latent feature map (B,C,H,W) or (B,L,C).
        return self._ensure_4d(quant)

    def forward(  # type: ignore[override]
        self,
        quant: torch.Tensor,
        input_image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_image is None:
            raise ValueError("VFFoundationAlignment requires input_image to compute target features.")

        z = self.decode_projection(quant)  # (B, C, H, W)
        aux_feature = self.compute_target(input_image)  # (B, C_aux, H', W')

        # Project to shared channel space
        if self.reverse_proj:
            z_proj = self.linear_proj(z)
            aux_proj = aux_feature
        else:
            aux_proj = self.linear_proj(aux_feature)
            z_proj = z

        # Match spatial shapes
        aux_proj, z_proj = self._match_spatial(aux_proj, z_proj)

        # Compute VF losses
        b, c, h, w = z_proj.shape
        z_flat = z_proj.view(b, c, -1)           # (B, C, N)
        aux_flat = aux_proj.view(b, aux_proj.shape[1], -1)

        z_norm = torch.nn.functional.normalize(z_flat, dim=1)    # (B, C, N)
        aux_norm = torch.nn.functional.normalize(aux_flat, dim=1)

        # Compute loss_1 using bmm (faster than einsum for this contraction).
        # bmm(Zᵀ, Z) is mathematically identical to einsum('bci,bcj->bij', Z, Z):
        #   result[b,i,j] = sum_c Z[b,c,i] * Z[b,c,j]
        # Out-of-place ops required: z_sim is a non-leaf tensor in the autograd
        # graph (gradients flow through linear_proj), so in-place modification
        # would corrupt the graph and raise a version-counter error during backward.
        z_sim = torch.bmm(z_norm.transpose(1, 2), z_norm)        # (B, N, N)
        aux_sim = torch.bmm(aux_norm.transpose(1, 2), aux_norm)   # (B, N, N)
        diff = (z_sim - aux_sim).abs()
        vf_loss_1 = torch.nn.functional.relu(diff - self.distmat_margin).mean()
        del z_sim, aux_sim, diff  # free large intermediates before vf_loss_2
        vf_loss_2 = torch.nn.functional.relu(
            1 - self.cos_margin - torch.nn.functional.cosine_similarity(aux_proj, z_proj, dim=1)
        ).mean()

        vf_loss = vf_loss_1 * self.distmat_weight + vf_loss_2 * self.cos_weight
        self.log_metric("alignment_loss", vf_loss.detach())
        self.log_metric("vf_loss_distmat", vf_loss_1.detach())
        self.log_metric("vf_loss_cos", vf_loss_2.detach())
        return vf_loss, z_proj
