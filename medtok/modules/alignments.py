# modelling/alignments.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np

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

    def generate_hog_image(self, hog_out: torch.Tensor) -> np.ndarray:
        """Generate HOG image according to HOG features."""
        assert hog_out.size(0) == 1 and hog_out.size(1) == 3, \
            'Check the input batch size and the channcel number, only support'\
            '"batch_size = 1".'
        hog_image = np.zeros([self.h, self.w])
        cell_gradient = np.array(hog_out.mean(dim=1).squeeze().detach().cpu())
        cell_width = self.pool / 2
        max_mag = np.array(cell_gradient).max()
        angle_gap = 360 / self.nbins

        for x in range(cell_gradient.shape[1]):
            for y in range(cell_gradient.shape[2]):
                cell_grad = cell_gradient[:, x, y]
                cell_grad /= max_mag
                angle = 0
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.pool +
                             magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.pool +
                             magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.pool -
                             magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.pool -
                             magnitude * cell_width * math.sin(angle_radian))
                    magnitude = 0 if magnitude < 0 else magnitude
                    cv2.line(hog_image, (y1, x1), (y2, x2),
                             int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return hog_image


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
        image_size: int,
        repa_model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        repa_patch_size: int = 14,
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
        
        # Use provided decoder
        self.decoder = decoder
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
        decoder: nn.Module,
        codebook_embed_dim: int,
        image_size: int,
        clip_model_name: str = 'vit_so400m_patch14_siglip_gap_224',
        clip_patch_size: int = 14,
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
        
        # Use provided decoder
        self.decoder = decoder
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
