import torch
import torch.nn as nn

from .base import AlignmentModule
from .losses import DistmatMarginLoss, CosineMarginLoss
from .utils import HOGGenerator, _Normalize, _Denormalize

try:
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    create_model = None


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------

class HOGAlignment(AlignmentModule):

    def __init__(
        self,
        decoder: nn.Module,
        codebook_embed_dim: int,
    ):
        super().__init__(
            name='hog',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=108,
        )
        self.hog_generator = HOGGenerator()

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.hog_generator(x)


# ---------------------------------------------------------------------------
# DINO
# ---------------------------------------------------------------------------

class DinoAlignment(AlignmentModule):

    def __init__(
        self,
        decoder: nn.Module,
        codebook_embed_dim: int,
        img_size: int,
        repa_model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        repa_patch_size: int = 14,
    ):
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for DinoAlignment. Please install timm.")

        repa_model = create_model(
            repa_model_name, pretrained=True,
            img_size=img_size, patch_size=repa_patch_size,
        )
        for p in repa_model.parameters():
            p.requires_grad = False
        repa_model.eval()

        super().__init__(
            name='dino',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=repa_model.embed_dim,
            input_transform=_Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
            ),
        )
        self.repa_model = repa_model

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.repa_model.forward_features(x)[:, self.repa_model.num_prefix_tokens:]


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

class ClipAlignment(AlignmentModule):

    def __init__(
        self,
        decoder: nn.Module,
        codebook_embed_dim: int,
        img_size: int,
        clip_model_name: str = 'vit_so400m_patch14_siglip_gap_224',
        clip_patch_size: int = 14,
    ):
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for ClipAlignment. Please install timm.")

        clip_model = create_model(
            clip_model_name, pretrained=True,
            img_size=img_size, patch_size=clip_patch_size,
        )
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_model.eval()

        super().__init__(
            name='clip',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=clip_model.embed_dim,
            input_transform=nn.Sequential(
                _Denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                _Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ),
        )
        self.clip_model = clip_model

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip_model.forward_features(x)[:, self.clip_model.num_prefix_tokens:]


# ---------------------------------------------------------------------------
# VF Foundation
# ---------------------------------------------------------------------------

class FoundationFeatureExtractor(nn.Module):
    """Lightweight wrapper to fetch frozen vision-foundation features.

    Supports MAE, DINOv2-L, and BiomedCLIP.
    Produces spatial feature maps shaped ``(B, C, H', W')``.
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
            import open_clip
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            )
            self.patch_size = 16
            self.base_size = 224
            self.feature_dim = 512
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
            if x.shape[-2:] != (224, 224):
                x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            tokens = self.model.forward_features(x)[:, 1:]
            feat_h = 224 // 14
            feat_w = 224 // 14
            return tokens.reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2)
        if self.model_type == "biomedclip":
            target_size = (self.base_size, self.base_size)
            if x.shape[-2:] != target_size:
                x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x_norm = (x - self.biomed_mean) / self.biomed_std
            emb = self.model.encode_image(x_norm)
            return emb.unsqueeze(-1).unsqueeze(-1)

        # MAE
        tokens = self.model.forward_features(x)[:, 1:]
        feat_h = h // self.patch_size
        feat_w = w // self.patch_size
        return tokens.reshape(b, feat_h, feat_w, -1).permute(0, 3, 1, 2)


class VFFoundationAlignment(AlignmentModule):
    """Align quantised tokens with frozen vision-foundation features.

    Uses a decoder pipeline (via the base class) and a two-part VF loss:
    ``DistmatMarginLoss`` (similarity-matrix distance with a margin) and
    ``CosineMarginLoss`` (per-location cosine margin).
    """

    def __init__(
        self,
        decoder: nn.Module,
        codebook_embed_dim: int,
        foundation_type: str = "dinov2",
        distmat_margin: float = 0.25,
        cos_margin: float = 0.5,
        distmat_weight: float = 1.0,
        cos_weight: float = 1.0,
    ):
        foundation = FoundationFeatureExtractor(foundation_type)

        super().__init__(
            name='vf',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=foundation.feature_dim,
            losses=[
                (DistmatMarginLoss(margin=distmat_margin), distmat_weight),
                (CosineMarginLoss(margin=cos_margin), cos_weight),
            ],
        )
        self.foundation_model = foundation

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.foundation_model(x)
