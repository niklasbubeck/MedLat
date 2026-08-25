import torch
import torch.nn as nn

from .base import AlignmentModule
from .losses import AlignmentLoss
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
        decoder=None,
        codebook_embed_dim=None,
        losses=None,
    ):
        super().__init__(
            name='hog',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=108,
            losses=losses,
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
        decoder=None,
        codebook_embed_dim=None,
        img_size: int = 224,
        repa_model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        repa_patch_size: int = 14,
        losses=None,
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
            losses=losses,
        )
        self.repa_model = repa_model
        self._teacher_img_size = img_size

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self._teacher_img_size or x.shape[-1] != self._teacher_img_size:
            x = nn.functional.interpolate(x, size=(self._teacher_img_size, self._teacher_img_size), mode='bilinear', align_corners=False)
        return self.repa_model.forward_features(x)[:, self.repa_model.num_prefix_tokens:]


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

class ClipAlignment(AlignmentModule):

    def __init__(
        self,
        decoder=None,
        codebook_embed_dim=None,
        img_size: int = 224,
        clip_model_name: str = 'vit_so400m_patch14_siglip_gap_224',
        clip_patch_size: int = 14,
        losses=None,
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
            losses=losses,
        )
        self.clip_model = clip_model
        self._teacher_img_size = img_size

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self._teacher_img_size or x.shape[-1] != self._teacher_img_size:
            x = nn.functional.interpolate(x, size=(self._teacher_img_size, self._teacher_img_size), mode='bilinear', align_corners=False)
        return self.clip_model.forward_features(x)[:, self.clip_model.num_prefix_tokens:]


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class MAEAlignment(AlignmentModule):

    def __init__(
        self,
        decoder=None,
        codebook_embed_dim=None,
        img_size: int = 224,
        model_name: str = 'hf-hub:timm/vit_large_patch16_224.mae',
        patch_size: int = 16,
        losses=None,
    ):
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for MAEAlignment. Please install timm.")

        mae_model = create_model(
            model_name, pretrained=True, dynamic_img_size=True,
        )
        for p in mae_model.parameters():
            p.requires_grad = False
        mae_model.eval()

        super().__init__(
            name='mae',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=mae_model.embed_dim,
            losses=losses,
        )
        self.mae_model = mae_model
        self._teacher_img_size = img_size

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self._teacher_img_size or x.shape[-1] != self._teacher_img_size:
            x = nn.functional.interpolate(x, size=(self._teacher_img_size, self._teacher_img_size), mode='bilinear', align_corners=False)
        return self.mae_model.forward_features(x)[:, self.mae_model.num_prefix_tokens:]


# ---------------------------------------------------------------------------
# BiomedCLIP
# ---------------------------------------------------------------------------

class BiomedClipAlignment(AlignmentModule):

    def __init__(
        self,
        decoder=None,
        codebook_embed_dim=None,
        base_size: int = 224,
        losses=None,
    ):
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        model.requires_grad_(False)
        model.eval()

        super().__init__(
            name='biomedclip',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=512,
            losses=losses,
        )
        self.biomed_model = model
        self.base_size = base_size
        mean = torch.tensor(list(model.visual.image_mean)).view(1, -1, 1, 1)
        std = torch.tensor(list(model.visual.image_std)).view(1, -1, 1, 1)
        self.register_buffer("biomed_mean", mean)
        self.register_buffer("biomed_std", std)

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.base_size, self.base_size):
            x = nn.functional.interpolate(
                x, size=(self.base_size, self.base_size),
                mode='bilinear', align_corners=False,
            )
        x = (x - self.biomed_mean) / self.biomed_std
        emb = self.biomed_model.encode_image(x)
        return emb.unsqueeze(-1).unsqueeze(-1)
