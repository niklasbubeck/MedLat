import torch
import torch.nn as nn

from .base import TokenizerAlignment
from .utils import HOGGenerator, _Denormalize


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------

class HOGAlignment(TokenizerAlignment):

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

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.hog_generator(x)


# ---------------------------------------------------------------------------
# DINO
# ---------------------------------------------------------------------------

class DinoAlignment(TokenizerAlignment):
    """DINOv2-based tokenizer alignment.

    Args:
        teacher: A :class:`TeacherModel` instance.  If ``None``, a
            :class:`TimmTeacher` is created from ``repa_model_name``.
    """

    def __init__(
        self,
        decoder=None,
        codebook_embed_dim=None,
        teacher=None,
        img_size: int = 224,
        repa_model_name: str = 'vit_large_patch14_dinov2.lvd142m',
        repa_patch_size: int = 14,
        losses=None,
    ):
        if teacher is None:
            from .teachers import TimmTeacher
            teacher = TimmTeacher(
                repa_model_name, img_size=img_size,
                patch_size=repa_patch_size,
            )

        super().__init__(
            name='dino',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=teacher.embed_dim,
            input_transform=teacher.input_transform,
            losses=losses,
        )
        self.teacher = teacher

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.teacher.extract_features(x)


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

class ClipAlignment(TokenizerAlignment):
    """CLIP / SigLIP tokenizer alignment.

    The default ``input_transform`` denormalises from ``[-1, 1]`` (common
    for diffusion pipelines) then applies the teacher's normalization.

    Args:
        teacher: A :class:`TeacherModel` instance.  If ``None``, a
            :class:`TimmTeacher` is created from ``clip_model_name``.
    """

    def __init__(
        self,
        decoder=None,
        codebook_embed_dim=None,
        teacher=None,
        img_size: int = 224,
        clip_model_name: str = 'vit_so400m_patch14_siglip_gap_224',
        clip_patch_size: int = 14,
        losses=None,
    ):
        if teacher is None:
            from .teachers import TimmTeacher
            teacher = TimmTeacher(
                clip_model_name, img_size=img_size,
                patch_size=clip_patch_size,
            )

        super().__init__(
            name='clip',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=teacher.embed_dim,
            input_transform=nn.Sequential(
                _Denormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                teacher.input_transform,
            ),
            losses=losses,
        )
        self.teacher = teacher

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.teacher.extract_features(x)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class MAEAlignment(TokenizerAlignment):
    """MAE tokenizer alignment.

    Args:
        teacher: A :class:`TeacherModel` instance.  If ``None``, a
            :class:`TimmTeacher` is created from ``model_name`` with
            ``dynamic_img_size=True``.
    """

    def __init__(
        self,
        decoder=None,
        codebook_embed_dim=None,
        teacher=None,
        img_size: int = 224,
        model_name: str = 'hf-hub:timm/vit_large_patch16_224.mae',
        patch_size: int = 16,
        losses=None,
    ):
        if teacher is None:
            from .teachers import TimmTeacher
            teacher = TimmTeacher(
                model_name, img_size=img_size, dynamic_img_size=True,
            )

        super().__init__(
            name='mae',
            decoder=decoder,
            codebook_embed_dim=codebook_embed_dim,
            target_dim=teacher.embed_dim,
            losses=losses,
        )
        self.teacher = teacher

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.teacher.extract_features(x)


# ---------------------------------------------------------------------------
# BiomedCLIP
# ---------------------------------------------------------------------------

class BiomedClipAlignment(TokenizerAlignment):

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

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.shape[-2:] != (self.base_size, self.base_size):
            x = nn.functional.interpolate(
                x, size=(self.base_size, self.base_size),
                mode='bilinear', align_corners=False,
            )
        x = (x - self.biomed_mean) / self.biomed_std
        emb = self.biomed_model.encode_image(x)
        return emb.unsqueeze(-1).unsqueeze(-1)
