"""SAMTeacher / MedSAMTeacher — teachers for SAM-family vision encoders.

SAM ViTs output ``(B, C, H, W)`` after a convolutional neck (C=256),
not the standard ``(B, L, D)`` patch tokens.  These subclasses handle
the reshape.
"""

import torch.nn as nn
import torch.nn.functional as F

from .base import TeacherModel


class SAMTeacher(TeacherModel):
    """Teacher wrapping a timm SAM-ViT (e.g. ``samvit_base_patch16.sa1b``).

    The neck reduces the channel dimension to 256 and outputs a 2-D
    feature map.  We flatten to ``(B, H*W, 256)``.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name, pretrained=True, img_size=1024):
        try:
            from timm import create_model
        except ImportError as e:
            raise RuntimeError("timm is required for SAMTeacher.") from e
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained,
                                  img_size=img_size)
        self.model.requires_grad_(False)
        self._img_size = img_size

    @property
    def embed_dim(self):
        return 256

    @property
    def input_transform(self):
        from medlat.alignments.utils import _Normalize
        return _Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)

    def extract_features(self, x):
        if x.shape[-2] != self._img_size or x.shape[-1] != self._img_size:
            x = F.interpolate(x, size=(self._img_size, self._img_size),
                              mode='bilinear', align_corners=False)
        out = self.model.forward_features(x)  # (B, C, H, W)
        B, C, H, W = out.shape
        return out.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)


class MedSAMTeacher(TeacherModel):
    """Teacher wrapping MedSAM's vision encoder from HuggingFace.

    Uses ``transformers.SamModel`` and extracts from the vision encoder,
    which outputs ``(B, H*W, C)`` with C=256 after the neck.
    """

    MEDSAM_MEAN = [0.485, 0.456, 0.406]
    MEDSAM_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name, img_size=1024):
        try:
            from transformers import SamModel
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for MedSAMTeacher."
            ) from e
        super().__init__()
        sam = SamModel.from_pretrained(model_name)
        self.vision_encoder = sam.vision_encoder
        self.neck = sam.shared_image_embedding if hasattr(sam, 'shared_image_embedding') else None
        self.vision_encoder.requires_grad_(False)
        self._img_size = img_size
        native = self.vision_encoder.config.image_size
        if img_size != native:
            self._resize_for(img_size, native)

    def _resize_for(self, img_size, native):
        """Run the 1024-native encoder at ``img_size``.

        HF's SAM rejects other input sizes and adds a fixed (1, 64, 64, C)
        absolute positional embedding. Resize that grid bicubically to the new
        patch grid and relax the size check; the relative position tables of the
        attention blocks are already interpolated by HF's ``get_rel_pos``.
        """
        enc = self.vision_encoder
        patch = enc.config.patch_size
        if img_size % patch:
            raise ValueError(f"img_size {img_size} is not a multiple of the patch size {patch}")
        grid = img_size // patch
        enc.patch_embed.image_size = (img_size, img_size)
        if enc.pos_embed is not None:
            pos = enc.pos_embed.data.permute(0, 3, 1, 2)  # (1, C, 64, 64)
            pos = F.interpolate(pos, size=(grid, grid), mode='bicubic', align_corners=False)
            enc.pos_embed = nn.Parameter(pos.permute(0, 2, 3, 1).contiguous(),
                                         requires_grad=False)

    @property
    def embed_dim(self):
        return 256

    @property
    def input_transform(self):
        from medlat.alignments.utils import _Normalize
        return _Normalize(self.MEDSAM_MEAN, self.MEDSAM_STD)

    def extract_features(self, x):
        if x.shape[-2] != self._img_size or x.shape[-1] != self._img_size:
            x = F.interpolate(x, size=(self._img_size, self._img_size),
                              mode='bilinear', align_corners=False)
        out = self.vision_encoder(x).last_hidden_state  # (B, C, H, W)
        B, C, H, W = out.shape
        return out.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
