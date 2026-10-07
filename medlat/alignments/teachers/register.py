"""Named teacher presets registered in the global model registry.

Usage::

    from medlat import get_model, available_models

    available_models("teacher.")
    # ('teacher.biomedclip', 'teacher.clip_b16', 'teacher.dinov2_b', ...)

    t = get_model("teacher.dinov2_b")
    t = get_model("teacher.rad_dino", img_size=256)
"""

from medlat.registry import register_model

from .timm_teacher import TimmTeacher
from .openclip_teacher import OpenCLIPTeacher
from .hf_teacher import HuggingFaceTeacher
from .sam_teacher import SAMTeacher, MedSAMTeacher

# ── natural, spatial ─────────────────────────────────────────────────


@register_model("teacher.dinov2_b",
                 description="DINOv2 ViT-B/14 (natural, self-supervised)")
def _dinov2_b(img_size=224, patch_size=14, **kw):
    return TimmTeacher("vit_base_patch14_dinov2.lvd142m",
                       img_size=img_size, patch_size=patch_size, **kw)


@register_model("teacher.dinov2_b_reg",
                 description="DINOv2 ViT-B/14 with register tokens")
def _dinov2_b_reg(img_size=224, patch_size=14, **kw):
    return TimmTeacher("vit_base_patch14_reg4_dinov2.lvd142m",
                       img_size=img_size, patch_size=patch_size, **kw)


@register_model("teacher.mae_b",
                 description="MAE ViT-B/16 (natural, self-supervised)")
def _mae_b(img_size=224, patch_size=16, **kw):
    return TimmTeacher("vit_base_patch16_224.mae",
                       img_size=img_size, patch_size=patch_size, **kw)


@register_model("teacher.sam_b",
                 description="SAM ViT-B/16 (segment-anything, 256-d after neck)")
def _sam_b(img_size=1024, **kw):
    return SAMTeacher("samvit_base_patch16.sa1b", img_size=img_size, **kw)


# ── natural, semantic ────────────────────────────────────────────────


@register_model("teacher.clip_b16",
                 description="CLIP ViT-B/16 (OpenAI, contrastive)")
def _clip_b16(img_size=224, patch_size=16, **kw):
    return TimmTeacher("vit_base_patch16_clip_224.openai",
                       img_size=img_size, patch_size=patch_size, **kw)


@register_model("teacher.siglip_b16",
                 description="SigLIP ViT-B/16 (WebLI, sigmoid contrastive)")
def _siglip_b16(img_size=224, patch_size=16, **kw):
    return TimmTeacher("vit_base_patch16_siglip_224.webli",
                       img_size=img_size, patch_size=patch_size, **kw)


# ── medical ──────────────────────────────────────────────────────────


@register_model("teacher.rad_dino",
                 description="RAD-DINO ViT-B/14 (radiology, self-supervised)")
def _rad_dino(img_size=224, **kw):
    return HuggingFaceTeacher("microsoft/rad-dino", img_size=img_size, **kw)


@register_model("teacher.biomedclip",
                 description="BiomedCLIP ViT-B/16 (PubMed, contrastive)")
def _biomedclip(img_size=224, **kw):
    return OpenCLIPTeacher(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        img_size=img_size, **kw,
    )


@register_model("teacher.medsiglip",
                 description="MedSigLIP ViT-SO (medical, 448px, gated)")
def _medsiglip(img_size=448, **kw):
    return HuggingFaceTeacher("google/medsiglip-448", img_size=img_size, **kw)


@register_model("teacher.medsam",
                 description="MedSAM ViT-B/16 (medical SAM, 256-d after neck)")
def _medsam(img_size=1024, **kw):
    return MedSAMTeacher("wanglab/medsam-vit-base", img_size=img_size, **kw)
