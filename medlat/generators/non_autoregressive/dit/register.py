import torch
import torch.nn as nn
from medlat.registry import register_model
from .models import DiT

__all__ = []

# ---------------------------------------------------------------------------
# REPA helpers
# ---------------------------------------------------------------------------

REPA_PAPER = "https://arxiv.org/abs/2410.06940"

_REPA_TEACHERS = {
    "dinov2s": ("vit_small_patch14_dinov2.lvd142m", 14),
    "dinov2b": ("vit_base_patch14_dinov2.lvd142m", 14),
    "dinov2l": ("vit_large_patch14_dinov2.lvd142m", 14),
}

def _make_repa(hidden_dim, teacher="dinov2b", img_size=224, **repa_kwargs):
    from medlat.alignments import REPAAlignment
    from medlat.alignments.teachers import TimmTeacher
    model_name, patch_size = _REPA_TEACHERS[teacher]
    t = TimmTeacher(model_name, img_size=img_size, patch_size=patch_size)
    return REPAAlignment(hidden_dim=hidden_dim, teacher=t, **repa_kwargs)

@register_model("dit.xl_1", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_XL_1(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

@register_model("dit.xl_2", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

@register_model("dit.xl_4", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

@register_model("dit.xl_8", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

@register_model("dit.l_1", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_L_1(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

@register_model("dit.l_2", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

@register_model("dit.l_4", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

@register_model("dit.l_8", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

@register_model("dit.b_1", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_B_1(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

@register_model("dit.b_2", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

@register_model("dit.b_4", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

@register_model("dit.b_8", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

@register_model("dit.s_1", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_S_1(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

@register_model("dit.s_2", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

@register_model("dit.s_4", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

@register_model("dit.s_8", paper_url="https://arxiv.org/abs/2212.09748")
def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# ---------------------------------------------------------------------------
# REPA — DiT-XL (depth=28, hidden=1152, alignment_layer=8)
# ---------------------------------------------------------------------------

@register_model("dit.repa_xl_2", paper_url=REPA_PAPER,
                 description="DiT-XL/2 + REPA with DINOv2-B teacher (default)")
def DiT_REPA_XL_2(**kwargs):
    repa = _make_repa(1152, "dinov2b")
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
               generator_alignment=repa, alignment_layer=8, **kwargs)

@register_model("dit.repa_xl_2_dinov2l", paper_url=REPA_PAPER,
                 description="DiT-XL/2 + REPA with DINOv2-L teacher")
def DiT_REPA_XL_2_DINOv2L(**kwargs):
    repa = _make_repa(1152, "dinov2l")
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
               generator_alignment=repa, alignment_layer=8, **kwargs)

@register_model("dit.repa_xl_2_dinov2s", paper_url=REPA_PAPER,
                 description="DiT-XL/2 + REPA with DINOv2-S teacher")
def DiT_REPA_XL_2_DINOv2S(**kwargs):
    repa = _make_repa(1152, "dinov2s")
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
               generator_alignment=repa, alignment_layer=8, **kwargs)

@register_model("dit.repa_xl_1", paper_url=REPA_PAPER,
                 description="DiT-XL/1 + REPA with DINOv2-B teacher")
def DiT_REPA_XL_1(**kwargs):
    repa = _make_repa(1152, "dinov2b")
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16,
               generator_alignment=repa, alignment_layer=8, **kwargs)


# ---------------------------------------------------------------------------
# REPA — DiT-L (depth=24, hidden=1024, alignment_layer=7)
# ---------------------------------------------------------------------------

@register_model("dit.repa_l_2", paper_url=REPA_PAPER,
                 description="DiT-L/2 + REPA with DINOv2-B teacher")
def DiT_REPA_L_2(**kwargs):
    repa = _make_repa(1024, "dinov2b")
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16,
               generator_alignment=repa, alignment_layer=7, **kwargs)

@register_model("dit.repa_l_2_dinov2l", paper_url=REPA_PAPER,
                 description="DiT-L/2 + REPA with DINOv2-L teacher")
def DiT_REPA_L_2_DINOv2L(**kwargs):
    repa = _make_repa(1024, "dinov2l")
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16,
               generator_alignment=repa, alignment_layer=7, **kwargs)


# ---------------------------------------------------------------------------
# REPA — DiT-B (depth=12, hidden=768, alignment_layer=4)
# ---------------------------------------------------------------------------

@register_model("dit.repa_b_2", paper_url=REPA_PAPER,
                 description="DiT-B/2 + REPA with DINOv2-B teacher")
def DiT_REPA_B_2(**kwargs):
    repa = _make_repa(768, "dinov2b")
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
               generator_alignment=repa, alignment_layer=4, **kwargs)

@register_model("dit.repa_b_2_dinov2s", paper_url=REPA_PAPER,
                 description="DiT-B/2 + REPA with DINOv2-S teacher")
def DiT_REPA_B_2_DINOv2S(**kwargs):
    repa = _make_repa(768, "dinov2s")
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
               generator_alignment=repa, alignment_layer=4, **kwargs)


# ---------------------------------------------------------------------------
# REPA — DiT-S (depth=12, hidden=384, alignment_layer=4)
# ---------------------------------------------------------------------------

@register_model("dit.repa_s_2", paper_url=REPA_PAPER,
                 description="DiT-S/2 + REPA with DINOv2-S teacher")
def DiT_REPA_S_2(**kwargs):
    repa = _make_repa(384, "dinov2s")
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6,
               generator_alignment=repa, alignment_layer=4, **kwargs)


# ---------------------------------------------------------------------------
# SRA helpers
# ---------------------------------------------------------------------------

SRA_PAPER = "https://github.com/vvvvvjdy/SRA"


def _make_sra(hidden_dim, teacher_layer, **sra_kwargs):
    from medlat.alignments import SRAAlignment
    return SRAAlignment(
        hidden_dim=hidden_dim,
        teacher_layer=teacher_layer,
        **sra_kwargs,
    )


# ---------------------------------------------------------------------------
# SRA — DiT-XL (depth=28, hidden=1152, student=8, teacher=16)
# ---------------------------------------------------------------------------

@register_model("dit.sra_xl_2", paper_url=SRA_PAPER,
                 description="DiT-XL/2 + SRA (self-distillation, student=8, teacher=16)")
def DiT_SRA_XL_2(**kwargs):
    sra = _make_sra(1152, teacher_layer=16)
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
               generator_alignment=sra, alignment_layer=8, **kwargs)


# ---------------------------------------------------------------------------
# SRA — DiT-L (depth=24, hidden=1024, student=7, teacher=14)
# ---------------------------------------------------------------------------

@register_model("dit.sra_l_2", paper_url=SRA_PAPER,
                 description="DiT-L/2 + SRA (self-distillation, student=7, teacher=14)")
def DiT_SRA_L_2(**kwargs):
    sra = _make_sra(1024, teacher_layer=14)
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16,
               generator_alignment=sra, alignment_layer=7, **kwargs)


# ---------------------------------------------------------------------------
# SRA — DiT-B (depth=12, hidden=768, student=4, teacher=8)
# ---------------------------------------------------------------------------

@register_model("dit.sra_b_2", paper_url=SRA_PAPER,
                 description="DiT-B/2 + SRA (self-distillation, student=4, teacher=8)")
def DiT_SRA_B_2(**kwargs):
    sra = _make_sra(768, teacher_layer=8)
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
               generator_alignment=sra, alignment_layer=4, **kwargs)


# ---------------------------------------------------------------------------
# SRA — DiT-S (depth=12, hidden=384, student=4, teacher=8)
# ---------------------------------------------------------------------------

@register_model("dit.sra_s_2", paper_url=SRA_PAPER,
                 description="DiT-S/2 + SRA (self-distillation, student=4, teacher=8)")
def DiT_SRA_S_2(**kwargs):
    sra = _make_sra(384, teacher_layer=8)
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6,
               generator_alignment=sra, alignment_layer=4, **kwargs)


# ---------------------------------------------------------------------------
# Dispersive Loss helpers
# ---------------------------------------------------------------------------

DISP_PAPER = "https://arxiv.org/abs/2506.09027"


def _make_disp(**disp_kwargs):
    from medlat.alignments import DispersiveLoss
    return DispersiveLoss(**disp_kwargs)


# ---------------------------------------------------------------------------
# Dispersive — standalone (first-quarter layer)
# ---------------------------------------------------------------------------

@register_model("dit.disp_xl_2", paper_url=DISP_PAPER,
                 description="DiT-XL/2 + Dispersive Loss (layer 7)")
def DiT_DISP_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
               dispersive_loss=_make_disp(), dispersive_layer=7, **kwargs)

@register_model("dit.disp_b_2", paper_url=DISP_PAPER,
                 description="DiT-B/2 + Dispersive Loss (layer 3)")
def DiT_DISP_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
               dispersive_loss=_make_disp(), dispersive_layer=3, **kwargs)


# ---------------------------------------------------------------------------
# REPA + Dispersive (combined)
# ---------------------------------------------------------------------------

@register_model("dit.repa_disp_xl_2", paper_url=REPA_PAPER,
                 description="DiT-XL/2 + REPA (DINOv2-B) + Dispersive Loss")
def DiT_REPA_DISP_XL_2(**kwargs):
    repa = _make_repa(1152, "dinov2b")
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
               generator_alignment=repa, alignment_layer=8,
               dispersive_loss=_make_disp(), dispersive_layer=7, **kwargs)

@register_model("dit.repa_disp_b_2", paper_url=REPA_PAPER,
                 description="DiT-B/2 + REPA (DINOv2-B) + Dispersive Loss")
def DiT_REPA_DISP_B_2(**kwargs):
    repa = _make_repa(768, "dinov2b")
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
               generator_alignment=repa, alignment_layer=4,
               dispersive_loss=_make_disp(), dispersive_layer=3, **kwargs)


# ---------------------------------------------------------------------------
# HASTE helpers
# ---------------------------------------------------------------------------

HASTE_PAPER = "https://arxiv.org/abs/2505.16792"


def _resolve_teacher(teacher, img_size=224):
    """Turn ``teacher`` into a TeacherModel.

    Accepts a TeacherModel instance (returned as-is), a legacy short key from
    ``_REPA_TEACHERS`` ("dinov2b", ...), or a global-registry teacher name
    ("mae_b", "rad_dino", ... — anything under ``teacher.*``). Keeps the
    legacy keys working so existing configs and registered variants are
    untouched.
    """
    from medlat.alignments.teachers import TeacherModel, TimmTeacher

    if isinstance(teacher, TeacherModel):
        return teacher
    if teacher in _REPA_TEACHERS:
        model_name, patch_size = _REPA_TEACHERS[teacher]
        return TimmTeacher(model_name, img_size=img_size, patch_size=patch_size)
    # late import: MODEL_REGISTRY is being populated while this module loads
    from medlat.registry import MODEL_REGISTRY
    name = teacher if str(teacher).startswith("teacher.") else f"teacher.{teacher}"
    return MODEL_REGISTRY.create(name)


def _make_haste(hidden_dim, num_attn_distill=4, teacher_attn_start=8,
                teacher="dinov2b", img_size=224, **haste_kwargs):
    from medlat.alignments import HASTEAlignment
    return HASTEAlignment(
        hidden_dim=hidden_dim, teacher=_resolve_teacher(teacher, img_size),
        num_attn_distill=num_attn_distill,
        teacher_attn_start=teacher_attn_start,
        **haste_kwargs,
    )


# ---------------------------------------------------------------------------
# HASTE — DiT-XL (depth=28, alignment=8, attn_distill=[4,5,6,7])
# ---------------------------------------------------------------------------

@register_model("dit.haste_xl_2", paper_url=HASTE_PAPER,
                 description="DiT-XL/2 + HASTE (DINOv2-B, attn distill layers 4-7)")
def DiT_HASTE_XL_2(teacher=None, **kwargs):
    from .haste import HASTEDiT
    haste = _make_haste(1152, num_attn_distill=4, teacher_attn_start=8,
                        teacher=teacher if teacher is not None else "dinov2b")
    return HASTEDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                    generator_alignment=haste, alignment_layer=8,
                    attn_distill_layers=[4, 5, 6, 7], **kwargs)


# ---------------------------------------------------------------------------
# HASTE — DiT-B (depth=12, alignment=4, attn_distill=[1,2,3])
# ---------------------------------------------------------------------------

@register_model("dit.haste_b_2", paper_url=HASTE_PAPER,
                 description="DiT-B/2 + HASTE (DINOv2-B, attn distill layers 1-3)")
def DiT_HASTE_B_2(teacher=None, **kwargs):
    from .haste import HASTEDiT
    haste = _make_haste(768, num_attn_distill=3, teacher_attn_start=9,
                        teacher=teacher if teacher is not None else "dinov2b")
    return HASTEDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12,
                    generator_alignment=haste, alignment_layer=4,
                    attn_distill_layers=[1, 2, 3], **kwargs)