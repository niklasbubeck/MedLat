"""Tests for AlignmentModule subclasses.

Optional-dependency tests are guarded with pytest.mark.skipif so the suite
passes cleanly regardless of whether timm / open_clip are installed.
"""

import pytest
import torch
import torch.nn as nn

try:
    import timm  # noqa: F401
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

requires_timm = pytest.mark.skipif(not TIMM_AVAILABLE, reason="timm not installed")


# ---------------------------------------------------------------------------
# Minimal stub decoder that satisfies the AlignmentModule decoder contract
# ---------------------------------------------------------------------------

class _StubDecoder(nn.Module):
    """Minimal decoder that matches the expected signature for HOGAlignment."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, interpolate_zq=None, H=None, W=None, D=None):
        return self.proj(x)


# ---------------------------------------------------------------------------
# HOGAlignment (no external deps)
# ---------------------------------------------------------------------------

def test_hog_alignment_constructs():
    from medlat.alignments import HOGAlignment
    decoder = _StubDecoder(embed_dim=64)
    align = HOGAlignment(decoder=decoder, codebook_embed_dim=32)
    assert align is not None


def test_hog_alignment_ensure_projection_dim_preserves_requires_grad():
    """Rebuilding to_pixel must keep the same requires_grad state."""
    from medlat.alignments import HOGAlignment
    decoder = _StubDecoder(embed_dim=64)
    align = HOGAlignment(decoder=decoder, codebook_embed_dim=32)

    # Freeze the projection head
    align.to_pixel.requires_grad_(False)
    assert not align.to_pixel.weight.requires_grad

    # Trigger a rebuild to a different output dim
    align.ensure_projection_dim(64)  # differs from default 108

    # requires_grad must still be False after rebuild
    assert not align.to_pixel.weight.requires_grad, (
        "ensure_projection_dim lost requires_grad=False after rebuilding to_pixel"
    )


def test_hog_alignment_ensure_projection_dim_trainable_preserved():
    """Rebuilt projection head stays trainable when it was trainable before."""
    from medlat.alignments import HOGAlignment
    decoder = _StubDecoder(embed_dim=64)
    align = HOGAlignment(decoder=decoder, codebook_embed_dim=32)

    align.to_pixel.requires_grad_(True)
    align.ensure_projection_dim(64)

    assert align.to_pixel.weight.requires_grad, (
        "ensure_projection_dim lost requires_grad=True after rebuilding to_pixel"
    )


# ---------------------------------------------------------------------------
# AlignmentModule._infer_grid_hw — square-grid contract
# ---------------------------------------------------------------------------

def test_infer_grid_hw_perfect_square():
    from medlat.alignments import HOGAlignment
    decoder = _StubDecoder()
    align = HOGAlignment(decoder=decoder, codebook_embed_dim=32)
    assert align._infer_grid_hw(256) == (16, 16)
    assert align._infer_grid_hw(64) == (8, 8)


def test_infer_grid_hw_non_square_raises():
    from medlat.alignments import HOGAlignment
    decoder = _StubDecoder()
    align = HOGAlignment(decoder=decoder, codebook_embed_dim=32)
    with pytest.raises(ValueError, match="square grid"):
        align._infer_grid_hw(192)  # 192 is not a perfect square


# ---------------------------------------------------------------------------
# DinoAlignment with composable VF losses (requires timm)
# ---------------------------------------------------------------------------

@requires_timm
def test_dino_alignment_with_vf_losses_constructs():
    from medlat.alignments import DinoAlignment, DistmatMarginLoss, CosineMarginLoss
    decoder = _StubDecoder(embed_dim=64)
    align = DinoAlignment(
        decoder=decoder, codebook_embed_dim=32,
        losses=[
            (DistmatMarginLoss(margin=0.25), 1.0),
            (CosineMarginLoss(margin=0.5), 1.0),
        ],
    )
    assert align is not None
    assert len(align.loss_modules) == 2


@requires_timm
def test_dino_alignment_with_vf_losses_forward_smoke():
    """Smoke test: forward pass with VF losses returns a scalar loss."""
    from medlat.alignments import DinoAlignment, DistmatMarginLoss, CosineMarginLoss
    decoder = _StubDecoder(embed_dim=64)
    align = DinoAlignment(
        decoder=decoder, codebook_embed_dim=32, img_size=64,
        losses=[
            (DistmatMarginLoss(margin=0.25), 1.0),
            (CosineMarginLoss(margin=0.5), 1.0),
        ],
    )
    align.eval()

    x_img = torch.randn(1, 3, 64, 64)
    x_latent = torch.randn(1, 64, 32)  # (B, L, codebook_embed_dim)
    with torch.no_grad():
        loss, _ = align(x_latent, input_image=x_img)
    assert loss.ndim == 0, "VF loss should be a scalar"
    assert loss.item() >= 0.0, "VF loss should be non-negative"


# ---------------------------------------------------------------------------
# DinoAlignment (requires timm)
# ---------------------------------------------------------------------------

@requires_timm
def test_dino_alignment_constructs():
    from medlat.alignments import DinoAlignment
    decoder = _StubDecoder(embed_dim=64)
    align = DinoAlignment(
        decoder=decoder,
        codebook_embed_dim=32,
        img_size=224,
    )
    assert align is not None


# ---------------------------------------------------------------------------
# SRAAlignment
# ---------------------------------------------------------------------------

def test_sra_alignment_constructs():
    from medlat.alignments import SRAAlignment
    align = SRAAlignment(hidden_dim=768, teacher_layer=8)
    assert align is not None
    assert align._ema_model is None
    assert align.teacher_layer == 8
    assert align.delta_max == 0.2


def test_sra_alignment_asserts_no_ema():
    """Forward without set_ema_model must raise AssertionError."""
    from medlat.alignments import SRAAlignment
    align = SRAAlignment(hidden_dim=64, teacher_layer=4)
    quant = torch.randn(1, 16, 64)
    img = torch.randn(1, 3, 32, 32)
    with pytest.raises(AssertionError, match="EMA model"):
        align(quant, input_image=img,
              noisy_latent=torch.randn(1, 4, 8, 8),
              timestep=torch.tensor([0.5]),
              class_labels=torch.tensor([0]))


def test_sra_alignment_ema_not_in_state_dict():
    """EMA model must NOT appear in SRA's state_dict."""
    from medlat.alignments import SRAAlignment
    align = SRAAlignment(hidden_dim=64, teacher_layer=4)
    ema = nn.Linear(64, 64)
    align.set_ema_model(ema)
    for key in align.state_dict():
        assert "ema" not in key.lower(), f"EMA leaked into state_dict: {key}"


# ---------------------------------------------------------------------------
# DispersiveLoss
# ---------------------------------------------------------------------------

def test_dispersive_loss_basic():
    from medlat.alignments import DispersiveLoss
    disp = DispersiveLoss(tau=0.5)
    z = torch.randn(4, 16, 64)
    loss = disp(z)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dispersive_loss_collapsed_representations():
    """Identical representations should produce a lower (more negative) loss
    than diverse ones — the loss penalises collapse."""
    from medlat.alignments import DispersiveLoss
    disp = DispersiveLoss(tau=0.5)
    diverse = torch.randn(8, 16, 64)
    collapsed = diverse[0:1].expand(8, -1, -1).clone()
    collapsed += torch.randn_like(collapsed) * 1e-4
    loss_diverse = disp(diverse)
    loss_collapsed = disp(collapsed)
    assert loss_collapsed > loss_diverse, (
        "Collapsed representations should have higher dispersive loss "
        "(near 0) — minimising the loss pushes toward diversity"
    )


@requires_timm
def test_dispersive_loss_dit_forward():
    """Dispersive loss fires during training and is absent during eval."""
    from medlat.generators.non_autoregressive.dit.models import DiT
    from medlat.alignments import DispersiveLoss

    model = DiT(
        img_size=64, vae_stride=16, patch_size=2,
        in_channels=4, hidden_size=384, depth=12,
        num_heads=6, num_classes=10,
        dispersive_loss=DispersiveLoss(), dispersive_layer=3,
    )
    x = torch.randn(4, 4, 4, 4)
    t = torch.rand(4)
    y = torch.randint(0, 10, (4,))

    model.train()
    _ = model(x, t, y)
    assert "dispersive_loss" in model._auxiliary_losses

    model.eval()
    with torch.no_grad():
        _ = model(x, t, y)
    assert "dispersive_loss" not in model._auxiliary_losses


@requires_timm
def test_sra_alignment_forward_smoke():
    """Full forward pass with a real DiT as EMA model."""
    from medlat.generators.non_autoregressive.dit.models import DiT
    from medlat.alignments import SRAAlignment

    sra = SRAAlignment(hidden_dim=384, teacher_layer=8)
    model = DiT(
        img_size=64, vae_stride=16, patch_size=2,
        in_channels=4, hidden_size=384, depth=12,
        num_heads=6, num_classes=10,
        generator_alignment=sra, alignment_layer=4,
    )
    import copy
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()
    sra.set_ema_model(ema)

    x = torch.randn(2, 4, 4, 4)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    img = torch.randn(2, 3, 64, 64)

    model.eval()
    with torch.no_grad():
        out = model(x, t, y, input_image=img)

    assert out.shape == (2, 8, 4, 4)
    assert "alignment_loss" in model._auxiliary_losses
    loss = model._auxiliary_losses["alignment_loss"]
    assert loss.ndim == 0
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# HASTEAlignment + HASTEDiT
# ---------------------------------------------------------------------------

@requires_timm
def test_haste_dit_block_returns_attn():
    """HASTEDiTBlock returns attention logits when asked."""
    from medlat.generators.non_autoregressive.dit.haste import HASTEDiTBlock
    block = HASTEDiTBlock(hidden_size=384, num_heads=6, cond_dim=768)
    x = torch.randn(2, 16, 384)
    c = torch.randn(2, 768)

    out_normal = block(x, c)
    assert isinstance(out_normal, torch.Tensor)

    out_attn, logits = block(x, c, return_attn=True)
    assert out_attn.shape == x.shape
    assert logits.shape == (2, 6, 16, 16)


@requires_timm
def test_haste_dit_forward_smoke():
    """HASTEDiT with HASTEAlignment produces alignment loss."""
    from medlat.generators.non_autoregressive.dit.haste import HASTEDiT
    from medlat.alignments import HASTEAlignment

    haste = HASTEAlignment(
        hidden_dim=384, img_size=56,
        teacher_model_name="vit_small_patch14_dinov2.lvd142m",
        teacher_patch_size=14,
        num_attn_distill=2, teacher_attn_start=8,
    )
    model = HASTEDiT(
        img_size=64, vae_stride=16, patch_size=2,
        in_channels=4, hidden_size=384, depth=12,
        num_heads=6, num_classes=10,
        generator_alignment=haste, alignment_layer=4,
        attn_distill_layers=[2, 3],
    )
    x = torch.randn(2, 4, 4, 4)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    img = torch.randn(2, 3, 64, 64)

    model.eval()
    with torch.no_grad():
        out = model(x, t, y, input_image=img)

    assert out.shape == (2, 8, 4, 4)
    assert "alignment_loss" in model._auxiliary_losses
    loss = model._auxiliary_losses["alignment_loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Teacher Registry (Phase 5)
# ---------------------------------------------------------------------------

def test_teacher_model_abc():
    """TeacherModel cannot be instantiated directly."""
    from medlat.alignments.teachers import TeacherModel
    with pytest.raises(TypeError):
        TeacherModel()


@requires_timm
def test_timm_teacher_constructs():
    from medlat.alignments.teachers import TimmTeacher
    teacher = TimmTeacher("vit_small_patch14_dinov2.lvd142m", img_size=56)
    assert teacher.embed_dim > 0
    assert teacher.supports_attention
    assert teacher.num_blocks > 0
    assert teacher.input_transform is not None


@requires_timm
def test_timm_teacher_extract_features():
    from medlat.alignments.teachers import TimmTeacher
    teacher = TimmTeacher("vit_small_patch14_dinov2.lvd142m", img_size=56)
    x = torch.randn(2, 3, 56, 56)
    with torch.no_grad():
        features = teacher.extract_features(x)
    assert features.ndim == 3
    assert features.shape[0] == 2
    assert features.shape[2] == teacher.embed_dim


@requires_timm
def test_timm_teacher_always_eval():
    """Teacher must stay in eval mode even after .train()."""
    from medlat.alignments.teachers import TimmTeacher
    teacher = TimmTeacher("vit_small_patch14_dinov2.lvd142m", img_size=56)
    teacher.train(True)
    assert not teacher.training


@requires_timm
def test_timm_teacher_attn_hook():
    from medlat.alignments.teachers import TimmTeacher
    teacher = TimmTeacher("vit_small_patch14_dinov2.lvd142m", img_size=56)
    hook = teacher.make_attn_hook(0)
    module = teacher.get_qkv_module(0)
    handle = module.register_forward_hook(hook)

    x = torch.randn(2, 3, 56, 56)
    with torch.no_grad():
        teacher.extract_features(x)

    assert hook.logits is not None
    assert hook.logits.ndim == 4
    hook.clear()
    assert hook.logits is None
    handle.remove()


@requires_timm
def test_create_teacher_factory():
    from medlat.alignments import create_teacher
    teacher = create_teacher("timm", "vit_small_patch14_dinov2.lvd142m", img_size=56)
    assert teacher.embed_dim > 0

    with pytest.raises(ValueError, match="Unknown teacher source"):
        create_teacher("nonexistent", "model")


def test_custom_teacher():
    from medlat.alignments.teachers import CustomTeacher

    backbone = nn.Linear(64, 128)

    def extract(model, x):
        B = x.shape[0]
        flat = x.reshape(B, -1)[:, :64]
        return model(flat).unsqueeze(1)  # (B, 1, 128)

    teacher = CustomTeacher(backbone, embed_dim=128, extract_fn=extract)
    assert teacher.embed_dim == 128
    assert not teacher.supports_attention
    assert teacher.input_transform is None

    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        features = teacher.extract_features(x)
    assert features.shape == (2, 1, 128)
    assert not backbone.weight.requires_grad


@requires_timm
def test_repa_with_explicit_teacher():
    """REPAAlignment accepts a pre-built teacher."""
    from medlat.alignments import REPAAlignment
    from medlat.alignments.teachers import TimmTeacher

    teacher = TimmTeacher("vit_small_patch14_dinov2.lvd142m", img_size=56)
    align = REPAAlignment(hidden_dim=384, teacher=teacher)
    assert align.teacher is teacher

    x_latent = torch.randn(2, 16, 384)
    x_img = torch.randn(2, 3, 56, 56)
    align.eval()
    with torch.no_grad():
        loss, pred = align(x_latent, input_image=x_img)
    assert loss.ndim == 0
    assert pred.shape[0] == 2


@requires_timm
def test_dino_with_explicit_teacher():
    """DinoAlignment accepts a pre-built teacher."""
    from medlat.alignments import DinoAlignment
    from medlat.alignments.teachers import TimmTeacher

    teacher = TimmTeacher("vit_small_patch14_dinov2.lvd142m", img_size=56)
    decoder = _StubDecoder(embed_dim=64)
    align = DinoAlignment(
        decoder=decoder, codebook_embed_dim=32,
        teacher=teacher, img_size=56,
    )
    assert align.teacher is teacher


@requires_timm
def test_haste_with_explicit_teacher():
    """HASTEAlignment accepts a pre-built teacher for attention distillation."""
    from medlat.alignments import HASTEAlignment
    from medlat.alignments.teachers import TimmTeacher

    teacher = TimmTeacher("vit_small_patch14_dinov2.lvd142m", img_size=56)
    align = HASTEAlignment(
        hidden_dim=384, teacher=teacher,
        num_attn_distill=2, teacher_attn_start=8,
    )
    assert align.teacher is teacher
    assert len(align._attn_hooks) == 2
