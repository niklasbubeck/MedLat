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
