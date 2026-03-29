"""Extended validate_compatibility tests."""

import pytest
import torch


IMG_SIZE = 32


# ---------------------------------------------------------------------------
# Positive cases — all 4 combinations
# ---------------------------------------------------------------------------

def test_continuous_non_ar_compatible():
    from medlat import get_model, validate_compatibility
    tok = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    gen = get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=3, num_classes=10)
    validate_compatibility(tok, gen)  # must not raise


def test_continuous_ar_compatible():
    from medlat import get_model, validate_compatibility
    tok = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    gen = get_model(
        "mar.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        patch_size=1,
        in_channels=3,
        buffer_size=IMG_SIZE // 4 * IMG_SIZE // 4,
    )
    validate_compatibility(tok, gen)  # must not raise


def test_discrete_non_ar_compatible():
    from medlat import get_model, validate_compatibility
    tok = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)
    gen = get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=3, num_classes=10)
    validate_compatibility(tok, gen)  # must not raise


def test_discrete_ar_compatible():
    from medlat import get_model, validate_compatibility
    tok = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)
    gen = get_model(
        "maskgit.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        num_tokens=8192,
        num_classes=10,
    )
    validate_compatibility(tok, gen)  # must not raise


# ---------------------------------------------------------------------------
# Negative cases — mismatch errors
# ---------------------------------------------------------------------------

def test_embed_dim_mismatch_raises():
    """Mismatched embed_dim / in_channels must raise ValueError with 'embed_dim' in message."""
    from medlat import get_model, validate_compatibility
    tok = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)   # embed_dim=3
    gen = get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=16, num_classes=10)
    with pytest.raises(ValueError, match="embed_dim"):
        validate_compatibility(tok, gen)


def test_codebook_size_mismatch_raises():
    """Mismatched codebook size must raise ValueError with 'codebook_size' in message."""
    from medlat import get_model, validate_compatibility
    tok = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)  # n_embed=8192
    gen = get_model(
        "maskgit.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        num_tokens=16384,  # intentionally wrong
        num_classes=10,
    )
    with pytest.raises(ValueError, match="codebook_size"):
        validate_compatibility(tok, gen)


def test_vae_stride_mismatch_raises():
    """Mismatched vae_stride must raise ValueError with 'vae_stride' in message."""
    from medlat import get_model, validate_compatibility
    tok = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)  # vae_stride=4
    gen = get_model(
        "mar.b",
        img_size=IMG_SIZE,
        vae_stride=8,  # intentionally wrong
        patch_size=1,
        in_channels=3,
        buffer_size=16,
    )
    with pytest.raises(ValueError, match="vae_stride"):
        validate_compatibility(tok, gen)
