"""
Round-trip encode/decode tests for all four GenWrapper routing combinations,
plus validate_compatibility checks.

All tests run on CPU with small (32×32) synthetic images so they complete
quickly in CI without a GPU.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures — one tiny tokenizer and one tiny generator per family.
# Models are built fresh per test to avoid shared state.
# ---------------------------------------------------------------------------

IMG_SIZE = 32
BATCH = 2


@pytest.fixture()
def continuous_tok():
    from medlat import get_model
    # f4 compression → latent 8×8, embed_dim=3
    return get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)


@pytest.fixture()
def discrete_tok():
    from medlat import get_model
    # f4 compression → latent 8×8, n_embed=8192, embed_dim=3
    return get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)


@pytest.fixture()
def diffusion_gen():
    from medlat import get_model
    # DiT-S/2 with in_channels=3 (matches both tokenizers above)
    return get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=3, num_classes=10)


@pytest.fixture()
def ar_continuous_gen():
    from medlat import get_model
    # MAR-B: autoregressive, continuous latents (in_channels=3)
    return get_model(
        "mar.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        patch_size=1,
        in_channels=3,
        buffer_size=IMG_SIZE // 4 * IMG_SIZE // 4,  # seq_len = 8*8 = 64
    )


@pytest.fixture()
def ar_discrete_gen():
    from medlat import get_model
    # MaskGIT-B: autoregressive, discrete tokens (codebook_size=8192)
    return get_model(
        "maskgit.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        num_tokens=8192,
        num_classes=10,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _images():
    return torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)


def _check_roundtrip(wrapper, images):
    """Encode then decode; verify output shape matches input."""
    wrapper.eval()
    with torch.no_grad():
        latents = wrapper.vae_encode(images)
        reconstructed = wrapper.vae_decode(latents)
    assert reconstructed.shape == images.shape, (
        f"Round-trip shape mismatch: expected {images.shape}, got {reconstructed.shape}"
    )


# ---------------------------------------------------------------------------
# Round-trip tests — one per routing combination
# ---------------------------------------------------------------------------

def test_continuous_diffusion_roundtrip(continuous_tok, diffusion_gen):
    """continuous tokenizer + non-autoregressive (DiT) generator."""
    from medlat import GenWrapper
    w = GenWrapper(diffusion_gen, continuous_tok)
    _check_roundtrip(w, _images())


def test_continuous_autoregressive_roundtrip(continuous_tok, ar_continuous_gen):
    """continuous tokenizer + autoregressive (MAR) generator."""
    from medlat import GenWrapper
    w = GenWrapper(ar_continuous_gen, continuous_tok)
    _check_roundtrip(w, _images())


def test_discrete_diffusion_roundtrip(discrete_tok, diffusion_gen):
    """discrete tokenizer + non-autoregressive (DiT) generator."""
    from medlat import GenWrapper
    w = GenWrapper(diffusion_gen, discrete_tok)
    _check_roundtrip(w, _images())


def test_discrete_autoregressive_roundtrip(discrete_tok, ar_discrete_gen):
    """discrete tokenizer + autoregressive (MaskGIT) generator."""
    from medlat import GenWrapper
    w = GenWrapper(ar_discrete_gen, discrete_tok)
    _check_roundtrip(w, _images())


# ---------------------------------------------------------------------------
# validate_compatibility — positive cases (should not raise)
# ---------------------------------------------------------------------------

def test_validate_compatibility_continuous_diffusion(continuous_tok, diffusion_gen):
    from medlat import validate_compatibility
    validate_compatibility(continuous_tok, diffusion_gen)  # must not raise


def test_validate_compatibility_continuous_ar(continuous_tok, ar_continuous_gen):
    from medlat import validate_compatibility
    validate_compatibility(continuous_tok, ar_continuous_gen)


def test_validate_compatibility_discrete_diffusion(discrete_tok, diffusion_gen):
    from medlat import validate_compatibility
    validate_compatibility(discrete_tok, diffusion_gen)


def test_validate_compatibility_discrete_ar(discrete_tok, ar_discrete_gen):
    from medlat import validate_compatibility
    validate_compatibility(discrete_tok, ar_discrete_gen)


# ---------------------------------------------------------------------------
# validate_compatibility — negative cases (wrong channel/codebook dims)
# ---------------------------------------------------------------------------

def test_validate_compatibility_channel_mismatch():
    """Mismatched embed_dim / in_channels should raise ValueError."""
    from medlat import get_model, validate_compatibility
    tok = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)   # embed_dim=3
    gen = get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=16, num_classes=10)
    with pytest.raises(ValueError, match="embed_dim"):
        validate_compatibility(tok, gen)


def test_validate_compatibility_codebook_mismatch():
    """Mismatched codebook size should raise ValueError."""
    from medlat import get_model, validate_compatibility
    tok = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)   # n_embed=8192
    gen = get_model(
        "maskgit.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        num_tokens=16384,   # intentionally wrong
        num_classes=10,
    )
    with pytest.raises(ValueError, match="codebook_size"):
        validate_compatibility(tok, gen)


def test_validate_compatibility_stride_mismatch():
    """Mismatched vae_stride should raise ValueError when both models expose it."""
    from medlat import get_model, validate_compatibility
    tok = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)   # vae_stride=4
    gen = get_model(
        "mar.b",
        img_size=IMG_SIZE,
        vae_stride=8,   # intentionally wrong
        patch_size=1,
        in_channels=3,
        buffer_size=16,
    )
    with pytest.raises(ValueError, match="vae_stride"):
        validate_compatibility(tok, gen)
