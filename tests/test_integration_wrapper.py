"""
Integration tests for GenWrapper encode/decode round-trips for all 4 routing
combinations: continuous+non-ar, continuous+ar, discrete+non-ar, discrete+ar.
Also tests scale_factor behaviour.
"""

import pytest
import torch


IMG_SIZE = 32
BATCH = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def continuous_tok():
    from medlat import get_model
    return get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)


@pytest.fixture()
def discrete_tok():
    from medlat import get_model
    return get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)


@pytest.fixture()
def diffusion_gen():
    from medlat import get_model
    return get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=3, num_classes=10)


@pytest.fixture()
def ar_continuous_gen():
    from medlat import get_model
    return get_model(
        "mar.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        patch_size=1,
        in_channels=3,
        buffer_size=IMG_SIZE // 4 * IMG_SIZE // 4,
    )


@pytest.fixture()
def ar_discrete_gen():
    from medlat import get_model
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
    wrapper.eval()
    with torch.no_grad():
        latents = wrapper.vae_encode(images)
        reconstructed = wrapper.vae_decode(latents)
    assert reconstructed.shape == images.shape, (
        f"Round-trip shape mismatch: expected {images.shape}, got {reconstructed.shape}"
    )
    return latents, reconstructed


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

def test_continuous_non_ar_roundtrip(continuous_tok, diffusion_gen):
    from medlat import GenWrapper
    w = GenWrapper(diffusion_gen, continuous_tok)
    _check_roundtrip(w, _images())


def test_continuous_ar_roundtrip(continuous_tok, ar_continuous_gen):
    from medlat import GenWrapper
    w = GenWrapper(ar_continuous_gen, continuous_tok)
    _check_roundtrip(w, _images())


def test_discrete_non_ar_roundtrip(discrete_tok, diffusion_gen):
    from medlat import GenWrapper
    w = GenWrapper(diffusion_gen, discrete_tok)
    _check_roundtrip(w, _images())


def test_discrete_ar_roundtrip(discrete_tok, ar_discrete_gen):
    from medlat import GenWrapper
    w = GenWrapper(ar_discrete_gen, discrete_tok)
    _check_roundtrip(w, _images())


# ---------------------------------------------------------------------------
# Scale factor tests
# ---------------------------------------------------------------------------

def test_scale_factor_auto_estimated(continuous_tok, diffusion_gen):
    """scale_factor is automatically estimated when not provided."""
    from medlat import GenWrapper
    w = GenWrapper(diffusion_gen, continuous_tok, scale_steps=3)
    w.train()

    x = _images()
    with torch.no_grad():
        for _ in range(3):
            w.vae_encode(x)

    # After training steps, step counter should have advanced
    assert w._scale_step_counter > 0


def test_manual_scale_factor_respected(continuous_tok, diffusion_gen):
    """Manually provided scale_factor is used for encoding/decoding."""
    from medlat import GenWrapper
    manual_sf = 2.0
    w = GenWrapper(diffusion_gen, continuous_tok, scale_factor=manual_sf)
    w.eval()

    x = _images()
    with torch.no_grad():
        z = w.vae_encode(x)
    # Encoded with scale_factor=2.0
    assert abs(w.scale_factor.item() - manual_sf) < 1e-6


def test_scale_factor_not_auto_when_provided(continuous_tok, diffusion_gen):
    """When scale_factor is manually provided, _auto_scale_factor is False."""
    from medlat import GenWrapper
    w = GenWrapper(diffusion_gen, continuous_tok, scale_factor=0.18215)
    assert not w._auto_scale_factor


def test_scale_factor_auto_when_not_provided(continuous_tok, diffusion_gen):
    """When scale_factor is not provided, _auto_scale_factor is True."""
    from medlat import GenWrapper
    w = GenWrapper(diffusion_gen, continuous_tok)
    assert w._auto_scale_factor
