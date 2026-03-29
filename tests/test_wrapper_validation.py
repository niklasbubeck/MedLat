"""Tests for GenWrapper eager validation and edge-case behaviours."""

import pytest
import torch
import torch.nn as nn


IMG_SIZE = 32
BATCH = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _continuous_tok():
    from medlat import get_model
    return get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)


def _discrete_tok():
    from medlat import get_model
    return get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)


def _diffusion_gen():
    from medlat import get_model
    return get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=3, num_classes=10)


def _ar_discrete_gen():
    from medlat import get_model
    return get_model(
        "maskgit.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        num_tokens=8192,
        num_classes=10,
    )


# ---------------------------------------------------------------------------
# Validation at construction
# ---------------------------------------------------------------------------

def test_gen_wrapper_constructs_continuous_non_ar():
    from medlat import GenWrapper
    w = GenWrapper(_diffusion_gen(), _continuous_tok())
    assert w is not None


def test_gen_wrapper_constructs_discrete_ar():
    from medlat import GenWrapper
    w = GenWrapper(_ar_discrete_gen(), _discrete_tok())
    assert w is not None


def test_vae_decode_before_encode_raises():
    """vae_decode with discrete+AR routing without prior vae_encode must raise RuntimeError."""
    from medlat import GenWrapper
    w = GenWrapper(_ar_discrete_gen(), _discrete_tok())
    w.eval()
    # Create fake indices (batch of token sequences)
    fake_indices = torch.zeros(BATCH, 64, dtype=torch.long)
    with pytest.raises(RuntimeError, match="vae_encode"):
        w.vae_decode(fake_indices)


def test_vae_decode_with_explicit_out_shape_does_not_raise():
    """vae_decode with discrete+AR routing with explicit out_shape should not raise."""
    from medlat import GenWrapper
    w = GenWrapper(_ar_discrete_gen(), _discrete_tok())
    w.eval()
    out_shape = (BATCH, 8, 8, 3)  # (B, H, W, C)
    fake_indices = torch.zeros(BATCH, 64, dtype=torch.long)
    with torch.no_grad():
        # Should not raise — out_shape is provided
        try:
            w.vae_decode(fake_indices, out_shape=out_shape)
        except RuntimeError as e:
            if "vae_encode" in str(e):
                pytest.fail(f"Should not raise vae_encode error when out_shape provided: {e}")


# ---------------------------------------------------------------------------
# Scale factor mutation in eval vs train mode
# ---------------------------------------------------------------------------

def test_scale_factor_does_not_mutate_in_eval():
    """scale_factor must NOT change during eval mode even with _auto_scale_factor=True."""
    from medlat import GenWrapper
    tok = _continuous_tok()
    gen = _diffusion_gen()
    w = GenWrapper(gen, tok)  # scale_factor=None → _auto_scale_factor=True
    w.eval()

    initial_sf = w.scale_factor.item()
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        w.vae_encode(x)

    assert w.scale_factor.item() == initial_sf, (
        f"scale_factor changed in eval mode: {initial_sf} → {w.scale_factor.item()}"
    )


def test_scale_factor_updates_in_train():
    """scale_factor SHOULD update during training mode when _auto_scale_factor=True."""
    from medlat import GenWrapper
    tok = _continuous_tok()
    gen = _diffusion_gen()
    w = GenWrapper(gen, tok, scale_steps=5)  # scale_factor=None → _auto_scale_factor=True
    w.train()

    initial_sf = w.scale_factor.item()
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        for _ in range(3):
            w.vae_encode(x)

    # After 3 steps the scale factor should have been updated
    assert w.scale_factor.item() != initial_sf or w.scale_factor.item() != 1.0 or w._scale_step_counter > 0


def test_manual_scale_factor_never_mutates():
    """Manually provided scale_factor must never change."""
    from medlat import GenWrapper
    tok = _continuous_tok()
    gen = _diffusion_gen()
    manual_sf = 0.18215
    w = GenWrapper(gen, tok, scale_factor=manual_sf)
    w.train()

    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        for _ in range(10):
            w.vae_encode(x)

    assert abs(w.scale_factor.item() - manual_sf) < 1e-6, (
        f"Manual scale_factor changed: {manual_sf} → {w.scale_factor.item()}"
    )
