"""Regression tests for specific bugs fixed in this PR."""

import pytest
import torch


IMG_SIZE = 32


def test_aekl_f4_d3_no_use_quant_conv_error():
    """Regression: AutoencoderKL must not raise TypeError for use_quant_conv."""
    from medlat import get_model
    m = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    assert m is not None


def test_aekl_f8_d4_succeeds():
    """Regression: f8 variant should also instantiate without error."""
    from medlat import get_model
    m = get_model("continuous.aekl.f8_d4", img_size=64)
    assert m is not None


def test_dcae_f32c32_no_use_quant_conv_error():
    """Regression: DCAE f32c32 must not raise NameError for use_quant_conv."""
    from medlat import get_model
    m = get_model("continuous.dcae.f32c32", img_size=IMG_SIZE)
    assert m is not None


def test_dcae_f64c128_no_use_quant_conv_error():
    """Regression: DCAE f64c128 must not raise NameError for use_quant_conv."""
    from medlat import get_model
    m = get_model("continuous.dcae.f64c128", img_size=IMG_SIZE)
    assert m is not None


def test_dcae_f128c512_no_use_quant_conv_error():
    """Regression: DCAE f128c512 must not raise NameError for use_quant_conv."""
    from medlat import get_model
    m = get_model("continuous.dcae.f128c512", img_size=IMG_SIZE)
    assert m is not None


def test_vq_f4_d3_e8192_succeeds():
    """Regression: discrete VQ model should instantiate cleanly."""
    from medlat import get_model
    m = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)
    assert m is not None


def test_aekl_f4_d3_encode_decode():
    """Check that encode/decode round-trip works without errors."""
    from medlat import get_model
    m = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    m.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        z, loss, _ = m.encode(x)
        rec = m.decode(z)
    assert rec.shape == x.shape


def test_vq_f4_d3_e8192_encode_decode():
    """Check that encode/decode round-trip works for VQModel."""
    from medlat import get_model
    m = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)
    m.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        quant, loss, info = m.encode(x)
        rec = m.decode(quant)
    assert rec.shape == x.shape
