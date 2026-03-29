"""Tests for abstract base classes and isinstance-based get_model_type()."""

import pytest
import torch
import torch.nn as nn


IMG_SIZE = 32


def test_autoencoderkl_is_continuous_first_stage():
    from medlat import get_model
    from medlat.base import ContinuousFirstStage
    m = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    assert isinstance(m, ContinuousFirstStage)


def test_vqmodel_is_discrete_first_stage():
    from medlat import get_model
    from medlat.base import DiscreteFirstStage
    m = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)
    assert isinstance(m, DiscreteFirstStage)


def test_get_model_type_continuous():
    from medlat import get_model
    from medlat.utils import get_model_type
    m = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    assert get_model_type(m) == "continuous"


def test_get_model_type_discrete():
    from medlat import get_model
    from medlat.utils import get_model_type
    m = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)
    assert get_model_type(m) == "discrete"


def test_get_model_type_non_autoregressive():
    from medlat import get_model
    from medlat.utils import get_model_type
    m = get_model("dit.s_2", img_size=IMG_SIZE, vae_stride=4, in_channels=3, num_classes=10)
    assert get_model_type(m) == "non-autoregressive"


def test_get_model_type_autoregressive():
    from medlat import get_model
    from medlat.utils import get_model_type
    m = get_model(
        "mar.b",
        img_size=IMG_SIZE,
        vae_stride=4,
        patch_size=1,
        in_channels=3,
        buffer_size=IMG_SIZE // 4 * IMG_SIZE // 4,
    )
    assert get_model_type(m) == "autoregressive"


def test_get_model_type_fallback_for_legacy_models():
    """A model using module-path fallback (no base class) should still work."""
    from medlat.utils import get_model_type

    class FakeContinuousModel(nn.Module):
        pass

    FakeContinuousModel.__module__ = "medlat.first_stage.continuous.fake"
    m = FakeContinuousModel()
    assert get_model_type(m) == "continuous"


def test_get_model_type_unknown_raises():
    from medlat.utils import get_model_type

    class UnknownModel(nn.Module):
        pass

    m = UnknownModel()
    with pytest.raises(ValueError, match="Cannot determine model type"):
        get_model_type(m)


def test_continuous_first_stage_has_embed_dim():
    from medlat import get_model
    m = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    assert hasattr(m, "embed_dim")
    assert isinstance(m.embed_dim, int)


def test_continuous_first_stage_has_vae_stride():
    from medlat import get_model
    m = get_model("continuous.aekl.f4_d3", img_size=IMG_SIZE)
    assert hasattr(m, "vae_stride")
    assert m.vae_stride is not None


def test_discrete_first_stage_has_n_embed_and_embed_dim():
    from medlat import get_model
    m = get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE)
    assert hasattr(m, "n_embed")
    assert hasattr(m, "embed_dim")
    assert m.n_embed == 8192
