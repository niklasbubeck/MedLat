"""Extended registry tests."""

import pytest


def test_available_models_prefix_continuous():
    from medlat import available_models
    models = list(available_models(prefix="continuous"))
    assert len(models) > 0
    for name in models:
        assert name.startswith("continuous")


def test_available_models_prefix_dit():
    from medlat import available_models
    models = list(available_models(prefix="dit"))
    assert len(models) > 0
    for name in models:
        assert name.startswith("dit")


def test_available_models_prefix_discrete():
    from medlat import available_models
    models = list(available_models(prefix="discrete"))
    assert len(models) > 0
    for name in models:
        assert name.startswith("discrete")


def test_available_models_no_prefix_returns_all():
    from medlat import available_models
    all_models = list(available_models())
    continuous_models = list(available_models(prefix="continuous"))
    assert len(all_models) > len(continuous_models)


def test_reregister_without_override_raises():
    from medlat import register_model, MODEL_REGISTRY
    import torch.nn as nn

    # First registration
    @register_model("test.reregister_test_model")
    def _dummy_factory(**kwargs):
        return nn.Linear(1, 1)

    # Re-registration without override should raise
    with pytest.raises(Exception):
        @register_model("test.reregister_test_model", override=False)
        def _dummy_factory2(**kwargs):
            return nn.Linear(2, 2)


def test_reregister_with_override_succeeds():
    from medlat import register_model, get_model
    import torch.nn as nn

    @register_model("test.override_test_model")
    def _factory_v1(**kwargs):
        return nn.Linear(1, 1)

    @register_model("test.override_test_model", override=True)
    def _factory_v2(**kwargs):
        return nn.Linear(2, 2)

    m = get_model("test.override_test_model")
    assert m.in_features == 2  # confirms v2 is active


def test_get_model_info_returns_model_info():
    from medlat import get_model_info, ModelInfo
    info = get_model_info("continuous.aekl.f4_d3")
    assert isinstance(info, ModelInfo)
    assert info.name == "continuous.aekl.f4_d3"


def test_get_model_info_has_expected_fields():
    from medlat import get_model_info
    info = get_model_info("continuous.aekl.f4_d3")
    assert hasattr(info, "name")
    assert hasattr(info, "code_url")
    assert hasattr(info, "paper_url")
