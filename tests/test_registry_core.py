"""Unit tests for medlat.registry — pure-Python registration logic.

These tests use a fresh ``ModelRegistry`` instance per test so they do not
interact with the global ``MODEL_REGISTRY`` populated at import time.
"""
from __future__ import annotations

import pytest

from medlat.registry import (
    ModelEntry,
    ModelInfo,
    ModelRegistry,
    available_models,
    clone_with,
    get_model_info,
    get_model_signature,
    register_model,
)


# Shared tiny class used as a model stand-in — no torch required.
class _Toy:
    def __init__(self, img_size=256, z_channels=3, ch_mult=(1, 2, 4)):
        self.img_size = img_size
        self.z_channels = z_channels
        self.ch_mult = ch_mult


# ---------------------------------------------------------------------------
# ModelInfo / ModelEntry dataclasses
# ---------------------------------------------------------------------------


def test_modelinfo_str_includes_all_populated_fields():
    info = ModelInfo(
        name="m.foo",
        description="d",
        code_url="c",
        paper_url="p",
        ckpt_path="k",
        metadata={"a": 1},
    )
    s = str(info)
    assert "m.foo" in s
    assert "d" in s and "c" in s and "p" in s and "k" in s
    assert "a" in s


def test_modelinfo_str_omits_empty_fields():
    info = ModelInfo(name="m.bar")
    s = str(info)
    assert "m.bar" in s
    assert "description" not in s
    assert "code_url" not in s


def test_modelentry_to_info_strips_builder():
    entry = ModelEntry(name="x", builder=lambda: 42, description="hi")
    info = entry.to_info()
    assert isinstance(info, ModelInfo)
    assert info.name == "x"
    assert info.description == "hi"
    # to_info must not expose a builder attribute
    assert not hasattr(info, "builder")


def test_modelentry_to_info_copies_metadata():
    md = {"a": 1}
    entry = ModelEntry(name="x", builder=lambda: None, metadata=md)
    info = entry.to_info()
    info.metadata["a"] = 2
    assert md["a"] == 1, "to_info() must deep-copy metadata"


# ---------------------------------------------------------------------------
# ModelRegistry core
# ---------------------------------------------------------------------------


def test_register_and_get_roundtrip():
    r = ModelRegistry()
    r.register("foo.bar", lambda: "built")
    entry = r.get("foo.bar")
    assert entry.name == "foo.bar"
    assert entry.builder() == "built"


def test_get_is_case_insensitive():
    r = ModelRegistry()
    r.register("Foo.Bar", lambda: 1)
    assert r.get("foo.bar").name == "Foo.Bar"
    assert r.get("FOO.BAR").name == "Foo.Bar"


def test_register_rejects_duplicate_without_override():
    r = ModelRegistry()
    r.register("m", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        r.register("m", lambda: 2)


def test_register_override_replaces_entry():
    r = ModelRegistry()
    r.register("m", lambda: 1)
    r.register("m", lambda: 2, override=True)
    assert r.get("m").builder() == 2


def test_register_preserves_canonical_case():
    r = ModelRegistry()
    r.register("MyModel", lambda: None)
    assert r.get("mymodel").name == "MyModel"
    assert "MyModel" in r.available()


def test_get_unknown_raises_keyerror_with_available():
    r = ModelRegistry()
    r.register("alpha", lambda: None)
    with pytest.raises(KeyError) as exc:
        r.get("beta")
    # message should list what *is* registered
    assert "alpha" in str(exc.value)


def test_create_instantiates_with_args():
    r = ModelRegistry()
    r.register("adder", lambda a, b=0: a + b)
    assert r.create("adder", 3, b=4) == 7


def test_available_returns_sorted_tuple():
    r = ModelRegistry()
    r.register("z.last", lambda: None)
    r.register("a.first", lambda: None)
    r.register("m.mid", lambda: None)
    names = r.available()
    assert isinstance(names, tuple)
    assert names == ("a.first", "m.mid", "z.last")


def test_available_prefix_filters():
    r = ModelRegistry()
    r.register("cont.aekl.f4", lambda: None)
    r.register("cont.aekl.f8", lambda: None)
    r.register("disc.vq.f4", lambda: None)
    assert r.available(prefix="cont.") == ("cont.aekl.f4", "cont.aekl.f8")
    assert r.available(prefix="disc.") == ("disc.vq.f4",)
    assert r.available(prefix="nope") == ()


def test_available_prefix_is_case_insensitive():
    r = ModelRegistry()
    r.register("Cont.AEKL.f4", lambda: None)
    assert r.available(prefix="cont.") == ("Cont.AEKL.f4",)


def test_get_info_returns_modelinfo_without_builder():
    r = ModelRegistry()
    r.register("m", lambda x: x, description="d", paper_url="p")
    info = r.get_info("m")
    assert isinstance(info, ModelInfo)
    assert info.description == "d"
    assert info.paper_url == "p"
    assert not hasattr(info, "builder")


# ---------------------------------------------------------------------------
# Decorator-style register_model and module-level helpers.
# These touch the global MODEL_REGISTRY — use guaranteed-unique names and
# clean up afterwards so test order is irrelevant.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_global_name():
    """Yield a unique name and delete it from the global registry afterwards."""
    from medlat.registry import MODEL_REGISTRY

    name = "__medlat_test__.temporary_entry"
    yield name
    # cleanup
    key = MODEL_REGISTRY._normalize(name)
    MODEL_REGISTRY._registry.pop(key, None)


def test_register_model_as_decorator(fresh_global_name):
    @register_model(fresh_global_name, description="decorator test")
    def builder(x: int = 5) -> int:
        return x * 2

    assert fresh_global_name in available_models()
    info = get_model_info(fresh_global_name)
    assert info.description == "decorator test"


def test_register_model_direct_call(fresh_global_name):
    def builder(x: int = 5) -> int:
        return x * 2

    register_model(fresh_global_name, builder, description="direct test")
    assert fresh_global_name in available_models()


def test_get_model_signature_marks_required_params(fresh_global_name):
    def builder(required_arg, optional_arg: int = 42, *args, **kwargs):
        return None

    register_model(fresh_global_name, builder)
    sig = get_model_signature(fresh_global_name)
    assert sig["required_arg"] == "<required>"
    assert sig["optional_arg"] == 42
    # *args / **kwargs must be stripped
    assert "args" not in sig
    assert "kwargs" not in sig


def test_get_model_signature_preserves_insertion_order(fresh_global_name):
    def builder(a, b=1, c=2):
        return None

    register_model(fresh_global_name, builder)
    sig = get_model_signature(fresh_global_name)
    assert list(sig.keys()) == ["a", "b", "c"]


def test_available_models_with_prefix():
    # Just sanity-check the module-level wrapper delegates correctly;
    # don't assume any specific entry exists in the populated global registry.
    all_names = list(available_models())
    if all_names:
        prefix = all_names[0][:3]
        filtered = list(available_models(prefix=prefix))
        # Every filtered entry should start with the same prefix (case-insensitive)
        for n in filtered:
            assert n.lower().startswith(prefix.lower())


# ---------------------------------------------------------------------------
# Config-snapshot provenance on .instantiate() — the backing tags that
# clone_with relies on.
# ---------------------------------------------------------------------------


def test_instantiate_attaches_medlat_name():
    r = ModelRegistry()
    r.register("toy.v1", lambda **kw: _Toy(**kw))
    m = r.create("toy.v1", z_channels=8)
    assert m._medlat_name == "toy.v1"


def test_instantiate_attaches_medlat_config_from_kwargs():
    r = ModelRegistry()
    r.register("toy.v1", lambda **kw: _Toy(**kw))
    m = r.create("toy.v1", z_channels=8, img_size=128)
    assert m._medlat_config == {"z_channels": 8, "img_size": 128}


def test_instantiate_binds_positional_args_to_names():
    r = ModelRegistry()
    # explicit signature (no **kw) so positional args can be bound by name
    r.register("toy.v2", lambda img_size=256, z_channels=3, ch=128: _Toy(img_size=img_size, z_channels=z_channels))
    m = r.create("toy.v2", 512, 16)  # positional
    assert m._medlat_config == {"img_size": 512, "z_channels": 16}


def test_instantiate_exposes_config_attribute_when_unset():
    r = ModelRegistry()
    r.register("toy.v1", lambda **kw: _Toy(**kw))
    m = r.create("toy.v1", z_channels=8)
    assert m.config == {"z_channels": 8}


def test_instantiate_preserves_preexisting_config_attribute():
    # HF-style models expose their own `.config`. The registry must not clobber it.
    class HFStyle:
        def __init__(self, **kw):
            self.config = {"pre_existing": True}

    r = ModelRegistry()
    r.register("hf.toy", lambda **kw: HFStyle(**kw))
    m = r.create("hf.toy", foo=1)
    # pre-existing config preserved
    assert m.config == {"pre_existing": True}
    # but the internal provenance tag is always the snapshot
    assert m._medlat_config == {"foo": 1}


def test_instantiate_silently_noop_on_primitive_return():
    # Builders returning primitives (ints, strings) can't accept attributes —
    # provenance tagging must fail silently and not crash.
    r = ModelRegistry()
    r.register("toy.int", lambda a, b=0: a + b)
    x = r.create("toy.int", 3, b=4)
    assert x == 7
    assert not hasattr(x, "_medlat_name")


def test_instantiate_flattens_var_keyword_catchall():
    # kwargs arriving via **kw must end up at the top level of the snapshot,
    # not nested under a "kw"/"kwargs" key.
    def builder(img_size=256, **kw):
        return _Toy(img_size=img_size)

    r = ModelRegistry()
    r.register("toy.kw", builder)
    m = r.create("toy.kw", img_size=128, extra="x", another=42)
    assert m._medlat_config == {"img_size": 128, "extra": "x", "another": 42}


# ---------------------------------------------------------------------------
# clone_with
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_toy():
    """Register a toy model on the *global* registry and clean up after the test.
    clone_with() resolves names through the global MODEL_REGISTRY, so we use it
    here rather than a scratch ModelRegistry."""
    from medlat.registry import MODEL_REGISTRY

    name = "__medlat_test__.clone_toy"
    MODEL_REGISTRY.register(name, lambda **kw: _Toy(**kw))
    yield name
    MODEL_REGISTRY._registry.pop(MODEL_REGISTRY._normalize(name), None)


def test_clone_with_overrides_single_field(registered_toy):
    from medlat import get_model

    m = get_model(registered_toy, z_channels=8)
    m2 = clone_with(m, z_channels=32)
    assert m2.z_channels == 32
    # other fields carry over from the original construction
    assert m2._medlat_config.get("z_channels") == 32


def test_clone_with_preserves_unchanged_fields(registered_toy):
    from medlat import get_model

    m = get_model(registered_toy, img_size=512, z_channels=8)
    m2 = clone_with(m, z_channels=16)
    assert m2.img_size == 512
    assert m2.z_channels == 16


def test_clone_with_returns_fresh_instance(registered_toy):
    from medlat import get_model

    m = get_model(registered_toy, z_channels=8)
    m2 = clone_with(m, z_channels=32)
    assert m is not m2
    # original is untouched
    assert m.z_channels == 8


def test_clone_with_rejects_non_registry_objects():
    with pytest.raises(ValueError, match="built via get_model"):
        clone_with("not a model", x=1)


def test_clone_with_rejects_unregistered_manual_model():
    # A model built directly (not via get_model) has no provenance tags.
    class Manual:
        pass

    with pytest.raises(ValueError, match="built via get_model"):
        clone_with(Manual(), x=1)


def test_clone_with_is_chainable(registered_toy):
    from medlat import get_model

    m = get_model(registered_toy, img_size=256, z_channels=3)
    m2 = clone_with(m, z_channels=16)
    m3 = clone_with(m2, img_size=512)
    assert m3.img_size == 512
    assert m3.z_channels == 16
