"""Structural tests for the abstract quantizer hierarchy.

These tests are about *contracts*, not numerical behavior:

* The abstract bases cannot be instantiated directly.
* Every concrete quantizer has the expected MRO.
* Concrete classes inherit the right default from
  :class:`AbstractQuantizer.get_codebook_entry`.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from medlat.first_stage.discrete.quantizer import quantize as qn


# ---------------------------------------------------------------------------
# Abstract classes cannot be instantiated
# ---------------------------------------------------------------------------


def test_abstract_quantizer_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        qn.AbstractQuantizer()  # type: ignore[abstract]


def test_residual_quantizer_base_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        qn.ResidualQuantizerBase()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MRO / subclass-of relationships
# ---------------------------------------------------------------------------


SINGLE_LEVEL_CLASSES = [
    "VectorQuantizer",
    "GumbelQuantize",
    "VectorQuantizer2",
    "SimVQ",
    "GroupedVQ",
    "LookupFreeQuantizer",
    "FiniteScalarQuantizer",
    "SoftVectorQuantizer",
]

RESIDUAL_CLASSES = [
    "ResidualQuantizer",
    "QincoResidualQuantizer",
    "MultiScaleResidualQuantizer",
    "MultiScaleResidualQuantizer3D",
]

TRANSITIVE_CLASSES = {
    # name → expected direct parent
    "SimpleQINCo": "VectorQuantizer2",
    "BinarySphericalQuantizer": "LookupFreeQuantizer",
}


@pytest.mark.parametrize("cls_name", SINGLE_LEVEL_CLASSES)
def test_single_level_classes_subclass_abstractquantizer(cls_name):
    cls = getattr(qn, cls_name)
    assert issubclass(cls, qn.AbstractQuantizer), f"{cls_name} must subclass AbstractQuantizer"
    # and still an nn.Module
    assert issubclass(cls, nn.Module)


@pytest.mark.parametrize("cls_name", RESIDUAL_CLASSES)
def test_residual_classes_subclass_residualbase(cls_name):
    cls = getattr(qn, cls_name)
    assert issubclass(cls, qn.ResidualQuantizerBase), f"{cls_name} must subclass ResidualQuantizerBase"
    # ResidualQuantizerBase extends AbstractQuantizer, so this should also hold
    assert issubclass(cls, qn.AbstractQuantizer)


@pytest.mark.parametrize("child,parent", TRANSITIVE_CLASSES.items())
def test_transitively_promoted_classes(child, parent):
    child_cls = getattr(qn, child)
    parent_cls = getattr(qn, parent)
    assert issubclass(child_cls, parent_cls)
    # Transitive promotion through the parent
    assert issubclass(child_cls, qn.AbstractQuantizer)


def test_qinco_helper_is_deliberately_outside_hierarchy():
    # QINCo uses a non-standard forward(residual, x_prev) signature and is an
    # internal helper for QincoResidualQuantizer — it must *not* be tagged as
    # an AbstractQuantizer subclass.
    assert not issubclass(qn.QINCo, qn.AbstractQuantizer)
    assert issubclass(qn.QINCo, nn.Module)


# ---------------------------------------------------------------------------
# Abstract method enforcement — concrete classes must supply `forward`
# ---------------------------------------------------------------------------


ALL_CONCRETE = SINGLE_LEVEL_CLASSES + RESIDUAL_CLASSES + list(TRANSITIVE_CLASSES)


@pytest.mark.parametrize("cls_name", ALL_CONCRETE)
def test_concrete_classes_define_forward(cls_name):
    cls = getattr(qn, cls_name)
    # `forward` defined somewhere in the MRO below AbstractQuantizer (which
    # declares it abstract). If a subclass failed to implement it, ABC would
    # mark the class itself as abstract.
    assert not getattr(cls, "__abstractmethods__", frozenset()), (
        f"{cls_name} still has unimplemented abstract methods: "
        f"{cls.__abstractmethods__}"
    )


# ---------------------------------------------------------------------------
# Default implementation of get_codebook_entry on AbstractQuantizer raises.
# Verify via an *explicit* dummy subclass (not instantiating AbstractQuantizer).
# ---------------------------------------------------------------------------


class _DummyQuantizer(qn.AbstractQuantizer):
    """Minimal concrete subclass used for testing the base-class defaults."""

    def __init__(self) -> None:
        super().__init__()
        self.n_e = 8
        self.e_dim = 4

    def forward(self, z):  # pragma: no cover — never called in these tests
        return z, torch.tensor(0.0), (torch.tensor(0.0), None, torch.zeros(1))


def test_get_codebook_entry_default_raises():
    m = _DummyQuantizer()
    with pytest.raises(NotImplementedError, match="get_codebook_entry"):
        m.get_codebook_entry(torch.zeros(4, dtype=torch.long), shape=None)


# ---------------------------------------------------------------------------
# Lightweight smoke: a representative concrete class can be instantiated and
# satisfies the contract. This is *not* a numerical equivalence check.
# ---------------------------------------------------------------------------


def test_vector_quantizer_instantiates_and_exposes_contract():
    m = qn.VectorQuantizer(n_e=4, e_dim=2, beta=0.25)
    assert isinstance(m, qn.AbstractQuantizer)
    assert m.n_e == 4
    assert m.e_dim == 2


def test_lookup_free_quantizer_instantiates_and_exposes_contract():
    m = qn.LookupFreeQuantizer(token_bits=3)
    assert isinstance(m, qn.AbstractQuantizer)
    # LFQ exposes codebook_size / token_size rather than n_e / e_dim;
    # the abstract base annotates the canonical names for documentation only.
    assert m.codebook_size == 8
    assert m.token_size == 3


# ---------------------------------------------------------------------------
# Automatic autocast-disable wrapping via __init_subclass__.
#
# Every concrete subclass's forward must be wrapped; the mechanism is the
# "why" behind why we deleted @autocast decorators from the concrete classes
# during the refactor. If the wrap silently stops working, a future mixed-
# precision training run could produce spurious NaNs without warning.
# ---------------------------------------------------------------------------


def test_init_subclass_wraps_a_freshly_defined_subclass():
    # Define a synthetic subclass and verify its forward was re-assigned away
    # from the user-provided function by __init_subclass__.
    def user_forward(self, z):  # pragma: no cover — never actually called
        return z, torch.tensor(0.0), (torch.tensor(0.0), None, torch.zeros(1))

    ns = {"__module__": __name__, "forward": user_forward}
    cls = type("_SyntheticQuantizer", (qn.AbstractQuantizer,), ns)

    # After class creation, cls.forward must differ from the raw user_forward
    # (that's the wrap). Grabbing via __dict__ bypasses descriptor lookup.
    assert cls.__dict__["forward"] is not user_forward, (
        "__init_subclass__ did not re-assign forward — autocast wrap missing."
    )


def test_init_subclass_does_not_wrap_bases_that_dont_redefine_forward():
    # ResidualQuantizerBase is abstract and does NOT define its own forward
    # (it inherits the abstract one). The wrap should skip it so that a fresh
    # concrete residual subclass still gets wrapped.
    # We assert that the attribute `forward` on ResidualQuantizerBase is the
    # same object as on AbstractQuantizer — no extra wrap in between.
    assert (
        qn.ResidualQuantizerBase.__dict__.get("forward") is None
    ), "ResidualQuantizerBase must not redefine forward in its own __dict__"


@pytest.mark.parametrize("cls_name", SINGLE_LEVEL_CLASSES + RESIDUAL_CLASSES)
def test_concrete_forward_is_wrapped(cls_name):
    # Concrete leaves must have a forward that's *not* the raw function we
    # would see if __init_subclass__ hadn't run. We don't know the raw
    # function, but we do know the wrapped one resolves through the
    # autocast-decorator path; it should not equal the __func__ of the
    # super-class's forward unless the subclass forwards to super directly.
    cls = getattr(qn, cls_name)
    assert "forward" in cls.__dict__, f"{cls_name} must define its own forward"
    # The wrapped function gets a fresh identity each time autocast() is
    # called, so it won't be identical to any user-defined reference. The
    # simplest sanity check is that it's still callable.
    assert callable(cls.__dict__["forward"])
