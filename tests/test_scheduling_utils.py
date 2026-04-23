"""Unit tests for medlat.scheduling.utils — EasyDict / mean_flat / log_state."""
from __future__ import annotations

import torch

from medlat.scheduling.utils import EasyDict, log_state, mean_flat


# ---------------------------------------------------------------------------
# EasyDict
# ---------------------------------------------------------------------------


def test_easydict_attribute_and_item_access_match():
    d = EasyDict({"foo": 1, "bar": "baz"})
    assert d.foo == 1
    assert d["foo"] == 1
    assert d.bar == "baz"
    assert d["bar"] == "baz"


def test_easydict_accepts_nested_values():
    d = EasyDict({"nested": {"x": 42}, "lst": [1, 2, 3]})
    assert d.nested == {"x": 42}
    assert d.lst == [1, 2, 3]


def test_easydict_empty():
    d = EasyDict({})
    # No attributes added; item access raises AttributeError for missing keys.
    import pytest

    with pytest.raises(AttributeError):
        _ = d["missing"]


# ---------------------------------------------------------------------------
# mean_flat
# ---------------------------------------------------------------------------


def test_mean_flat_2d():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    torch.testing.assert_close(mean_flat(x), torch.tensor([2.0, 5.0]))


def test_mean_flat_high_rank():
    x = torch.ones(2, 3, 4, 5)
    out = mean_flat(x)
    assert out.shape == (2,)
    torch.testing.assert_close(out, torch.ones(2))


# ---------------------------------------------------------------------------
# log_state
# ---------------------------------------------------------------------------


def test_log_state_sorts_keys():
    s = log_state({"zebra": 1, "apple": 2, "mango": 3})
    # Output lines should appear in sorted order
    lines = s.split("\n")
    assert lines[0].startswith("apple")
    assert lines[1].startswith("mango")
    assert lines[2].startswith("zebra")


def test_log_state_renders_objects_as_class_names():
    class Widget:
        def __repr__(self):
            return "<object Widget at 0xdeadbeef>"

    s = log_state({"w": Widget()})
    assert "[Widget]" in s


def test_log_state_renders_scalars_as_values():
    s = log_state({"n": 42, "s": "hi"})
    assert "n: 42" in s
    assert "s: hi" in s
