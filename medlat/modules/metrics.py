"""
Shared metric-logging mixin for MedLat modules.

Anything that wants a ``log_metric`` / ``get_metrics`` / ``reset_metrics``
surface — quantizers, first-stage autoencoders, schedulers — can pick this
mixin up without reimplementing the plumbing. Latest-value semantics: each
``log_metric(key, value)`` call overwrites the previous value for that key;
nothing is buffered.

Tensor values are detached so logging cannot hold onto a computation graph;
non-tensor values pass through unchanged.
"""
from typing import Any, Dict

import torch

__all__ = ["MetricLoggerMixin"]


class MetricLoggerMixin:
    """Latest-value scalar metric sink for arbitrary classes.

    Subclasses inherit the three public methods below. The underlying
    ``_metrics`` dict is created lazily on first :meth:`log_metric` call, so
    instances that never log anything carry no per-instance overhead.

    Example::

        class MyModel(MetricLoggerMixin, nn.Module):
            def forward(self, x):
                out = ...
                self.log_metric("my_stat", out.mean())
                return out

        m = MyModel()
        m(x)
        wandb.log(m.get_metrics())
    """

    def log_metric(self, key: str, value: Any) -> None:
        """Record the latest value for ``key``; overwrites any prior value.

        Tensor values are ``.detach()``-ed so logging never retains a
        computation graph. Non-tensor values (Python scalars, strings,
        small lists, …) pass through unchanged.
        """
        if not hasattr(self, "_metrics"):
            self._metrics: Dict[str, Any] = {}
        if isinstance(value, torch.Tensor):
            value = value.detach()
        self._metrics[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Return a shallow copy of the current metric dict.

        Callers can safely mutate the returned dict without affecting the
        internal state. Returns ``{}`` if nothing has been logged yet.
        """
        return dict(getattr(self, "_metrics", {}))

    def reset_metrics(self) -> None:
        """Clear every logged metric. No-op if nothing has been logged."""
        if hasattr(self, "_metrics"):
            self._metrics.clear()
