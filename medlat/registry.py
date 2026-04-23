from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def _capture_call(
    builder: Callable[..., Any],
    args: tuple,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a kwargs-only snapshot of how ``builder`` was called.

    Positional arguments are bound to their parameter names via
    :func:`inspect.signature`, and any values that arrived through a
    ``**kwargs`` catch-all are hoisted to the top level.  ``*args`` overflow (if
    any) is dropped — it can't be replayed by name and is vanishingly rare for
    MedLat factories. Falls back to the raw kwargs dict if signature binding
    fails for any reason.
    """
    try:
        sig = inspect.signature(builder)
        bound = sig.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return dict(kwargs)

    captured: Dict[str, Any] = {}
    for param_name, value in bound.arguments.items():
        param = sig.parameters[param_name]
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue  # can't round-trip as a kwarg
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            # flatten **kwargs catch-all into top-level keys
            if isinstance(value, dict):
                captured.update(value)
            continue
        captured[param_name] = value
    return captured


def _attach_provenance(
    result: Any,
    name: str,
    config: Dict[str, Any],
) -> None:
    """Tag ``result`` with the registry name + config it was built from.

    Always sets ``_medlat_name`` and ``_medlat_config`` — these are the source
    of truth that :func:`clone_with` reads from. Also exposes a friendlier
    ``config`` attribute, but only if the model doesn't already define one
    (so we don't clobber HuggingFace-style configs on wrapped models).

    Silently no-ops if ``result`` doesn't accept attribute assignment (e.g. a
    primitive returned by a test builder).
    """
    try:
        setattr(result, "_medlat_name", name)
        setattr(result, "_medlat_config", dict(config))
    except (AttributeError, TypeError):
        return  # immutable / __slots__-locked result — nothing more we can do

    if not hasattr(result, "config"):
        try:
            setattr(result, "config", dict(config))
        except (AttributeError, TypeError):
            pass


@dataclass(slots=True)
class ModelInfo:
    """Display-only metadata for a registered model (no builder)."""

    name: str
    description: Optional[str] = None
    code_url: Optional[str] = None
    paper_url: Optional[str] = None
    ckpt_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Model: {self.name}"]
        if self.description:
            lines.append(f"  description: {self.description}")
        if self.code_url:
            lines.append(f"  code_url: {self.code_url}")
        if self.paper_url:
            lines.append(f"  paper_url: {self.paper_url}")
        if self.ckpt_path:
            lines.append(f"  ckpt_path: {self.ckpt_path}")
        if self.metadata:
            lines.append(f"  metadata: {self.metadata}")
        return "\n".join(lines)


@dataclass(slots=True)
class ModelEntry:
    """Metadata container for a registered model."""

    name: str
    builder: Callable[..., Any]
    code_url: Optional[str] = None
    description: Optional[str] = None
    paper_url: Optional[str] = None
    ckpt_path: Optional[str] = None  # Path to original model weights
    metadata: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying builder and tag the result with provenance.

        After construction, the returned object carries:

        * ``_medlat_name`` — the registry name used to build it;
        * ``_medlat_config`` — the kwargs dict the builder was called with
          (with positional args bound to their parameter names and any
          ``**kwargs`` catch-all flattened to the top level);
        * ``config`` — same as ``_medlat_config``, set only if the object
          doesn't already expose a ``config`` attribute.

        The provenance is what lets :func:`clone_with` rebuild the same model
        with a field overridden. If the builder returns a non-mutable type,
        tagging silently no-ops so the core behavior is unchanged.
        """
        result = self.builder(*args, **kwargs)
        snapshot = _capture_call(self.builder, args, kwargs)
        _attach_provenance(result, self.name, snapshot)
        return result

    def to_info(self) -> ModelInfo:
        """Return display-only info (no builder)."""
        return ModelInfo(
            name=self.name,
            description=self.description,
            code_url=self.code_url,
            paper_url=self.paper_url,
            ckpt_path=self.ckpt_path,
            metadata=dict(self.metadata),
        )


class ModelRegistry:
    """Central registry that keeps track of model builder callables.

    Names are compared case-insensitively after stripping whitespace — that is,
    ``"DiT.XL_2"`` and ``"dit.xl_2"`` refer to the same entry.  The canonical
    (original-case) name is preserved in :class:`ModelEntry` for display.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, ModelEntry] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower()

    def register(
        self,
        name: str,
        builder: Callable[..., Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        code_url: Optional[str] = None,
        description: Optional[str] = None,
        paper_url: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        override: bool = False,
    ) -> ModelEntry:
        """Register ``builder`` under ``name`` and return the stored entry.

        Args:
            name: identifier used by :func:`get_model`. Case-insensitive.
            builder: any callable that returns an instantiated model.
            metadata: free-form dict attached to the entry.
            code_url: optional URL to the model's reference implementation.
            description: short human-readable description.
            paper_url: optional URL to the model's paper.
            ckpt_path: optional path / URL of an official checkpoint.
            override: if ``False`` (default) re-registering the same name raises
                :class:`ValueError`; pass ``True`` to replace an existing entry.

        Raises:
            ValueError: if ``name`` is already registered and ``override=False``.
        """
        key = self._normalize(name)
        if not override and key in self._registry:
            raise ValueError(f"Model '{name}' already registered.")
        # Store the canonical (original-case) name in the entry so that
        # error messages and available_models() can show it as registered.
        entry = ModelEntry(
            name=name,
            builder=builder,
            code_url=code_url,
            description=description,
            paper_url=paper_url,
            ckpt_path=ckpt_path,
            metadata=metadata or {},
        )
        self._registry[key] = entry
        return entry

    def get(self, name: str) -> ModelEntry:
        """Return the :class:`ModelEntry` registered under ``name``.

        Raises:
            KeyError: if the name is not registered; the error message lists
                every currently-registered name.
        """
        key = self._normalize(name)
        try:
            return self._registry[key]
        except KeyError as exc:
            canonical_names = sorted(e.name for e in self._registry.values())
            raise KeyError(
                f"Unknown model '{name}'. Available models: {canonical_names}"
            ) from exc

    def get_info(self, name: str) -> ModelInfo:
        """Return display-only metadata for a model (no builder)."""
        return self.get(name).to_info()

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the model registered under ``name`` with the given args."""
        return self.get(name).instantiate(*args, **kwargs)

    def available(self, prefix: Optional[str] = None) -> Iterable[str]:
        """Return a sorted tuple of registered model names.

        Args:
            prefix: if given, only names whose normalised form starts with
                ``prefix`` (case-insensitive) are returned.
        """
        if prefix is None:
            return tuple(sorted(e.name for e in self._registry.values()))
        normalized = self._normalize(prefix)
        return tuple(
            sorted(e.name for key, e in self._registry.items() if key.startswith(normalized))
        )


MODEL_REGISTRY = ModelRegistry()


def register_model(
    name: str,
    builder: Optional[Callable[..., Any]] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    code_url: Optional[str] = None,
    description: Optional[str] = None,
    paper_url: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    override: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a model builder, either directly or as a decorator.

    Usage::

        @register_model("my.vqgan.v1", description="My tweaked VQ-GAN")
        def build_vqgan(img_size: int = 256, **kw):
            return VQGAN(img_size=img_size, **kw)

        # equivalent direct form:
        register_model("my.vqgan.v1", build_vqgan, description="My tweaked VQ-GAN")

    See :meth:`ModelRegistry.register` for a description of the metadata
    arguments. Returns the decorator (or the wrapped function, when used
    directly).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY.register(
            name,
            fn,
            metadata=metadata,
            code_url=code_url,
            description=description,
            paper_url=paper_url,
            ckpt_path=ckpt_path,
            override=override,
        )
        return fn

    if builder is not None:
        return decorator(builder)
    return decorator


def get_model(name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a registered model."""
    return MODEL_REGISTRY.create(name, *args, **kwargs)


def get_model_signature(name: str) -> Dict[str, Any]:
    """Return the parameter signature of a registered model builder.

    Introspects the builder function registered under ``name`` and returns a
    dict mapping each parameter name to its default value.  Parameters with no
    default are represented by the sentinel string ``'<required>'``.

    ``**kwargs`` catch-alls are omitted — they are implementation details of
    individual builders, not part of the public interface.

    Example::

        from medlat import get_model_signature

        sig = get_model_signature("dit.xl_2")
        # → {'img_size': '<required>', 'vae_stride': '<required>',
        #    'in_channels': '<required>', 'num_classes': 10, ...}

        # Discover required parameters at a glance:
        required = [k for k, v in sig.items() if v == '<required>']

    Args:
        name: registered model identifier (case-insensitive).

    Returns:
        Ordered dict of parameter names → default values (or ``'<required>'``).
    """
    entry = MODEL_REGISTRY.get(name)
    sig = inspect.signature(entry.builder)
    result: Dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            result[param_name] = "<required>"
        else:
            result[param_name] = param.default
    return result


def get_model_info(name: str) -> ModelInfo:
    """Return display-only metadata for a registered model (no builder)."""
    return MODEL_REGISTRY.get_info(name)


def available_models(prefix: Optional[str] = None) -> Iterable[str]:
    """List the registered model identifiers."""
    return MODEL_REGISTRY.available(prefix=prefix)


def clone_with(model: Any, **overrides: Any) -> Any:
    """Rebuild ``model`` via the registry with one or more kwargs overridden.

    This is the fast iteration primitive: tweak a single field without
    constructing a new config dict or re-reading the builder's source.  The
    returned object is a **fresh instance** — random initialisation differs
    from the original, and any trained weights on the input are *not* carried
    over.

    Example::

        from medlat import get_model, clone_with

        tok = get_model("discrete.vq.f4_d3_e8192")
        tok_big = clone_with(tok, z_channels=16)       # only z_channels changes
        tok_big_highres = clone_with(tok_big, img_size=512)

    Args:
        model: an object produced by :func:`get_model` (or anything exposing
            the ``_medlat_name`` and ``_medlat_config`` provenance tags that
            :meth:`ModelEntry.instantiate` attaches).
        **overrides: kwargs to merge on top of the original construction dict.

    Raises:
        ValueError: if ``model`` does not carry the registry provenance tags —
            typically because it wasn't built via :func:`get_model`.
    """
    name = getattr(model, "_medlat_name", None)
    if name is None:
        raise ValueError(
            f"clone_with() requires a model built via get_model(); got "
            f"{type(model).__name__}, which has no registered provenance. "
            "If you built the model manually, call get_model(...) to enable cloning."
        )
    base_config: Dict[str, Any] = getattr(model, "_medlat_config", {}) or {}
    merged = {**base_config, **overrides}
    return get_model(name, **merged)

