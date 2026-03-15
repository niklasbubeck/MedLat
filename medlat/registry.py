from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional


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
        return self.builder(*args, **kwargs)

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
    """Central registry that keeps track of model builder callables."""

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
        key = self._normalize(name)
        if not override and key in self._registry:
            raise ValueError(f"Model '{name}' already registered.")
        entry = ModelEntry(
            name=key,
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
        key = self._normalize(name)
        try:
            return self._registry[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown model '{name}'. Available models: {sorted(self._registry)}"
            ) from exc

    def get_info(self, name: str) -> ModelInfo:
        """Return display-only metadata for a model (no builder)."""
        return self.get(name).to_info()

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name).instantiate(*args, **kwargs)

    def available(self, prefix: Optional[str] = None) -> Iterable[str]:
        if prefix is None:
            return tuple(sorted(self._registry))
        normalized = self._normalize(prefix)
        return tuple(sorted(name for name in self._registry if name.startswith(normalized)))


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
    """
    Helper that registers a model builder either directly or as a decorator.
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


def get_model_info(name: str) -> ModelInfo:
    """Return display-only metadata for a registered model (no builder)."""
    return MODEL_REGISTRY.get_info(name)


def available_models(prefix: Optional[str] = None) -> Iterable[str]:
    """List the registered model identifiers."""
    return MODEL_REGISTRY.available(prefix=prefix)

