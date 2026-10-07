"""Abstract base class for generative schedulers (diffusion and flow matching)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch

class BaseScheduler(ABC):
    """Shared base class for GaussianDiffusionScheduler and FlowMatchingScheduler.

    Concrete subclasses must implement training_losses and p_sample_loop.
    Advanced / paradigm-specific methods (e.g. ddim_sample_loop, get_drift)
    are accessible via __getattr__ fall-through on each concrete subclass.
    """

    @abstractmethod
    def training_losses(
        self,
        model: Any,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss for one batch.
        Returns a dict that always contains 'loss'. May also contain 'mse', 'vb', etc.
        """
        ...

    @abstractmethod
    def p_sample_loop(
        self,
        model: Any,
        shape: Tuple,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        sampler: str = "default",
        **sampler_kwargs,
    ) -> torch.Tensor:
        """Generate a batch of samples from noise. Returns tensor of shape `shape`."""
        ...

    def _collect_auxiliary_losses(
        self, model: Any, terms: Dict[str, torch.Tensor],
    ) -> None:
        """Collect auxiliary losses (alignment, etc.) from the model after forward.

        If the model exposes ``get_auxiliary_losses()``, each returned entry
        is added to ``terms`` and summed into ``terms['loss']``.  Models
        without the method are silently skipped.
        """
        if hasattr(model, "get_auxiliary_losses"):
            for key, value in model.get_auxiliary_losses().items():
                terms[key] = value
                terms["loss"] = terms["loss"] + value
