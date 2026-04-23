"""
medlat.scheduling — unified generative scheduler package.

Replaces the separate medlat.diffusion and medlat.transport packages.
Both paradigms now subclass BaseScheduler and are created through a
single create_scheduler() factory.

    from medlat.scheduling import create_scheduler, available_schedulers, scheduler_info

    # Discover what is available
    print(available_schedulers())
    # → ('diffusion', 'flow', 'self_flow')

    print(scheduler_info("diffusion"))
    # → SchedulerInfo(name='diffusion', samplers=['ddpm', 'ddim'], ...)

    # Gaussian diffusion
    sched = create_scheduler("diffusion", steps=1000, noise_schedule="cosine")

    # Flow matching
    sched = create_scheduler("flow", path_type="Linear", prediction="velocity")

    # Self-Flow (dual-timestep)
    sched = create_scheduler("self_flow", masking_strategy="complexity")

    # Uniform training / sampling interface:
    terms   = sched.training_losses(model, x_start, model_kwargs=cond)
    samples = sched.p_sample_loop(model, shape, model_kwargs=cond)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseScheduler
from .gaussian import GaussianDiffusionScheduler, create_gaussian_diffusion
from .flow import FlowMatchingScheduler, create_transport, Transport, Sampler, ModelType, WeightType, PathType
from .self_flow import DualTimestepScheduler, create_dual_timestep_scheduler


# ---------------------------------------------------------------------------
# Scheduler catalog
# ---------------------------------------------------------------------------

@dataclass
class SchedulerInfo:
    """Human-readable metadata for a scheduler type.

    Attributes:
        name:             Identifier passed to ``create_scheduler()``.
        description:      One-paragraph summary of the paradigm.
        samplers:         Valid ``sampler=`` strings for ``p_sample_loop()``,
                          in order of typical usage.
        required_kwargs:  Kwargs that **must** be supplied to ``create_scheduler()``.
        optional_kwargs:  Kwargs with defaults, as ``{name: (type_hint, default, description)}``.
        paper_url:        Primary reference.
    """

    name: str
    description: str
    samplers: List[str]
    required_kwargs: List[str]
    optional_kwargs: Dict[str, Tuple[str, Any, str]]
    paper_url: Optional[str] = None


_SCHEDULER_CATALOG: Dict[str, SchedulerInfo] = {
    "diffusion": SchedulerInfo(
        name="diffusion",
        description=(
            "Gaussian denoising diffusion probabilistic model (DDPM/DDIM). "
            "Learns to reverse a fixed Markov noising chain over T steps. "
            "Supports stochastic (DDPM) and deterministic (DDIM) sampling."
        ),
        samplers=["ddpm", "ddim"],
        required_kwargs=[],
        optional_kwargs={
            "steps":                   ("int",  1000,    "Total diffusion timesteps."),
            "noise_schedule":          ("str",  "linear","Beta schedule: 'linear' or 'cosine'."),
            "learn_sigma":             ("bool", False,   "Whether the model outputs learned variance."),
            "predict_xstart":          ("bool", False,   "Predict x₀ instead of ε."),
            "use_kl":                  ("bool", False,   "Use rescaled KL loss instead of MSE."),
            "rescale_timesteps":       ("bool", False,   "Scale timesteps to [0, 1000] range."),
            "timestep_respacing":      ("str",  "",      "Comma-sep step counts or 'ddimN' for striding."),
            "latent_dim":              ("int",  None,    "Effective data dim `m` for Esser et al. (2024) timestep shift. None disables."),
            "base_dim":                ("int",  4096,    "Reference dim `n` for the Esser shift (paper baseline)."),
        },
        paper_url="https://arxiv.org/abs/2006.11239",
    ),

    "flow": SchedulerInfo(
        name="flow",
        description=(
            "Flow matching / continuous normalising flows. "
            "Learns a velocity field that transports noise to data along a straight "
            "(or curved) ODE path.  Much simpler loss than diffusion; typically "
            "fewer NFEs at inference.  Also accessible as 'transport'."
        ),
        samplers=["euler", "heun", "dopri5", "sde"],
        required_kwargs=[],
        optional_kwargs={
            "path_type":    ("str",  "Linear",   "Interpolation path: 'Linear', 'GVP', or 'VP'."),
            "prediction":   ("str",  "velocity", "Model output type: 'velocity', 'score', or 'noise'."),
            "loss_weight":  ("str",  "velocity", "Loss weighting: 'velocity', 'likelihood', or 'none'."),
            "train_eps":    ("float", 1e-5,      "Min timestep during training."),
            "sample_eps":   ("float", 1e-3,      "Min timestep during sampling."),
            "use_lognorm":  ("bool", False,       "Sample t from logit-normal instead of uniform."),
        },
        paper_url="https://arxiv.org/abs/2210.02747",
    ),

    "self_flow": SchedulerInfo(
        name="self_flow",
        description=(
            "Dual-Timestep Self-Flow (DTS). "
            "Splits each sample's token sequence into anchor (low-noise) and masked "
            "(high-noise) groups, sampling independent per-token timesteps. "
            "Optionally biases masking toward high-variance (diagnostically informative) "
            "patches via masking_strategy='complexity'. "
            "Particularly effective for medical images. (Chefer et al., ICML 2025)"
        ),
        samplers=["euler", "heun"],
        required_kwargs=[],
        optional_kwargs={
            "patch_size":        ("int",   2,        "Spatial patch size for patchify/unpatchify."),
            "mask_ratio":        ("float", 0.25,     "Fraction of tokens assigned high noise."),
            "t_anchor_range":    ("tuple", (0.0, 0.4), "Timestep range for anchor (low-noise) tokens."),
            "t_masked_range":    ("tuple", (0.6, 1.0), "Timestep range for masked (high-noise) tokens."),
            "use_lognorm":       ("bool",  False,    "Sample t from logit-normal instead of uniform."),
            "masking_strategy":  ("str",   "random", "'random' or 'complexity' (variance-weighted)."),
        },
        paper_url="https://arxiv.org/abs/2603.06507",
    ),
}

# "transport" is an alias for "flow"
_SCHEDULER_CATALOG["transport"] = _SCHEDULER_CATALOG["flow"]


def available_schedulers() -> Tuple[str, ...]:
    """Return the registered scheduler type names.

    Example::

        from medlat.scheduling import available_schedulers
        print(available_schedulers())
        # → ('diffusion', 'flow', 'self_flow', 'transport')
    """
    return tuple(sorted(k for k in _SCHEDULER_CATALOG if k != "transport"))


def scheduler_info(name: str) -> SchedulerInfo:
    """Return metadata for a scheduler type.

    Args:
        name: one of the values returned by :func:`available_schedulers`.

    Returns:
        :class:`SchedulerInfo` with description, valid samplers, and kwargs.

    Raises:
        KeyError: if ``name`` is not a known scheduler type.

    Example::

        from medlat.scheduling import scheduler_info
        info = scheduler_info("diffusion")
        print(info.samplers)        # ['ddpm', 'ddim']
        print(info.optional_kwargs) # {'steps': ('int', 1000, '...'), ...}
    """
    key = name.strip().lower()
    if key not in _SCHEDULER_CATALOG:
        raise KeyError(
            f"Unknown scheduler '{name}'. "
            f"Available: {available_schedulers()}"
        )
    return _SCHEDULER_CATALOG[key]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_scheduler(scheduler_type: str = "diffusion", **kwargs: Any) -> BaseScheduler:
    """Instantiate a generative scheduler by type.

    Args:
        scheduler_type: ``'diffusion'``, ``'flow'`` (alias ``'transport'``), or
                        ``'self_flow'``.  Call :func:`available_schedulers` to
                        list all options, or :func:`scheduler_info` for kwargs.
        **kwargs:       Forwarded verbatim to the scheduler factory.

    Returns:
        A :class:`~medlat.scheduling.base.BaseScheduler`-compatible object.
    """
    key = scheduler_type.strip().lower()
    if key == "diffusion":
        return create_gaussian_diffusion(**kwargs)
    elif key in ("flow", "transport"):
        return create_transport(**kwargs)
    elif key == "self_flow":
        return create_dual_timestep_scheduler(**kwargs)
    else:
        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}'. "
            f"Available: {available_schedulers()}"
        )


__all__ = [
    # Base / concrete classes
    "BaseScheduler",
    "GaussianDiffusionScheduler",
    "FlowMatchingScheduler",
    "DualTimestepScheduler",
    # Factories
    "create_gaussian_diffusion",
    "create_transport",
    "create_dual_timestep_scheduler",
    "create_scheduler",
    # Discovery
    "SchedulerInfo",
    "available_schedulers",
    "scheduler_info",
    # Flow enums
    "Transport",
    "Sampler",
    "ModelType",
    "WeightType",
    "PathType",
]
