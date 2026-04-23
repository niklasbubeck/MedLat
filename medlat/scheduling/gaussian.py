"""Gaussian diffusion scheduler — public API for the scheduling package."""
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
from .base import BaseScheduler
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


class GaussianDiffusionScheduler(BaseScheduler):
    """Wraps SpacedDiffusion under the BaseScheduler interface.

    Provides training_losses and p_sample_loop with 'ddpm' and 'ddim' samplers.
    All advanced SpacedDiffusion / GaussianDiffusion methods (ddim_sample,
    p_sample_loop_progressive, calc_bpd_loop, etc.) are accessible directly
    via attribute fall-through.
    """

    def __init__(self, diffusion: SpacedDiffusion) -> None:
        self._diffusion = diffusion

    def training_losses(
        self,
        model: Any,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss for one batch.
        Returns a dict with keys 'loss' and optionally 'mse' / 'vb'.
        """
        return self._diffusion.training_losses(
            model, x_start, t=t, model_kwargs=model_kwargs, noise=noise
        )

    def p_sample_loop(
        self,
        model: Any,
        shape: Tuple,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        sampler: str = "ddpm",
        **sampler_kwargs,
    ) -> torch.Tensor:
        """Generate a batch of samples from noise.

        Args:
            sampler: 'ddpm' (default, stochastic) or 'ddim' (deterministic).
                     Pass eta=0.0 to ddim for fully deterministic sampling.
        """
        if sampler == "ddpm":
            return self._diffusion.p_sample_loop(
                model, shape,
                noise=noise, model_kwargs=model_kwargs,
                device=device, progress=progress,
                **sampler_kwargs,
            )
        elif sampler == "ddim":
            return self._diffusion.ddim_sample_loop(
                model, shape,
                noise=noise, model_kwargs=model_kwargs,
                device=device, progress=progress,
                **sampler_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown sampler '{sampler}' for GaussianDiffusionScheduler. "
                "Choose 'ddpm' or 'ddim'."
            )

    def __getattr__(self, name: str):
        """Fall through to SpacedDiffusion for advanced / low-level access."""
        return getattr(self._diffusion, name)


def create_gaussian_diffusion(
    *,
    steps: int = 1000,
    learn_sigma: bool = False,
    sigma_small: bool = False,
    noise_schedule: str = "linear",
    use_kl: bool = False,
    predict_xstart: bool = False,
    rescale_timesteps: bool = False,
    rescale_learned_sigmas: bool = False,
    timestep_respacing: str = "",
    latent_dim: Optional[int] = None,
    base_dim: int = 4096,
) -> GaussianDiffusionScheduler:
    """Factory for GaussianDiffusionScheduler.

    Args:
        steps:                  Total diffusion timesteps.
        learn_sigma:            Whether the model outputs learned variance.
        sigma_small:            Use the smaller fixed variance schedule.
        noise_schedule:         'linear' or 'cosine'.
        use_kl:                 Use rescaled KL loss instead of MSE.
        predict_xstart:         Model predicts x_0 rather than epsilon.
        rescale_timesteps:      Pass timesteps scaled to 0-1000 range.
        rescale_learned_sigmas: Rescale the learned sigmas.
        timestep_respacing:     Comma-separated step counts for respacing,
                                or 'ddimN' for DDIM-style striding.
        latent_dim:             Effective data dimension ``m`` for the
                                dimension-aware Esser et al. (2024) timestep
                                shift. When set, training and sampling use a
                                shifted timestep distribution. ``None``
                                (default) preserves the original schedule.
        base_dim:               Reference dimension ``n`` paired with
                                ``latent_dim``. Defaults to 4096 per the paper.

    Returns:
        GaussianDiffusionScheduler ready for training_losses / p_sample_loop.
    """
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        latent_dim=latent_dim,
        base_dim=base_dim,
    )
    return GaussianDiffusionScheduler(diffusion)
