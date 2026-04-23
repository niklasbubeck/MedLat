import math
from typing import Optional, Union

import torch as th


__all__ = [
    "EasyDict",
    "compute_esser_alpha",
    "esser_shift",
    "log_state",
    "mean_flat",
]


def compute_esser_alpha(latent_dim: int, base_dim: int = 4096) -> float:
    """Dimension-dependent timestep-shift factor from Esser et al. (2024).

    α = √(m / n), where ``m`` is the effective data dimension the model is
    operating in and ``n`` is a reference dimension (the paper uses 4096 as
    the baseline).

    Args:
        latent_dim: effective dimension ``m`` of the data the scheduler sees
            (for an image latent, this is typically ``C · H · W``).
        base_dim: reference dimension ``n``; defaults to 4096 per the paper.

    Returns:
        Scalar ``α > 0`` suitable to pass as ``alpha=...`` to :func:`esser_shift`.
    """
    if latent_dim <= 0 or base_dim <= 0:
        raise ValueError(
            f"latent_dim={latent_dim} and base_dim={base_dim} must both be positive"
        )
    return math.sqrt(latent_dim / base_dim)


def esser_shift(
    t: Union[th.Tensor, float],
    alpha: Optional[float] = None,
    *,
    latent_dim: Optional[int] = None,
    base_dim: int = 4096,
) -> Union[th.Tensor, float]:
    """Apply the Esser et al. (2024) timestep-shift transform.

    .. math::
       t' = \\frac{\\alpha \\, t}{1 + (\\alpha - 1) \\, t}

    where ``α = √(m / n)`` scales the shift with the effective data
    dimension. Higher ``α`` (larger data) biases the distribution toward
    later (noisier) timesteps; ``α = 1`` is the identity.

    Inputs and outputs are both in ``[0, 1]``; endpoints ``t=0`` and ``t=1``
    are preserved for any ``α``.

    Args:
        t: input time(s) in ``[0, 1]``. Torch tensor or Python scalar.
        alpha: shift factor. If omitted, must be derivable from
            ``latent_dim`` / ``base_dim``.
        latent_dim: convenience — when given, ``α`` is computed via
            :func:`compute_esser_alpha` and ``alpha`` is ignored.
        base_dim: paired with ``latent_dim``; defaults to 4096.

    Returns:
        Shifted time, same type and shape as ``t``.
    """
    if latent_dim is not None:
        alpha = compute_esser_alpha(latent_dim, base_dim)
    if alpha is None:
        raise ValueError(
            "esser_shift needs either `alpha` or `latent_dim` to determine the shift factor."
        )
    return (alpha * t) / (1 + (alpha - 1) * t)


class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))

def log_state(state):
    result = []

    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")

    return '\n'.join(result)
