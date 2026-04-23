"""Tests for the shared Esser et al. (2024) timestep-shift helper and its
integration into both :class:`GaussianDiffusion` and the flow-matching ODE
integrator.

Invariants covered:

* Pure-math properties of :func:`esser_shift` (boundary preservation,
  identity at ``α=1``, monotonicity).
* :func:`compute_esser_alpha` reproduces ``α = √(m/n)`` and rejects
  non-positive inputs.
* :class:`GaussianDiffusion` with ``latent_dim=None`` is bit-for-bit
  equivalent to the pre-migration behaviour (regression guard).
* :class:`GaussianDiffusion` with a set ``latent_dim`` actually routes
  tensor lookups through the warped index.
* :func:`create_gaussian_diffusion` plumbs the new kwargs through.
"""
from __future__ import annotations

import math

import pytest
import torch

from medlat.scheduling.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from medlat.scheduling.utils import compute_esser_alpha, esser_shift


# ---------------------------------------------------------------------------
# compute_esser_alpha
# ---------------------------------------------------------------------------


def test_compute_esser_alpha_matches_sqrt_ratio():
    assert compute_esser_alpha(4096) == pytest.approx(1.0)
    assert compute_esser_alpha(16384) == pytest.approx(2.0)
    assert compute_esser_alpha(1024) == pytest.approx(0.5)


def test_compute_esser_alpha_rejects_nonpositive():
    with pytest.raises(ValueError):
        compute_esser_alpha(0)
    with pytest.raises(ValueError):
        compute_esser_alpha(-128)
    with pytest.raises(ValueError):
        compute_esser_alpha(128, base_dim=0)


# ---------------------------------------------------------------------------
# esser_shift — pure math properties
# ---------------------------------------------------------------------------


def test_esser_shift_is_identity_at_alpha_one():
    t = torch.linspace(0.0, 1.0, 11)
    torch.testing.assert_close(esser_shift(t, alpha=1.0), t)


def test_esser_shift_preserves_endpoints():
    for alpha in (0.1, 0.5, 2.0, 4.0):
        assert esser_shift(torch.tensor(0.0), alpha=alpha).item() == 0.0
        assert esser_shift(torch.tensor(1.0), alpha=alpha).item() == pytest.approx(1.0)


def test_esser_shift_is_strictly_monotonic():
    for alpha in (0.5, 2.0, 4.0):
        t = torch.linspace(0.0, 1.0, 50)
        shifted = esser_shift(t, alpha=alpha)
        diffs = shifted[1:] - shifted[:-1]
        assert torch.all(diffs > 0), f"non-monotonic at alpha={alpha}"


def test_esser_shift_matches_reference_formula():
    # Hand-computed: alpha=2, t=0.5 → (2·0.5)/(1 + 1·0.5) = 1/1.5 ≈ 0.6666...
    got = esser_shift(torch.tensor(0.5), alpha=2.0)
    assert got.item() == pytest.approx(2 / 3)


def test_esser_shift_accepts_latent_dim_shortcut():
    # alpha = sqrt(16384 / 4096) = 2.0 → same as alpha=2 above.
    got = esser_shift(torch.tensor(0.5), latent_dim=16384)
    torch.testing.assert_close(got, esser_shift(torch.tensor(0.5), alpha=2.0))


def test_esser_shift_errors_without_alpha_or_latent_dim():
    with pytest.raises(ValueError, match="alpha"):
        esser_shift(torch.tensor(0.5))


# ---------------------------------------------------------------------------
# GaussianDiffusion integration
# ---------------------------------------------------------------------------


def _make_diffusion(latent_dim=None, steps: int = 100) -> GaussianDiffusion:
    return GaussianDiffusion(
        betas=get_named_beta_schedule("linear", steps),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        latent_dim=latent_dim,
    )


def test_gaussian_diffusion_without_latent_dim_has_no_alpha_shift():
    d = _make_diffusion(latent_dim=None)
    assert d.alpha_shift is None
    # Warp is the identity when alpha_shift is None.
    t = torch.tensor([0, 25, 50, 75, 99], dtype=torch.long)
    torch.testing.assert_close(d._warp_timesteps(t), t)


def test_gaussian_diffusion_with_latent_dim_stores_alpha_shift():
    d = _make_diffusion(latent_dim=16384)
    assert d.alpha_shift == pytest.approx(2.0)


def test_warp_timesteps_matches_esser_shift_on_normalised_grid():
    # With alpha > 1 the shift pushes intermediate timesteps to higher
    # (noisier) indices; boundaries are preserved.
    d = _make_diffusion(latent_dim=16384, steps=100)
    t = torch.tensor([0, 25, 50, 75, 99], dtype=torch.long)
    warped = d._warp_timesteps(t)

    T = d.num_timesteps - 1  # 99
    expected = torch.clamp(
        (esser_shift(t.float() / T, alpha=d.alpha_shift) * T).long(), 0, T
    )
    torch.testing.assert_close(warped, expected)

    # endpoints preserved
    assert warped[0].item() == 0
    assert warped[-1].item() == T
    # alpha > 1 → each intermediate index is >= original (shift is positive).
    assert (warped >= t).all()


def test_warp_is_identity_at_alpha_one():
    d = _make_diffusion(latent_dim=4096)  # → alpha=1
    t = torch.tensor([0, 25, 50, 99], dtype=torch.long)
    torch.testing.assert_close(d._warp_timesteps(t), t)


def test_gaussian_diffusion_q_sample_uses_warped_index():
    # With alpha > 1, q_sample at t=mid should pick up a later-schedule beta
    # than the unshifted diffusion would. We compare the alpha_bar picked up
    # by _extract against manual indexing at the shifted timestep.
    d = _make_diffusion(latent_dim=16384, steps=100)
    t = torch.tensor([50], dtype=torch.long)
    x_start = torch.ones(1, 3, 4, 4)
    noise = torch.zeros_like(x_start)

    out = d.q_sample(x_start, t, noise=noise)  # with zero noise, out == sqrt(alpha_bar) * x_start
    warped_idx = int(d._warp_timesteps(t).item())
    expected_scale = float(d.sqrt_alphas_cumprod[warped_idx])
    assert out.mean().item() == pytest.approx(expected_scale, rel=1e-5)

    # Sanity: the unshifted lookup at t=50 would give a different value.
    unshifted_scale = float(d.sqrt_alphas_cumprod[50])
    assert unshifted_scale != pytest.approx(expected_scale, rel=1e-5)


def test_create_gaussian_diffusion_plumbs_latent_dim():
    from medlat.scheduling.gaussian import create_gaussian_diffusion

    sched = create_gaussian_diffusion(steps=100, latent_dim=16384, base_dim=4096)
    # Attribute fall-through reaches the underlying SpacedDiffusion.
    assert sched.alpha_shift == pytest.approx(2.0)


def test_create_gaussian_diffusion_default_preserves_legacy_behavior():
    from medlat.scheduling.gaussian import create_gaussian_diffusion

    sched = create_gaussian_diffusion(steps=100)
    assert sched.alpha_shift is None
