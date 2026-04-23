"""Unit tests for medlat.scheduling.path — flow-matching coupling plans.

These plans are pure math (no learnable parameters), so we can verify them
against known boundary conditions and interior identities.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medlat.scheduling.path import (
    GVPCPlan,
    ICPlan,
    VPCPlan,
    expand_t_like_x,
)


BATCH = 4
CHANNELS, H, W = 2, 3, 3


# ---------------------------------------------------------------------------
# expand_t_like_x
# ---------------------------------------------------------------------------


def test_expand_t_like_x_for_4d_data():
    t = torch.arange(BATCH, dtype=torch.float32)
    x = torch.zeros(BATCH, CHANNELS, H, W)
    out = expand_t_like_x(t, x)
    assert out.shape == (BATCH, 1, 1, 1)
    # Batch values preserved
    assert torch.all(out.squeeze() == t)


def test_expand_t_like_x_for_1d_data():
    t = torch.randn(BATCH)
    x = torch.zeros(BATCH)
    out = expand_t_like_x(t, x)
    assert out.shape == (BATCH,)


# ---------------------------------------------------------------------------
# ICPlan (linear coupling)
# ---------------------------------------------------------------------------


def _sample_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(BATCH, CHANNELS, H, W, generator=g)
    x1 = torch.randn(BATCH, CHANNELS, H, W, generator=g)
    return x0, x1


def test_icplan_alpha_sigma_at_t_zero():
    plan = ICPlan()
    t = torch.zeros(BATCH)
    alpha, d_alpha = plan.compute_alpha_t(t)
    sigma, d_sigma = plan.compute_sigma_t(t)
    torch.testing.assert_close(alpha, torch.zeros(BATCH))
    assert d_alpha == 1
    torch.testing.assert_close(sigma, torch.ones(BATCH))
    assert d_sigma == -1


def test_icplan_alpha_sigma_at_t_one():
    plan = ICPlan()
    t = torch.ones(BATCH)
    alpha, _ = plan.compute_alpha_t(t)
    sigma, _ = plan.compute_sigma_t(t)
    torch.testing.assert_close(alpha, torch.ones(BATCH))
    torch.testing.assert_close(sigma, torch.zeros(BATCH))


def test_icplan_plan_at_t_zero_returns_x0():
    plan = ICPlan()
    x0, x1 = _sample_batch()
    t = torch.zeros(BATCH)
    _, xt, _ = plan.plan(t, x0, x1)
    torch.testing.assert_close(xt, x0)


def test_icplan_plan_at_t_one_returns_x1():
    plan = ICPlan()
    x0, x1 = _sample_batch()
    t = torch.ones(BATCH)
    _, xt, _ = plan.plan(t, x0, x1)
    torch.testing.assert_close(xt, x1)


def test_icplan_velocity_is_constant_x1_minus_x0():
    # Linear path: ut(t) = d_alpha_t * x1 + d_sigma_t * x0 = x1 - x0  (independent of t)
    plan = ICPlan()
    x0, x1 = _sample_batch()
    for t_val in (0.0, 0.3, 0.5, 0.8, 1.0):
        t = torch.full((BATCH,), t_val)
        _, _, ut = plan.plan(t, x0, x1)
        torch.testing.assert_close(ut, x1 - x0)


def test_icplan_plan_returns_expected_shapes():
    plan = ICPlan()
    x0, x1 = _sample_batch()
    t = torch.rand(BATCH)
    t_out, xt, ut = plan.plan(t, x0, x1)
    assert t_out.shape == (BATCH,)
    assert xt.shape == x0.shape
    assert ut.shape == x0.shape


@pytest.mark.parametrize(
    "form", ["constant", "SBDM", "sigma", "linear", "decreasing", "inccreasing-decreasing"]
)
def test_icplan_compute_diffusion_valid_forms(form):
    plan = ICPlan()
    x = torch.randn(BATCH, CHANNELS, H, W)
    t = torch.rand(BATCH)
    out = plan.compute_diffusion(x, t, form=form, norm=1.0)
    # Should be a tensor or a scalar-broadcastable value
    assert out is not None


def test_icplan_compute_diffusion_unknown_form_raises():
    plan = ICPlan()
    x = torch.randn(BATCH, CHANNELS, H, W)
    t = torch.rand(BATCH)
    with pytest.raises(NotImplementedError):
        plan.compute_diffusion(x, t, form="nonsense")


# ---------------------------------------------------------------------------
# GVPCPlan (geometric / sine-cosine)
# ---------------------------------------------------------------------------


def test_gvpcplan_alpha_sigma_at_t_zero_and_one():
    plan = GVPCPlan()
    t0 = torch.zeros(BATCH)
    t1 = torch.ones(BATCH)
    alpha0, _ = plan.compute_alpha_t(t0)
    sigma0, _ = plan.compute_sigma_t(t0)
    alpha1, _ = plan.compute_alpha_t(t1)
    sigma1, _ = plan.compute_sigma_t(t1)
    torch.testing.assert_close(alpha0, torch.zeros(BATCH))
    torch.testing.assert_close(sigma0, torch.ones(BATCH))
    torch.testing.assert_close(alpha1, torch.ones(BATCH))
    torch.testing.assert_close(sigma1, torch.zeros(BATCH), atol=1e-6, rtol=1e-6)


def test_gvpcplan_plan_boundary_conditions():
    plan = GVPCPlan()
    x0, x1 = _sample_batch()
    _, xt0, _ = plan.plan(torch.zeros(BATCH), x0, x1)
    _, xt1, _ = plan.plan(torch.ones(BATCH), x0, x1)
    torch.testing.assert_close(xt0, x0)
    torch.testing.assert_close(xt1, x1, atol=1e-6, rtol=1e-6)


def test_gvpcplan_xt_norm_preserved_for_orthogonal_inputs():
    # alpha^2 + sigma^2 = sin^2 + cos^2 = 1, so for x0, x1 orthogonal with
    # unit norms, ||xt||^2 = alpha^2||x1||^2 + sigma^2||x0||^2 = 1.
    plan = GVPCPlan()
    x0 = torch.tensor([1.0, 0.0])
    x1 = torch.tensor([0.0, 1.0])
    for t_val in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = torch.tensor([t_val])
        xt = plan.compute_xt(t, x0[None], x1[None])
        torch.testing.assert_close(
            xt.norm(dim=-1), torch.tensor([1.0]), atol=1e-6, rtol=1e-6
        )


# ---------------------------------------------------------------------------
# VPCPlan (variance-preserving)
# ---------------------------------------------------------------------------


def test_vpcplan_constructs_with_default_sigmas():
    plan = VPCPlan()
    assert plan.sigma_min == 0.1
    assert plan.sigma_max == 20.0


def test_vpcplan_at_t_one_recovers_x1():
    # At t=1: log_mean_coeff(1)=0 → alpha=1, sigma=0 → xt = x1
    plan = VPCPlan()
    x0, x1 = _sample_batch()
    _, xt, _ = plan.plan(torch.ones(BATCH), x0, x1)
    torch.testing.assert_close(xt, x1, atol=1e-5, rtol=1e-5)
