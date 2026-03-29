"""
SiT: Scalable Interpolant Transformers
Paper: https://arxiv.org/abs/2401.08740
"Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers"
Ma et al., 2024

SiT uses the same DiT architecture but is designed for flow matching / stochastic interpolants
rather than DDPM. The model predicts a velocity (or noise) field with learn_sigma=False,
making it compatible with ODE/SDE-based samplers (e.g. DPM-Solver, DDIM, Euler).

Integration note: SiT models are drop-in replacements for DiT in GenWrapper.
Switch your noise scheduler / loss to flow matching to get the full benefit.
"""

from medlat.registry import register_model
from medlat.generators.non_autoregressive.dit.models import DiT

# ---------------------------------------------------------------------------
# XL variants  (depth=28, hidden=1152, heads=16)
# ---------------------------------------------------------------------------

@register_model(
    "sit.xl_1",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-XL with patch size 1 — flow-matching DiT, no sigma head",
)
def SiT_XL_1(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@register_model(
    "sit.xl_2",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-XL with patch size 2 — flow-matching DiT, no sigma head",
)
def SiT_XL_2(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


@register_model(
    "sit.xl_4",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-XL with patch size 4 — flow-matching DiT, no sigma head",
)
def SiT_XL_4(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


@register_model(
    "sit.xl_8",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-XL with patch size 8 — flow-matching DiT, no sigma head",
)
def SiT_XL_8(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


# ---------------------------------------------------------------------------
# L variants  (depth=24, hidden=1024, heads=16)
# ---------------------------------------------------------------------------

@register_model(
    "sit.l_1",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-L with patch size 1",
)
def SiT_L_1(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)


@register_model(
    "sit.l_2",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-L with patch size 2",
)
def SiT_L_2(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


@register_model(
    "sit.l_4",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-L with patch size 4",
)
def SiT_L_4(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


@register_model(
    "sit.l_8",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-L with patch size 8",
)
def SiT_L_8(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


# ---------------------------------------------------------------------------
# B variants  (depth=12, hidden=768, heads=12)
# ---------------------------------------------------------------------------

@register_model(
    "sit.b_1",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-B with patch size 1",
)
def SiT_B_1(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)


@register_model(
    "sit.b_2",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-B with patch size 2",
)
def SiT_B_2(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


@register_model(
    "sit.b_4",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-B with patch size 4",
)
def SiT_B_4(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


@register_model(
    "sit.b_8",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-B with patch size 8",
)
def SiT_B_8(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


# ---------------------------------------------------------------------------
# S variants  (depth=12, hidden=384, heads=6)
# ---------------------------------------------------------------------------

@register_model(
    "sit.s_1",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-S with patch size 1",
)
def SiT_S_1(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)


@register_model(
    "sit.s_2",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-S with patch size 2",
)
def SiT_S_2(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


@register_model(
    "sit.s_4",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-S with patch size 4",
)
def SiT_S_4(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


@register_model(
    "sit.s_8",
    paper_url="https://arxiv.org/abs/2401.08740",
    description="SiT-S with patch size 8",
)
def SiT_S_8(**kwargs):
    kwargs.setdefault("learn_sigma", False)
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
