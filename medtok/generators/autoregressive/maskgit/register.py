import torch 
import torch.nn as nn
from medtok.registry import register_model
from .transformer import MaskGIT
from functools import partial


@register_model("maskgit.b")
def MaskGIT_B(**kwargs):
    model = MaskGIT(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model("maskgit.l")
def MaskGIT_L(**kwargs):
    model = MaskGIT(
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model("maskgit.h")
def MaskGIT_H(**kwargs):
    model = MaskGIT(
        embed_dim=1280, depth=20, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model