import torch
from medtok.registry import register_model
from .titok import TiTok

from typing import Optional
from medtok.first_stage.discrete.vq_models import VQModel



###### Configs #######

"""
- S_128: 512, 8, 8, 128
- S_256: 512, 8, 8, 256
- S_512: 512, 8, 8, 512
- B_64: 768, 12, 12, 64
- B_128: 768, 12, 12, 128
- B_256: 768, 12, 12, 256
- B_512: 768, 12, 12, 512
- L_32: 1024, 24, 16, 32
- L_64: 1024, 24, 16, 64
- L_128: 1024, 24, 16, 128
- L_256: 1024, 24, 16, 256
- L_512: 1024, 24, 16, 512
"""


@register_model("token.titok.s_128")
def TiTok_S_128(
    img_size: int | tuple[int, ...],
    patch_size: int | tuple[int, ...] = 16, 
    hidden_size=512,
    in_channels=3,
    out_channels=3,
    depth=8,
    num_heads=8,
    num_latent_tokens=128,
    token_size=12,
    codebook_size=4096,
    quantizer_loss_weight=1.0,
    pixel_vqgan: Optional[VQModel] = None, # VQModel = PretrainedTokenizer("/vol/miltank/users/bubeckn/1d-tokenizer/maskgit-vqgan-imagenet-f16-256.bin")
    stage="1", 
    quantize_mode="vq",
    **kwargs,
):
    return TiTok(
        img_size=img_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        num_heads=num_heads,
        num_latent_tokens=num_latent_tokens,
        token_size=token_size,
        codebook_size=codebook_size,
        quantizer_loss_weight=quantizer_loss_weight,
        pixel_vqgan=pixel_vqgan,
        stage=stage,
        quantize_mode=quantize_mode,
        **kwargs,
    )

@register_model("token.titok.s_128_e2e")
def TiTok_S_128_E2E(**kwargs):
    return TiTok_S_128(num_latent_tokens=128, patch_size=8, stage="e2e", **kwargs)

@register_model("token.titok.s_256_e2e")
def TiTok_S_256_E2E(**kwargs):
    return TiTok_S_128(num_latent_tokens=256, patch_size=8, stage="e2e", **kwargs)

@register_model("token.titok.s_512_e2e")
def TiTok_S_512_E2E(**kwargs):
    return TiTok_S_128(num_latent_tokens=512, patch_size=8, stage="e2e", **kwargs)


@register_model("token.titok.b_64")
def TiTok_B_64(img_size: int | tuple[int, ...],
                patch_size: int | tuple[int, ...] = 16, 
                hidden_size=768,
                in_channels=3,
                out_channels=3,
                depth=12,
                num_heads=12,
                num_latent_tokens=64,
                token_size=12,
                codebook_size=4096,
                quantizer_loss_weight=1.0,
                pixel_vqgan: Optional[VQModel] = None, # VQModel = PretrainedTokenizer("/vol/miltank/users/bubeckn/1d-tokenizer/maskgit-vqgan-imagenet-f16-256.bin"), 
                stage="1", 
                quantize_mode="vq",
                **kwargs):
    return TiTok(img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, in_channels=in_channels, out_channels=out_channels, depth=depth, num_heads=num_heads, num_latent_tokens=num_latent_tokens, token_size=token_size, codebook_size=codebook_size, quantizer_loss_weight=quantizer_loss_weight, pixel_vqgan=pixel_vqgan, stage=stage, quantize_mode=quantize_mode, **kwargs)

@register_model("token.titok.b_128_p8_e2e")
def TiTok_B_128_p8_E2E(**kwargs):
    return TiTok_B_64(num_latent_tokens=128, patch_size=8, stage="e2e", **kwargs)

@register_model("token.titok.b_256_p8_e2e")
def TiTok_B_256_p8_E2E(**kwargs):
    return TiTok_B_64(num_latent_tokens=256, patch_size=8, stage="e2e", **kwargs)

@register_model("token.titok.b_512_p8_e2e")
def TiTok_B_512_p8_E2E(**kwargs):
    return TiTok_B_64(num_latent_tokens=512, patch_size=8, stage="e2e", **kwargs)

@register_model("token.titok.l_32")
def TiTok_L_32(img_size: int | tuple[int, ...],
                patch_size: int | tuple[int, ...] = 16, 
                hidden_size=1024,
                in_channels=3,
                out_channels=3,
                depth=24,
                num_heads=16,
                num_latent_tokens=32,
                token_size=12,
                codebook_size=4096,
                quantizer_loss_weight=1.0,
                pixel_vqgan: Optional[VQModel] = None, # VQModel = PretrainedTokenizer("/vol/miltank/users/bubeckn/1d-tokenizer/maskgit-vqgan-imagenet-f16-256.bin"), 
                stage="1", 
                quantize_mode="vq",
                **kwargs):
    return TiTok(img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, in_channels=in_channels, out_channels=out_channels, depth=depth, num_heads=num_heads, num_latent_tokens=num_latent_tokens, token_size=token_size, codebook_size=codebook_size, quantizer_loss_weight=quantizer_loss_weight, pixel_vqgan=pixel_vqgan, stage=stage, quantize_mode=quantize_mode, **kwargs)

@register_model("token.titok.l_64_e2e")
def TiTok_L_64_E2E(**kwargs):
    return TiTok_L_32(num_latent_tokens=64, stage="e2e", **kwargs)

@register_model("token.titok.l_128_e2e")
def TiTok_L_128_E2E(**kwargs):
    return TiTok_L_32(num_latent_tokens=128, stage="e2e", **kwargs)

@register_model("token.titok.l_256_e2e")
def TiTok_L_256_E2E(**kwargs):
    return TiTok_L_32(num_latent_tokens=256, stage="e2e", **kwargs)

@register_model("token.titok.l_512_e2e")
def TiTok_L_512_E2E(**kwargs):
    return TiTok_L_32(num_latent_tokens=512, stage="e2e", **kwargs)