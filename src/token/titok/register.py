import torch
from src.registry import register_model
from .titok import TiTok

from typing import Optional
from src.discrete.vq_models import VQModel

@register_model("token.titok.s_128")
def TiTok_S_128(image_size: int | tuple[int, ...],
                patch_size: int | tuple[int, ...], 
                hidden_size=512,
                in_channels=3,
                out_channels=3,
                depth=8,
                num_heads=8,
                num_latent_tokens=128,
                token_size=12,
                codebook_size=4096,
                quantizer_loss_weight=1.0,
                pixel_vqgan: Optional[VQModel] = None, # VQModel = PretrainedTokenizer("/vol/miltank/users/bubeckn/1d-tokenizer/maskgit-vqgan-imagenet-f16-256.bin"), 
                stage="1", 
                quantize_mode="vq",
                **kwargs):
    return TiTok(image_size=image_size, patch_size=patch_size, hidden_size=hidden_size, in_channels=in_channels, out_channels=out_channels, depth=depth, num_heads=num_heads, num_latent_tokens=num_latent_tokens, token_size=token_size, codebook_size=codebook_size, quantizer_loss_weight=quantizer_loss_weight, pixel_vqgan=pixel_vqgan, stage=stage, quantize_mode=quantize_mode, **kwargs)

@register_model("token.titok.b_64")
def TiTok_B_64(image_size: int | tuple[int, ...],
                patch_size: int | tuple[int, ...], 
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
    return TiTok(image_size=image_size, patch_size=patch_size, hidden_size=hidden_size, in_channels=in_channels, out_channels=out_channels, depth=depth, num_heads=num_heads, num_latent_tokens=num_latent_tokens, token_size=token_size, codebook_size=codebook_size, quantizer_loss_weight=quantizer_loss_weight, pixel_vqgan=pixel_vqgan, stage=stage, quantize_mode=quantize_mode, **kwargs)

@register_model("token.titok.l_32")
def TiTok_L_32(image_size: int | tuple[int, ...],
                patch_size: int | tuple[int, ...], 
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
    return TiTok(image_size=image_size, patch_size=patch_size, hidden_size=hidden_size, in_channels=in_channels, out_channels=out_channels, depth=depth, num_heads=num_heads, num_latent_tokens=num_latent_tokens, token_size=token_size, codebook_size=codebook_size, quantizer_loss_weight=quantizer_loss_weight, pixel_vqgan=pixel_vqgan, stage=stage, quantize_mode=quantize_mode, **kwargs)

