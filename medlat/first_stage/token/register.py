import torch
from medlat.registry import register_model
from .titok.titok import TiTok
from .maetok.modules.vit_models import MAETokViTEncoder, MAETokViTDecoder
from .maetok.maetok import MaskAEModel

from typing import Optional
from medlat.first_stage.discrete.vq_models import VQModel
from medlat.first_stage.discrete.quantizer.quantize import SoftVectorQuantizer
from medlat.modules.alignments import *


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


###### MAETOK #######
@register_model("token.maetok.s_128")
def MAETok_S_128(img_size: int = 256, 
                 patch_size: int = 16,
                 codebook_embed_dim: int = 32,
                 num_latent_tokens: int = 128,
                 enc_embed_dim: int = 512,
                 dec_embed_dim: int = 512,
                 enc_depth: int = 8,
                 dec_depth: int = 8,
                 enc_num_heads: int = 8,
                 dec_num_heads: int = 8,
                 aux_hog_dec: bool = True,
                 aux_dino_dec: bool = True,
                 aux_clip_dec: bool = True,
                 aux_biomed_clip_dec: bool = False, **kwargs):  
    return MaskAEModel(img_size=img_size, base_img_size=img_size, patch_size=patch_size, codebook_embed_dim=codebook_embed_dim, num_latent_tokens=num_latent_tokens, enc_embed_dim=enc_embed_dim, dec_embed_dim=dec_embed_dim, enc_depth=enc_depth, dec_depth=dec_depth, enc_num_heads=enc_num_heads, dec_num_heads=dec_num_heads, aux_hog_dec=aux_hog_dec, aux_dino_dec=aux_dino_dec, aux_clip_dec=aux_clip_dec, aux_biomed_clip_dec=aux_biomed_clip_dec, **kwargs)


@register_model("token.maetok.s_256")
def MAETok_S_256(**kwargs):
    return MAETok_S_128(num_latent_tokens=256, **kwargs)

@register_model("token.maetok.s_512")
def MAETok_S_512(**kwargs):
    return MAETok_S_128(num_latent_tokens=512, **kwargs)


@register_model("token.maetok.b_128_p16",
                code_url="https://github.com/Hhhhhhao/continuous_tokenizer",
                paper_url="https://arxiv.org/pdf/2502.03444",)
def MAETok_B_128(img_size: int = 256, 
                 patch_size: int = 16,
                 codebook_embed_dim: int = 32,
                 num_latent_tokens: int = 128,
                 aux_hog_dec: bool = True,
                 aux_dino_dec: bool = True,
                 aux_clip_dec: bool = True,
                 aux_biomed_clip_dec: bool = False,
                 **kwargs):
    return MaskAEModel(img_size=img_size, base_img_size=img_size, patch_size=patch_size, codebook_embed_dim=codebook_embed_dim, num_latent_tokens=num_latent_tokens, aux_hog_dec=aux_hog_dec, aux_dino_dec=aux_dino_dec, aux_clip_dec=aux_clip_dec, aux_biomed_clip_dec=aux_biomed_clip_dec, **kwargs)

@register_model("token.maetok.b_128_p8")
def MAETok_B_128_p8(**kwargs):
    return MAETok_B_128(num_latent_tokens=128, patch_size=8, **kwargs)

@register_model("token.maetok.b_256_p8")
def MAETok_B_256_p8(**kwargs):
    return MAETok_B_128(num_latent_tokens=256, patch_size=8, **kwargs)

@register_model("token.maetok.b_512_p8")
def MAETok_B_512_p8(**kwargs):
    return MAETok_B_128(num_latent_tokens=512, patch_size=8, **kwargs)


@register_model("token.maetok.l_128")
def MAETok_L_128(img_size: int = 256, 
                 patch_size: int = 16,
                 codebook_embed_dim: int = 32,
                 num_latent_tokens: int = 128,
                 enc_embed_dim: int = 1024,
                 dec_embed_dim: int = 1024,
                 enc_depth: int = 24,
                 dec_depth: int = 24,
                 enc_num_heads: int = 16,
                 dec_num_heads: int = 16,
                 aux_hog_dec: bool = True,
                 aux_dino_dec: bool = True,
                 aux_clip_dec: bool = True,
                 aux_biomed_clip_dec: bool = False, **kwargs):  
    return MaskAEModel(img_size=img_size, base_img_size=img_size, patch_size=patch_size, codebook_embed_dim=codebook_embed_dim, num_latent_tokens=num_latent_tokens, enc_embed_dim=enc_embed_dim, dec_embed_dim=dec_embed_dim, enc_depth=enc_depth, dec_depth=dec_depth, enc_num_heads=enc_num_heads, dec_num_heads=dec_num_heads, aux_hog_dec=aux_hog_dec, aux_dino_dec=aux_dino_dec, aux_clip_dec=aux_clip_dec, aux_biomed_clip_dec=aux_biomed_clip_dec, **kwargs)

@register_model("token.maetok.l_256")
def MAETok_L_256(**kwargs):
    return MAETok_L_128(num_latent_tokens=256, **kwargs)

@register_model("token.maetok.l_512")
def MAETok_L_512(**kwargs):
    return MAETok_L_128(num_latent_tokens=512, **kwargs)



###### SoftVQ #######

@register_model("token.softvq.s_t32_d32")
def SoftVQ_S_T32_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 32,
    embed_dim_encoder: int = 384,
    embed_dim_decoder: int = 384,
    depth_encoder: int = 12,
    depth_decoder: int = 12,
    num_heads_encoder: int = 6,
    num_heads_decoder: int = 6,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )
    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, pre_post_layer="linear", **kwargs)

@register_model("token.softvq.s_t64_d32")
def SoftVQ_S_T64_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 64,
    embed_dim_encoder: int = 384,
    embed_dim_decoder: int = 384,
    depth_encoder: int = 12,
    depth_decoder: int = 12,
    num_heads_encoder: int = 6,
    num_heads_decoder: int = 6,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )
    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, pre_post_layer="linear", **kwargs)


@register_model("token.softvq.b_t32_d32")
def SoftVQ_B_T32_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 32,
    embed_dim_encoder: int = 768,
    embed_dim_decoder: int = 768,
    depth_encoder: int = 12,
    depth_decoder: int = 12,
    num_heads_encoder: int = 12,
    num_heads_decoder: int = 12,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )
    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, pre_post_layer="linear", **kwargs)

@register_model("token.softvq.b_t64_d32")
def SoftVQ_B_T64_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 64,
    embed_dim_encoder: int = 768,
    embed_dim_decoder: int = 768,
    depth_encoder: int = 12,
    depth_decoder: int = 12,
    num_heads_encoder: int = 12,
    num_heads_decoder: int = 12,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )
    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, pre_post_layer="linear", **kwargs)

@register_model("token.softvq.bl_t32_d32")
def SoftVQ_BL_T32_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 32,
    embed_dim_encoder: int = 768,
    embed_dim_decoder: int = 1024,
    depth_encoder: int = 12,
    depth_decoder: int = 24,
    num_heads_encoder: int = 12,
    num_heads_decoder: int = 16,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )
    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, pre_post_layer="linear", **kwargs)

@register_model("token.softvq.bl_t64_d32")
def SoftVQ_BL_T64_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 64,
    embed_dim_encoder: int = 768,
    embed_dim_decoder: int = 1024,
    depth_encoder: int = 12,
    depth_decoder: int = 24,
    num_heads_encoder: int = 12,
    num_heads_decoder: int = 16,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )
    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, **kwargs)

@register_model("token.softvq.l_t32_d32")
def SoftVQ_L_T32_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 32,
    embed_dim_encoder: int = 1024,
    embed_dim_decoder: int = 1024,
    depth_encoder: int = 24,
    depth_decoder: int = 24,
    num_heads_encoder: int = 16,
    num_heads_decoder: int = 16,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )
    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, pre_post_layer="linear", **kwargs)

@register_model("token.softvq.l_t64_d32")
def SoftVQ_L_T64_D32(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    num_latent_tokens: int = 64,
    embed_dim_encoder: int = 1024,
    embed_dim_decoder: int = 1024,
    depth_encoder: int = 24,
    depth_decoder: int = 24,
    num_heads_encoder: int = 16,
    num_heads_decoder: int = 16,
    mlp_ratio: float = 4.0,
    # quantizer config
    n_e: int = 8192,
    e_dim: int = 32,
    entropy_loss_weight: float = 0.01,
    entropy_loss_temperature: float = 0.01,
    entropy_gamma: float = 1.0,
    tau: float = 0.07,
    use_norm: bool = True,
    # alignment config
    foundation_type: str = "dinov2",
    **kwargs
    ):

    encoder = MAETokViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
    )
    decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
    )

    aux_dino_decoder = MAETokViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_latent_tokens=num_latent_tokens,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=mlp_ratio,
        codebook_embed_dim=e_dim,
        to_pixel='identity',
    )

    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    alignment = DinoAlignment(decoder=aux_dino_decoder, codebook_embed_dim=e_dim, img_size=img_size)
    return VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer, alignment=alignment, pre_post_layer="linear", **kwargs)

