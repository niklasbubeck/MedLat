from .maetok import MaskAEModel
from medtok.registry import register_model


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
