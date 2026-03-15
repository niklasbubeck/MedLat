from medlat.registry import register_model
from .models import MaskedAutoencoderViT


@register_model("token.vmae.t_p8_d16")
def VMAE_T_P8_D16(**kwargs):
    """Tiny VMAE, patch size 8, latent dim 16 (enc/dec 96d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=96,
        depth=12,
        num_heads=8,
        decoder_embed_dim=96,
        decoder_depth=12,
        decoder_num_heads=8,
        mlp_ratio=4,
        latent_dim=16,
        **kwargs,
    )


@register_model("token.vmae.t_p8_d16_asym")
def VMAE_T_P8_D16_ASYM(**kwargs):
    """Tiny-asymmetric VMAE, patch size 8, latent dim 16 (enc 96d, dec 192d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=96,
        depth=12,
        num_heads=8,
        decoder_embed_dim=192,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        latent_dim=16,
        **kwargs,
    )


@register_model("token.vmae.s_p8_d32")
def VMAE_S_P8_D32(**kwargs):
    """Small VMAE, patch size 8, latent dim 32 (enc/dec 192d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=192,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        latent_dim=32,
        **kwargs,
    )


@register_model("token.vmae.s_p8_d32_flex")
def VMAE_S_P8_D32_FLEX(**kwargs):
    """Flexible small VMAE, patch size 8, latent dim 32 (enc/dec 192d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=192,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        latent_dim=32,
        **kwargs,
    )


@register_model("token.vmae.s_p8_d16")
def VMAE_S_P8_D16(**kwargs):
    """Small VMAE, patch size 8, latent dim 16 (enc 192d, dec 384d, MLP down)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=12,
        decoder_num_heads=24,
        mlp_ratio=4,
        latent_dim=16,
        down_nonlinear=True,
        **kwargs,
    )


@register_model("token.vmae.s_p8_d16_prev")
def VMAE_S_P8_D16_PREV(**kwargs):
    """Previous small VMAE variant, patch size 8, latent dim 16 (enc/dec 192d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=192,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        latent_dim=16,
        **kwargs,
    )


@register_model("token.vmae.s_p8_d16_flex")
def VMAE_S_P8_D16_FLEX(**kwargs):
    """Flexible small VMAE, patch size 8, latent dim 16 (enc 192d, dec 384d, MLP down)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=12,
        decoder_num_heads=24,
        mlp_ratio=4,
        latent_dim=16,
        down_nonlinear=True,
        **kwargs,
    )


@register_model("token.vmae.s_p8_d16_alt")
def VMAE_S_P8_D16_ALT(**kwargs):
    """Alternative small VMAE config, patch size 8, latent dim 16 (enc/dec 192d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=192,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        latent_dim=16,
        **kwargs,
    )


@register_model("token.vmae.b_p8_d16_prev")
def VMAE_B_P8_D16_PREV(**kwargs):
    """Base VMAE (previous large variant), patch size 8, latent dim 16 (enc/dec 384d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=16,
        decoder_embed_dim=384,
        decoder_depth=12,
        decoder_num_heads=16,
        mlp_ratio=4,
        latent_dim=16,
        **kwargs,
    )


@register_model("token.vmae.s_p16_d32")
def VMAE_S_P16_D32(**kwargs):
    """Small VMAE, patch size 16, latent dim 32 (enc/dec 192d)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=192,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        latent_dim=32,
        **kwargs,
    )


@register_model("token.vmae.b_p16_d32")
def VMAE_B_P16_D32(**kwargs):
    """Base VMAE, patch size 16, latent dim 32 (enc/dec 384d, downsizing at layer 4)."""
    return MaskedAutoencoderViT(
        img_size=128,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        latent_dim=32,
        finetune_downsample_layer=4,
        **kwargs,
    )
