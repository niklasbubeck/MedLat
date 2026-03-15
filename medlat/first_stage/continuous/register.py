from medlat.first_stage.continuous.modules.ldm_modules import Encoder as LDMEncoder, Decoder as LDMDecoder
from medlat.first_stage.continuous.modules.maisi_modules import MaisiEncoder, MaisiDecoder
from medlat.first_stage.continuous.modules.dcae_modules.dcae_modules import DCAEEncoder, DCAEDecoder
from medlat.registry import register_model
from medlat.first_stage.continuous.vae_models import AutoencoderKL, AutoencoderKLTransformer
from medlat.modules.alignments import *
from medlat.modules.vit_core import GenericViTEncoder, GenericViTDecoder


@register_model(f"continuous.aekl.f4_d3", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f4/config.yaml", 
paper_url="https://arxiv.org/pdf/2112.10752",)
def AEKL_f4_d3(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=3,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 4
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

@register_model(f"continuous.aekl.f4_d8")
def AEKL_f4_d8(**kwargs):
    return AEKL_f4_d3(z_channels=8, **kwargs)

@register_model(f"continuous.aekl.f4_d16")
def AEKL_f4_d16(**kwargs):
    return AEKL_f4_d3(z_channels=16, **kwargs)

@register_model(f"continuous.aekl.f4_d32")
def AEKL_f4_d32(**kwargs):
    return AEKL_f4_d3(z_channels=32, **kwargs)

@register_model(f"continuous.aekl.f8_d4", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f8/config.yaml", 
paper_url="https://arxiv.org/pdf/2112.10752",)
def AEKL_f8_d4(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=4,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 8.
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
        kwargs: Can include 'ddconfig' dict or any ddconfig key directly
    """
    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)


@register_model(f"continuous.aekl.f8_d8")
def AEKL_f8_d8(**kwargs):
    return AEKL_f8_d4(z_channels=8, **kwargs)

@register_model(f"continuous.aekl.f8_d16")
def AEKL_f8_d16(**kwargs):
    return AEKL_f8_d4(z_channels=16, **kwargs)

@register_model(f"continuous.aekl.f8_d32")
def AEKL_f8_d32(**kwargs):
    return AEKL_f8_d4(z_channels=32, **kwargs)

@register_model(f"continuous.aekl.f16_d8", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_16x16x16.yaml", 
paper_url="https://arxiv.org/pdf/2112.10752",
description="ATTENTION: There are two different official configurations with z=8 and z=16 depending on the repo, we use z=8 here.")
def AEKL_f16_d8(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=8,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 16
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

@register_model(f"continuous.aekl.f16_d16")
def AEKL_f16_d16(**kwargs):
    return AEKL_f16_d8(z_channels=16, **kwargs)

@register_model(f"continuous.aekl.f16_d32")
def AEKL_f16_d32(**kwargs):
    return AEKL_f16_d8(z_channels=32, **kwargs)

@register_model(f"continuous.aekl.f16_d64")
def AEKL_f16_d64(**kwargs):
    return AEKL_f16_d8(z_channels=64, **kwargs)

@register_model(f"continuous.aekl.f32_d64", 
code_url="https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f32/config.yaml",
paper_url="https://arxiv.org/pdf/2112.10752",)
def AEKL_f32_d64(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=64,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[16, 8],
    dropout=0.0,
    **kwargs):
    """
    AutoencoderKL with compression factor 32
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

@register_model(f"continuous.maisi.f4_d3", 
code_url="https://github.com/monai/monai/blob/main/monai/networks/nets/maisi_autoencoderkl.py",
paper_url="https://arxiv.org/pdf/2507.05622",)
def Maisi_f4_d4(
    z_channels: int = 4,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_ch: int = 1,
    num_res_blocks: list[int] = [2, 2, 2],
    num_channels: list[int] = [64, 128, 256],
    attention_levels: list[bool] = [False, False, False],
    norm_num_groups: int = 32,
    norm_eps: float = 1e-6,
    with_encoder_nonlocal_attn: bool = False,
    with_decoder_nonlocal_attn: bool = False,
    include_fc: bool = False,
    use_combined_linear: bool = False,
    use_flash_attention: bool = False,
    use_convtranspose: bool = False,
    num_splits: int = 1,
    dim_split: int = 1,
    double_z: bool = True,
    norm_float16: bool = True,
    print_info: bool = False,
    save_mem: bool = True,
    **kwargs):

    encoder = MaisiEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=z_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            double_z=double_z,
            print_info=print_info,
            save_mem=save_mem,
        )

    decoder = MaisiDecoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=z_channels,
            out_channels=out_ch,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
            num_splits=num_splits,
            dim_split=dim_split,
            norm_float16=norm_float16,
            double_z=double_z,
            print_info=print_info,
            save_mem=save_mem,
        )

    return AutoencoderKL(encoder=encoder, decoder=decoder, **kwargs)

@register_model(f"continuous.medvae.f8_d16")
def MedVAE_f8_d16(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=16,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
    **kwargs
):

    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)

    alignment = VFFoundationAlignment(latent_channels=z_channels, foundation_type="biomedclip")
    return AutoencoderKL(encoder=encoder, decoder=decoder, alignment=alignment, **kwargs)


@register_model(f"continuous.medvae.f8_d32")
def MedVAE_f8_d32(**kwargs):
    return MedVAE_f8_d16(z_channels=32, **kwargs)


@register_model(f"continuous.vavae.f8_d32_dinov2")
def VAVAE_f8_d32_dinov2(**kwargs):
    return VAVAE_f8_d16_dinov2(z_channels=32, **kwargs)

@register_model(f"continuous.vavae.f8_d16_dinov2")
def VAVAE_f8_d16_dinov2(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=16,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[],
    dropout=0.0,
    **kwargs
):

    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)

    alignment = VFFoundationAlignment(latent_channels=z_channels, foundation_type="dinov2")
    return AutoencoderKL(encoder=encoder, decoder=decoder, alignment=alignment, **kwargs)


@register_model(f"continuous.vavae.f16_d16_mae",
                code_url="https://github.com/hustvl/LightningDiT/tree/2725fed42a14898744433809949834e26957bcdd",
                paper_url="https://arxiv.org/pdf/2501.01423",)
def VAVAE_f16_d16_mae(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=16,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    """
    VAVAE with compression factor 16
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)

    alignment = VFFoundationAlignment(latent_channels=z_channels, foundation_type="mae")
    return AutoencoderKL(encoder=encoder, decoder=decoder, alignment=alignment, **kwargs)

@register_model(f"continuous.vavae.f16_d32_mae",
                code_url="https://github.com/hustvl/LightningDiT/tree/2725fed42a14898744433809949834e26957bcdd",
                paper_url="https://arxiv.org/pdf/2501.01423",)
def VAVAE_f16_d32_mae(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=32,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    return VAVAE_f16_d16_mae(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout, **kwargs)

@register_model(f"continuous.vavae.f16_d64_mae",
                code_url="https://github.com/hustvl/LightningDiT/tree/2725fed42a14898744433809949834e26957bcdd",
                paper_url="https://arxiv.org/pdf/2501.01423",)
def VAVAE_f16_d64_mae(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=64,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    return VAVAE_f16_d16_mae(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout, **kwargs)

@register_model(f"continuous.vavae.f16_d16_dinov2",
                code_url="https://github.com/hustvl/LightningDiT/tree/2725fed42a14898744433809949834e26957bcdd",
                paper_url="https://arxiv.org/pdf/2501.01423",)
def VAVAE_f16_d16_dinov2(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=16,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    """
    VAVAE with compression factor 16
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = LDMEncoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)
    decoder = LDMDecoder(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout)

    alignment = VFFoundationAlignment(latent_channels=z_channels, foundation_type="dinov2")
    return AutoencoderKL(encoder=encoder, decoder=decoder, alignment=alignment, **kwargs)

@register_model(f"continuous.vavae.f16_d32_dinov2",
                code_url="https://github.com/hustvl/LightningDiT/tree/2725fed42a14898744433809949834e26957bcdd",
                paper_url="https://arxiv.org/pdf/2501.01423",)
def VAVAE_f16_d32_dinov2(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=32,
    in_channels=3,
    out_ch=3,
    ch=128,
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    return VAVAE_f16_d16_dinov2(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout, **kwargs)

@register_model(f"continuous.vavae.f16_d64_dinov2",
                code_url="https://github.com/hustvl/LightningDiT/tree/2725fed42a14898744433809949834e26957bcdd",
                paper_url="https://arxiv.org/pdf/2501.01423",)
def VAVAE_f16_d64_dinov2(
    img_size=256,
    dims=2,
    ## Encoder decoder config
    double_z=True,
    z_channels=64,
    in_channels=3,
    out_ch=3,
    ch=128,
    
    ch_mult=[1, 1, 2, 2, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    **kwargs):
    return VAVAE_f16_d16_dinov2(img_size=img_size, dims=dims, double_z=double_z, z_channels=z_channels, in_channels=in_channels, out_ch=out_ch, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout, **kwargs)


@register_model(
    f"continuous.dcae.f32c32",
    code_url="https://github.com/mit-han-lab/efficientvit",
    paper_url="https://arxiv.org/pdf/2410.10733",
    ckpt_path="https://huggingface.co/mit-han-lab/dc-ae-f32c32-in-1.0/resolve/main/model.safetensors",
)
def DCAE_f32c32(
    img_size=256,
    dims=2,
    in_channels=3,
    z_channels=32,
    width_list=(128, 256, 512, 512, 1024, 1024),
    depth_list=(0, 4, 8, 2, 2, 2),
    block_type=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
    norm="trms2d",
    act="silu",
    decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d"],
    decoder_act=["relu", "relu", "relu", "silu", "silu", "silu"],
    decoder_depth_list=(0, 5, 10, 2, 2, 2),
    double_z=False,
    pre_post_layer="none",
    project_out_conv_only=True,
    **kwargs):
    """
    DCAE with compression factor 32 and 32 latent channels.
    double_z=False, use_quant_conv=False and project_out_conv_only=True match pretrained DCAE checkpoint keys.
    Args:
        dims (int): Number of dimensions (2 for 2D, 3 for 3D)
    """
    encoder = DCAEEncoder(
        in_channels=in_channels,
        z_channels=z_channels,
        width_list=width_list,
        depth_list=depth_list,
        block_type=block_type,
        norm=norm,
        act=act,
        double_z=double_z,
        dims=dims,
        img_size=img_size,
        project_out_conv_only=project_out_conv_only,
    )
    decoder = DCAEDecoder(
        in_channels=in_channels,
        z_channels=z_channels,
        width_list=width_list,
        depth_list=decoder_depth_list,
        block_type=block_type,
        norm=decoder_norm,
        act=decoder_act,
        dims=dims,
        img_size=img_size,
    )
    return AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        use_quant_conv=use_quant_conv,
        double_z=double_z,
        **kwargs,
    )


@register_model(
    f"continuous.dcae.f64c128",
    code_url="https://github.com/mit-han-lab/efficientvit",
    paper_url="https://arxiv.org/pdf/2410.10733",
    ckpt_path="https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0/resolve/main/model.safetensors",
)
def DCAE_f64c128(
    img_size=256,
    dims=2,
    in_channels=3,
    z_channels=128,
    width_list=(128, 256, 512, 512, 1024, 1024, 2048),
    depth_list=(0, 4, 8, 2, 2, 2, 2),
    block_type=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
    norm="trms2d",
    act="silu",
    decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d"],
    decoder_act=["relu", "relu", "relu", "silu", "silu", "silu", "silu"],
    decoder_depth_list=(0, 5, 10, 2, 2, 2, 2),
    double_z=False,
    pre_post_layer="none",
    project_out_conv_only=True,
    **kwargs):
    """
    DCAE with compression factor 64 and 128 latent channels
    """
    encoder = DCAEEncoder(
        in_channels=in_channels,
        z_channels=z_channels,
        width_list=width_list,
        depth_list=depth_list,
        block_type=block_type,
        norm=norm,
        act=act,
        double_z=double_z,
        dims=dims,
        img_size=img_size,
        project_out_conv_only=project_out_conv_only,
    )
    decoder = DCAEDecoder(
        in_channels=in_channels,
        z_channels=z_channels,
        width_list=width_list,
        depth_list=decoder_depth_list,
        block_type=block_type,
        norm=decoder_norm,
        act=decoder_act,
        dims=dims,
        img_size=img_size,
    )
    return AutoencoderKL(
        encoder=encoder, 
        decoder=decoder,
        double_z=double_z,
        use_quant_conv=use_quant_conv,
        **kwargs,
    )


@register_model(
    f"continuous.dcae.f128c512",
    code_url="https://github.com/mit-han-lab/efficientvit",
    paper_url="https://arxiv.org/pdf/2410.10733",
    ckpt_path="https://huggingface.co/mit-han-lab/dc-ae-f128c512-in-1.0/resolve/main/model.safetensors",
)
def DCAE_f128c512(
    img_size=256,
    dims=2,
    in_channels=3,
    z_channels=512,
    width_list=(128, 256, 512, 512, 1024, 1024, 2048, 2048),
    depth_list=(0, 4, 8, 2, 2, 2, 2, 2),
    block_type=["ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU"],
    norm="trms2d",
    act="silu",
    decoder_norm=["bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d", "trms2d"],
    decoder_act=["relu", "relu", "relu", "silu", "silu", "silu", "silu", "silu"],
    decoder_depth_list=(0, 5, 10, 2, 2, 2, 2, 2),
    double_z=False,
    pre_post_layer="none",
    project_out_conv_only=True,
    **kwargs):
    """
    DCAE with compression factor 128 and 512 latent channels
    """
    encoder = DCAEEncoder(
        in_channels=in_channels,
        z_channels=z_channels,
        width_list=width_list,
        depth_list=depth_list,
        block_type=block_type,
        norm=norm,
        act=act,
        double_z=double_z,
        dims=dims,
        img_size=img_size,
        project_out_conv_only=project_out_conv_only,
    )
    decoder = DCAEDecoder(
        in_channels=in_channels,
        z_channels=z_channels,
        width_list=width_list,
        depth_list=decoder_depth_list,
        block_type=block_type,
        norm=decoder_norm,
        act=decoder_act,
        dims=dims,
        img_size=img_size,
        )
    return AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        double_z=double_z,
        use_quant_conv=use_quant_conv,
        **kwargs,
    )


@register_model(f"discrete.hcvq.vae.S_16")
def SDVQ_Transformer_S_16(
    img_size: int = 256,
    patch_size: int = 16,
    in_channels: int = 3,
    embed_dim_encoder: int = 384,
    embed_dim_decoder: int = 384,
    depth_encoder: int = 12,
    depth_decoder: int = 12,
    num_heads_encoder: int = 6,
    num_heads_decoder: int = 6,
    mlp_ratio: float = 4.0,
    mask_ratio_mu: float = 0.0,
    masking: str = "none",
    e_dim: int = 32,
    **kwargs
    ):

    encoder = GenericViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim_encoder,
        depth=depth_encoder,
        num_heads=num_heads_encoder,
        mlp_ratio=mlp_ratio,
        pos_type="learned",
        use_rope=True,
        num_prefix_tokens=1,
        num_latent_tokens=0,
        mask_ratio_mu=mask_ratio_mu,
        masking=masking,
        double_z=True,
    )
    decoder = GenericViTDecoder(
        img_size=img_size,
        patch_size=patch_size,
        out_channels=in_channels,
        embed_dim=embed_dim_decoder,
        depth=depth_decoder,
        num_heads=num_heads_decoder,
        mlp_ratio=mlp_ratio,
        pos_type="learned",
        use_rope=True,
        num_prefix_tokens=1,
        num_latent_tokens=0,
        to_pixel="conv",
        token_dim=None,   # PostQuantLayer Is Done in the VQModel
    )

    return AutoencoderKLTransformer(encoder=encoder, decoder=decoder, pre_post_layer="linear", embed_dim=e_dim, channel_dim=2, **kwargs)