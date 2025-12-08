from medtok.continuous.modules.ldm_modules import Encoder as LDMEncoder, Decoder as LDMDecoder
from medtok.continuous.modules.maisi_modules import MaisiEncoder, MaisiDecoder
from medtok.registry import register_model
from medtok.continuous.vae_models import AutoencoderKL


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