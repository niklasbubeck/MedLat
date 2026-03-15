import torch 
import torch.nn as nn
from medlat.registry import register_model
from medlat.first_stage.discrete.modules.ldm_modules import Encoder, Decoder
from einops.layers.torch import Rearrange, Reduce
from medlat.first_stage.discrete.vq_models import VQModel, VQModelTransformer
from medlat.first_stage.discrete.quantizer import *
from medlat.modules.alignments import *
from medlat.first_stage.token.maetok.modules.vit_models import MAETokViTEncoder, MAETokViTDecoder
from medlat.modules.vit_core import GenericViTEncoder, GenericViTDecoder

@register_model("discrete.vq.f4_d3_e8192")
def VQ_f4_d3_e8192(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=3,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        n_e=8192,
        e_dim=3,
        beta=0.25,
        use_ema=False,
        ema_decay=0.99,
        ema_eps=1e-5,
        **kwargs
    ):
    """
    Instantiate a VQModel (f4 config) with flexible, efficient parameter override.
    https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/vq-f4/config.yaml
    """
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = VectorQuantizer2(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.vq.f4_d8_e8192")
def VQ_f4_d8_e8192(**kwargs):
    return VQ_f4_d3_e8192(z_channels=8, **kwargs)

@register_model(f"discrete.vq.f4_d16_e8192")
def VQ_f4_d16_e8192(**kwargs):
    return VQ_f4_d3_e8192(z_channels=16, **kwargs)

@register_model(f"discrete.vq.f4_d32_e8192")
def VQ_f4_d32_e8192(**kwargs):
    return VQ_f4_d3_e8192(z_channels=32, **kwargs)

@register_model(f"discrete.vq.f8_d4_e16384")
def VQ_f8_d4_e16384(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
        # --- quantizer config ---
        n_e=16384,
        e_dim=4,
        beta=0.25,
        use_ema=False,
        ema_decay=0.99,
        ema_eps=1e-5,
        **kwargs
    ):
    """
    Instantiate a VQModel (f8 config) with flexible, efficient parameter override.
    https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/vq-f8/config.yaml
    """
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = VectorQuantizer2(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.vq.f8_d8_e16384")
def VQ_f8_d8_e16384(**kwargs):
    return VQ_f8_d4_e16384(z_channels=8, **kwargs)

@register_model(f"discrete.vq.f8_d16_e16384")
def VQ_f8_d16_e16384(**kwargs):
    return VQ_f8_d4_e16384(z_channels=16, **kwargs)

@register_model(f"discrete.vq.f8_d32_e16384")
def VQ_f8_d32_e16384(**kwargs):
    return VQ_f8_d4_e16384(z_channels=32, **kwargs)

@register_model(f"discrete.vq.f16_d8_e16384")
def VQ_f16_d8_e16384(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=8,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        # --- quantizer config ---
        n_e=16384,
        e_dim=8,
        beta=0.25,
        use_ema=False,
        ema_decay=0.99,
        ema_eps=1e-5,
        **kwargs
    ):
    """
    Instantiate a VQModel (f16 config) with flexible, efficient parameter override.
    https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/vq-f16/config.yaml
    """
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = VectorQuantizer2(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.vq.f16_d16_e16384")
def VQ_f16_d16_e16384(**kwargs):
    return VQ_f16_d8_e16384(z_channels=16, **kwargs)

@register_model(f"discrete.vq.f16_d32_e16384")
def VQ_f16_d32_e16384(**kwargs):
    return VQ_f16_d8_e16384(z_channels=32, **kwargs)

@register_model(f"discrete.vq.f16_d64_e16384")
def VQ_f16_d64_e16384(**kwargs):
    return VQ_f16_d8_e16384(z_channels=64, **kwargs)

@register_model(f"discrete.maskgit.vq.f16_d256_e1024",
code_url="https://github.com/google-research/maskgit/blob/main/maskgit/nets/vqgan_tokenizer.py",
paper_url="https://arxiv.org/abs/2202.04200",)
def MaskGITVQ_f16_d256_e1024(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        n_e=1024,
        e_dim=256,
        beta=0.25,
        entropy_loss_ratio=0.1,
        entropy_loss_type="softmax",
        entropy_temperature=0.01,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = VectorQuantizer2(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        entropy_loss_ratio=entropy_loss_ratio,
        entropy_loss_type=entropy_loss_type,
        entropy_temperature=entropy_temperature,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.msrq.f16_d32_e4096",
code_url="https://github.com/FoundationVision/VAR/blob/main/models/quant.py",
paper_url="https://arxiv.org/pdf/2404.02905",)
def MSRQ_f16_d32_e4096(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=32,
        in_channels=3,
        out_ch=3,
        ch=160,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
        # --- quantizer config ---
        n_e=4096,
        e_dim=32,
        beta=0.25,
        use_ema=False,
        ema_decay=0.99,
        ema_eps=1e-5,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        quant_resi=0.5,
        share_quant_resi=4,
        using_znorm=False,
        **kwargs
    ):
    """
    Instantiate a MSRQModel (EMA, f8 config) with flexible, efficient parameter override.
    """
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = MultiScaleResidualQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        v_patch_nums=v_patch_nums,
        quant_resi=quant_resi,
        share_quant_resi=share_quant_resi,
        using_znorm=using_znorm,
        dims=dims,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.lfq.f16_d10_b10")
def LFQ_f16_d10_b10(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=10,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=10,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = LookupFreeQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.lfq.f8_d10_b10")
def LFQ_f8_d10_b10(**kwargs):
    return LFQ_f16_d10_b10(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f4_d10_b10")
def LFQ_f4_d10_b10(**kwargs):
    return LFQ_f16_d10_b10(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f16_d12_b12")
def LFQ_f16_d12_b12(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=12,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=12,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = LookupFreeQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.lfq.f8_d12   _b12")
def LFQ_f8_d12_b12(**kwargs):
    return LFQ_f16_d12_b12(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f4_d12_b12")
def LFQ_f4_d12_b12(**kwargs):
    return LFQ_f16_d12_b12(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f16_d14_b14")
def LFQ_f16_d14_b14(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=14,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=14,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = LookupFreeQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.lfq.f8_d14_b14")
def LFQ_f8_d14_b14(**kwargs):
    return LFQ_f16_d14_b14(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f4_d14_b14")
def LFQ_f4_d14_b14(**kwargs):
    return LFQ_f16_d14_b14(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f16_d16_b16")
def LFQ_f16_d16_b16(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=16,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=16,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = LookupFreeQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.lfq.f8_d16_b16")
def LFQ_f8_d16_b16(**kwargs):
    return LFQ_f16_d16_b16(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f4_d16_b16")
def LFQ_f4_d16_b16(**kwargs):
    return LFQ_f16_d16_b16(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f16_d18_b18")
def LFQ_f16_d18_b18(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=18,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=18,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = LookupFreeQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.lfq.f8_d18_b18")
def LFQ_f8_d18_b18(**kwargs):
    return LFQ_f16_d18_b18(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.lfq.f4_d18_b18")
def LFQ_f4_d18_b18(**kwargs):
    return LFQ_f16_d18_b18(ch_mult=[1, 2, 4], **kwargs)

@register_model("discrete.simple_qinco.f4_d3_e8192")
def SimpleQINCo_f4_d3_e8192(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=3,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        n_e=8192,
        e_dim=3,
        beta=0.25,
        hidden_dim=256,
        num_layers=3,
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = SimpleQINCo(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.simple_qinco.f8_d4_e16384")
def SimpleQINCo_f8_d4_e16384(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
        # --- quantizer config ---
        n_e=16384,
        e_dim=4,
        beta=0.25,
        hidden_dim=256,
        num_layers=3,
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = SimpleQINCo(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.simple_qinco.f16_d8_e16384")
def SimpleQINCo_f16_d8_e16384(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=8,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        # --- quantizer config ---
        n_e=16384,
        e_dim=8,
        beta=0.25,
        hidden_dim=256,
        num_layers=3,
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = SimpleQINCo(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model("discrete.simvq.f4_d3_e8192")
def SimVQ_f4_d3_e8192(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=3,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        n_e=8192,
        e_dim=3,
        beta=0.25,
        commitment_weight=1.0,
        rotation_trick=True,
        codebook_transform=None,
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = SimVQ(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        rotation_trick=rotation_trick,
        commitment_weight=commitment_weight,
        codebook_transform=codebook_transform,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.simvq.f8_d4_e16384")
def SimVQ_f8_d4_e16384(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
        # --- quantizer config ---
        n_e=16384,
        e_dim=4,
        beta=0.25,
        commitment_weight=1.0,
        rotation_trick=True,
        codebook_transform=None,
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = SimVQ(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        rotation_trick=rotation_trick,
        commitment_weight=commitment_weight,
        codebook_transform=codebook_transform,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.simvq.f16_d8_e16384")
def SimVQ_f16_d8_e16384(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=8,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        # --- quantizer config ---
        n_e=16384,
        e_dim=8,
        beta=0.25,
        commitment_weight=1.0,
        rotation_trick=True,
        codebook_transform=None,
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = SimVQ(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        rotation_trick=rotation_trick,
        commitment_weight=commitment_weight,
        codebook_transform=codebook_transform,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model("discrete.fsq.f4_d3_l8192")
def FSQ_f4_d3_l8192(
        # --- encoder/decoder config (IDENTICAL) ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=3,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,

        # --- FSQ quantizer config ---
        levels=[32, 16, 16],  # 8192 total states
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )

    quantizer = FiniteScalarQuantizer(
        levels=levels,
        dim=z_channels,
    )

    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model("discrete.fsq.f8_d4_l16384")
def FSQ_f8_d4_l16384(
        # --- encoder/decoder config (IDENTICAL) ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,

        # --- FSQ quantizer config ---
        levels=[16, 8, 8, 16],  # 16384 total states
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )

    quantizer = FiniteScalarQuantizer(
        levels=levels,
        dim=z_channels,
    )

    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model("discrete.fsq.f16_d8_l16384")
def FSQ_f16_d8_l16384(
        # --- encoder/decoder config (IDENTICAL) ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=8,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,

        # --- FSQ quantizer config ---
        levels=[4, 4, 4, 4, 4, 4, 2, 2],  # 16384 total states
        **kwargs
    ):

    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )

    quantizer = FiniteScalarQuantizer(
        levels=levels,
        dim=z_channels,
    )

    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.fsq.f16_d256_e8",
code_url="https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb",
paper_url="https://arxiv.org/pdf/2309.15505",
description="The original version employs on maskgits VQVAE config")
def FSQ_f16_d256_e8(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        levels=[8, 6, 5],
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = FiniteScalarQuantizer(
        levels=levels,
        dim=z_channels,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.fsq.f16_d256_e10",
code_url="https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb",
paper_url="https://arxiv.org/pdf/2309.15505",
description="The original version employs on maskgits VQVAE config levels from table1")
def FSQ_f16_d256_e10(**kwargs):
    return FSQ_f16_d256_e8(levels=[8, 5, 5, 5], **kwargs)

@register_model(f"discrete.fsq.f16_d256_e12",
code_url="https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb",
paper_url="https://arxiv.org/pdf/2309.15505",
description="The original version employs on maskgits VQVAE config levels from table1")
def FSQ_f16_d256_e12(**kwargs):
    return FSQ_f16_d256_e8(levels=[7, 5, 5, 5, 5], **kwargs)

@register_model(f"discrete.fsq.f16_d256_e14",
code_url="https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb",
paper_url="https://arxiv.org/pdf/2309.15505",
description="The original version employs on maskgits VQVAE config levels from table1")
def FSQ_f16_d256_e14(**kwargs):
    return FSQ_f16_d256_e8(levels=[8, 8, 8, 6, 5], **kwargs)

@register_model(f"discrete.fsq.f16_d256_e16",
code_url="https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb",
paper_url="https://arxiv.org/pdf/2309.15505",
description="The original version employs on maskgits VQVAE config levels from table1")
def FSQ_f16_d256_e16(**kwargs):
    return FSQ_f16_d256_e8(levels=[8, 8, 8, 5, 5, 5], **kwargs)


@register_model(f"discrete.bsq.f16_d10_b10")
def BSQ_f16_d10_b10(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=10,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=10,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = BinarySphericalQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.bsq.f8_d10_b10")
def BSQ_f8_d10_b10(**kwargs):
    return BSQ_f16_d10_b10(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f4_d10_b10")
def BSQ_f4_d10_b10(**kwargs):
    return BSQ_f16_d10_b10(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f16_d12_b12")
def BSQ_f16_d12_b12(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=12,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=12,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = BinarySphericalQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.bsq.f8_d12_b12")
def BSQ_f8_d12_b12(**kwargs):
    return BSQ_f16_d12_b12(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f4_d12_b12")
def BSQ_f4_d12_b12(**kwargs):
    return BSQ_f16_d12_b12(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f16_d14_b14")
def BSQ_f16_d14_b14(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=14,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=14,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = BinarySphericalQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.bsq.f8_d14_b14")
def BSQ_f8_d14_b14(**kwargs):
    return BSQ_f16_d14_b14(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f4_d14_b14")
def BSQ_f4_d14_b14(**kwargs):
    return BSQ_f16_d14_b14(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f16_d16_b16")
def BSQ_f16_d16_b16(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=16,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=16,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = BinarySphericalQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.bsq.f8_d16_b16")
def BSQ_f8_d16_b16(**kwargs):
    return BSQ_f16_d16_b16(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f4_d16_b16")
def BSQ_f4_d16_b16(**kwargs):
    return BSQ_f16_d16_b16(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f16_d18_b18")
def BSQ_f16_d18_b18(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=18,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        token_bits=18,
        commitment_cost=0.25,
        entropy_loss_weight=0.02,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = BinarySphericalQuantizer(
        token_bits=token_bits,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.bsq.f8_d18_b18")
def BSQ_f8_d18_b18(**kwargs):
    return BSQ_f16_d18_b18(ch_mult=[1, 2, 2, 4], **kwargs)

@register_model(f"discrete.bsq.f4_d18_b18")
def BSQ_f4_d18_b18(**kwargs):
    return BSQ_f16_d18_b18(ch_mult=[1, 2, 4], **kwargs)

@register_model(f"discrete.rqvae.f4_d3_e8192")
def RQVAE_f4_d3_e8192(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=3,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        quantizer_class=VectorQuantizer2,
        num_quantizers=4,
        n_e=8192,
        e_dim=3,
        beta=0.25,
        shared_codebook=True,
        quantize_dropout=False,
        dropout_start_level=0,
        **kwargs
):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = ResidualQuantizer(
        quantizer_class=quantizer_class,
        num_quantizers=num_quantizers,
        quantizer_kwargs_list=[
            {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
            }
        ] * num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.rqvae.f8_d4_e16384")
def RQVAE_f8_d4_e16384(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        quantizer_class=VectorQuantizer2,
        num_quantizers=4,
        n_e=16384,
        e_dim=4,
        beta=0.25,
        shared_codebook=True,
        quantize_dropout=False,
        dropout_start_level=0,
        **kwargs
):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = ResidualQuantizer(
        quantizer_class=quantizer_class,
        num_quantizers=num_quantizers,
        quantizer_kwargs_list=[
            {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
            }
        ] * num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.rqvae.f16_d8_e16384")
def RQVAE_f16_d8_e16384(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=8,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        # --- quantizer config ---
        quantizer_class=VectorQuantizer2,
        num_quantizers=4,
        n_e=16384,
        e_dim=8,
        beta=0.25,
        shared_codebook=True,
        quantize_dropout=False,
        dropout_start_level=0,
        **kwargs
):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = ResidualQuantizer(
        quantizer_class=quantizer_class,
        num_quantizers=num_quantizers,
        quantizer_kwargs_list=[
            {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
            }
        ] * num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.rsimple_qinco.f4_d3_e8192")
def RSimpleQINCo_f4_d3_e8192(**kwargs):
    return RQVAE_f4_d3_e8192(**kwargs, quantizer_class=SimpleQINCo)

@register_model(f"discrete.rsimple_qinco.f8_d4_e16384")
def RSimpleQINCo_f8_d4_e16384(**kwargs):
    return RQVAE_f8_d4_e16384(**kwargs, quantizer_class=SimpleQINCo)

@register_model(f"discrete.rsimple_qinco.f16_d8_e16384")
def RSimpleQINCo_f16_d8_e16384(**kwargs):
    return RQVAE_f16_d8_e16384(**kwargs, quantizer_class=SimpleQINCo)


@register_model(f"discrete.qinco.f4_d3_e8192")
def QINCo_f4_d3_e8192(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=3,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        quantizer_class=QINCo,
        num_quantizers=4,
        n_e=8192,
        e_dim=3,
        beta=0.25,
        top_a=64,
        hidden_dim=256,
        num_layers=3,
        shared_codebook=True,
        quantize_dropout=False,
        dropout_start_level=0,
        **kwargs
):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = QincoResidualQuantizer(
        quantizer_class=quantizer_class,
        num_quantizers=num_quantizers,
        quantizer_kwargs_list=[
            {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
                "top_a": top_a,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
            }
        ] * num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.qinco.f8_d4_e16384")
def QINCo_f8_d4_e16384(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        quantizer_class=QINCo,
        num_quantizers=4,
        n_e=16384,
        e_dim=4,
        beta=0.25,
        top_a=64,
        hidden_dim=256,
        num_layers=3,
        shared_codebook=True,
        quantize_dropout=False,
        dropout_start_level=0,
        **kwargs
):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = QincoResidualQuantizer(
        quantizer_class=quantizer_class,
        num_quantizers=num_quantizers,
        quantizer_kwargs_list=[
            {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
                "top_a": top_a,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
            }
        ] * num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"discrete.qinco.f16_d8_e16384")
def QINCo_f16_d8_e16384(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=8,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        quantizer_class=QINCo,
        num_quantizers=4,
        n_e=16384,
        e_dim=8,
        beta=0.25,
        top_a=64,
        hidden_dim=256,
        num_layers=3,
        shared_codebook=True,
        quantize_dropout=False,
        dropout_start_level=0,
        **kwargs
):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    quantizer = QincoResidualQuantizer(
        quantizer_class=quantizer_class,
        num_quantizers=num_quantizers,
        quantizer_kwargs_list=[
            {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
                "top_a": top_a,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
            }
        ] * num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"continuous.soft_vq.f4_d3_e8192")
def SoftVQ_f4_d3_e8192(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=3,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        n_e=8192,
        e_dim=3,
        entropy_loss_weight=0.01,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0,
        tau=0.07,
        use_norm=True,
        **kwargs
    ):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
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
    alignment = VFFoundationAlignment(latent_channels=z_channels, foundation_type="dinov2")
    return VQModel(encoder, decoder, quantizer, alignment=alignment, **kwargs)

@register_model(f"continuous.wqvae.f8_d4_e16384")
def WQVAE_f8_d4_e16384(
    # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        # --- quantizer config ---
        quantizer_class=WaveletResidualQuantizer,
        num_quantizers=4,
        n_e=16384,
        e_dim=4,
        beta=0.25,
        shared_codebook=True,
        quantize_dropout=False,
        dropout_start_level=0,
        wavelet="db1",
        wavelet_levels=1,
        subbands=["LL", "LH", "HL", "HH"],
        **kwargs
):
    encoder = Encoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    decoder = Decoder(
        img_size=img_size,
        dims=dims,
        double_z=double_z,
        z_channels=z_channels,
        in_channels=in_channels,
        out_ch=out_ch,
        ch=ch,
        ch_mult=[ch_mult[0], ch_mult[1], ch_mult[2], ch_mult[2]],
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        dropout=dropout
    )
    
    # --- Create identical quantizer kwargs for each level ---
    quantizer_kwargs_list = [
        {
            "n_e": n_e,
            "e_dim": e_dim,
            "beta": beta
        }
        for _ in range(num_quantizers)
    ]
    
    quantizer = quantizer_class(
        quantizer_class=VectorQuantizer,  # Base VQ class (your existing one)
        num_quantizers=num_quantizers,
        wavelet=wavelet,
        wavelet_levels=wavelet_levels,
        subbands=subbands,
        quantizer_kwargs_list=quantizer_kwargs_list,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"discrete.hcvq.residual_vq.S_16")
def RQVAE_Transformer_S_16(
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
    # --- quantizer config ---
    quantizer_class=VectorQuantizer2,
    num_quantizers=4,
    n_e=16384,
    e_dim=32,
    beta=0.25,
    shared_codebook=True,
    quantize_dropout=False,
    dropout_start_level=0,
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
    quantizer = ResidualQuantizer(
        quantizer_class=quantizer_class,
        num_quantizers=num_quantizers,
        quantizer_kwargs_list=[
            {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
            }
        ] * num_quantizers,
        shared_codebook=shared_codebook,
        quantize_dropout=quantize_dropout,
        dropout_start_level=dropout_start_level,
    )
    return VQModelTransformer(encoder=encoder, decoder=decoder, quantizer=quantizer, pre_post_layer="linear", **kwargs)


@register_model(f"discrete.hcvq.soft_vq.S_16")
def SoftVQ_Transformer_S_16(
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
    # --- quantizer config ---
    n_e=16384,
    e_dim=32,
    entropy_loss_weight=0.01,
    entropy_loss_temperature=0.01,
    entropy_gamma=1.0,
    tau=0.07,
    use_norm=True,
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
    quantizer = SoftVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
        tau=tau,
        use_norm=use_norm,
    )
    return VQModelTransformer(encoder=encoder, decoder=decoder, quantizer=quantizer, pre_post_layer="linear", **kwargs)


@register_model(f"discrete.hcvq.grouped_vq.S_16")
def GroupedVQ_Transformer_S_16(
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
    # --- quantizer config ---
    quantizer_class=VectorQuantizer2,
    n_e=16384,
    e_dim=32,
    beta=0.25,
    groups=5,
    split_dim=1,
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
    quantizer = GroupedVQ(
        quantizer_class=quantizer_class,
        quantizer_kwargs_list=[
                {
                "n_e": n_e,
                "e_dim": e_dim,
                "beta": beta,
                }
            ] * groups,
        groups=groups,
        split_dim=split_dim,
    )
    return VQModelTransformer(encoder=encoder, decoder=decoder, quantizer=quantizer, pre_post_layer="linear", **kwargs)


@register_model(f"discrete.hcvq.sd_vq.S_16")
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
    # --- quantizer config ---
    n_e=16384,
    e_dim=32,
    beta=0.25,
    use_ema=False,
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
    quantizer = VectorQuantizer2(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        use_ema=use_ema,
    )
    return VQModelTransformer(encoder=encoder, decoder=decoder, quantizer=quantizer, pre_post_layer="linear", **kwargs)

@register_model(f"discrete.hcvq.msrq.S_16")
def MSRQ_Transformer_S_16(
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
    # --- quantizer config ---
    dims=3,
    n_e=16384,
    e_dim=32,
    beta=0.25,
    use_ema=False,
    ema_decay=0.99,
    ema_eps=1e-5,
    v_patch_nums=((1,1,1), (2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6), (8,8,8), (10, 11, 9)),
    quant_resi=0.5,
    share_quant_resi=4,
    using_znorm=False,
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
    

    quantizer = MultiScaleResidualQuantizer3D(
        n_e=n_e,
        e_dim=e_dim,
        dims=dims,
        using_znorm=using_znorm,
        beta=beta,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps,
        v_patch_nums=v_patch_nums,
        quant_resi=quant_resi,
        share_quant_resi=share_quant_resi,
    )
    return VQModelTransformer(encoder=encoder, decoder=decoder, quantizer=quantizer, pre_post_layer="linear", **kwargs)