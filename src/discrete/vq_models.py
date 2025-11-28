# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
from torch import nn
from src.utils import instantiate_from_config, init_from_ckpt
from typing import Optional, Sequence, Union, List, Any, Dict, Tuple
from ..modules import Encoder, Decoder, ViTEncoder, ViTDecoder, get_conv_layer
from .quantizer import *
from ..registry import register_model

__all__ = ["VQModel", "VQModel_f4", "VQModel_f8", "VQModel_f16", "MSRQModel_f8", "ViTVQModel_Small", "ViTVQModel_Base", "ViTVQModel_Large", "LFQModel_f4"]

"""
For a versatile selection of Quantizers, please refer to: https://github.com/lucidrains/vector-quantize-pytorch
You can define it using the OmegaConf setup when training from source and config files.
"""

"""
The VQModel gives you a lot of flexibility in terms of quantizer choices, and the possibility to use scaling as introduced in VAR.
There for you have different options to choose how to setup your Quantizer.

1. Train full VAR style with residuals for the scaling and scaling features:
    quantizer_cfg = {
        "_target_": "src.models.first_stage.discrete.quantizer.EMAVectorQuantizerWithVAR",
        "n_e": 8192,
        "e_dim": 3,
        "beta": 0.25,
        "remap": None,
        "v_patch_nums": (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        "quant_resi": 0.5,
        "share_quant_resi": 4,
        "using_znorm": False,
    }

    just define the quant_residual settings.

2. Train full VAR style WITHOU THE RESIDUALS but still get the VAR scaling features:
    quantizer_cfg = {
        "_target_": "src.models.first_stage.discrete.quantizer.EMAVectorQuantizerWithVAR",
        "n_e": 8192,
        "e_dim": 3,
        "beta": 0.25,
        "remap": None,
        "v_patch_nums": (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        "using_znorm": False,
    }
    dont define the quant_residual settings, NOW YOU CAN USE THE STANDARD VQVAE,VQGAN WEIGHTS AND STILL USE THE VAR SCALING FEATURES.

3. Train standard VQVAE / VQGAN:
    quantizer_cfg = {
        "_target_": "src.models.first_stage.discrete.quantizer.EMAVectorQuantizer",
        "n_e": 8192,
        "e_dim": 3,
        "beta": 0.25,
        "remap": None,
    }

    just dont define any var settings. You will get no VAR scaling features and default to VQVAE / VQGAN.
"""

_REGISTRY_PREFIX = "discrete.vq."

@register_model("discrete.vq.model")
class VQModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        ckpt_path=None,
        # Additional parameters
        quant_conv_ks=1,   ## in var its 3, but VQVAE / VQGAN uses 1
    ):
        super().__init__()
        self.embed_dim = quantizer.e_dim
        self.n_embed = quantizer.n_e
        self.dims = getattr(encoder, "dims", 2)
        conv_layer = get_conv_layer(self.dims)
        self.encoder = encoder
        self.decoder = decoder
        self.z_channels = getattr(encoder, "z_channels", None)
        self.quantizer = quantizer

        if self.z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        self.quant_conv = conv_layer(self.z_channels, self.embed_dim, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = conv_layer(self.embed_dim, self.z_channels, quant_conv_ks, stride=1, padding=quant_conv_ks//2)

        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)

    def _check_msrq_features(self, method_name):
        if not self.quantizer.has_msrq_features:
            raise NotImplementedError(
                f"Method {method_name} requires MSRQ features. Please initialize VQModel_Combined with "
                "v_patch_nums, quant_resi, share_quant_resi, and using_znorm parameters."
            )

    def lock_parameters(self):
        """Lock the parameters of the model to prevent them from being updated during training."""
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, out_shape=None):
        quant_b = self.quantizer.get_codebook_entry(code_b, shape=out_shape)
        # Move channel dimension (which is last) to the second
        if quant_b.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W)
            quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
        elif quant_b.dim() == 5:
            # (B, D, H, W, C) -> (B, C, D, H, W)
            quant_b = quant_b.permute(0, 4, 1, 2, 3).contiguous()
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    # VAR/MSQR-specific methods
    def fhat_to_img(self, f_hat: torch.Tensor):
        """Convert quantized features to image"""
        self._check_msrq_features('fhat_to_img')
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
        """Convert image to multi-scale indices"""
        self._check_msrq_features('img_to_idxBl')
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """Convert multi-scale indices to image"""
        self._check_msrq_features('idxBl_to_img')
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantizer.embedding(idx_Bl).transpose(1, 2).view(B, self.embed_dim, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """Convert embeddings to image"""
        self._check_msrq_features('embed_to_img')
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantizer.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantizer.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        """Convert image to reconstructed image at multiple scales"""
        self._check_msrq_features('img_to_reconstructed_img')
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantizer.ema_vocab_hit_SV' in state_dict and state_dict['quantizer.ema_vocab_hit_SV'].shape[0] != self.quantizer.ema_vocab_hit_SV.shape[0]:
            state_dict['quantizer.ema_vocab_hit_SV'] = self.quantizer.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


@register_model("discrete.vq.f4")
def VQModel_f4(
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
        remap=None,
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
        remap=remap,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"{_REGISTRY_PREFIX}f8")
def VQModel_f8(
        # --- encoder/decoder config ---
        img_size=256,
        dims=2,
        double_z=False,
        z_channels=4,
        in_channels=1,
        out_ch=1,
        ch=128,
        ch_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
        # --- quantizer config ---
        n_e=16384,
        e_dim=4,
        beta=0.25,
        remap=None,
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
        remap=remap,
        dims=dims,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)


@register_model(f"{_REGISTRY_PREFIX}f16")
def VQModel_f16(
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
        remap=None,
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
        remap=remap,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_eps=ema_eps
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}msrq_f8",
code_url="https://github.com/FoundationVision/VAR/blob/main/models/quant.py",
paper_url="https://arxiv.org/pdf/2404.02905",)
def MSRQModel_f8(
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
        remap=None,
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
    quantizer = MSRQEMAVectorQuantizer(
        n_e=n_e,
        e_dim=e_dim,
        beta=beta,
        remap=remap,
        v_patch_nums=v_patch_nums,
        quant_resi=quant_resi,
        share_quant_resi=share_quant_resi,
        using_znorm=using_znorm,
        dims=dims
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}vit_xsmall")
def ViTVQModel_XSmall(
        # --- encoder/decoder (ViT) config ---
        image_size: int,
        patch_size=None,
        z_channels=256,
        dim=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        channels=3,
        # --- quantizer config ---
        n_e=8192,
        e_dim=32,
        beta=0.25,
        remap=None,
        **kwargs
    ):
    encoder = EncoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    decoder = DecoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    quantizer = ViTVectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta, remap=remap)
    return VQModel(encoder, decoder, quantizer, vit=True, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}vit_small")
def ViTVQModel_Small(
        # --- encoder/decoder (ViT) config ---
        image_size: int,
        patch_size=None,
        z_channels=512,
        dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        channels=3,
        # --- quantizer config ---
        n_e=8192,
        e_dim=32,
        beta=0.25,
        remap=None,
        **kwargs
    ):
    encoder = EncoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    decoder = DecoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    quantizer = ViTVectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta, remap=remap)
    return VQModel(encoder, decoder, quantizer, vit=True, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}vit_base")
def ViTVQModel_Base(
        # --- encoder/decoder (ViT) config ---
        image_size: int,
        patch_size=None,
        z_channels=768,
        dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        channels=3,
        # --- quantizer config ---
        n_e=8192,
        e_dim=32,
        beta=0.25,
        remap=None,
        **kwargs
    ):
    encoder = EncoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    decoder = DecoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    quantizer = ViTVectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta, remap=remap)
    return VQModel(encoder, decoder, quantizer, vit=True, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}vit_large")
def ViTVQModel_Large(
        # --- encoder/decoder (ViT) config ---
        image_size: int,
        patch_size=None,
        z_channels=1280,
        dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        channels=3,
        # --- quantizer config ---
        n_e=8192,
        e_dim=32,
        beta=0.25,
        remap=None,
        **kwargs
    ):
    encoder = EncoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    decoder = DecoderVisionTransformer(
        image_size=image_size,
        patch_size=8 if patch_size is None else patch_size,
        z_channels=z_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        channels=channels
    )
    quantizer = ViTVectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta, remap=remap)
    return VQModel(encoder, decoder, quantizer, vit=True, **kwargs)

@register_model(f"{_REGISTRY_PREFIX}lfq_f4d3b10")
def LFQModel_f4(
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
        token_bits=10,
        beta=0.25,
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
        beta=beta,
        commitment_cost=commitment_cost,
        entropy_loss_weight=entropy_loss_weight,
        entropy_loss_temperature=entropy_loss_temperature,
        entropy_gamma=entropy_gamma,
    )
    return VQModel(encoder, decoder, quantizer, **kwargs)

