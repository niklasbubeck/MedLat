# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
from torch import nn
from medlat.utils import init_from_ckpt
from typing import Optional, Sequence, Union, List, Any, Dict, Tuple
from medlat.first_stage.discrete.modules.ldm_modules import get_conv_layer
from medlat.modules.alignments import AlignmentModule
from einops import rearrange
__all__ = ["VQModel", "VQModelTransformer"]


class VQModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        alignment: AlignmentModule = None,
        ckpt_path=None,
        # Additional parameters
        quant_conv_ks=1,   ## in var its 3, but VQVAE / VQGAN uses 1
        pre_post_layer="conv",
    ):
        super().__init__()
        self.embed_dim = quantizer.e_dim
        self.n_embed = quantizer.n_e
        self.dims = getattr(encoder, "dims", 2)
        self.encoder = encoder
        self.decoder = decoder
        self.alignment = alignment
        self.z_channels = getattr(encoder, "z_channels", None)
        self.quantizer = quantizer

        if self.z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        self.vae_stride = getattr(encoder, "vae_stride", None)
        if pre_post_layer == "conv":
            conv_layer = get_conv_layer(self.dims)
            self.quant_conv = conv_layer(self.z_channels, self.embed_dim, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
            self.post_quant_conv = conv_layer(self.embed_dim, self.z_channels, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        elif pre_post_layer == "none":
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        else:
            raise ValueError(f"Invalid pre_post_layer: {pre_post_layer}")


        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)

    # def _check_msrq_features(self, method_name):
    #     if not self.quantizer.has_msrq_features:
    #         raise NotImplementedError(
    #             f"Method {method_name} requires MSRQ features. Please initialize VQModel_Combined with "
    #             "v_patch_nums, quant_resi, share_quant_resi, and using_znorm parameters."
    #         )

    def lock_parameters(self):
        """Lock the parameters of the model to prevent them from being updated during training."""
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        h = self.encoder(x)
        # Allow encoders that return (features, aux)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def quantize(self, h):
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info
    
    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h, None, None

    def decode_from_prequant(self, h):
        quant, emb_loss, info = self.quantizer(h)
        return self.decode(quant)
        
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

        if self.alignment is not None:
            alignment_loss, _ = self.alignment(quant, input)
            diff = diff + alignment_loss
            
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    # # VAR/MSQR-specific methods
    # def fhat_to_img(self, f_hat: torch.Tensor):
    #     """Convert quantized features to image"""
    #     self._check_msrq_features('fhat_to_img')
    #     return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    # def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:
    #     """Convert image to multi-scale indices"""
    #     self._check_msrq_features('img_to_idxBl')
    #     h = self.encoder(inp_img_no_grad)
    #     if isinstance(h, tuple):
    #         h = h[0]
    #     f = self.quant_conv(h)
    #     return self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    # def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
    #     """Convert multi-scale indices to image"""
    #     self._check_msrq_features('idxBl_to_img')
    #     B = ms_idx_Bl[0].shape[0]
    #     ms_h_BChw = []
    #     for idx_Bl in ms_idx_Bl:
    #         l = idx_Bl.shape[1]
    #         pn = round(l ** 0.5)
    #         ms_h_BChw.append(self.quantizer.embedding(idx_Bl).transpose(1, 2).view(B, self.embed_dim, pn, pn))
    #     return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    # def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
    #     """Convert embeddings to image"""
    #     self._check_msrq_features('embed_to_img')
    #     if last_one:
    #         return self.decoder(self.post_quant_conv(self.quantizer.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
    #     else:
    #         return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantizer.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    # def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
    #     """Convert image to reconstructed image at multiple scales"""
    #     self._check_msrq_features('img_to_reconstructed_img')
    #     h = self.encoder(x)
    #     if isinstance(h, tuple):
    #         h = h[0]
    #     f = self.quant_conv(h)
    #     ls_f_hat_BChw = self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
    #     if last_one:
    #         return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
    #     else:
    #         return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    # def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
    #     if 'quantizer.ema_vocab_hit_SV' in state_dict and state_dict['quantizer.ema_vocab_hit_SV'].shape[0] != self.quantizer.ema_vocab_hit_SV.shape[0]:
    #         state_dict['quantizer.ema_vocab_hit_SV'] = self.quantizer.ema_vocab_hit_SV
    #     return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


class VQModelTransformer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        alignment: AlignmentModule = None,
        ckpt_path=None,
        pre_post_layer="linear",
    ):
        super().__init__()
        self.embed_dim = quantizer.e_dim
        self.n_embed = quantizer.n_e
        self.dims = getattr(encoder, "dims", 2)
        self.encoder = encoder
        self.decoder = decoder
        self.alignment = alignment
        self.z_channels = getattr(encoder, "z_channels", None)
        self.quantizer = quantizer

        if self.z_channels is None:
            raise ValueError(f"Encoder {encoder.__class__.__name__} must define z_channels.")

        self.vae_stride = getattr(encoder, "vae_stride", None)
        if pre_post_layer == "linear":
            self.quant_conv = nn.Linear(self.z_channels, self.embed_dim)
            self.post_quant_conv = nn.Linear(self.embed_dim, self.z_channels)
        elif pre_post_layer == "none":
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
        else:
            raise ValueError(f"Invalid pre_post_layer: {pre_post_layer}")


        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)


    def lock_parameters(self):
        """Lock the parameters of the model to prevent them from being updated during training."""
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        h, aux = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info, aux

    def quantize(self, h):
        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h, aux = self.encoder(x)
        h = self.quant_conv(h)
        return h, None, None, aux

    def decode_from_prequant(self, h, aux=None):
        quant, emb_loss, info = self.quantizer(h)
        return self.decode(quant, aux=aux)
        
    def decode(self, quant, aux=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, ids_restore=aux["ids_restore"] if aux is not None else None)
        return dec

    def decode_code(self, code_b, out_shape=None, aux=None):
        quant_b = self.quantizer.get_codebook_entry(code_b, shape=out_shape)
        # Move channel dimension (which is last) to the second
        if quant_b.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W)
            quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
        elif quant_b.dim() == 5:
            # (B, D, H, W, C) -> (B, C, D, H, W)
            quant_b = quant_b.permute(0, 4, 1, 2, 3).contiguous()
        dec = self.decode(quant_b, aux=aux)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind), aux = self.encode(input)
        dec = self.decode(quant, aux=aux)

        if self.alignment is not None:
            alignment_loss, _ = self.alignment(quant, input)
            diff = diff + alignment_loss
            
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff