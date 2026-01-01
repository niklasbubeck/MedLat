"""This file contains the model definition of TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

""" 
taken from http://github.com/bytedance/1d-tokenizer/blob/main/modeling/titok.py

"""

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from medtok.first_stage.token.titok.modules import TiTokEncoder, TiTokDecoder, Pixel_Encoder, Pixel_Decoder, Pixel_Quantizer
from medtok.first_stage.modules.gaussian_dist import DiagonalGaussianDistribution
import json
from omegaconf import OmegaConf
from pathlib import Path
from medtok.first_stage.discrete.vq_models import VQModel
from medtok.first_stage.discrete.quantizer.quantize import VectorQuantizer2
from typing import Optional
from medtok.utils import init_from_ckpt
__all__ = ["TiTok"]




class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Encoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu"), weights_only=False), strict=True)
        print("Loaded pretrained weights")
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_loss, (_,_, codebook_indices) = self.quantize(hidden_states)
        return quantized_states, codebook_loss, (None,None, codebook_indices)
    
    @torch.no_grad()
    def decode_code(self, codebook_indices, **kwargs):
        return self.decode(codebook_indices)
    

    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)



"""
Important NOTE: 
The token size should be thought about together with the quantizer object!

Output of the encoder: [B, C=token_size, 1 = num_latent_tokens]
In the quantizer the shape is rearranged to [B, 1, num_latent_tokens, token_size]
In the quantizer the shape is flattend to [-1, e_dim] 
implicitly translating to two cases: 

You can think of TiTOK as a learned, low-dimensional interface into a richer codebook:
- The codebook has many expressive atoms (e.g. 1024 entries of 256 dimensions)
- The encoder finds just a few best-fitting atoms to describe the whole image
- The decoder (or transformer) then decodes the image from those sparse tokens

"""


class TiTok(nn.Module):
    def __init__(self, img_size=256,
                 patch_size=16, 
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
                 dims=2,
                 ckpt_path=None):
        super().__init__()
        # This should be False for stage1 and True for stage2.
        self.stage=stage
        assert stage in ["1", "2", "e2e"]
        if isinstance(img_size, int):
            img_size = (img_size,) * dims
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * dims

        assert len(img_size) == len(patch_size)
        assert not (self.stage == "1" and pixel_vqgan is None), "For stage 1, pixel_vqgan is required."
        self.dims = len(img_size)

        self.n_embed = codebook_size
        self.embed_dim = token_size
        print(f"num latent tokens: {num_latent_tokens}")
        self.quantize_mode = quantize_mode
        self.quantizer_loss_weight = quantizer_loss_weight
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        if self.stage == "2" and self.quantize_mode not in ["vq"]:
            raise ValueError("Only support finetune_decoder with vq quantization for now.")
                
        self.encoder = TiTokEncoder(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_latent_tokens=num_latent_tokens,
            token_size=token_size,
            quantize_mode=quantize_mode,
            is_legacy= not self.stage == "e2e",
        )
        cs = pixel_vqgan.quantize.n_e if pixel_vqgan is not None and self.stage != "e2e" else codebook_size
        self.decoder = TiTokDecoder(
            img_size=img_size,
            out_channels=out_channels,
            codebook_size=cs,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_latent_tokens=num_latent_tokens,
            token_size=token_size,
            quantize_mode=quantize_mode,
            is_legacy= not self.stage == "e2e",
        )
        
        self.num_latent_tokens = num_latent_tokens
        scale = self.encoder.hidden_size ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.hidden_size))
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer2(
                n_e=codebook_size,
                e_dim=token_size,
                legacy=False, # fixes beta weighting error of taming
                beta=0.25,
                use_norm=True,
                )
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        if self.stage == "2":
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
            
        # Freeze the pixel_vqgan is frozen when loaded
        if pixel_vqgan is not None:
            self.pixel_vqgan = pixel_vqgan.eval()
            self.pixel_vqgan.requires_grad_(False)

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.stage == "2":
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, loss, (perplexity, min_encodings, min_encoding_indices) = self.quantize(z)
                loss = torch.tensor(0.0)
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq":
                z_quantized, loss, (perplexity, min_encodings, min_encoding_indices) = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                # result_dict = posteriors
        return z_quantized, loss, (perplexity, min_encodings, min_encoding_indices)
    
    
    def decode(self, z_quantized): 
        decoded = self.decoder(z_quantized)
        if self.stage == "2":
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded = self.pixel_decoder(quantized_states)
        return decoded
    

    def decode_code(self, tokens, out_shape=None):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            quant_b = self.quantize.get_codebook_entry(tokens, shape=out_shape)
            # Move channel dimension (which is last) to the second
            if quant_b.dim() == 4:
                # (B, H, W, C) -> (B, C, H, W)
                quant_b = quant_b.permute(0, 3, 1, 2).contiguous()
            elif quant_b.dim() == 5:
                # (B, D, H, W, C) -> (B, C, D, H, W)
                quant_b = quant_b.permute(0, 4, 1, 2, 3).contiguous()
        elif self.quantize_mode == "vae":
            quant_b = tokens
        decoded = self.decode(quant_b)
        return decoded
    
    def proxy_code_loss(self, decoded, x, q_loss):
        _, _, (_, _, idx) = self.pixel_vqgan.encode(x)
        idx = idx.reshape(decoded.shape[0], -1) # B x N
        decoded = decoded.reshape(decoded.shape[0], decoded.shape[1], -1)
        print(idx.shape, decoded.shape)
        ce_loss = F.cross_entropy(decoded, idx, reduction='mean')
        total_loss = ce_loss + q_loss * self.quantizer_loss_weight
        return total_loss.float()
    
    
    def forward(self, x):
        z_quantized, loss, (_, _, _) = self.encode(x)
        decoded = self.decode(z_quantized)
        return decoded, loss

    