import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .modules import MAETokViTEncoder, MAETokViTDecoder
from medtok.modules.alignments import HOGAlignment, DinoAlignment, ClipAlignment
from medtok.registry import register_model

__all__ = ['MaskAEModel']

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class MaskAEModel(nn.Module):
    def __init__(self, 
                 img_size: int = 256,
                 patch_size: int = 16,
                 base_img_size: int = 256,
                 codebook_embed_dim: int = 32,
                 dec_use_movq: bool = False,
                 num_latent_tokens: int = 128,
                 to_pixel: str = 'linear',
                 enc_embed_dim: int = 768,
                 dec_embed_dim: int = 768,
                 enc_depth: int = 12,
                 dec_depth: int = 12,
                 enc_num_heads: int = 12,
                 dec_num_heads: int = 12,
                 enc_mlp_ratio: float = 4.,
                 dec_mlp_ratio: float = 4.,
                 enc_drop_path_rate: float = 0.0,
                 dec_drop_path_rate: float = 0.05,
                 dec_cls_token: bool = False,
                 use_ape: bool = False,
                 use_rope: bool = True,
                 rope_mixed: bool = True,
                 rope_theta: float = 10.0,
                 repa_patch_size: int = 16,
                 repa_model: str = 'vit_large_patch14_dinov2.lvd142m',
                 repa_proj_dim: int = 1024,
                 enc_token_drop: float = 0.4,
                 enc_token_drop_max: float = 0.6,
                 aux_dec_model: str = 'vit_tinytiny_patch14_dinov2_movq2',
                 aux_loss_mask: bool = True,
                 aux_dec_cls_token: bool = True,
                 aux_hog_dec: bool = True,
                 aux_dino_dec: bool = True,
                 aux_clip_dec: bool = True,
                 aux_biomed_clip_dec: bool = False,
                 ):
        super().__init__()
        self.img_size = img_size
        self.base_img_size = base_img_size
        self.dec_use_movq = dec_use_movq
        self.codebook_embed_dim = codebook_embed_dim
        self.num_latent_tokens = num_latent_tokens
        self.to_pixel = to_pixel
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_num_heads = enc_num_heads
        self.dec_num_heads = dec_num_heads
        self.enc_mlp_ratio = enc_mlp_ratio
        self.dec_mlp_ratio = dec_mlp_ratio
        self.patch_size = patch_size
        self.enc_drop_path_rate = enc_drop_path_rate
        self.dec_drop_path_rate = dec_drop_path_rate
        self.dec_cls_token = dec_cls_token
        self.use_ape = use_ape
        self.use_rope = use_rope
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.repa_patch_size = repa_patch_size
        self.repa_model = repa_model
        self.repa_proj_dim = repa_proj_dim
        self.enc_token_drop = enc_token_drop
        self.enc_token_drop_max = enc_token_drop_max
        self.aux_dec_model = aux_dec_model  # Keep for backward compatibility but not used directly
        self.aux_loss_mask = aux_loss_mask
        self.aux_dec_cls_token = aux_dec_cls_token
        self.aux_hog_dec = aux_hog_dec
        self.aux_dino_dec = aux_dino_dec
        self.aux_clip_dec = aux_clip_dec
        self.aux_biomed_clip_dec = aux_biomed_clip_dec

        self.encoder = MAETokViTEncoder(
            in_channels=3,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.enc_embed_dim,
            depth=self.enc_depth,
            num_heads=self.enc_num_heads,
            mlp_ratio=self.enc_mlp_ratio,
            num_latent_tokens=self.num_latent_tokens,
            use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
            token_drop=self.enc_token_drop,
            token_drop_max=self.enc_token_drop_max,
            base_img_size=self.base_img_size,
            drop_path_rate=self.enc_drop_path_rate
        )

        self.decoder = MAETokViTDecoder(
            in_channels=3,
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.dec_embed_dim,
            depth=self.dec_depth,
            num_heads=self.dec_num_heads,
            mlp_ratio=self.dec_mlp_ratio,
            num_latent_tokens=self.num_latent_tokens,
            use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
            cls_token=self.dec_cls_token,
            codebook_embed_dim=self.codebook_embed_dim,
            to_pixel=self.to_pixel,
            base_img_size=self.base_img_size,
            use_movq=self.dec_use_movq,
            drop_path_rate=self.dec_drop_path_rate
        )           
        # Add missing quant_conv layer
        self.quant_conv = nn.Linear(self.encoder.embed_dim, self.codebook_embed_dim)
        
        # Add missing post_quant_conv for main decoder
        self.post_quant_conv = nn.Linear(self.codebook_embed_dim, self.decoder.embed_dim)
        
        # Initialize alignment modules
        self.aux_hog_decoder = None
        if self.aux_hog_dec:
            print('Using HOG decoder')
            use_movq_hog = 'movq' in self.aux_dec_model if self.aux_dec_model else False
            # Create decoder for HOG alignment
            aux_hog_decoder_model = MAETokViTDecoder(
                in_channels=3,
                embed_dim=384,  # Default decoder dimensions
                depth=12,
                num_heads=6,
                mlp_ratio=4.0,
                img_size=self.img_size,
                patch_size=self.patch_size,
                drop_path_rate=0.0,
                num_latent_tokens=self.num_latent_tokens,
                to_pixel='identity',
                codebook_embed_dim=self.codebook_embed_dim,
                rope_theta=self.rope_theta,
                rope_mixed=self.rope_mixed,
                use_rope=self.use_rope,
                use_ape=self.use_ape,
                cls_token=self.aux_dec_cls_token,
                base_img_size=self.base_img_size,
                use_movq=use_movq_hog,
            )
            self.aux_hog_decoder = HOGAlignment(
                decoder=aux_hog_decoder_model,
                codebook_embed_dim=self.codebook_embed_dim,
                use_movq=use_movq_hog,
            )
        
        self.aux_dino_decoder = None
        if self.aux_dino_dec:
            print('Using DINO decoder')
            use_movq_dino = 'movq' in self.aux_dec_model if self.aux_dec_model else False
            # Create decoder for DINO alignment
            aux_dino_decoder_model = MAETokViTDecoder(
                in_channels=3,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4.0,
                img_size=self.img_size,
                patch_size=self.repa_patch_size,
                drop_path_rate=0.0,
                num_latent_tokens=self.num_latent_tokens,
                to_pixel='identity',
                codebook_embed_dim=self.codebook_embed_dim,
                rope_theta=self.rope_theta,
                rope_mixed=self.rope_mixed,
                use_rope=self.use_rope,
                use_ape=self.use_ape,
                cls_token=self.aux_dec_cls_token,
                base_img_size=self.base_img_size,
                use_movq=use_movq_dino,
            )
            self.aux_dino_decoder = DinoAlignment(
                decoder=aux_dino_decoder_model,
                codebook_embed_dim=self.codebook_embed_dim,
                img_size=self.img_size,
                repa_model_name=self.repa_model,
                repa_patch_size=self.repa_patch_size,
                use_movq=use_movq_dino,
            )
        
        self.aux_clip_decoder = None
        if self.aux_clip_dec:
            print('Using CLIP decoder')
            use_movq_clip = 'movq' in self.aux_dec_model if self.aux_dec_model else False
            # Create decoder for CLIP alignment
            aux_clip_decoder_model = MAETokViTDecoder(
                in_channels=3,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4.0,
                img_size=self.img_size,
                patch_size=self.repa_patch_size,
                drop_path_rate=0.0,
                num_latent_tokens=self.num_latent_tokens,
                to_pixel='identity',
                codebook_embed_dim=self.codebook_embed_dim,
                rope_theta=self.rope_theta,
                rope_mixed=self.rope_mixed,
                use_rope=self.use_rope,
                use_ape=self.use_ape,
                cls_token=self.aux_dec_cls_token,
                base_img_size=self.base_img_size,
                use_movq=use_movq_clip,
            )
            self.aux_clip_decoder = ClipAlignment(
                decoder=aux_clip_decoder_model,
                codebook_embed_dim=self.codebook_embed_dim,
                img_size=self.img_size,
                clip_model_name='vit_so400m_patch14_siglip_gap_224',
                clip_patch_size=self.repa_patch_size,
                use_movq=use_movq_clip,
            )
            
        # self.aux_biomed_clip_decoder = self.aux_biomed_clip_dec
        # if self.aux_biomed_clip_decoder:
        #     self.biomed_clip_model, _, _ = open_clip.create_model_and_transforms(
        #     model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        # )
        #     for param in self.biomed_clip_model.parameters():
        #         param.requires_grad = False
        #     self.biomed_clip_model.dynamic_img_size = True
        #     self.biomed_clip_model.eval()

        #     print('Using Biomed CLIP decoder: ', self.aux_dec_model)
        #     self.decoder_biomed_clip = MAETokViTDecoder(
        #         in_channels=3,
        #         img_size=[224,352],
        #         patch_size=repa_patch_size,
        #         embed_dim=384,
        #         depth=6,
        #         num_heads=6,
        #         mlp_ratio=4.,
        #         num_latent_tokens=self.num_latent_tokens,
        #         use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
        #         cls_token=self.aux_dec_cls_token,
        #         codebook_embed_dim=self.codebook_embed_dim,
        #         to_pixel='identity',
        #         base_img_size=self.base_img_size,
        #         use_movq=self.dec_use_movq,
        #     )           
        #     self.post_quant_conv_clip = nn.Linear(self.codebook_embed_dim, self.decoder_biomed_clip.embed_dim)
        #     self.to_pixel_clip = nn.Linear(self.decoder_biomed_clip.embed_dim, self.biomed_clip_model.embed_dim)



    def encode(self, x):
        if self.training:
            h, mask = self.encoder(x, return_mask=True)
        else:
            h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = torch.tensor(0.)
        info = None
        if self.training:
            return quant, emb_loss, info, mask
        else:
            return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.dec_use_movq:
            dec = self.decoder(quant, interpolate_zq=tmp_quant, H=h, W=w, D=None)
        else:
            dec = self.decoder(quant, interpolate_zq=None, H=h, W=w, D=None)
        return dec

    def forward(self, input):
        b, _, h, w, *_  = input.size()
        if self.training:
            quant, diff, info, mask = self.encode(input)
        else:
            quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input)
        # Compute alignment losses using alignment modules
        if self.training:
            ## get rid of the cls token in mask 
            if mask is not None:
                mask = mask[:, 1:, :]
            # HOG alignment loss
            if self.aux_hog_decoder is not None:
                hog_loss, _ = self.aux_hog_decoder(quant, input_image=input, mask=mask if self.aux_loss_mask else None)
                hog_rec_loss = hog_loss
            else:
                hog_rec_loss = torch.tensor(0.0, device=input.device)
            
            # DINO alignment loss
            if self.aux_dino_decoder is not None:
                dino_loss, _ = self.aux_dino_decoder(quant, input_image=input, mask=mask if self.aux_loss_mask else None)
                dino_rec_loss = dino_loss
            else:
                dino_rec_loss = torch.tensor(0.0, device=input.device)
            
            # CLIP alignment loss
            if self.aux_clip_decoder is not None:
                clip_loss, _ = self.aux_clip_decoder(quant, input_image=input, mask=mask if self.aux_loss_mask else None)
                clip_rec_loss = clip_loss
            else:
                clip_rec_loss = torch.tensor(0.0, device=input.device)
        
        # Combine auxiliary losses with main loss
        if self.training:
            total_aux_loss = hog_rec_loss + dino_rec_loss + clip_rec_loss
            diff = diff + total_aux_loss
        
        return dec, diff

    def get_last_layer(self):
        # Access the last layer of the to_pixel projection
        if hasattr(self.decoder.to_pixel, 'proj'):
            if isinstance(self.decoder.to_pixel.proj, nn.Linear):
                return self.decoder.to_pixel.proj
            elif isinstance(self.decoder.to_pixel.proj, (nn.Conv2d, nn.Conv3d)):
                return self.decoder.to_pixel.proj
        return None
