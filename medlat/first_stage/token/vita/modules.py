import torch
from torch import nn
from typing import List, Sequence, Tuple, Union
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from typing import Optional

# from networks.unetr_blocks import UnetOutBlock, UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock


import numpy as np

from .utils.imaging_model_related import Masker, sincos_pos_embed, patchify, unpatchify


class PatchEmbed(nn.Module):
    def __init__(self, im_shape: list[int], 
                 in_channels: int = 1, 
                 patch_size: list[int] = [1, 16, 16], 
                 out_channels: int = 256, 
                 flatten: bool = True, 
                 bias: bool = True,
                 norm_layer: Optional[nn.Module] = None, **kwargs):
        super().__init__()
        
        assert len(patch_size) == 3, "Patch size should be 3D"
        assert in_channels == 1, "Only supporting input channel size as 1"
        print(f"im_shape: {im_shape}, patch_size: {patch_size}")
        self.im_shape = im_shape
        if len(im_shape) == 3:
            self.im_shape = im_shape.unsqueeze(1) # (S, H, W) -> (S, 1, H, W)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flatten = flatten
        
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.grid_size = (im_shape[0], 
                          im_shape[1] // patch_size[0], 
                          im_shape[2] // patch_size[1], 
                          im_shape[3] // patch_size[2]) # (S, t, h, w)
        self.num_patches = np.prod(self.grid_size)
        print(f"grid_size: {self.grid_size}, num_patches: {self.num_patches}")
    
    def forward(self, x):
        """ 
        input: (B, S, T, H, W)
        output: (B * S, out_channels, t, h, w) or (B, num_patches, out_channels) if flatten is True, 
                where num_patches = S * T * H * W / np.prod(patch_size)
        """
        x_ = x.reshape(-1, *self.im_shape[-3:]) # (B*S, T, H, W)
        x_ = x_.unsqueeze(1) # (B*S, 1, T, H, W)
        x_ = self.proj(x_) # (B*S, out_channels, t, h, w)
        
        if self.flatten:
            x__ = x_.flatten(2) # (B*S, out_channels, t*h*w)
            x__ = x__.moveaxis(1, -1) # (B*S, t*h*w, out_channels)
            x_ = x__.reshape(x.shape[0], -1, x__.shape[-1]) # (B, S*t*h*w, out_channels)
        else:
            x_ = x_.moveaxis(1, -1)
            
        x = self.norm(x_)
        return x

class ImagingMaskedEncoder(nn.Module):
    def __init__(
        self,
        patch_embed_cls: str = "PatchEmbed",
        patch_size: Tuple[int, ...] = (5, 8, 8),
        patch_in_channels: int = 1,
        pixel_unshuffle_scale: int = 1,
        mask_type: str = "random",
        mask_ratio: float = 0.0,
        circular_pe: bool = False,
        use_enc_pe: bool = True,
        mask_loss: bool = True,
        shift_size: Tuple[int, ...] = (0, 0, 0),
        enc_embed_dim: int = 1025, # should be divisible by 8 or 6 for one modality, +1 for two modalities
        enc_depth: int = 6,
        enc_num_heads: int = 5,
        mlp_ratio: float = 4.0,
        grad_checkpointing: bool = False,
        img_shape: tuple = None,
        use_both_axes: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.enc_embed_dim = enc_embed_dim
        self.patch_size = patch_size
        self.circular_pe = circular_pe
        self.use_enc_pe = use_enc_pe
        self.mask_loss = mask_loss
        self.shift_size = shift_size
        self.patch_embed_cls = globals()[patch_embed_cls]
        self.img_shape = img_shape
        self.patch_in_channels = patch_in_channels
        self.use_both_axes = use_both_axes
        self.patch_p_num = np.prod(patch_size) * self.patch_in_channels
        self.grad_checkpointing = grad_checkpointing
        self.patch_embed = PatchEmbed(img_shape, 
                                    in_channels=patch_in_channels, 
                                    patch_size=patch_size, 
                                    out_channels=enc_embed_dim, )
        print(f"patch_embed.num_patches: {self.patch_embed.num_patches}")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed.out_channels))
        self.enc_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, self.patch_embed.out_channels), 
                                      requires_grad=False)
        self.encoder = nn.ModuleList([Block(dim=self.patch_embed.out_channels, 
                                            num_heads=enc_num_heads, 
                                            mlp_ratio=mlp_ratio, 
                                            qkv_bias=True,)
                                      for i in range(enc_depth)])
        self.encoder_norm = nn.LayerNorm(self.patch_embed.out_channels)
        # --------------------------------------------------------------------------
        # MAE Masker
        self.using_masker = False if mask_ratio == 0.0 else True
        if self.using_masker:
            self.masker = Masker(mask_type=mask_type, 
                                 mask_ratio=mask_ratio, 
                                 grid_size=patch_embed.grid_size)
        self.initialize_parameters()
    
    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding
        enc_pos_embed = sincos_pos_embed(self.enc_embed_dim, self.patch_embed.grid_size, cls_token=True,
                                        use_both_axes=self.use_both_axes, circular_pe=self.circular_pe)
        if not self.use_enc_pe:
            enc_pos_embed[:, 0] = 0
        self.enc_pos_embed.data.copy_(enc_pos_embed.unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm"s trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, "cls_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        """Forward pass of encoder
        input: [B, S, T, H, W] torch.Tensor
        output:
            latent: [B, 1 + length * mask_ratio, embed_dim] torch.Tensor
            mask: [B, 1 + length * mask_ratio] torch.Tensor
            ids_restore: [B, 1 + length * mask_ratio] torch.Tensor
        """
        # Embed patches: (B, S, T, H, W) -> (B, S * T * num_patches, embed_dim)
        print(f"x.shape: {x.shape}")
        x = self.patch_embed(x)
        print(f"x.shape: {x.shape}")
        
        # Add positional embedding: (B, S * T * num_patches, embed_dim)
        if self.use_enc_pe:
            enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
            x = x + enc_pos_embed[:, 1:, :]
            cls_token = self.cls_token + enc_pos_embed[:, :1, :] # (1, 1, embed_dim)
        else:
            cls_token = self.cls_token
        
        # Mask patches: length -> length * mask_ratio
        if self.using_masker:
            x, mask, ids_restore = self.masker(x)
        else:
            mask, ids_restore = None, None
            
        # Append cls token: (B, 1 + length * mask_ratio, em bed_dim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer encoder
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.encoder:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.encoder:
                x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_with_skip_connection(self, x):
        # Embed patches: (B, S, T, H, W) -> (B, S * num_patches, embed_dim)
        x = self.patch_embed(x)

        # Add positional embedding: (B, S * num_patches, embed_dim)
        if self.use_enc_pe:
            enc_pos_embed = self.enc_pos_embed.repeat(x.shape[0], 1, 1)
            x = x + enc_pos_embed[:, 1:, :]
            cls_token = self.cls_token + enc_pos_embed[:, :1, :] # (1, 1, embed_dim)
        else:
            cls_token = self.cls_token
        
        # Append cls token: (B, 1 + length * mask_ratio, embed_dim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply MAE encoder
        hidden_latents = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.encoder:
                hidden_latents.append(x[:, 1:, :])
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.encoder:
                hidden_latents.append(x[:, 1:, :])
                x = blk(x)
        encoder_output = self.encoder_norm(x)
        return encoder_output, hidden_latents


class Layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(Layer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Relu(Layer):
    """A linear layer followed by a ReLU activation function."""
    def __init__(self, in_size, out_size, dropout=False, **kwargs):
        super(Relu, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio, norm_layer=nn.LayerNorm, grad_checkpointing: bool = False):
        super(ViTDecoder, self).__init__()
        self.network = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(dim)
        self.grad_checkpointing = grad_checkpointing
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply transformer decoder
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.network:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.network:
                x = blk(x)
        x = self.norm(x)
        return x
        

class LinearDecoder(nn.Module):
    def __init__(self, in_size, dim, depth, layer_type:Layer=Relu):
        super(LinearDecoder, self).__init__()
        self.network = nn.ModuleList([layer_type(in_size, dim // 2, dropout=0.1)])
        self.network.extend([
            layer_type(dim // 2 ** i, dim // 2 ** (i + 1), dropout=0.1) for i in range(1, depth)])
        self.fc = nn.Linear(dim // 2 ** depth, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.flatten(start_dim=1)
        for layer in self.network:
            x = layer(x)
        x = self.fc(x)
        return x


class UNETR_decoder(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        upsample_kernel_sizes: Union[list, Sequence[int]],
        feature_size: int = 16,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name="batch")

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name="batch", spatial_dims=2)

        """

        super().__init__()

        patch_size = (1, *patch_size)
        self.grid_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = hidden_size
        self.slice_num = img_size[0]
        self.output_channel = out_channels * self.slice_num # times slice num
        self.upsample_kernel_sizes = upsample_kernel_sizes
        assert len(self.upsample_kernel_sizes) == 3, "Only support UNETR decoder depth equals 3"
            
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=self.upsample_kernel_sizes[1:],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=self.upsample_kernel_sizes[2],
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[1],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=self.upsample_kernel_sizes[0],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=self.output_channel)

    def proj_feat(self, x, hidden_size, grid_size):
        new_view = (x.size(0), *grid_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(grid_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in: torch.Tensor, x: torch.Tensor, hidden_states_out: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass of UNETR decoder.

        Args:
            x_in (torch.Tensor): images in the shape of (batch, slice, time, height, width)
            x (torch.Tensor): latent features extracted from the encoder
            hidden_states_out (List[torch.Tensor]): the output of each layer of encoder

        Returns:
            torch.Tensor: segmentation probability in the same shape as x_in
        """
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[0]
        proj_x2 = self.proj_feat(x2, self.hidden_size//self.grid_size[0], self.grid_size) # (batch, hidden_size, s, 2, 16, 16)
        proj_x2 = proj_x2.view(proj_x2.shape[0], -1, *self.grid_size[1:])
        enc2 = self.encoder2(proj_x2)
        x3 = hidden_states_out[1]
        proj_x3 = self.proj_feat(x3, self.hidden_size//self.grid_size[0], self.grid_size) # (batch, hidden_size, s, 2, 16, 16)
        proj_x3 = proj_x3.view(proj_x3.shape[0], -1, *self.grid_size[1:])
        enc3 = self.encoder3(proj_x3)
        
        proj_x = self.proj_feat(x, self.hidden_size//self.grid_size[0], self.grid_size)
        dec3 = proj_x.view(proj_x.shape[0], -1, *self.grid_size[1:])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        seg_out = self.out(out)
        seg_pred = seg_out.view(seg_out.shape[0], -1, self.slice_num, *seg_out.shape[2:]) # (B, 4, slice, T, H, W)
        return seg_pred


class ImagingMaskedDecoder(nn.Module):
    def __init__(self, 
                grid_size: tuple,
                head_out_dim: int,
                decoder_num_patches: int,
                use_enc_pe: bool = True,
                enc_embed_dim: int = 1025,
                dec_embed_dim: int = 1025,
                dec_num_heads: int = 5,
                dec_depth: int = 2,
                mlp_ratio: float = 4.0,
                use_both_axes: bool = False,
                *args, **kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.dec_embed_dim = dec_embed_dim
        self.use_both_axes = use_both_axes
        self.decoder_num_patches = decoder_num_patches
        self.use_enc_pe = use_enc_pe

        print(f"grid_size: {grid_size}, head_out_dim: {head_out_dim}, decoder_num_patches: {decoder_num_patches}, use_enc_pe: {use_enc_pe}, enc_embed_dim: {enc_embed_dim}, dec_embed_dim: {dec_embed_dim}, dec_num_heads: {dec_num_heads}, dec_depth: {dec_depth}, mlp_ratio: {mlp_ratio}, use_both_axes: {use_both_axes}")
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.decoder_num_patches, dec_embed_dim), requires_grad=False)
        self.decoder = ViTDecoder(dim=dec_embed_dim, 
                                  num_heads=dec_num_heads,
                                  depth=dec_depth,
                                  mlp_ratio=mlp_ratio,)
        self.recon_head = nn.Linear(in_features=dec_embed_dim,
                                    out_features=head_out_dim,)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim), requires_grad=True)
        self.initialize_parameters()
    
    def initialize_parameters(self):
        dec_pos_embed = sincos_pos_embed(self.dec_embed_dim, self.grid_size, cls_token=True,
                                         use_both_axes=self.use_both_axes)
        if not self.use_enc_pe:
            dec_pos_embed[:, 0] = 0
        self.dec_pos_embed.data.copy_(dec_pos_embed.unsqueeze(0))
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        """Forward pass of reconstruction decoder
        input:
            x: [B, 1 + length * mask_ratio, embed_dim] torch.Tensor
            ids_restore: [B, 1 + length * mask_ratio] torch.Tensor
        output:
            pred: [B, length, embed_dim] torch.Tensor
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        dec_pos_embed = self.dec_pos_embed.repeat(x.shape[0], 1, 1)
        
        # Append mask tokens and add positional embedding in schuffled order
        if ids_restore is not None:
            mask_token_n = ids_restore.shape[1] + 1 - x.shape[1]
            mask_tokens = self.mask_token.repeat(x.shape[0], mask_token_n, 1)
            x_shuffle = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_restore = torch.gather(x_shuffle, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_shuffle.shape[-1]))
            x_restore_ = x_restore + dec_pos_embed[:, 1:, :]
        else:
            x_restore_ = x[:, 1:, :] + dec_pos_embed[:, 1:, :]
        cls_tok_ = x[:, :1, :] + dec_pos_embed[:, :1, :]
        
        # Reconstruction decoder
        x = torch.cat([cls_tok_, x_restore_], dim=1) # add class token
        x = self.decoder(x) # apply transformer decoder
        
        # Reconstruction head
        x = self.recon_head(x)
        x = x[:, 1:, :] # remove cls token
        x = torch.sigmoid(x) # scale x to [0, 1] for reconstruction task
        return x


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CosineSimilarity(dim=1)
        
    # def forward(self, p1, p2, z1, z2):
    #     """y is detached from the computation of the gradient.
    #     This is inspired by paper: Chen and He, et al. 2020. Exploring simple siamese representation learning."""
    #     return -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
    
    def forward(self, p2, z1):
        """y is detached from the computation of the gradient.
        This is inspired by paper: Chen and He, et al. 2020. Exploring simple siamese representation learning."""
        return -self.criterion(p2, z1).mean()

class ReconstructionCriterion(torch.nn.Module):
    def __init__(self, loss_types, loss_weights: Optional[list[float]] = None, mask_loss: bool = False, **kwargs):
        super().__init__()
        self.max_value = 1.0
        self.loss_types = loss_types
        self.loss_weights = loss_weights if loss_weights is not None else [1.0] * len(self.loss_types)
        self.loss_fcts = []
        loss_fct_dict = {"mse": torch.nn.MSELoss(reduction="none"), "cl": ContrastiveLoss()}
        if not isinstance(self.loss_types, list):
            self.loss_fcts = [loss_fct_dict[self.loss_types]]
        else:
            self.loss_fcts += [loss_fct_dict[loss_name] for loss_name in self.loss_types]

        self.use_mask_in_loss = mask_loss
        
    def forward(self, x, y, mask: Optional[torch.Tensor] = None, **kwargs):
        """Compute reconstruction loss
        x: [B, L, D] torch.Tensor, reconstructed image
        y: [B, L, D] torch.Tensor, reference image
        mask: [B, L] torch.Tensor, mask for encoder, where 0 is keep, 1 is remove
        
        loss_dict: dict, loss values
        psnr_value: float, psnr value
        """
        assert x.shape[-1] == y.shape[-1]
        if self.use_mask_in_loss:
            masked_x = x[mask == 1]
            masked_y = y[mask == 1]
        else:
            masked_x = x
            masked_y = y
        # --------------------------------------------------------------------------
        # Calculate losses
        total_loss = 0.0
        loss_dict = {}
        for i, loss_name in enumerate(self.loss_types):
            self.loss_fct = self.loss_fcts[i]
            if loss_name == "cl":
                p2, z1 = kwargs["p2"], kwargs["z1"]
                loss = self.loss_fct(p2, z1)
                # p1, p2, z1, z2 = kwargs["p1"], kwargs["p2"], kwargs["z1"], kwargs["z2"]
                # loss = self.loss_fct(p1, p2, z1, z2)
            else:    
                loss = self.loss_fct(masked_x, masked_y)
                loss = loss.mean()
            loss_dict[loss_name] = loss
            total_loss += self.loss_weights[i] * loss
        loss_dict["loss"] = total_loss
        return loss_dict