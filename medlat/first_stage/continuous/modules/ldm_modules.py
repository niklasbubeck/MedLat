# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
import torch.nn as nn
import numpy as np

### GENERAL ###

def get_conv_layer(dims):
    if dims == 2:
        return nn.Conv2d
    elif dims == 3:
        return nn.Conv3d
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")


def get_interpolate_mode(dims):
    if dims == 2:
        return "nearest"
    elif dims == 3:
        return "nearest"
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, dims=2):
        super().__init__()
        self.with_conv = with_conv
        self.dims = dims
        if self.with_conv:
            conv_layer = get_conv_layer(dims)
            kernel_size = 3
            self.conv = conv_layer(
                in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=get_interpolate_mode(self.dims))
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, dims=2):
        super().__init__()
        self.with_conv = with_conv
        self.dims = dims
        if self.with_conv:
            conv_layer = get_conv_layer(dims)
            kernel_size = 3
            self.conv = conv_layer(
                in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad_size = (0, 1) * self.dims  # Creates appropriate padding tuple for 2D/3D
            x = torch.nn.functional.pad(x, pad_size, mode="constant", value=0)
            x = self.conv(x)
        else:
            if self.dims == 2:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            else:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        dims=2
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.dims = dims
        conv_layer = get_conv_layer(dims)

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_layer(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_layer(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_layer(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = conv_layer(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            temb_proj = self.temb_proj(nonlinearity(temb))
            # Broadcast temb to match spatial dimensions
            if self.dims == 2:
                h = h + temb_proj[:, :, None, None]
            else:  # dims == 3
                h = h + temb_proj[:, :, None, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, dims=2):
        super().__init__()
        self.in_channels = in_channels
        self.dims = dims
        conv_layer = get_conv_layer(dims)

        self.norm = Normalize(in_channels)
        self.q = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = conv_layer(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dims == 2:
            b, c, h, w = q.shape
            q = q.reshape(b, c, h * w)
            k = k.reshape(b, c, h * w)
            v = v.reshape(b, c, h * w)
        else:  # dims == 3
            b, c, d, h, w = q.shape
            q = q.reshape(b, c, d * h * w)
            k = k.reshape(b, c, d * h * w)
            v = v.reshape(b, c, d * h * w)

        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)

        if self.dims == 2:
            h_ = h_.reshape(b, c, h, w)
        else:  # dims == 3
            h_ = h_.reshape(b, c, d, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        img_size=256,
        z_channels=16,
        double_z=True,
        dims=2,
        ignore_mid_attn=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.vae_stride = 2 ** (len(ch_mult) - 1)
        self.num_res_blocks = num_res_blocks
        
        # Handle img_size: support both int (backward compat) and tuple/list (non-cubic)
        if isinstance(img_size, (int, float)):
            # Backward compatibility: convert int to tuple
            if dims == 2:
                self.resolution = (int(img_size),) * 2
            else:  # dims == 3
                self.resolution = (int(img_size),) * 3
        elif isinstance(img_size, (tuple, list)):
            self.resolution = tuple(int(s) for s in img_size)
            if len(self.resolution) != dims:
                raise ValueError(f"img_size tuple length {len(self.resolution)} doesn't match dims {dims}")
        else:
            raise ValueError(f"img_size must be int or tuple/list, got {type(img_size)}")
        
        self.in_channels = in_channels
        self.dims = dims
        self.ignore_mid_attn = ignore_mid_attn
        self.z_channels = z_channels

        conv_layer = get_conv_layer(dims)

        # downsampling
        self.conv_in = conv_layer(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = list(self.resolution)  # Track resolution as list for easy modification
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        dims=dims
                    )
                )
                block_in = block_out
                # Check if any dimension matches attn_resolutions
                if any(res in attn_resolutions for res in curr_res):
                    attn.append(AttnBlock(block_in, dims=dims))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, dims=dims)
                # Downsample each dimension
                curr_res = [r // 2 for r in curr_res]
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims
        )
        if not ignore_mid_attn:
            self.mid.attn_1 = AttnBlock(block_in, dims=dims)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_layer(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        if not self.ignore_mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        img_size=256,
        z_channels=16,
        give_pre_end=False,
        dims=2,
        ignore_mid_attn=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        # Handle img_size: support both int (backward compat) and tuple/list (non-cubic)
        if isinstance(img_size, (int, float)):
            # Backward compatibility: convert int to tuple
            if dims == 2:
                self.resolution = (int(img_size),) * 2
            else:  # dims == 3
                self.resolution = (int(img_size),) * 3
        elif isinstance(img_size, (tuple, list)):
            self.resolution = tuple(int(s) for s in img_size)
            if len(self.resolution) != dims:
                raise ValueError(f"img_size tuple length {len(self.resolution)} doesn't match dims {dims}")
        else:
            raise ValueError(f"img_size must be int or tuple/list, got {type(img_size)}")
        
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.dims = dims
        self.ignore_mid_attn = ignore_mid_attn
        conv_layer = get_conv_layer(dims)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # Calculate curr_res for each dimension
        curr_res = [r // (2 ** (self.num_resolutions - 1)) for r in self.resolution]
        
        # Adjust z_shape based on dimensions (non-cubic support)
        if dims == 2:
            self.z_shape = (1, z_channels, curr_res[0], curr_res[1])
        else:  # dims == 3
            self.z_shape = (1, z_channels, curr_res[0], curr_res[1], curr_res[2])
            
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = conv_layer(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims
        )
        if not ignore_mid_attn:
            self.mid.attn_1 = AttnBlock(block_in, dims=dims)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            dims=dims
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        dims=dims
                    )
                )
                block_in = block_out
                # Check if any dimension matches attn_resolutions
                if any(res in attn_resolutions for res in curr_res):
                    attn.append(AttnBlock(block_in, dims=dims))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, dims=dims)
                # Upsample each dimension
                curr_res = [r * 2 for r in curr_res]
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_layer(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        if not self.ignore_mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self) -> nn.Parameter:
        return self.conv_out.weight

### FOR AEKL ###

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, channel_dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=channel_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.tensor([0.0], device=self.mean.device)
        else:
            reduce_dims = list(range(1, self.mean.ndim))  # all dims except batch
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dims
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dims
                )

    def nll(self, sample):
        if self.deterministic:
            return torch.tensor([0.0], device=self.mean.device)
        logtwopi = np.log(2.0 * np.pi)
        reduce_dims = list(range(1, self.mean.ndim))  # sum over everything but batch
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=reduce_dims
        )

    def mode(self):
        return self.mean
    

# ### ViTVQ ### 

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)
#     elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
#         w = m.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

# class ViTEncoder(nn.Module):
#     def __init__(self, image_size: Union[Tuple[int, ...], int], patch_size: Union[Tuple[int, ...], int],
#                  dim: int, depth: int = 12, num_heads: int = 12, mlp_ratio: int = 4, channels: int = 3, dims: int = 2, **kwargs) -> None:
#         super().__init__()
#         if isinstance(image_size, int):
#             image_size = (image_size,) * dims
#         if isinstance(patch_size, int):
#             patch_size = (patch_size,) * dims

#         assert all(i % p == 0 for i, p in zip(image_size, patch_size)), 'Input dimensions must be divisible by the patch size.'
#         pos_embedding = get_sincos_pos_embed(dim, tuple(i // p for i, p in zip(image_size, patch_size)), dims)

#         if dims == 2:
#             self.to_patch_embedding = nn.Sequential(
#                 nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
#                 Rearrange('b c h w -> b (h w) c'),
#             )
#         elif dims == 3:
#             self.to_patch_embedding = nn.Sequential(
#                 nn.Conv3d(channels, dim, kernel_size=patch_size, stride=patch_size),
#                 Rearrange('b c d h w -> b (d h w) c'),
#             )
#         else:
#             raise ValueError("dims must be 2 or 3.")

#         self.pos_embedding = nn.Parameter(torch.from_numpy(pos_embedding).float().unsqueeze(0), requires_grad=False)
#         self.encoder_blocks = nn.ModuleList([
#             Block(dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
#                   proj_drop=0.1, attn_drop=0.1) for _ in range(depth)])

#         self.apply(init_weights)

#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         x = self.to_patch_embedding(x)
#         x = x + self.pos_embedding
#         for block in self.encoder_blocks:
#             x = block(x)
#         return x

# class ViTDecoder(nn.Module):
#     def __init__(self, image_size: Union[Tuple[int, ...], int], patch_size: Union[Tuple[int, ...], int],
#                  dim: int, depth: int, num_heads: int, mlp_ratio: int, channels: int = 3, dims: int = 2, **kwargs) -> None:
#         super().__init__()
#         if isinstance(image_size, int):
#             image_size = (image_size,) * dims
#         if isinstance(patch_size, int):
#             patch_size = (patch_size,) * dims

#         assert all(o % p == 0 for o, p in zip(image_size, patch_size)), 'Output dimensions must be divisible by the patch size.'
#         pos_embedding = get_sincos_pos_embed(dim, tuple(o // p for o, p in zip(image_size, patch_size)), dims)

#         self.decoder_blocks = nn.ModuleList([
#             Block(dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
#                   proj_drop=0.1, attn_drop=0.1) for _ in range(depth)])
#         self.pos_embedding = nn.Parameter(torch.from_numpy(pos_embedding).float().unsqueeze(0), requires_grad=False)

#         if dims == 2:
#             self.to_pixel = nn.Sequential(
#                 Rearrange('b (h w) c -> b c h w', h=image_size[0] // patch_size[0]),
#                 nn.ConvTranspose2d(dim, channels, kernel_size=patch_size, stride=patch_size)
#             )
#         elif dims == 3:
#             self.to_pixel = nn.Sequential(
#                 Rearrange('b (d h w) c -> b c d h w', d=image_size[0] // patch_size[0]),
#                 nn.ConvTranspose3d(dim, channels, kernel_size=patch_size, stride=patch_size)
#             )
#         else:
#             raise ValueError("dims must be 2 or 3.")

#         self.apply(init_weights)

#     def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
#         x = token + self.pos_embedding
#         for block in self.decoder_blocks:
#             x = block(x)
#         x = self.to_pixel(x)
#         return x

#     def get_last_layer(self) -> nn.Parameter:
#         return self.to_pixel[-1].weight