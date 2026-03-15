from __future__ import annotations
import torch
from typing import Tuple, Sequence
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import numpy as np


def _to_tuple(value: int | Sequence[int], dims: int) -> Tuple[int, ...]:
    """
    Normalize an integer / sequence into a tuple of length ``dims``.

    Accepts:
    - scalar int: replicated ``dims`` times
    - tuple / list of ints: must already have length ``dims``
    """
    if isinstance(value, (tuple, list)):
        if len(value) != dims:
            raise ValueError(f"Expected {dims} values, got {len(value)}")
        return tuple(int(v) for v in value)
    return (int(value),) * dims


def _infer_dims(img_size: int | Tuple[int, ...]) -> int:
    return 3 if isinstance(img_size, tuple) and len(img_size) == 3 else 2


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class PatchEmbed(nn.Module):
    """Image/Volume to Patch Embedding - Supports both 2D and 3D"""
    def __init__(self, to_embed='linear', img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        
        # Handle img_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.dims = 2
        elif len(img_size) == 2:
            self.img_size = img_size
            self.dims = 2
        elif len(img_size) == 3:
            self.img_size = img_size
            self.dims = 3
        else:
            raise ValueError(f"img_size must be int or tuple of length 2 or 3, got {img_size}")
        
        # Handle patch_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(patch_size, int):
            if self.dims == 2:
                self.patch_size = (patch_size, patch_size)
            else:  # 3D
                self.patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            if self.dims == 2:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 2 not compatible with 3D img_size {img_size}")
        elif len(patch_size) == 3:
            if self.dims == 3:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 3 not compatible with 2D img_size {img_size}")
        else:
            raise ValueError(f"patch_size must be int or tuple of length 2 or 3, got {patch_size}")
        
        # Calculate grid size and number of patches
        if self.dims == 2:
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:  # 3D
            self.grid_size = (self.img_size[0] // self.patch_size[0], 
                             self.img_size[1] // self.patch_size[1], 
                             self.img_size[2] // self.patch_size[2])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.flatten = flatten
        
        if to_embed == 'conv':
            # Create appropriate convolution layer
            if self.dims == 2:
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            else:  # 3D
                self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        elif to_embed == 'siren':
            self.proj = nn.Sequential(
                    SineLayer(in_chans, embed_dim, is_first=True, omega_0=30.),
                    SineLayer(embed_dim, embed_dim * 2, is_first=False, omega_0=30)
                )
        
        elif to_embed == 'linear':
            self.proj = nn.Linear(in_chans, embed_dim)
        elif to_embed == 'identity':
            self.proj = nn.Identity()
        else:
            raise NotImplementedError

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        if self.dims == 2:
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        else:  # 3D
            B, C, D, H, W = x.shape
            assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
                f"Input volume size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        x = self.proj(x)
        
        if self.flatten:
            if self.dims == 2:
                x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            else:  # 3D
                x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNC
        
        x = self.norm(x)
        return x


class ToPixel(nn.Module):
    def __init__(self, to_pixel='linear', img_size=256, out_channels=3, in_dim=512, patch_size=16) -> None:
        super().__init__()
        self.to_pixel_name = to_pixel
        
        # Handle img_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
            self.dims = 2
        elif len(img_size) == 2:
            self.img_size = img_size
            self.dims = 2
        elif len(img_size) == 3:
            self.img_size = img_size
            self.dims = 3
        else:
            raise ValueError(f"img_size must be int or tuple of length 2 or 3, got {img_size}")
        
        # Handle patch_size: can be int (2D), tuple of 2 (2D), or tuple of 3 (3D)
        if isinstance(patch_size, int):
            if self.dims == 2:
                self.patch_size = (patch_size, patch_size)
            else:  # 3D
                self.patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            if self.dims == 2:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 2 not compatible with 3D img_size {img_size}")
        elif len(patch_size) == 3:
            if self.dims == 3:
                self.patch_size = patch_size
            else:
                raise ValueError(f"patch_size tuple of length 3 not compatible with 2D img_size {img_size}")
        else:
            raise ValueError(f"patch_size must be int or tuple of length 2 or 3, got {patch_size}")
        
        # Calculate grid size and number of patches
        if self.dims == 2:
            self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:  # 3D
            self.grid_size = (self.img_size[0] // self.patch_size[0], 
                             self.img_size[1] // self.patch_size[1], 
                             self.img_size[2] // self.patch_size[2])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.out_channels = out_channels
        if to_pixel == 'linear':
            if self.dims == 2:
                self.model = nn.Linear(in_dim, out_channels * self.patch_size[0] * self.patch_size[1])
            else:  # 3D
                self.model = nn.Linear(in_dim, out_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        elif to_pixel == 'conv':
            if self.dims == 2:
                self.model = nn.Sequential(
                    Rearrange("b (h w) c -> b c h w", h=self.grid_size[0], w=self.grid_size[1]),
                    nn.Conv2d(in_dim, self.patch_size[0] * self.patch_size[1] * out_channels, 1, padding=0),
                    Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_size[0], p2=self.patch_size[1]),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)
                    )
            else:  # 3D
                self.model = nn.Sequential(
                    Rearrange("b (d h w) c -> b c d h w", d=self.grid_size[0], h=self.grid_size[1], w=self.grid_size[2]),
                    nn.Conv3d(in_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * out_channels, 1, padding=0),
                    Rearrange("b (p1 p2 p3 c) d h w -> b c (d p1) (h p2) (w p3)", p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2]),
                    nn.Conv3d(out_channels, out_channels, 3, padding=1)
                    )
        elif to_pixel == 'siren':
            if self.dims == 2:
                self.model = nn.Sequential(
                    SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                    SineLayer(in_dim * 2, self.img_size[0] // self.patch_size[0] * self.patch_size[0] * out_channels, is_first=False, omega_0=30)
                )
            else:  # 3D
                self.model = nn.Sequential(
                    SineLayer(in_dim, in_dim * 2, is_first=True, omega_0=30.),
                    SineLayer(in_dim * 2, self.img_size[0] // self.patch_size[0] * self.patch_size[0] * out_channels, is_first=False, omega_0=30)
                )
        elif to_pixel == 'identity' or to_pixel == 'none':
            self.model = nn.Identity()
        else:
            raise NotImplementedError

    def get_last_layer(self):
        if self.to_pixel_name == 'linear':
            return self.model.weight
        elif self.to_pixel_name == 'siren':
            return self.model[1].linear.weight
        elif self.to_pixel_name == 'conv':
            return self.model[-1].weight
        else:
            return None

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * out_channels) for 2D 
        (N, L, patch_size**3 * out_channels) for 3D
        imgs: (N, out_channels, H, W) for 2D 
            (N, out_channels, D, H, W) for 3D
        """
        B = x.shape[0]
        C = self.out_channels

        if self.dims == 2:
            p_h, p_w = self.patch_size
            H, W = self.img_size
            h, w = H // p_h, W // p_w
            assert h * w == x.shape[1], f"Expected {h*w} patches, got {x.shape[1]}"
            imgs = rearrange(
                x, "b (h w) (ph pw c) -> b c (h ph) (w pw)",
                h=h, w=w, ph=p_h, pw=p_w, c=C
            )

        else:  # 3D
            p_d, p_h, p_w = self.patch_size
            D, H, W = self.img_size
            d, h, w = D // p_d, H // p_h, W // p_w
            assert d * h * w == x.shape[1], f"Expected {d*h*w} patches, got {x.shape[1]}"
            imgs = rearrange(
                x, "b (d h w) (pd ph pw c) -> b c (d pd) (h ph) (w pw)",
                d=d, h=h, w=w, pd=p_d, ph=p_h, pw=p_w, c=C
            )
        return imgs


    def forward(self, x):
        if self.to_pixel_name == 'linear':
            x = self.model(x)
            x = self.unpatchify(x)
        elif self.to_pixel_name == 'siren':
            x = self.model(x)
            if self.dims == 2:
                x = x.view(x.shape[0], self.out_channels, self.patch_size[0] * int(self.num_patches ** 0.5),
                           self.patch_size[0] * int(self.num_patches ** 0.5))
            else:  # 3D
                d = self.img_size[0] // self.patch_size[0]
                h = self.img_size[1] // self.patch_size[1]
                w = self.img_size[2] // self.patch_size[2]
                x = x.view(x.shape[0], self.out_channels, self.patch_size[0] * d,
                           self.patch_size[1] * h, self.patch_size[2] * w)
        elif self.to_pixel_name == 'conv':
            x = self.model(x)
        elif self.to_pixel_name == 'identity':
            x = self.model(x)
        elif self.to_pixel_name == 'none':
            x = self.model(x)
            x = self.unpatchify(x)
        return x