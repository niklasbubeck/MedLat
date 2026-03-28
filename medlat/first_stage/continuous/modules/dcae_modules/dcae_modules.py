"""
DCAE (Deep Convolutional Autoencoder) modules following the same pattern as ldm_modules.py
Adopted from EfficientViT's DCAE implementation
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Tuple

logger = logging.getLogger(__name__)

# Import DCAE building blocks
try:
    from .nn.act import build_act
    from .nn.norm import build_norm
    from .nn.ops import (
        ChannelDuplicatingPixelUnshuffleUpSampleLayer,
        ConvLayer,
        ConvPixelShuffleUpSampleLayer,
        ConvPixelUnshuffleDownSampleLayer,
        EfficientViTBlock,
        IdentityLayer,
        InterpolateConvUpSampleLayer,
        OpSequential,
        PixelUnshuffleChannelAveragingDownSampleLayer,
        ResBlock,
        ResidualBlock,
    )
except ImportError:
    raise ImportError(
        "DCAE modules require the dcae package. Please ensure dcae is installed and in your path."
    )


def build_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    norm: Optional[str],
    act: Optional[str],
    dims: int = 2,
) -> nn.Module:
    """Build a single block (ResBlock or EfficientViTBlock). EfficientViT blocks are 2D only.

    EViT_GLU / EViTS5_GLU use EfficientViTBlock: LiteMLA (lightweight linear attention over
    spatial dim, with Q,K,V from convs) + GLUMBConv (convolutions with GLU). Not a
    standard Vision Transformer backbone; see EfficientViTBlock docstring in nn.ops.
    """
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            dims=dims,
        )
        block = ResidualBlock(main_block, IdentityLayer())
    elif block_type == "EViT_GLU":
        if dims != 2:
            raise ValueError("EfficientViT blocks (EViT_GLU, EViTS5_GLU) are only supported for 2D (dims=2). Use ResBlock for 3D.")
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=())
    elif block_type == "EViTS5_GLU":
        if dims != 2:
            raise ValueError("EfficientViT blocks (EViT_GLU, EViTS5_GLU) are only supported for 2D (dims=2). Use ResBlock for 3D.")
        assert in_channels == out_channels
        block = EfficientViTBlock(in_channels, norm=norm, act_func=act, local_module="GLUMBConv", scales=(5,))
    else:
        raise ValueError(f"block_type {block_type} is not supported")
    return block


def build_stage_main(
    width: int,
    depth: int,
    block_type: Union[str, List[str]],
    norm: str,
    act: str,
    input_width: int,
    dims: int = 2,
) -> List[nn.Module]:
    """Build the main blocks for a stage."""
    assert isinstance(block_type, str) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            in_channels=width if d > 0 else input_width,
            out_channels=width,
            norm=norm,
            act=act,
            dims=dims,
        )
        stage.append(block)
    return stage


def build_downsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    dims: int = 2,
) -> nn.Module:
    """Build a downsampling block."""
    if block_type == "Conv":
        block = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_bias=True,
            norm=None,
            act_func=None,
            dims=dims,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2, dims=dims
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")

    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2, dims=dims
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    dims: int = 2,
) -> nn.Module:
    """Build an upsampling block."""
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2, dims=dims
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2, dims=dims
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")

    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2, dims=dims
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


class DCAEEncoder(nn.Module):
    """
    DCAE Encoder following the same pattern as ldm_modules.Encoder
    
    Args:
        in_channels: Number of input channels
        z_channels: Number of latent channels (output channels)
        width_list: List of channel widths for each stage
        depth_list: List of depths (number of blocks) for each stage
        block_type: Block type string or list of block types per stage
        norm: Normalization type (e.g., "trms2d", "bn2d")
        act: Activation type (e.g., "silu", "relu")
        downsample_block_type: Type of downsampling block ("Conv" or "ConvPixelUnshuffle")
        downsample_match_channel: Whether to match channels in downsampling
        downsample_shortcut: Shortcut type for downsampling ("averaging" or None)
        out_norm: Output normalization type
        out_act: Output activation type
        out_shortcut: Output shortcut type ("averaging" or None)
        double_z: Whether to output double channels (for VAE)
        dims: Number of spatial dimensions (2 or 3)
        img_size: Input image size (for computing vae_stride)
    """
    def __init__(
        self,
        *,
        in_channels=3,
        z_channels=32,
        width_list=(128, 256, 512, 512, 1024, 1024),
        depth_list=(2, 2, 2, 2, 2, 2),
        block_type="ResBlock",
        norm="trms2d",
        act="silu",
        downsample_block_type="ConvPixelUnshuffle",
        downsample_match_channel=True,
        downsample_shortcut="averaging",
        out_norm=None,
        out_act=None,
        out_shortcut="averaging",
        project_out_conv_only=False,
        double_z=False,
        dims=2,
        img_size=256,
        **ignore_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.dims = dims
        self.num_stages = len(width_list)
        self.vae_stride = 2 ** (self.num_stages - 1)
        
        assert len(depth_list) == self.num_stages
        assert len(width_list) == self.num_stages
        assert isinstance(block_type, str) or (
            isinstance(block_type, list) and len(block_type) == self.num_stages
        )

        # Project in
        project_in_factor = 1 if depth_list[0] > 0 else 2
        project_in_out_channels = width_list[0] if depth_list[0] > 0 else width_list[1]
        
        if project_in_factor == 1:
            self.project_in = ConvLayer(
                in_channels=in_channels,
                out_channels=project_in_out_channels,
                kernel_size=3,
                stride=1,
                use_bias=True,
                norm=None,
                act_func=None,
                dims=self.dims,
            )
        elif project_in_factor == 2:
            self.project_in = build_downsample_block(
                block_type=downsample_block_type,
                in_channels=in_channels,
                out_channels=project_in_out_channels,
                shortcut=None,
                dims=self.dims,
            )
        else:
            raise ValueError(f"downsample factor {project_in_factor} is not supported for encoder project in")

        # Build stages
        self.stages = nn.ModuleList()
        for stage_id, (width, depth) in enumerate(zip(width_list, depth_list)):
            current_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage_blocks = build_stage_main(
                width=width,
                depth=depth,
                block_type=current_block_type,
                norm=norm,
                act=act,
                input_width=width,
                dims=self.dims,
            )

            if stage_id < self.num_stages - 1 and depth > 0:
                downsample_block = build_downsample_block(
                    block_type=downsample_block_type,
                    in_channels=width,
                    out_channels=width_list[stage_id + 1] if downsample_match_channel else width,
                    shortcut=downsample_shortcut,
                    dims=self.dims,
                )
                stage_blocks.append(downsample_block)

            self.stages.append(OpSequential(stage_blocks))

        # Project out
        out_channels = 2 * z_channels if double_z else z_channels
        if project_out_conv_only:
            # Single conv at op_list.0 to match pretrained DCAE checkpoint keys
            project_out_layers = [
                ConvLayer(
                    in_channels=width_list[-1],
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    use_bias=True,
                    norm=None,
                    act_func=None,
                    dims=self.dims,
                ),
            ]
        else:
            project_out_layers = [
                build_norm(out_norm, num_features=width_list[-1], dims=self.dims) if out_norm else IdentityLayer(),
                build_act(out_act) if out_act else IdentityLayer(),
                ConvLayer(
                    in_channels=width_list[-1],
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    use_bias=True,
                    norm=None,
                    act_func=None,
                    dims=self.dims,
                ),
            ]

        if out_shortcut == "averaging":
            shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
                in_channels=width_list[-1], out_channels=out_channels, factor=1, dims=self.dims
            )
            self.project_out = ResidualBlock(OpSequential(project_out_layers), shortcut_block)
        else:
            self.project_out = OpSequential(project_out_layers)

    def forward(self, x):
        x = self.project_in(x)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x


class DCAEDecoder(nn.Module):
    """
    DCAE Decoder following the same pattern as ldm_modules.Decoder
    
    Args:
        in_channels: Number of input channels (for output)
        z_channels: Number of latent channels (input channels)
        width_list: List of channel widths for each stage
        depth_list: List of depths (number of blocks) for each stage
        block_type: Block type string or list of block types per stage
        norm: Normalization type (string or list per stage)
        act: Activation type (string or list per stage)
        upsample_block_type: Type of upsampling block ("ConvPixelShuffle" or "InterpolateConv")
        upsample_match_channel: Whether to match channels in upsampling
        upsample_shortcut: Shortcut type for upsampling ("duplicating" or None)
        in_shortcut: Input shortcut type ("duplicating" or None)
        out_norm: Output normalization type
        out_act: Output activation type
        dims: Number of spatial dimensions (2 or 3)
        img_size: Input image size (for computing z_shape)
    """
    def __init__(
        self,
        *,
        in_channels=3,
        z_channels=32,
        width_list=(128, 256, 512, 512, 1024, 1024),
        depth_list=(2, 2, 2, 2, 2, 2),
        block_type="ResBlock",
        norm="trms2d",
        act="silu",
        upsample_block_type="ConvPixelShuffle",
        upsample_match_channel=True,
        upsample_shortcut="duplicating",
        in_shortcut="duplicating",
        out_norm="trms2d",
        out_act="relu",
        dims=2,
        img_size=256,
        **ignore_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.dims = dims
        self.num_stages = len(width_list)
        
        assert len(depth_list) == self.num_stages
        assert len(width_list) == self.num_stages
        assert isinstance(block_type, str) or (
            isinstance(block_type, list) and len(block_type) == self.num_stages
        )
        assert isinstance(norm, str) or (isinstance(norm, list) and len(norm) == self.num_stages)
        assert isinstance(act, str) or (isinstance(act, list) and len(act) == self.num_stages)

        # Compute z_shape
        curr_res = img_size // 2 ** (self.num_stages - 1)
        if dims == 2:
            self.z_shape = (1, z_channels, curr_res, curr_res)
        else:  # dims == 3
            self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
        
        logger.info("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)
        ))

        # Project in
        project_in_block = ConvLayer(
            in_channels=z_channels,
            out_channels=width_list[-1],
            kernel_size=3,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            dims=self.dims,
        )

        if in_shortcut == "duplicating":
            shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
                in_channels=z_channels, out_channels=width_list[-1], factor=1, dims=self.dims
            )
            self.project_in = ResidualBlock(project_in_block, shortcut_block)
        else:
            self.project_in = project_in_block

        # Build stages (in reverse order)
        self.stages = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(width_list, depth_list)))):
            stage_blocks = []
            
            if stage_id < self.num_stages - 1 and depth > 0:
                upsample_block = build_upsample_block(
                    block_type=upsample_block_type,
                    in_channels=width_list[stage_id + 1],
                    out_channels=width if upsample_match_channel else width_list[stage_id + 1],
                    shortcut=upsample_shortcut,
                    dims=self.dims,
                )
                stage_blocks.append(upsample_block)

            current_block_type = block_type[stage_id] if isinstance(block_type, list) else block_type
            current_norm = norm[stage_id] if isinstance(norm, list) else norm
            current_act = act[stage_id] if isinstance(act, list) else act

            stage_blocks.extend(
                build_stage_main(
                    width=width,
                    depth=depth,
                    block_type=current_block_type,
                    norm=current_norm,
                    act=current_act,
                    input_width=(
                        width if upsample_match_channel else width_list[min(stage_id + 1, self.num_stages - 1)]
                    ),
                    dims=self.dims,
                )
            )
            self.stages.insert(0, OpSequential(stage_blocks))
        
        self.stages = nn.ModuleList(self.stages)

        # Project out
        project_out_factor = 1 if depth_list[0] > 0 else 2
        project_out_in_channels = width_list[0] if depth_list[0] > 0 else width_list[1]
        
        project_out_layers = [
            build_norm(out_norm, num_features=project_out_in_channels, dims=self.dims) if out_norm else IdentityLayer(),
            build_act(out_act) if out_act else IdentityLayer(),
        ]

        if project_out_factor == 1:
            project_out_layers.append(
                ConvLayer(
                    in_channels=project_out_in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    use_bias=True,
                    norm=None,
                    act_func=None,
                    dims=self.dims,
                )
            )
        elif project_out_factor == 2:
            project_out_layers.append(
                build_upsample_block(
                    block_type=upsample_block_type,
                    in_channels=project_out_in_channels,
                    out_channels=in_channels,
                    shortcut=None,
                    dims=self.dims,
                )
            )
        else:
            raise ValueError(f"upsample factor {project_out_factor} is not supported for decoder project out")
        
        self.project_out = OpSequential(project_out_layers)

    def forward(self, z):
        x = self.project_in(z)
        for stage in reversed(self.stages):
            if len(stage.op_list) == 0:
                continue
            x = stage(x)
        x = self.project_out(x)
        return x

    def get_last_layer(self) -> nn.Parameter:
        """Get the last layer weights (for loss computation)."""
        return self.project_out.op_list[-1].conv.weight if hasattr(self.project_out.op_list[-1], 'conv') else None
