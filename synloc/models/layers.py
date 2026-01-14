"""
Basic building blocks for YOLOX-Pose.

Standalone implementations replacing mmcv.cnn dependencies.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block.

    Replaces mmcv.cnn.ConvModule with a simpler, standalone implementation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Default: 1.
        padding: Convolution padding. If None, auto-computed as kernel_size // 2.
        groups: Convolution groups. Default: 1.
        bias: Whether to use bias. Default: False (disabled when using BatchNorm).
        norm: Whether to use BatchNorm. Default: True.
        act: Activation type ('silu', 'swish', 'relu', 'none'). Default: 'silu'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = False,
        norm: bool = True,
        act: str = 'silu'
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001) if norm else nn.Identity()

        if act in ('silu', 'swish'):
            self.act = nn.SiLU(inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'none':
            self.act = nn.Identity()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block.

    Consists of a depthwise convolution followed by a pointwise (1x1) convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride. Default: 1.
        padding: Depthwise convolution padding. If None, auto-computed.
        norm: Whether to use BatchNorm. Default: True.
        act: Activation type. Default: 'silu'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: bool = True,
        act: str = 'silu'
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        # Depthwise convolution
        self.depthwise = ConvBNAct(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, norm=norm, act=act
        )
        # Pointwise convolution
        self.pointwise = ConvBNAct(
            in_channels, out_channels, 1, norm=norm, act=act
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class ChannelAttention(nn.Module):
    """Channel attention module.

    Applies global average pooling followed by a 1x1 conv and hard sigmoid
    to compute channel-wise attention weights.

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class DarknetBottleneck(nn.Module):
    """Basic bottleneck block used in Darknet.

    Each block consists of two ConvBNAct layers with a residual connection.
    First conv is 1x1, second is 3x3.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expansion: Hidden channel expansion ratio. Default: 0.5.
        add_identity: Whether to add residual connection. Default: True.
        use_depthwise: Whether to use depthwise separable conv. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)

        if use_depthwise:
            self.conv2 = DepthwiseSeparableConv(hidden_channels, out_channels, 3)
        else:
            self.conv2 = ConvBNAct(hidden_channels, out_channels, 3)

        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    Splits input into two paths: one goes through bottleneck blocks,
    the other is a shortcut. Both paths are concatenated and processed
    by a final convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expand_ratio: Channel expansion ratio. Default: 0.5.
        num_blocks: Number of bottleneck blocks. Default: 1.
        add_identity: Whether to add identity in blocks. Default: True.
        use_depthwise: Whether to use depthwise conv in blocks. Default: False.
        channel_attention: Whether to add channel attention. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        channel_attention: bool = False
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention

        self.main_conv = ConvBNAct(in_channels, mid_channels, 1)
        self.short_conv = ConvBNAct(in_channels, mid_channels, 1)
        self.final_conv = ConvBNAct(2 * mid_channels, out_channels, 1)

        self.blocks = nn.Sequential(*[
            DarknetBottleneck(
                mid_channels,
                mid_channels,
                expansion=1.0,
                add_identity=add_identity,
                use_depthwise=use_depthwise
            ) for _ in range(num_blocks)
        ])

        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)

        return self.final_conv(x_final)


class Focus(nn.Module):
    """Focus layer that reduces spatial dimensions by 2x while increasing channels.

    Slices input into 4 patches and concatenates them along channel dimension,
    effectively converting spatial information into channel information.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size. Default: 1.
        stride: Convolution stride. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1
    ):
        super().__init__()
        self.conv = ConvBNAct(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x: Tensor) -> Tensor:
        # Shape: (b, c, h, w) -> (b, 4c, h/2, w/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]

        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right),
            dim=1
        )
        return self.conv(x)


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling bottleneck layer.

    Applies multiple max pooling operations with different kernel sizes
    and concatenates results for multi-scale feature aggregation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_sizes: Tuple of pooling kernel sizes. Default: (5, 9, 13).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, ...] = (5, 9, 13)
    ):
        super().__init__()
        mid_channels = in_channels // 2

        self.conv1 = ConvBNAct(in_channels, mid_channels, 1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])

        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        return self.conv2(x)
