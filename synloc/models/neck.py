"""
YOLOX Path Aggregation Feature Pyramid Network (PAFPN).

Standalone implementation extracted from mmpose.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

from .layers import ConvBNAct, DepthwiseSeparableConv, CSPLayer


class YOLOXPAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network used in YOLOX.

    Combines top-down and bottom-up feature aggregation with CSP blocks.

    Args:
        in_channels: Number of input channels per scale. E.g., [128, 256, 512].
        out_channels: Number of output channels (same for all scales).
        num_csp_blocks: Number of bottlenecks in CSP blocks. Default: 3.
        use_depthwise: Whether to use depthwise separable conv. Default: False.

    Example:
        >>> neck = YOLOXPAFPN(in_channels=[128, 256, 512], out_channels=128)
        >>> features = (torch.randn(1, 128, 80, 80),
        ...             torch.randn(1, 256, 40, 40),
        ...             torch.randn(1, 512, 20, 20))
        >>> outputs = neck(features)
        >>> for out in outputs:
        ...     print(out.shape)
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Top-down pathway: reduce channels and CSP blocks
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()

        for idx in range(len(in_channels) - 1, 0, -1):
            # 1x1 conv to reduce channels
            self.reduce_layers.append(
                ConvBNAct(in_channels[idx], in_channels[idx - 1], 1)
            )
            # CSP block for feature fusion
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise
                )
            )

        # Bottom-up pathway: downsample and CSP blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        for idx in range(len(in_channels) - 1):
            # 3x3 conv with stride 2 to downsample
            if use_depthwise:
                self.downsamples.append(
                    DepthwiseSeparableConv(in_channels[idx], in_channels[idx], 3, stride=2)
                )
            else:
                self.downsamples.append(
                    ConvBNAct(in_channels[idx], in_channels[idx], 3, stride=2, padding=1)
                )
            # CSP block for feature fusion
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise
                )
            )

        # Output convolutions to unify channel dimensions
        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvBNAct(in_channels[i], out_channels, 1)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Forward pass.

        Args:
            inputs: Tuple of feature tensors from backbone, ordered from
                    smallest to largest spatial size.

        Returns:
            Tuple of feature tensors with unified channel dimensions.
        """
        assert len(inputs) == len(self.in_channels)

        # Top-down pathway
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]

            # Reduce channels of higher level feature
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            # Upsample and concatenate with lower level feature
            upsample_feat = self.upsample(feat_high)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # Bottom-up pathway
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            # Downsample and concatenate with higher level feature
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        # Apply output convolutions
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
