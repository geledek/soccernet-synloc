"""
CSPDarknet backbone for YOLOX-Pose.

Standalone implementation extracted from mmpose.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Tuple, List, Optional, Sequence

from .layers import ConvBNAct, DepthwiseSeparableConv, CSPLayer, Focus, SPPBottleneck


class CSPDarknet(nn.Module):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch: Architecture type ('P5' or 'P6'). Default: 'P5'.
        deepen_factor: Depth multiplier for CSP blocks. Default: 1.0.
        widen_factor: Width multiplier for channels. Default: 1.0.
        out_indices: Output from which stages. Default: (2, 3, 4).
        frozen_stages: Stages to freeze (-1 means none). Default: -1.
        use_depthwise: Whether to use depthwise separable conv. Default: False.
        spp_kernel_sizes: Kernel sizes for SPP layer. Default: (5, 9, 13).
        norm_eval: Whether to set norm layers to eval mode. Default: False.

    Example:
        >>> model = CSPDarknet(deepen_factor=0.33, widen_factor=0.5)  # Small variant
        >>> x = torch.randn(1, 3, 640, 640)
        >>> outputs = model(x)
        >>> for out in outputs:
        ...     print(out.shape)
    """

    # Architecture settings: [in_channels, out_channels, num_blocks, add_identity, use_spp]
    ARCH_SETTINGS = {
        'P5': [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 1024, 3, False, True]
        ],
        'P6': [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, False, True]
        ]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        spp_kernel_sizes: Tuple[int, ...] = (5, 9, 13),
        norm_eval: bool = False
    ):
        super().__init__()
        arch_setting = self.ARCH_SETTINGS[arch]

        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(
                f'frozen_stages must be in range(-1, {len(arch_setting) + 1}). '
                f'Got {frozen_stages}'
            )

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval

        # Stem layer (Focus)
        self.stem = Focus(
            3,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3
        )
        self.layers = ['stem']

        # Build stages
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)

            stage = []

            # Downsample convolution
            if use_depthwise:
                conv_layer = DepthwiseSeparableConv(in_channels, out_channels, 3, stride=2)
            else:
                conv_layer = ConvBNAct(in_channels, out_channels, 3, stride=2, padding=1)
            stage.append(conv_layer)

            # SPP layer (only in last stage)
            if use_spp:
                spp = SPPBottleneck(out_channels, out_channels, kernel_sizes=spp_kernel_sizes)
                stage.append(spp)

            # CSP layer
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise
            )
            stage.append(csp_layer)

            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

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

    def _freeze_stages(self):
        """Freeze stages for transfer learning."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Tuple of feature tensors from specified output indices.
        """
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


def get_backbone_channels(widen_factor: float) -> List[int]:
    """Get output channel sizes for a given width factor.

    Args:
        widen_factor: Width multiplier.

    Returns:
        List of output channel sizes for stages 2, 3, 4.
    """
    base_channels = [256, 512, 1024]
    return [int(c * widen_factor) for c in base_channels]
