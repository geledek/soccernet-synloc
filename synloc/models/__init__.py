"""Model components for YOLOX-Pose."""

from .layers import ConvBNAct, DepthwiseSeparableConv, CSPLayer, DarknetBottleneck
from .layers import Focus, SPPBottleneck, ChannelAttention
from .backbone import CSPDarknet
from .neck import YOLOXPAFPN
from .yoloxpose import YOLOXPose
from .head_simcc import SimCCHead, SimCCLoss

__all__ = [
    'ConvBNAct', 'DepthwiseSeparableConv', 'CSPLayer', 'DarknetBottleneck',
    'Focus', 'SPPBottleneck', 'ChannelAttention', 'CSPDarknet', 'YOLOXPAFPN',
    'YOLOXPose', 'SimCCHead', 'SimCCLoss'
]
