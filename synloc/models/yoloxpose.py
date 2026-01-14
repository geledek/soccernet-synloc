"""
Complete YOLOX-Pose model for SoccerNet SynLoc challenge.

Combines backbone, neck, and head into a single model with
variant configurations matching the original mmpose implementation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union

from .backbone import CSPDarknet, get_backbone_channels
from .neck import YOLOXPAFPN
from .head import YOLOXPoseHead


class YOLOXPose(nn.Module):
    """YOLOX-Pose model for single-stage athlete detection and localization.

    Predicts bounding boxes and 2 keypoints (pelvis, pelvis_ground) per athlete.
    The pelvis_ground keypoint is projected to world coordinates for BEV localization.

    Args:
        variant: Model size variant ('tiny', 's', 'm', 'l'). Default: 's'.
        num_keypoints: Number of keypoints. Default: 2.
        num_classes: Number of classes. Default: 1 (person).
        pretrained_backbone: Path to pretrained backbone weights. Default: None.

    Example:
        >>> model = YOLOXPose(variant='s', num_keypoints=2)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> # Training mode: get raw predictions
        >>> cls, obj, bbox, kpt, vis = model(x)
        >>> # Inference mode: get decoded results
        >>> results = model.predict(x, input_size=(640, 640))
    """

    # Variant configurations: (deepen_factor, widen_factor)
    # These match the original YOLOX configurations
    VARIANTS = {
        'tiny': {'deepen': 0.33, 'widen': 0.375},
        's': {'deepen': 0.33, 'widen': 0.5},
        'm': {'deepen': 0.67, 'widen': 0.75},
        'l': {'deepen': 1.0, 'widen': 1.0},
        'x': {'deepen': 1.33, 'widen': 1.25},
    }

    def __init__(
        self,
        variant: str = 's',
        num_keypoints: int = 2,
        num_classes: int = 1,
        pretrained_backbone: Optional[str] = None
    ):
        super().__init__()

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(self.VARIANTS.keys())}")

        cfg = self.VARIANTS[variant]
        self.variant = variant
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes

        # Backbone
        self.backbone = CSPDarknet(
            arch='P5',
            deepen_factor=cfg['deepen'],
            widen_factor=cfg['widen'],
            out_indices=(2, 3, 4),
            frozen_stages=-1,
            use_depthwise=False
        )

        # Get channel sizes for neck/head
        in_channels = get_backbone_channels(cfg['widen'])
        out_channels = int(256 * cfg['widen'])

        # Neck (PAFPN)
        self.neck = YOLOXPAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_csp_blocks=3,
            use_depthwise=False
        )

        # Head
        self.head = YOLOXPoseHead(
            num_keypoints=num_keypoints,
            in_channels=out_channels,
            num_classes=num_classes,
            widen_factor=cfg['widen'],
            feat_channels=256,
            featmap_strides=(8, 16, 32)
        )

        # Load pretrained backbone if provided
        if pretrained_backbone:
            self.load_backbone_weights(pretrained_backbone)

    def forward(self, x: Tensor) -> Tuple[List[Tensor], ...]:
        """Forward pass for training.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Tuple of (cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis)
            where each is a list of tensors per scale.
        """
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)

    def predict(
        self,
        x: Tensor,
        input_size: Tuple[int, int],
        score_thr: float = 0.01,
        nms_thr: float = 0.65,
        max_per_img: int = 100
    ) -> List[Dict[str, Tensor]]:
        """Forward pass for inference with decoding.

        Args:
            x: Input images (B, 3, H, W).
            input_size: Input size (width, height) for clamping predictions.
            score_thr: Score threshold for filtering. Default: 0.01.
            nms_thr: NMS IoU threshold. Default: 0.65.
            max_per_img: Maximum detections per image. Default: 100.

        Returns:
            List of dicts per image with keys:
                - 'bboxes': Tensor (N, 4) as (x1, y1, x2, y2)
                - 'scores': Tensor (N,) detection confidence
                - 'labels': Tensor (N,) class labels
                - 'keypoints': Tensor (N, K, 2) keypoint coordinates
                - 'keypoint_scores': Tensor (N, K) keypoint visibility scores
        """
        features = self.backbone(x)
        features = self.neck(features)
        return self.head.predict(
            features,
            input_size=input_size,
            score_thr=score_thr,
            nms_thr=nms_thr,
            max_per_img=max_per_img
        )

    def load_backbone_weights(self, checkpoint_path: str):
        """Load pretrained backbone weights.

        Supports loading from:
        - YOLOX MMDetection checkpoints
        - Our standalone checkpoints
        - Full model checkpoints (extracts backbone)

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Extract backbone weights
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                backbone_state[new_key] = value
            elif not any(key.startswith(p) for p in ['neck.', 'head.', 'bbox_head.']):
                backbone_state[key] = value

        if backbone_state:
            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            if missing:
                print(f"Missing keys in backbone: {missing}")
            if unexpected:
                print(f"Unexpected keys in backbone: {unexpected}")
        else:
            print("Warning: No backbone weights found in checkpoint")

    def load_weights(self, checkpoint_path: str):
        """Load full model weights.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Try to load directly
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_yoloxpose(
    variant: str = 's',
    num_keypoints: int = 2,
    pretrained: Optional[str] = None
) -> YOLOXPose:
    """Build YOLOX-Pose model.

    Args:
        variant: Model size ('tiny', 's', 'm', 'l'). Default: 's'.
        num_keypoints: Number of keypoints. Default: 2.
        pretrained: Path to pretrained weights. Default: None.

    Returns:
        YOLOXPose model instance.
    """
    model = YOLOXPose(variant=variant, num_keypoints=num_keypoints)
    if pretrained:
        model.load_weights(pretrained)
    return model


# Model info for different variants
MODEL_INFO = {
    'tiny': {'params': '~5M', 'gflops_640': '~10', 'input_sizes': [640, 960, 1280]},
    's': {'params': '~9M', 'gflops_640': '~18', 'input_sizes': [640, 960, 1280]},
    'm': {'params': '~25M', 'gflops_640': '~48', 'input_sizes': [640, 960, 1280]},
    'l': {'params': '~54M', 'gflops_640': '~108', 'input_sizes': [640, 960, 1280]},
}
