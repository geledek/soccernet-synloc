"""
YOLOX-Pose detection head for athlete localization.

Simplified standalone implementation extracted from mmpose.
Predicts classification scores, bounding boxes, keypoint offsets and visibilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Sequence, Union, Dict

from .layers import ConvBNAct


class YOLOXPoseHeadModule(nn.Module):
    """YOLOXPose head module for multi-scale prediction.

    Outputs classification, objectness, bbox, keypoints and visibility predictions
    at multiple feature map scales.

    Args:
        num_keypoints: Number of keypoints per instance. Default: 2 (pelvis, pelvis_ground).
        in_channels: Number of input channels from neck.
        num_classes: Number of object classes. Default: 1 (person).
        widen_factor: Width multiplier for channels. Default: 1.0.
        feat_channels: Number of hidden channels. Default: 256.
        stacked_convs: Number of stacked conv layers per branch. Default: 2.
        featmap_strides: Feature map strides. Default: [8, 16, 32].
    """

    def __init__(
        self,
        num_keypoints: int = 2,
        in_channels: int = 256,
        num_classes: int = 1,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = (8, 16, 32)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels

        self._init_layers()

    def _init_layers(self):
        """Initialize prediction branches."""
        self._init_cls_branch()
        self._init_reg_branch()
        self._init_pose_branch()
        self._init_weights()

    def _init_cls_branch(self):
        """Initialize classification branch."""
        self.conv_cls = nn.ModuleList()
        self.out_cls = nn.ModuleList()

        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(ConvBNAct(chn, self.feat_channels, 3))
            self.conv_cls.append(nn.Sequential(*stacked_convs))
            self.out_cls.append(nn.Conv2d(self.feat_channels, self.num_classes, 1))

    def _init_reg_branch(self):
        """Initialize bbox regression branch."""
        self.conv_reg = nn.ModuleList()
        self.out_bbox = nn.ModuleList()
        self.out_obj = nn.ModuleList()

        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(ConvBNAct(chn, self.feat_channels, 3))
            self.conv_reg.append(nn.Sequential(*stacked_convs))
            self.out_bbox.append(nn.Conv2d(self.feat_channels, 4, 1))
            self.out_obj.append(nn.Conv2d(self.feat_channels, 1, 1))

    def _init_pose_branch(self):
        """Initialize keypoint prediction branch."""
        self.conv_pose = nn.ModuleList()
        self.out_kpt = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()

        for _ in self.featmap_strides:
            # Deeper branch for pose
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(ConvBNAct(chn, self.feat_channels, 3))
            self.conv_pose.append(nn.Sequential(*stacked_convs))
            self.out_kpt.append(nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(self.feat_channels, self.num_keypoints, 1))

    def _init_weights(self):
        """Initialize weights with prior probability for objectness."""
        bias_init = -math.log((1 - 0.01) / 0.01)  # prior prob = 0.01
        for conv_cls, conv_obj in zip(self.out_cls, self.out_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[List[Tensor], ...]:
        """Forward pass through all branches.

        Args:
            x: Tuple of feature maps from neck.

        Returns:
            Tuple of (cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis)
            where each is a list of tensors per scale.
        """
        cls_scores, bbox_preds, objectnesses = [], [], []
        kpt_offsets, kpt_vis = [], []

        for i in range(len(x)):
            cls_feat = self.conv_cls[i](x[i])
            reg_feat = self.conv_reg[i](x[i])
            pose_feat = self.conv_pose[i](x[i])

            cls_scores.append(self.out_cls[i](cls_feat))
            objectnesses.append(self.out_obj[i](reg_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            kpt_offsets.append(self.out_kpt[i](pose_feat))
            kpt_vis.append(self.out_kpt_vis[i](pose_feat))

        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis


class YOLOXPoseHead(nn.Module):
    """Complete YOLOX-Pose head with decoding and loss computation.

    Args:
        num_keypoints: Number of keypoints. Default: 2.
        in_channels: Input channels from neck.
        num_classes: Number of classes. Default: 1.
        widen_factor: Width multiplier. Default: 1.0.
        feat_channels: Hidden channels. Default: 256.
        featmap_strides: Feature map strides. Default: (8, 16, 32).
    """

    def __init__(
        self,
        num_keypoints: int = 2,
        in_channels: int = 256,
        num_classes: int = 1,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        featmap_strides: Sequence[int] = (8, 16, 32)
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.featmap_sizes = None
        self.mlvl_priors = None

        self.head_module = YOLOXPoseHeadModule(
            num_keypoints=num_keypoints,
            in_channels=in_channels,
            num_classes=num_classes,
            widen_factor=widen_factor,
            feat_channels=feat_channels,
            featmap_strides=featmap_strides
        )

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[List[Tensor], ...]:
        """Forward pass.

        Args:
            feats: Feature maps from neck.

        Returns:
            Raw predictions from head module.
        """
        return self.head_module(feats)

    def predict(
        self,
        feats: Tuple[Tensor, ...],
        input_size: Tuple[int, int],
        score_thr: float = 0.01,
        nms_thr: float = 0.65,
        max_per_img: int = 100
    ) -> List[Dict[str, Tensor]]:
        """Decode predictions and apply NMS.

        Args:
            feats: Feature maps from neck.
            input_size: Input image size (width, height).
            score_thr: Score threshold for filtering. Default: 0.01.
            nms_thr: NMS IoU threshold. Default: 0.65.
            max_per_img: Maximum detections per image. Default: 100.

        Returns:
            List of dicts with 'bboxes', 'scores', 'keypoints', 'keypoint_scores'.
        """
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = self.forward(feats)

        batch_size = cls_scores[0].size(0)
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # Generate priors (anchor points)
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self._generate_priors(featmap_sizes, device, dtype)
            self.featmap_sizes = featmap_sizes

        flatten_priors = torch.cat(self.mlvl_priors)

        # Get strides for each prior
        mlvl_strides = []
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides):
            h, w = featmap_size
            mlvl_strides.append(flatten_priors.new_full((h * w,), stride))
        flatten_stride = torch.cat(mlvl_strides)

        # Flatten predictions
        flatten_cls = self._flatten_predictions(cls_scores).sigmoid()
        flatten_obj = self._flatten_predictions(objectnesses).sigmoid()
        flatten_bbox = self._flatten_predictions(bbox_preds)
        flatten_kpt = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()

        # Decode predictions
        flatten_bbox = self.decode_bbox(flatten_bbox, flatten_priors, flatten_stride)
        flatten_kpt = self.decode_kpt(flatten_kpt, flatten_priors, flatten_stride)

        # Process each image
        results_list = []
        for b in range(batch_size):
            bboxes = flatten_bbox[b]
            scores = flatten_cls[b] * flatten_obj[b]
            keypoints = flatten_kpt[b]
            kpt_scores = flatten_kpt_vis[b]

            # Get max class score
            scores, labels = scores.max(dim=-1)

            # Filter by score
            mask = scores > score_thr
            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            keypoints = keypoints[mask]
            kpt_scores = kpt_scores[mask]

            if bboxes.numel() > 0:
                # Apply NMS
                keep = self._nms(bboxes, scores, nms_thr)
                keep = keep[:max_per_img]

                bboxes = bboxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                keypoints = keypoints[keep]
                kpt_scores = kpt_scores[keep]

                # Clamp to image bounds
                bboxes[:, 0::2].clamp_(0, input_size[0])
                bboxes[:, 1::2].clamp_(0, input_size[1])

            results_list.append({
                'bboxes': bboxes,
                'scores': scores,
                'labels': labels,
                'keypoints': keypoints,
                'keypoint_scores': kpt_scores
            })

        return results_list

    def _generate_priors(
        self,
        featmap_sizes: List[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype
    ) -> List[Tensor]:
        """Generate anchor points for all feature map scales."""
        mlvl_priors = []
        for (h, w), stride in zip(featmap_sizes, self.featmap_strides):
            shift_x = torch.arange(0, w, device=device, dtype=dtype) * stride + stride / 2
            shift_y = torch.arange(0, h, device=device, dtype=dtype) * stride + stride / 2
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            priors = torch.stack([shift_x.flatten(), shift_y.flatten()], dim=-1)
            mlvl_priors.append(priors)
        return mlvl_priors

    def decode_bbox(
        self,
        pred_bboxes: Tensor,
        priors: Tensor,
        stride: Tensor
    ) -> Tensor:
        """Decode bbox predictions to (x1, y1, x2, y2) format.

        Args:
            pred_bboxes: Raw bbox predictions (B, N, 4) as (dx, dy, log_w, log_h).
            priors: Anchor points (N, 2).
            stride: Strides per anchor (N,).

        Returns:
            Decoded bboxes (B, N, 4) as (x1, y1, x2, y2).
        """
        stride = stride.view(1, -1, 1)
        priors = priors.view(1, -1, 2)

        xys = pred_bboxes[..., :2] * stride + priors
        whs = pred_bboxes[..., 2:].exp() * stride

        x1 = xys[..., 0] - whs[..., 0] / 2
        y1 = xys[..., 1] - whs[..., 1] / 2
        x2 = xys[..., 0] + whs[..., 0] / 2
        y2 = xys[..., 1] + whs[..., 1] / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def decode_kpt(
        self,
        pred_kpt: Tensor,
        priors: Tensor,
        stride: Tensor
    ) -> Tensor:
        """Decode keypoint predictions to (x, y) coordinates.

        Args:
            pred_kpt: Raw keypoint predictions (B, N, K*2).
            priors: Anchor points (N, 2).
            stride: Strides per anchor (N,).

        Returns:
            Decoded keypoints (B, N, K, 2).
        """
        stride = stride.view(1, -1, 1, 1)
        priors = priors.view(1, -1, 1, 2)

        pred_kpt = pred_kpt.reshape(*pred_kpt.shape[:2], self.num_keypoints, 2)
        return pred_kpt * stride + priors

    def _flatten_predictions(self, preds: List[Tensor]) -> Tensor:
        """Flatten multi-scale predictions to (B, N, C)."""
        preds = [p.permute(0, 2, 3, 1).flatten(1, 2) for p in preds]
        return torch.cat(preds, dim=1)

    def _nms(self, bboxes: Tensor, scores: Tensor, iou_thr: float) -> Tensor:
        """Apply non-maximum suppression."""
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        _, order = scores.sort(descending=True)
        keep = []

        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            # Compute IoU with rest
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = (iou <= iou_thr).nonzero(as_tuple=False).squeeze(-1)
            order = order[inds + 1]

        return torch.tensor(keep, device=bboxes.device)
