"""
Loss functions for YOLOX-Pose training.

Includes BCE, IoU, and OKS (Object Keypoint Similarity) losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class BCELoss(nn.Module):
    """Binary Cross Entropy loss with optional label smoothing.

    Args:
        reduction: Reduction mode ('mean', 'sum', 'none').
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pred: Tensor, target: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        """Compute BCE loss.

        Args:
            pred: Predictions (before sigmoid).
            target: Targets (0 or 1).
            weight: Optional per-element weights.

        Returns:
            Loss value.
        """
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class IoULoss(nn.Module):
    """IoU-based loss for bounding box regression.

    Supports IoU, GIoU, DIoU, and CIoU variants.

    Args:
        mode: Loss mode ('iou', 'giou', 'diou', 'ciou').
        reduction: Reduction mode.
        eps: Epsilon for numerical stability.
    """

    def __init__(
        self,
        mode: str = 'iou',
        reduction: str = 'mean',
        eps: float = 1e-7
    ):
        super().__init__()
        self.mode = mode
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute IoU loss.

        Args:
            pred: Predicted boxes (N, 4) as (x1, y1, x2, y2).
            target: Target boxes (N, 4) as (x1, y1, x2, y2).

        Returns:
            Loss value.
        """
        # Ensure proper ordering
        pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = target.unbind(-1)

        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        if self.mode == 'iou':
            loss = 1 - iou
        elif self.mode == 'giou':
            # Enclosing box
            enclose_x1 = torch.min(pred_x1, target_x1)
            enclose_y1 = torch.min(pred_y1, target_y1)
            enclose_x2 = torch.max(pred_x2, target_x2)
            enclose_y2 = torch.max(pred_y2, target_y2)
            enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + self.eps

            giou = iou - (enclose_area - union_area) / enclose_area
            loss = 1 - giou
        elif self.mode in ('diou', 'ciou'):
            # Center distance
            pred_cx = (pred_x1 + pred_x2) / 2
            pred_cy = (pred_y1 + pred_y2) / 2
            target_cx = (target_x1 + target_x2) / 2
            target_cy = (target_y1 + target_y2) / 2
            center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

            # Enclosing box diagonal
            enclose_x1 = torch.min(pred_x1, target_x1)
            enclose_y1 = torch.min(pred_y1, target_y1)
            enclose_x2 = torch.max(pred_x2, target_x2)
            enclose_y2 = torch.max(pred_y2, target_y2)
            enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps

            diou = iou - center_dist / enclose_diag

            if self.mode == 'diou':
                loss = 1 - diou
            else:  # ciou
                pred_w = pred_x2 - pred_x1
                pred_h = pred_y2 - pred_y1
                target_w = target_x2 - target_x1
                target_h = target_y2 - target_y1

                v = (4 / (torch.pi ** 2)) * (
                    torch.atan(target_w / (target_h + self.eps)) -
                    torch.atan(pred_w / (pred_h + self.eps))
                ) ** 2
                alpha = v / (1 - iou + v + self.eps)
                ciou = diou - alpha * v
                loss = 1 - ciou
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class OKSLoss(nn.Module):
    """Object Keypoint Similarity loss for keypoint regression.

    OKS measures the similarity between predicted and ground truth keypoints,
    taking into account the object scale (area).

    Args:
        num_keypoints: Number of keypoints.
        sigmas: Per-keypoint sigmas for OKS computation. If None, uses uniform.
        reduction: Reduction mode.
    """

    def __init__(
        self,
        num_keypoints: int = 2,
        sigmas: Optional[Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.reduction = reduction

        if sigmas is None:
            # Default sigmas for SynLoc (pelvis, pelvis_ground)
            sigmas = torch.tensor([0.089, 0.089])  # Same as pelvis in COCO
        self.register_buffer('sigmas', sigmas)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        target_vis: Tensor,
        areas: Tensor
    ) -> Tensor:
        """Compute OKS loss.

        Args:
            pred: Predicted keypoints (N, K, 2).
            target: Target keypoints (N, K, 2).
            target_vis: Target visibility (N, K).
            areas: Object areas (N,).

        Returns:
            Loss value.
        """
        # Compute squared distances
        dist2 = ((pred - target) ** 2).sum(dim=-1)  # (N, K)

        # Scale by object area and sigma
        scale = areas.unsqueeze(-1) + 1e-8  # (N, 1)
        sigmas_sq = (self.sigmas ** 2).unsqueeze(0)  # (1, K)

        # OKS for each keypoint
        oks = torch.exp(-dist2 / (2 * scale * sigmas_sq))  # (N, K)

        # Weight by visibility
        visible_mask = target_vis > 0
        if visible_mask.sum() > 0:
            loss = 1 - (oks * visible_mask).sum() / visible_mask.sum()
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss


class YOLOXPoseLoss(nn.Module):
    """Combined loss for YOLOX-Pose training.

    Combines:
    - BCE loss for classification
    - BCE loss for objectness
    - IoU loss for bounding boxes
    - OKS loss for keypoints
    - BCE loss for keypoint visibility

    Args:
        num_keypoints: Number of keypoints.
        cls_weight: Classification loss weight.
        obj_weight: Objectness loss weight.
        bbox_weight: Bounding box loss weight.
        kpt_weight: Keypoint loss weight.
        vis_weight: Visibility loss weight.
    """

    def __init__(
        self,
        num_keypoints: int = 2,
        cls_weight: float = 1.0,
        obj_weight: float = 1.0,
        bbox_weight: float = 5.0,
        kpt_weight: float = 30.0,
        vis_weight: float = 1.0
    ):
        super().__init__()
        self.cls_loss = BCELoss(reduction='none')
        self.obj_loss = BCELoss(reduction='none')
        self.bbox_loss = IoULoss(mode='giou', reduction='none')
        self.kpt_loss = OKSLoss(num_keypoints=num_keypoints, reduction='none')
        self.vis_loss = BCELoss(reduction='none')

        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.bbox_weight = bbox_weight
        self.kpt_weight = kpt_weight
        self.vis_weight = vis_weight

    def forward(
        self,
        cls_pred: Tensor,
        obj_pred: Tensor,
        bbox_pred: Tensor,
        kpt_pred: Tensor,
        vis_pred: Tensor,
        cls_target: Tensor,
        obj_target: Tensor,
        bbox_target: Tensor,
        kpt_target: Tensor,
        vis_target: Tensor,
        areas: Tensor,
        pos_mask: Tensor
    ) -> dict:
        """Compute all losses.

        Args:
            cls_pred: Classification predictions.
            obj_pred: Objectness predictions.
            bbox_pred: Bounding box predictions.
            kpt_pred: Keypoint predictions.
            vis_pred: Visibility predictions.
            cls_target: Classification targets.
            obj_target: Objectness targets.
            bbox_target: Bounding box targets.
            kpt_target: Keypoint targets.
            vis_target: Visibility targets.
            areas: Object areas.
            pos_mask: Positive sample mask.

        Returns:
            Dict of loss values.
        """
        num_pos = pos_mask.sum().clamp(min=1)

        # Objectness loss (all samples)
        loss_obj = self.obj_loss(obj_pred, obj_target).sum() / num_pos

        losses = {'loss_obj': self.obj_weight * loss_obj}

        if num_pos > 0:
            # Classification loss (positive samples only)
            loss_cls = self.cls_loss(
                cls_pred[pos_mask], cls_target
            ).sum() / num_pos
            losses['loss_cls'] = self.cls_weight * loss_cls

            # Bounding box loss
            loss_bbox = self.bbox_loss(
                bbox_pred[pos_mask], bbox_target
            ).sum() / num_pos
            losses['loss_bbox'] = self.bbox_weight * loss_bbox

            # Keypoint loss
            loss_kpt = self.kpt_loss(
                kpt_pred[pos_mask], kpt_target, vis_target, areas
            )
            losses['loss_kpt'] = self.kpt_weight * loss_kpt

            # Visibility loss
            loss_vis = self.vis_loss(
                vis_pred[pos_mask], vis_target
            ).mean()
            losses['loss_vis'] = self.vis_weight * loss_vis

        return losses
