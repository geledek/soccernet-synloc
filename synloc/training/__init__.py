"""Training utilities for YOLOX-Pose."""

from .losses import BCELoss, IoULoss, OKSLoss, YOLOXPoseLoss
from .trainer import SynLocTrainer

__all__ = ['BCELoss', 'IoULoss', 'OKSLoss', 'YOLOXPoseLoss', 'SynLocTrainer']
