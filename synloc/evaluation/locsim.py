"""
mAP-LocSim evaluation metric for BEV athlete localization.

Adapted from sskit/coco.py for standalone use.
Uses localization similarity instead of IoU for matching.
"""

import numpy as np
from xtcocotools.cocoeval import COCOeval
from xtcocotools.coco import COCO
from typing import Dict, Optional, List
import json


class LocSimCOCOeval(COCOeval):
    """COCO evaluation with LocSim metric for BEV localization.

    LocSim = exp(log(0.05) * dist² / tau²)

    Where dist is Euclidean distance in world coordinates (meters).

    Args:
        coco_gt: COCO ground truth object.
        coco_dt: COCO detection results object.
        iou_type: Evaluation type (ignored, always uses LocSim).
    """

    locsim_tau = 1.0  # Distance threshold for LocSim (meters)

    def __init__(self, coco_gt, coco_dt, iou_type='bbox'):
        super().__init__(coco_gt, coco_dt, iou_type)
        self.score_key = 'score'

    def computeIoU(self, img_id, cat_id):
        """Compute LocSim matrix instead of IoU.

        Returns:
            LocSim matrix of shape (num_dt, num_gt).
        """
        p = self.params
        gt = self._gts[img_id, cat_id] if p.useCats else \
            [_ for cId in p.catIds for _ in self._gts[img_id, cId]]
        dt = self._dts[img_id, cat_id] if p.useCats else \
            [_ for cId in p.catIds for _ in self._dts[img_id, cId]]

        if len(gt) == 0 or len(dt) == 0:
            return []

        # Sort detections by score
        inds = np.argsort([-d[self.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[:p.maxDets[-1]]

        # Get BEV positions
        if hasattr(p, 'position_from_keypoint_index'):
            # Project keypoints to ground
            img = self.cocoGt.loadImgs(int(img_id))[0]
            bev_dt = self._keypoints_to_bev(dt, img, p.position_from_keypoint_index)
        else:
            bev_dt = np.array([det['position_on_pitch'][:2] for det in dt])

        bev_gt = np.array([det['position_on_pitch'][:2] for det in gt])

        # Compute pairwise distances
        dist2 = ((bev_dt[:, None, :] - bev_gt[None, :, :]) ** 2).sum(-1)

        # Convert to LocSim
        locsim = np.exp(np.log(0.05) * dist2 / self.locsim_tau ** 2)

        return locsim

    def _keypoints_to_bev(self, detections, img_info, kpt_idx):
        """Project keypoints to BEV coordinates.

        Args:
            detections: List of detection dicts with 'keypoints'.
            img_info: Image info dict with camera params.
            kpt_idx: Index of keypoint to use for projection.

        Returns:
            BEV positions array (N, 2).
        """
        from ..data.camera import image_to_ground

        # Extract keypoint positions
        positions = []
        for det in detections:
            kps = np.array(det['keypoints']).reshape(-1, 3)
            positions.append(kps[kpt_idx, :2])
        positions = np.array(positions, dtype=np.float32)

        # Normalize to image coordinates
        w, h = float(img_info['width']), float(img_info['height'])
        normalized = ((positions - ((w - 1) / 2, (h - 1) / 2)) / w).astype(np.float32)

        # Project to ground
        import torch
        camera_matrix = torch.tensor(img_info['camera_matrix'], dtype=torch.float32)
        undist_poly = torch.tensor(img_info['undist_poly'], dtype=torch.float32)
        bev = image_to_ground(camera_matrix, undist_poly, torch.from_numpy(normalized))

        return bev[:, :2].numpy()

    def accumulate(self, p=None):
        """Accumulate evaluation results and compute additional metrics."""
        if p is None:
            p = self.params
        super().accumulate(p)

        # Extract precision/recall at LocSim=0.5
        iou_idx = np.where(p.iouThrs == 0.5)[0]
        if len(iou_idx) == 0:
            iou_idx = [0]
        iou = iou_idx[0]
        area = p.areaRngLbl.index('all')
        dets = np.argmax(p.maxDets)

        precision = np.squeeze(self.eval['precision'][iou, :, 0, area, dets])
        scores = np.squeeze(self.eval['scores'][iou, :, 0, area, dets])
        recall = p.recThrs

        # Compute F1 scores
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        self.eval['precision_50'] = precision
        self.eval['recall_50'] = recall
        self.eval['f1_50'] = f1
        self.eval['scores_50'] = scores

    def frame_accuracy(self, threshold: float) -> float:
        """Compute frame accuracy at given score threshold.

        Frame accuracy = fraction of images where all athletes are correctly detected.

        Args:
            threshold: Score threshold for detections.

        Returns:
            Frame accuracy value.
        """
        rng = self.params.areaRng[self.params.areaRngLbl.index('all')]
        iou_idx = np.where(self.params.iouThrs == 0.5)[0]
        if len(iou_idx) == 0:
            iou_idx = [0]

        ok = bad = 0
        for e in self.evalImgs:
            if e is None:
                continue
            if e['aRng'] == rng:
                matches = (e['dtMatches'][iou_idx[0]] > -1)
                if len(matches) > 0:
                    matched_scores = np.array(e['dtScores'])[matches]
                    if (matched_scores > threshold).sum() == len(e['gtIds']):
                        ok += 1
                    else:
                        bad += 1
                elif len(e['gtIds']) == 0:
                    ok += 1
                else:
                    bad += 1

        return ok / (ok + bad) if (ok + bad) > 0 else 0.0

    def summarize(self):
        """Print evaluation summary with additional LocSim metrics."""
        super().summarize()

        # Find optimal score threshold from F1
        if hasattr(self.params, 'score_threshold'):
            threshold = self.params.score_threshold
        else:
            i = self.eval['f1_50'].argmax()
            threshold = (self.eval['scores_50'][i] + self.eval['scores_50'][max(0, i - 1)]) / 2

        # Get metrics at threshold
        i = np.searchsorted(-self.eval['scores_50'], -threshold, 'right') - 1
        i = max(0, min(i, len(self.eval['precision_50']) - 1))

        stats = [
            self.eval['precision_50'][i],
            self.eval['recall_50'][i],
            self.eval['f1_50'][i],
            threshold,
            self.frame_accuracy(threshold)
        ]
        self.stats = np.concatenate([self.stats, stats])

        print()
        print(f'  Precision      @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[0]:5.3f}')
        print(f'  Recall         @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[1]:5.3f}')
        print(f'  F1             @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[2]:5.3f}')
        print(f'  Frame Accuracy @[ LocSim=0.5 | ScoreTh={threshold:5.3f} ]       = {stats[4]:5.3f}')
        print(f'  mAP-LocSim     @[ LocSim=0.50:0.95 | ScoreTh={threshold:5.3f} ] = {self.stats[0]:5.3f}')

        return {
            'mAP_locsim': float(self.stats[0]),
            'precision': float(stats[0]),
            'recall': float(stats[1]),
            'f1': float(stats[2]),
            'score_threshold': float(threshold),
            'frame_accuracy': float(stats[4])
        }


def evaluate_predictions(
    gt_file: str,
    results: List[Dict],
    position_from_keypoint_index: int = 1,
    score_threshold: Optional[float] = None
) -> Dict:
    """Evaluate predictions against ground truth.

    Args:
        gt_file: Path to ground truth COCO annotation file.
        results: List of detection results in COCO format.
        position_from_keypoint_index: Keypoint index for BEV projection.
        score_threshold: Fixed score threshold (if None, auto-select via F1).

    Returns:
        Evaluation metrics dict.
    """
    # Load ground truth
    coco_gt = COCO(gt_file)

    # Load results
    coco_dt = coco_gt.loadRes(results)

    # Run evaluation
    coco_eval = LocSimCOCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.useSegm = None
    coco_eval.params.position_from_keypoint_index = position_from_keypoint_index

    if score_threshold is not None:
        coco_eval.params.score_threshold = score_threshold

    coco_eval.evaluate()
    coco_eval.accumulate()
    metrics = coco_eval.summarize()

    return metrics
