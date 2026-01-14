"""
Inference utilities for YOLOX-Pose.

Run inference on images and format results for submission.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def run_inference(
    model,
    dataloader: DataLoader,
    device: str = 'cuda',
    score_thr: float = 0.01,
    nms_thr: float = 0.65,
    max_per_img: int = 100
) -> List[Dict]:
    """Run inference on a dataset.

    Args:
        model: YOLOX-Pose model.
        dataloader: Data loader for inference.
        device: Device for inference.
        score_thr: Score threshold for detections.
        nms_thr: NMS IoU threshold.
        max_per_img: Maximum detections per image.

    Returns:
        List of results in COCO format.
    """
    model.eval()
    model = model.to(device)

    all_results = []
    det_id = 1

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            images = batch['image'].to(device)
            img_ids = batch['img_id']
            orig_sizes = batch['orig_size']

            # Get input size
            _, _, h, w = images.shape
            input_size = (w, h)

            # Run inference
            results_batch = model.predict(
                images,
                input_size=input_size,
                score_thr=score_thr,
                nms_thr=nms_thr,
                max_per_img=max_per_img
            )

            # Format results
            for img_id, orig_size, results in zip(img_ids, orig_sizes, results_batch):
                # Scale back to original image size
                scale_x = orig_size[0] / w
                scale_y = orig_size[1] / h

                bboxes = results['bboxes'].cpu().numpy()
                scores = results['scores'].cpu().numpy()
                keypoints = results['keypoints'].cpu().numpy()
                kpt_scores = results['keypoint_scores'].cpu().numpy()

                for i in range(len(bboxes)):
                    # Scale bbox
                    x1, y1, x2, y2 = bboxes[i]
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    bbox_w, bbox_h = x2 - x1, y2 - y1

                    # Scale keypoints
                    kpts = keypoints[i].copy()
                    kpts[:, 0] *= scale_x
                    kpts[:, 1] *= scale_y

                    # Format keypoints as [x1, y1, v1, x2, y2, v2, ...]
                    kpts_flat = []
                    for k in range(len(kpts)):
                        kpts_flat.extend([
                            float(kpts[k, 0]),
                            float(kpts[k, 1]),
                            float(kpt_scores[i, k])
                        ])

                    result = {
                        'id': det_id,
                        'image_id': int(img_id),
                        'category_id': 1,
                        'bbox': [float(x1), float(y1), float(bbox_w), float(bbox_h)],
                        'area': float(bbox_w * bbox_h),
                        'score': float(scores[i]),
                        'keypoints': kpts_flat,
                    }
                    all_results.append(result)
                    det_id += 1

    return all_results


def format_results_for_submission(
    results: List[Dict],
    score_threshold: float,
    position_from_keypoint_index: int = 1,
    output_dir: str = '.'
) -> Tuple[str, str]:
    """Format results for challenge submission.

    Creates:
    - results.json: Detection results
    - metadata.json: Score threshold and keypoint index

    Args:
        results: List of detection results.
        score_threshold: Score threshold used.
        position_from_keypoint_index: Keypoint index for BEV projection.
        output_dir: Output directory.

    Returns:
        Paths to results.json and metadata.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save metadata
    metadata = {
        'score_threshold': score_threshold,
        'position_from_keypoint_index': position_from_keypoint_index
    }
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(results_path), str(metadata_path)


def create_submission_zip(
    results_path: str,
    metadata_path: str,
    output_path: str = 'submission.zip'
) -> str:
    """Create submission zip file.

    Args:
        results_path: Path to results.json.
        metadata_path: Path to metadata.json.
        output_path: Output zip file path.

    Returns:
        Path to created zip file.
    """
    import zipfile

    with zipfile.ZipFile(output_path, 'w') as zipf:
        zipf.write(results_path, 'results.json')
        zipf.write(metadata_path, 'metadata.json')

    return output_path


def visualize_predictions(
    image: np.ndarray,
    results: Dict,
    keypoint_names: List[str] = ['pelvis', 'pelvis_ground'],
    score_thr: float = 0.3
) -> np.ndarray:
    """Visualize predictions on image.

    Args:
        image: Input image (H, W, 3).
        results: Detection results dict.
        keypoint_names: Names for keypoints.
        score_thr: Score threshold for visualization.

    Returns:
        Image with drawn predictions.
    """
    import cv2

    image = image.copy()
    bboxes = results['bboxes']
    scores = results['scores']
    keypoints = results['keypoints']
    kpt_scores = results['keypoint_scores']

    colors = [(0, 255, 0), (255, 0, 0)]  # Green for pelvis, Red for pelvis_ground

    for i in range(len(bboxes)):
        if scores[i] < score_thr:
            continue

        # Draw bbox
        x1, y1, x2, y2 = map(int, bboxes[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw score
        cv2.putText(
            image, f'{scores[i]:.2f}',
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

        # Draw keypoints
        for k in range(len(keypoints[i])):
            x, y = map(int, keypoints[i, k])
            conf = kpt_scores[i, k] if kpt_scores is not None else 1.0
            color = colors[k % len(colors)]
            cv2.circle(image, (x, y), 5, color, -1)

    return image
