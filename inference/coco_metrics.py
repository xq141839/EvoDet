"""Dependency-free COCO-style evaluation (mAP / AP50).

Implements the standard COCO protocol (101-point interpolated Average
Precision) without relying on ``pycocotools``, following [44] (Lin et al.,
"Microsoft COCO: Common Objects in Context", ECCV 2014).

* ``AP@t`` — AP at a single IoU threshold ``t``;
* ``mAP``  — mean over ``t ∈ {0.50, 0.55, …, 0.95}``;
* ``AP50`` — AP at ``t = 0.50``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["evaluate_coco", "box_iou"]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two ``xyxy`` boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _pr_curve(
    gt_boxes: List[np.ndarray], pred_boxes: np.ndarray, pred_scores: np.ndarray, iou_thr: float
) -> Tuple[np.ndarray, np.ndarray]:
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return np.zeros(len(pred_scores)), np.zeros(len(pred_scores))
    matched = [False] * n_gt
    tp = np.zeros(len(pred_scores), dtype=bool)
    fp = np.zeros(len(pred_scores), dtype=bool)
    order = np.argsort(-pred_scores)
    for i in order:
        best_iou, best_j = 0.0, -1
        for j in range(n_gt):
            if matched[j]:
                continue
            iou = box_iou(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = True
            matched[best_j] = True
        else:
            fp[i] = True
    tp_cum = np.cumsum(tp).astype(np.float64)
    fp_cum = np.cumsum(fp).astype(np.float64)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return recall, precision


def _ap101(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated Average Precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    points = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for t in points:
        idx = np.where(mrec >= t)[0]
        if idx.size == 0:
            continue
        ap += mpre[idx[0]]
    return ap / 101.0


def evaluate_coco(
    gt_by_cat: Dict[int, List[np.ndarray]],
    pred_by_cat: Dict[int, Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    iou_thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05)),
) -> Dict:
    """Evaluate COCO mAP / AP50.

    Args:
        gt_by_cat: ``{class_idx: [gt_box_xyxy, ...]}``.
        pred_by_cat: ``{class_idx: (pred_boxes [N,4], pred_scores [N])}``.
        num_classes: number of categories.
        iou_thresholds: IoU thresholds for the mAP average.

    Returns:
        dict with ``per_class`` (mAP per class), ``mAP`` and ``AP50``.
    """
    per_class_ap = {c: [] for c in range(num_classes)}
    for thr in iou_thresholds:
        for c in range(num_classes):
            pred_boxes, pred_scores = pred_by_cat.get(c, (np.zeros((0, 4)), np.zeros(0)))
            gt_boxes = gt_by_cat.get(c, [])
            if len(gt_boxes) == 0:
                # Following COCO, categories with no GT are ignored in the mean.
                per_class_ap[c].append(float("nan"))
                continue
            recall, precision = _pr_curve(gt_boxes, pred_boxes, pred_scores, thr)
            per_class_ap[c].append(_ap101(recall, precision))

    per_class_map = {}
    valid_maps = []
    for c in range(num_classes):
        vals = np.array([v for v in per_class_ap[c] if not np.isnan(v)])
        if vals.size:
            m = float(vals.mean())
            per_class_map[c] = m
            valid_maps.append(m)
        else:
            per_class_map[c] = float("nan")

    map_mean = float(np.mean(valid_maps)) if valid_maps else float("nan")

    # AP50 = AP at IoU 0.5.
    ap50 = {}
    valid_ap50 = []
    for c in range(num_classes):
        pred_boxes, pred_scores = pred_by_cat.get(c, (np.zeros((0, 4)), np.zeros(0)))
        gt_boxes = gt_by_cat.get(c, [])
        if len(gt_boxes) == 0:
            ap50[c] = float("nan")
            continue
        recall, precision = _pr_curve(gt_boxes, pred_boxes, pred_scores, 0.5)
        v = _ap101(recall, precision)
        ap50[c] = v
        valid_ap50.append(v)

    ap50_mean = float(np.mean(valid_ap50)) if valid_ap50 else float("nan")

    return {
        "per_class_map": per_class_map,
        "mAP": map_mean,
        "per_class_ap50": ap50,
        "AP50": ap50_mean,
    }
