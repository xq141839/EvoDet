#!/usr/bin/env python3
"""EvoDet evaluation on Endoscapes (COCO mAP / AP50).

Loads ground truth from ``endoscapes/<split>/annotation_coco.json``, runs the
full EvoDet pipeline (S²-Align + C²-Detector with Margin-NMS), and reports the
COCO-style ``mAP`` (IoU 0.50–0.95) and ``AP50``.

Example
-------
    python inference/test_endo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
        --split val --device cuda
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import evodet  # noqa: F401  (registers EvoDet & submodules)
from evodet import load_evodet, evodet_postprocess
from evodet.utils import ENDO_CLASS_NAMES, load_image_tensor
from inference.coco_metrics import evaluate_coco


def parse_args():
    p = argparse.ArgumentParser(description="EvoDet Endoscapes evaluation")
    p.add_argument("-c", "--config", type=str, default="configs/evodet_endo.yml")
    p.add_argument("-r", "--resume", type=str, required=True, help="checkpoint path")
    p.add_argument("--endo-base", type=str, default="endoscapes")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("-d", "--device", type=str, default="cuda")
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument("--iou-thresh", type=float, default=0.7, help="Margin-NMS τ_n")
    p.add_argument("--max-images", type=int, default=0, help="0 = all images")
    return p.parse_args()


def load_gt(base: str, split: str):
    """Load Endoscapes COCO GT. Returns ``(gt_by_cat, images, class_names)``."""
    ann_file = os.path.join(base, split, "annotation_coco.json")
    with open(ann_file, "r") as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda x: x["id"])
    class_names = [c["name"].replace("_", " ").replace("-", " ").title() for c in cats]
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    gt_by_cat = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_cat[id_to_idx[ann["category_id"]]].append(
            np.array([x, y, x + w, y + h], dtype=np.float32)
        )

    images = coco["images"]
    return gt_by_cat, images, class_names


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[1/4] loading ground truth ({args.split}) ...")
    gt_by_cat, images, class_names = load_gt(args.endo_base, args.split)
    num_classes = len(class_names)
    print(f"      categories: {class_names}")
    print(f"      images: {len(images)}")

    print("[2/4] loading model ...")
    model = load_evodet(args.config, args.resume, device=str(device))
    num_classes = model.s2align.num_classes

    print("[3/4] running inference ...")
    pred_by_cat = defaultdict(lambda: ([], []))
    for i, img_info in enumerate(images):
        if args.max_images > 0 and i >= args.max_images:
            break
        path = os.path.join(args.endo_base, args.split, img_info["file_name"])
        if not os.path.exists(path):
            continue
        tensor, orig_size, _ = load_image_tensor(path, size=(640, 640), device=str(device))
        with torch.no_grad():
            outputs = model(tensor)
        results = evodet_postprocess(
            outputs, orig_size, num_classes,
            iou_threshold=args.iou_thresh, score_threshold=args.score_thresh,
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        for c in range(num_classes):
            m = labels == c
            if m.any():
                pred_by_cat[c][0].append(boxes[m])
                pred_by_cat[c][1].append(scores[m])
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{len(images)}")

    pred_by_cat = {
        c: (
            np.concatenate(pred_by_cat[c][0]) if pred_by_cat[c][0] else np.zeros((0, 4)),
            np.concatenate(pred_by_cat[c][1]) if pred_by_cat[c][1] else np.zeros(0),
        )
        for c in range(num_classes)
    }

    print("[4/4] evaluating ...")
    metrics = evaluate_coco(dict(gt_by_cat), pred_by_cat, num_classes)

    print("\n=== Endoscapes detection results (COCO) ===")
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else f"cls{c}"
        m = metrics["per_class_map"].get(c, float("nan"))
        a = metrics["per_class_ap50"].get(c, float("nan"))
        print(f"  {name:<16s} mAP={m:.3f}  AP50={a:.3f}")
    print(f"  {'-':-<40s}")
    print(f"  {'mean':<16s} mAP={metrics['mAP']:.3f}  AP50={metrics['AP50']:.3f}")


if __name__ == "__main__":
    main()
