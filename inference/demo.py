#!/usr/bin/env python3
"""EvoDet inference demo (single image or image folder).

Runs the full EvoDet pipeline — S²-Align + C²-Detector (Margin-NMS) — and
renders the detections. Optionally overlays the S²-Align spatial prior anchors
and attention maps to inspect the label-evolution priors.

Examples
--------
    # single image
    python inference/demo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
        --input endoscapes/val/100_27200.jpg -o vis/

    # whole folder
    python inference/demo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
        --input endoscapes/val/ --ext .jpg -o vis/

    # also dump S²-Align attention maps / anchors
    python inference/demo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
        --input endoscapes/val/100_27200.jpg -o vis/ --show-prior
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import evodet  # noqa: F401  (registers EvoDet & submodules)
from evodet import load_evodet, evodet_postprocess
from evodet.utils import ENDO_CLASS_NAMES, ENDO_COLORS, load_image_tensor


def parse_args():
    p = argparse.ArgumentParser(description="EvoDet inference demo")
    p.add_argument("-c", "--config", type=str, default="configs/evodet_endo.yml")
    p.add_argument("-r", "--resume", type=str, default=None, help="checkpoint path")
    p.add_argument("--input", type=str, required=True, help="image path or folder")
    p.add_argument("--ext", type=str, default=".jpg", help="image extension for folders")
    p.add_argument("-o", "--output-dir", type=str, default="vis")
    p.add_argument("-d", "--device", type=str, default="cuda")
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--iou-thresh", type=float, default=0.7, help="Margin-NMS τ_n")
    p.add_argument("--show-prior", action="store_true", help="dump S²-Align priors")
    return p.parse_args()


def draw_detections(img_bgr, results, class_names):
    """Draw predicted boxes (class + margin score) on a BGR image."""
    for label, box, score in zip(
        results["labels"].cpu().numpy(),
        results["boxes"].cpu().numpy(),
        results["scores"].cpu().numpy(),
    ):
        name = class_names[int(label)] if int(label) < len(class_names) else f"cls{int(label)}"
        color = ENDO_COLORS.get(name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img_bgr, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img_bgr


def save_prior(prior, orig_wh, class_names, out_dir, name):
    """Render S²-Align attention maps and anchor boxes (one image per class)."""
    attention = prior["attention_map"][0].cpu().numpy()  # [C, H, W]
    anchors = prior["anchor_boxes"][0].cpu().numpy()      # [C, 4] normalised xyxy
    w, h = orig_wh
    for c in range(attention.shape[0]):
        heat = attention[c]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        heat = cv2.resize(heat, (w, h))
        heatmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = np.full((h, w, 3), 0, np.uint8)
        x1, y1, x2, y2 = (anchors[c] * np.array([w, h, w, h])).astype(int)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        heatmap = cv2.addWeighted(overlay, 0.6, heatmap, 0.7, 0)
        cv2.putText(heatmap, class_names[c], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out = os.path.join(out_dir, f"prior_{name}_{class_names[c].replace(' ', '_')}.jpg")
        cv2.imwrite(out, heatmap)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_evodet(args.config, args.resume, device=str(device))
    num_classes = model.s2align.num_classes
    class_names = ENDO_CLASS_NAMES if num_classes == len(ENDO_CLASS_NAMES) else [f"cls{i}" for i in range(num_classes)]

    if os.path.isdir(args.input):
        paths = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(args.ext.lower())
        )
    else:
        paths = [args.input]

    for path in paths:
        tensor, orig_size, orig_wh = load_image_tensor(path, size=(640, 640), device=str(device))
        with torch.no_grad():
            outputs = model(tensor)
        results = evodet_postprocess(
            outputs, orig_size, num_classes,
            iou_threshold=args.iou_thresh, score_threshold=args.score_thresh,
        )[0]

        img = cv2.imread(path)
        img = draw_detections(img, results, class_names)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, img)

        if args.show_prior and "prior" in outputs:
            save_prior(outputs["prior"], orig_wh, class_names, args.output_dir,
                       os.path.splitext(os.path.basename(path))[0])

        n = results["labels"].numel()
        print(f"[demo] {os.path.basename(path)} -> {out_path} ({n} detections)")

    print(f"[demo] done. outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
