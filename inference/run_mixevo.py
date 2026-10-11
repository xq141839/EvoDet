#!/usr/bin/env python3
"""MixEvo label-evolution demo on Endoscapes.

Runs the S²-Align + C²-Detector pipeline over a weakly-labeled pool and an
unlabeled pool, then applies MixEvo's multi-criterion pseudo-label filtering
(``J(IoU, s_i⁺) ≥ τ_e`` and ``s_i⁺ ≥ τ_c``) and reports the effective-rank
trace until convergence (``Δerank < δ``).

This is an *inference/evolution* demonstration (no training): the two pools are
built from the val split (the first ``--weak-n`` images form the weakly-labeled
pool, the next ``--unlabeled-n`` images the unlabeled pool).

Example
-------
    python inference/run_mixevo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
        --weak-n 40 --unlabeled-n 80 --device cuda
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision.ops import roi_align

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import evodet  # noqa: F401
from evodet import load_evodet, MixEvo
from evodet.utils import ENDO_CLASS_NAMES, load_image_tensor
from evodet.c2detector import box_cxcywh_to_xyxy


def parse_args():
    p = argparse.ArgumentParser(description="MixEvo evolution demo")
    p.add_argument("-c", "--config", type=str, default="configs/evodet_endo.yml")
    p.add_argument("-r", "--resume", type=str, required=True)
    p.add_argument("--endo-base", type=str, default="endoscapes")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("-d", "--device", type=str, default="cuda")
    p.add_argument("--weak-n", type=int, default=40, help="#weakly-labeled images")
    p.add_argument("--unlabeled-n", type=int, default=80, help="#unlabeled images")
    p.add_argument("--candidate-thresh", type=float, default=0.05)
    p.add_argument("--tau-e", type=float, default=0.3, help="evolving threshold τ_e")
    p.add_argument("--tau-c", type=float, default=0.5, help="confidence threshold τ_c")
    p.add_argument("--delta", type=float, default=0.3, help="convergence threshold δ")
    p.add_argument("-o", "--output", type=str, default="mixevo_trace.json")
    return p.parse_args()


def run_candidates(model, image_paths, device, num_classes, candidate_thresh):
    """Return MixEvo candidate tuples for a pool of images."""
    candidates = []
    for path in image_paths:
        tensor, orig_size, _ = load_image_tensor(path, size=(640, 640), device=str(device))
        with torch.no_grad():
            c2, c3, c4, sem_feats = model.backbone(tensor)
            prior = model.s2align(sem_feats)
            feats = model.encoder([c2, c3, c4])
            det = model.decoder(feats, prototypes=model.s2align.prototypes)

        # target confidence s_i⁺ and predicted category
        logits = det["pred_logits"][0]          # [Nq, C]
        s_pos = logits.sigmoid()
        target_score, category = s_pos.max(-1)  # [Nq]
        boxes = box_cxcywh_to_xyxy(det["pred_boxes"][0])  # [Nq, 4] normalised

        # RoI features from the stride-16 encoder memory for effective rank
        mem = feats[1]                          # [1, D, H, W]
        rois = torch.cat(
            [torch.zeros(boxes.shape[0], 1, device=boxes.device), boxes], dim=1
        )
        roi_feat = roi_align(mem, rois, output_size=(1, 1), spatial_scale=1.0) \
            .squeeze(-1).squeeze(-1)            # [Nq, D]

        m = target_score >= candidate_thresh
        if not m.any():
            continue
        anchor_boxes = prior["anchor_boxes"][0]  # [C, 4]
        candidates.append(
            (boxes[m], target_score[m], category[m], anchor_boxes, roi_feat[m])
        )
    return candidates


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ann_file = os.path.join(args.endo_base, args.split, "annotation_coco.json")
    with open(ann_file) as f:
        coco = json.load(f)
    images = [os.path.join(args.endo_base, args.split, im["file_name"]) for im in coco["images"]]
    images = [p for p in images if os.path.exists(p)]

    model = load_evodet(args.config, args.resume, device=str(device))
    num_classes = model.s2align.num_classes

    weak_paths = images[: args.weak_n]
    unlabeled_paths = images[args.weak_n : args.weak_n + args.unlabeled_n]
    print(f"[1/3] weak pool={len(weak_paths)}, unlabeled pool={len(unlabeled_paths)}")

    weak_candidates = run_candidates(model, weak_paths, device, num_classes, args.candidate_thresh)
    unlabeled_candidates = run_candidates(model, unlabeled_paths, device, num_classes, args.candidate_thresh)
    print(f"[2/3] candidates: weak={len(weak_candidates)} images, "
          f"unlabeled={len(unlabeled_candidates)} images")

    mixevo = MixEvo(
        evolving_threshold=args.tau_e,
        confidence_threshold=args.tau_c,
        convergence_threshold=args.delta,
    )
    trace, promoted = mixevo.evolve(weak_candidates, unlabeled_candidates)

    print(f"[3/3] evolution finished: {len(trace)} iterations, {len(promoted)} promoted pseudo-boxes")
    print(f"\n{'iter':>4s} {'#promoted':>10s} {'#recycled':>10s} {'erank':>8s} {'Δerank':>8s} {'converged':>9s}")
    for t in trace:
        print(f"{t.iteration:4d} {len(t.promoted):10d} {t.recycled:10d} "
              f"{t.erank:8.3f} {t.delta_erank:8.3f} {str(t.converged):>9s}")

    out = {
        "iterations": [
            {
                "iteration": t.iteration,
                "n_promoted": len(t.promoted),
                "n_recycled": t.recycled,
                "erank": t.erank,
                "delta_erank": t.delta_erank,
                "converged": t.converged,
            }
            for t in trace
        ],
        "n_promoted_total": len(promoted),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[trace] saved to {args.output}")


if __name__ == "__main__":
    main()
