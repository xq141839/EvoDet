#!/usr/bin/env python3
"""Prepare Endoscapes into a COCO-format subset for EvoDet / DEIMv2.

Endoscapes provides three annotation granularities (bounding boxes, image-level
labels and unlabeled frames). This script converts the box-labeled splits into
a compact COCO dataset with ``category_id ∈ {0..5}`` (the six Endoscapes
classes), which is the format expected by the DETR-style DEIM detector.

Classes (order): Cystic Plate, Calot Triangle, Cystic Artery, Cystic Duct,
Gallbladder, Tool.

Example
-------
    python scripts/prepare_endo.py --endo-base endoscapes --out endo_coco
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

ENDO_CLASSES = ["cystic plate", "calot triangle", "cystic artery",
                "cystic duct", "gallbladder", "tool"]


def normalize(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def make_coco_dict():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n.title()} for i, n in enumerate(ENDO_CLASSES)],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Endoscapes -> COCO")
    p.add_argument("--endo-base", type=str, default="endoscapes")
    p.add_argument("--out", type=str, default="endo_coco")
    p.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    p.add_argument("--copy-images", action="store_true", help="copy instead of symlink")
    return p.parse_args()


def main():
    args = parse_args()
    out_images = os.path.join(args.out, "images")
    out_ann = os.path.join(args.out, "annotations")
    for split in args.splits:
        os.makedirs(os.path.join(out_images, split), exist_ok=True)
    os.makedirs(out_ann, exist_ok=True)

    for split in args.splits:
        ann_file = os.path.join(args.endo_base, split, "annotation_coco.json")
        if not os.path.exists(ann_file):
            print(f"[skip] {split}: no annotation file at {ann_file}")
            continue

        with open(ann_file) as f:
            endo = json.load(f)

        # old category_id -> new 0..5
        name_to_new = {normalize(n): i for i, n in enumerate(ENDO_CLASSES)}
        old_to_new = {}
        for cat in endo.get("categories", []):
            key = normalize(cat["name"])
            if key in name_to_new:
                old_to_new[cat["id"]] = name_to_new[key]

        coco = make_coco_dict()
        img_id, ann_id = 1, 1
        img_anns = defaultdict(list)
        for ann in endo.get("annotations", []):
            if ann["category_id"] in old_to_new:
                img_anns[ann["image_id"]].append(
                    {"cat_id": old_to_new[ann["category_id"]], "bbox": ann["bbox"]}
                )

        imported = 0
        for info in endo.get("images", []):
            if info["id"] not in img_anns:
                continue
            src = os.path.join(args.endo_base, split, info["file_name"])
            if not os.path.exists(src):
                continue
            dst_name = f"{img_id:08d}.jpg"
            dst = os.path.join(out_images, split, dst_name)
            if args.copy_images:
                shutil.copy2(src, dst)
            elif not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

            coco["images"].append(
                {"id": img_id, "file_name": dst_name,
                 "width": info["width"], "height": info["height"]}
            )
            for a in img_anns[info["id"]]:
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": a["cat_id"],
                    "bbox": a["bbox"], "area": a["bbox"][2] * a["bbox"][3],
                    "iscrowd": 0,
                })
                ann_id += 1
            img_id += 1
            imported += 1

        out_file = os.path.join(out_ann, f"instances_{split}.json")
        with open(out_file, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"[done] {split}: {imported} images, {len(coco['annotations'])} boxes -> {out_file}")


if __name__ == "__main__":
    main()
