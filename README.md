# EvoDet: Self-Evolving Mix-Granularity Medical Image Detection from Incomplete Annotations

<p align="center">
  <b>Qing Xu</b> · <b>Chao Wang</b> · <b>Xuanying Li</b> · <b>Wenting Duan</b> · <b>Zhen Chen</b><br>
</p>

Official inference / reference implementation of **EvoDet**, a self-evolving
framework for medical image detection from incomplete annotations via
mix-granularity learning. 

---

## Repository structure

```
EvoDet/
├── README.md                 # this file
├── requirements.txt          # environment (PyTorch 2.5.1)
├── configs/
│   └── evodet_endo.yml       # EvoDet config for Endoscapes (6 classes)
├── engine/                   # vendored framework (DINOv3 backbone + DEIM detector)
│   ├── backbone/             #   DINOv3 / ViT-Tiny + multi-scale adapter backbone
│   ├── deim/                 #   DEIM decoder / encoder / postprocessor / criterion
│   ├── core/                 #   YAMLConfig registry (model loading)
│   ├── data/                 #   COCO dataset + evaluator
│   └── ...
├── evodet/                   # the three EvoDet modules (new code)
│   ├── s2align.py
│   ├── c2detector.py
│   ├── mixevo.py
│   ├── evodet.py
│   ├── losses.py
│   └── utils.py
├── inference/
│   ├── demo.py               # image / folder inference + visualization
│   ├── test_endo.py          # Endoscapes COCO mAP / AP50 evaluation
│   ├── run_mixevo.py         # MixEvo label-evolution demo (effective-rank trace)
│   └── coco_metrics.py       # dependency-free COCO mAP / AP50
├── scripts/
│   └── prepare_endo.py       # Endoscapes -> COCO subset preparation
└── weights/                  # place checkpoints here (see below)
```

---

## Installation

```bash
conda create -n evodet python=3.11 -y
conda activate evodet
pip install -r requirements.txt
```

Requirements (`torch==2.5.1`, `torchvision==0.20.1`, …). See
[`requirements.txt`](requirements.txt).

---

## Quick start

### 1. Inference demo (image / folder)

```bash
python inference/demo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
    --input endoscapes/val/100_27200.jpg -o vis/ --device cuda

# whole folder + dump S²-Align spatial priors / attention maps
python inference/demo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
    --input endoscapes/val/ --ext .jpg -o vis/ --show-prior
```

### 2. Endoscapes evaluation (COCO mAP / AP50)

```bash
python inference/test_endo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
    --split val --device cuda
```

### 3. MixEvo label evolution (effective-rank trace)

```bash
python inference/run_mixevo.py -c configs/evodet_endo.yml -r weights/evodet_endo.pth \
    --weak-n 40 --unlabeled-n 80 --device cuda
```

### 4. Data preparation (Endoscapes -> COCO)

```bash
python scripts/prepare_endo.py --endo-base endoscapes --out endo_coco --splits train val
```

---

## Dataset

[**Endoscapes**](https://doi.org/10.1038/s41597-025-04634-y) (Mascagni et al.,
*Sci. Data* 2025) is a laparoscopic cholecystectomy benchmark with three
annotation granularities — bounding boxes, image-level labels and unlabeled
frames — forming a natural incomplete-annotation setting. The six classes are:

| id | class |
|---|---|
| 0 | Cystic Plate |
| 1 | Calot Triangle |
| 2 | Cystic Artery |
| 3 | Cystic Duct |
| 4 | Gallbladder |
| 5 | Tool |

The COCO subset is built by [`scripts/prepare_endo.py`](scripts/prepare_endo.py).

---

## Configuration

[`configs/evodet_endo.yml`](configs/evodet_endo.yml) is self-contained and
mirrors the Endoscapes multi-class configuration, adding the
`EvoDet` / `EvoDetBackbone` / `C2Decoder` / `S2Align` blocks. `num_classes = 6`.

