"""EvoDet — Self-Evolving Mix-Granularity Medical Image Detection.

Top-level model composing the three modules:

* :class:`EvoDetBackbone` — the frozen VFM encoder (DINOv3 / ViT-Tiny) that also
  returns the raw patch embeddings consumed by S²-Align;
* :class:`S2Align` — Semantic-Spatial Alignment (spatial prior anchors);
* :class:`C2Decoder` — Complementary Confidence-Calibrated Detector;
* :class:`MixEvo` — Mix-Granularity Collaborative Label Evolution.

The model is *registered* into the DEIMv2 ``engine`` registry, so it is built
and loaded through the exact same ``YAMLConfig`` mechanism as the plain DEIM
detector ("the endo set"), e.g.::

    import evodet                       # registers EvoDet / EvoDetBackbone / ...
    from engine.core import YAMLConfig
    cfg = YAMLConfig("configs/evodet_endo.yml")
    model = cfg.model                   # full EvoDet model
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.core import register
from engine.backbone.dinov3_adapter import DINOv3STAs
from engine.backbone.vit_tiny import VisionTransformer

from .s2align import S2Align
from .c2detector import C2Decoder, decode_c2

__all__ = ["EvoDetBackbone", "EvoDet", "load_evodet", "evodet_postprocess"]


@register()
class EvoDetBackbone(DINOv3STAs):
    """Frozen VFM encoder that additionally exposes the raw patch embeddings.

    Identical to :class:`DINOv3STAs` (multi-scale ``c2, c3, c4`` outputs for the
    detector) except that it also returns the frozen VFM patch features
    ``sem_feats`` (``[1/8, 1/16, 1/32]`` scales) required by S²-Align.
    """

    def forward(self, x):
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        bs, _, _, _ = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, d, H, W]
            resize_H = int(H_c * 2 ** (num_scales - i))
            resize_W = int(W_c * 2 ** (num_scales - i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4, sem_feats


@register()
class EvoDet(nn.Module):
    """EvoDet framework: S²-Align + C²-Detector (+ MixEvo at inference time).

    Args are injected from the YAML config through the DEIMv2 registry:
    ``backbone``, ``encoder``, ``decoder`` and ``s2align``.
    """

    __inject__ = ["backbone", "encoder", "decoder", "s2align"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: C2Decoder,
        s2align: S2Align,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.s2align = s2align

    def forward(self, x: torch.Tensor, targets=None) -> dict:
        c2, c3, c4, sem_feats = self.backbone(x)

        # Module 1: S²-Align -> spatial prior anchors + semantic maps.
        prior = self.s2align(sem_feats)

        # Multi-scale adapter (1x1 conv + transformer + FPN/PAN).
        feats = self.encoder([c2, c3, c4])

        # Module 2: C²-Detector -> boxes, target confidence s_i⁺, background s_i⁻.
        det = self.decoder(feats, prototypes=self.s2align.prototypes)
        det["prior"] = prior
        return det

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


# ---------------------------------------------------------------------- #
# Post-processing (Margin-NMS)                                           #
# ---------------------------------------------------------------------- #
def evodet_postprocess(
    outputs: dict,
    orig_size: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.05,
) -> List[dict]:
    """Margin-NMS post-processing of the full :class:`EvoDet` output."""
    return decode_c2(
        outputs,
        orig_size,
        num_classes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )


# ---------------------------------------------------------------------- #
# Model loading                                                          #
# ---------------------------------------------------------------------- #
def load_evodet(
    config_path: str,
    weights_path: Optional[str],
    device: str = "cuda",
    use_ema: bool = True,
    strict: bool = False,
) -> nn.Module:
    """Build and load an :class:`EvoDet` model from a YAML config + checkpoint.

    Uses the DEIMv2 ``YAMLConfig`` mechanism (the "endo set" loading path) and
    loads the checkpoint state with ``strict=False`` so the C²-Detector's
    negative-query / prototype-projection parameters (absent from a plain DEIM
    checkpoint) keep their default initialisation.

    Args:
        config_path: path to the EvoDet YAML config.
        weights_path: path to a DEIM/DEIMv2 checkpoint (``model`` or ``ema``).
        device: target device.
        use_ema: prefer the EMA weights when present in the checkpoint.

    Returns:
        the model in ``eval()`` mode, moved to ``device``.
    """
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path)
    model: nn.Module = cfg.model

    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location="cpu")
        if use_ema and isinstance(checkpoint.get("ema", None), dict) and "module" in checkpoint["ema"]:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if not strict and (missing or unexpected):
            n_missing = len(missing)
            n_unexpected = len(unexpected)
            print(
                f"[EvoDet] loaded checkpoint with strict=False "
                f"({n_missing} missing / {n_unexpected} unexpected keys; "
                f"expected for C²-Detector / S²-Align extensions)."
            )

    return model.to(device).eval()
