"""Complementary Confidence-Calibrated Detector (C²-Detector).

EvoDet module 2/3. Standard query-based detectors evaluate each prediction with
a single foreground confidence, so visually similar backgrounds and
low-contrast lesion boundaries can produce foreground scores comparable to true
targets. C²-Detector couples object queries with *learnable negative queries* to
decompose each localization into a complementary target-background confidence::

    {b_i, s_i⁺, s_i⁻} = Φ_det( MLP( CA( SA(q_i), E ) ), n_i )

where ``q_i = q_i^learn + W_p · p_c`` is a prototype-enhanced object query,
``n_i`` a learnable negative query, ``b_i`` the predicted box, and ``s_i⁺`` /
``s_i⁻`` the target and background confidences.

The detector further replaces single-score ranking with **Margin-NMS**, which
selects and suppresses candidates using a margin-based confidence (Sec. III-C)::

    r_i = s_i⁺   if IoU(b_m, b_i) < τ_n  ∧  s_i⁺ > s_i⁻
    r_i = 0      otherwise

The implementation subclasses the DEIMv2 ``DEIMTransformer`` (the multi-scale
adapter / decoder backbone) and adds:

* ``prototype_proj`` — ``W_p`` mapping prototypes into the query space,
* ``negative_queries`` — learnable negative queries ``n_i``,
* a negative branch (self-attn → cross-attn → MLP) producing ``s_i⁻`` and
  negative boxes for IoU-based pairing,
* :func:`margin_nms` / :func:`decode_c2` for margin-based post-processing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

from engine.core import register
from engine.deim.box_ops import box_cxcywh_to_xyxy
from engine.deim.deim_decoder import DEIMTransformer
from engine.deim.deim_utils import MLP, RMSNorm

__all__ = ["C2Decoder", "C2Detector", "margin_nms", "decode_c2"]


@register()
class C2Decoder(DEIMTransformer):
    """DEIM decoder augmented with prototype-enhanced queries and a
    complementary negative-query branch (C²-Detector).

    Args are identical to ``DEIMTransformer`` plus the C²-specific ones below.
    """

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="silu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        # ---- C²-Detector extensions ----
        prototype_dim: int = 256,
        num_negative_queries: int = 64,
        neg_nhead: int = 8,
    ):
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            mlp_act=mlp_act,
            use_gateway=use_gateway,
            share_bbox_head=share_bbox_head,
            share_score_head=share_score_head,
        )

        # Semantic prototype -> query-space projection W_p.
        self.prototype_proj = nn.Linear(prototype_dim, hidden_dim, bias=False)

        # Learnable negative queries n_i and their complementary branch.
        self.negative_queries = nn.Embedding(num_negative_queries, hidden_dim)
        self.neg_self_attn = nn.MultiheadAttention(hidden_dim, neg_nhead, dropout=0.0, batch_first=True)
        self.neg_cross_attn = nn.MultiheadAttention(hidden_dim, neg_nhead, dropout=0.0, batch_first=True)
        self.neg_norm1 = RMSNorm(hidden_dim)
        self.neg_norm2 = RMSNorm(hidden_dim)
        self.neg_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.neg_score_head = nn.Linear(hidden_dim, 1)

        self._reset_c2_parameters()

    def _reset_c2_parameters(self):
        nn.init.xavier_uniform_(self.prototype_proj.weight)
        nn.init.normal_(self.negative_queries.weight, std=0.02)
        nn.init.constant_(self.neg_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.neg_bbox_head.layers[-1].bias, 0)

    # ------------------------------------------------------------------ #
    # Prototype enhancement q_i = q_i^learn + W_p · p_c                    #
    # ------------------------------------------------------------------ #
    def prototype_enhancement(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Map prototypes into the query space and tile them across queries.

        Args:
            prototypes: ``[C, d]`` semantic prototypes.

        Returns:
            ``[1, num_queries, hidden_dim]`` broadcastable additive offset.
        """
        p = self.prototype_proj(prototypes)          # [C, hidden_dim]
        C = p.shape[0]
        reps = self.num_queries // C
        emb = p.repeat_interleave(reps, dim=0)       # [num_queries, hidden_dim]
        if emb.shape[0] < self.num_queries:
            emb = torch.cat([emb, p[: self.num_queries - emb.shape[0]]], dim=0)
        return emb.unsqueeze(0)

    # ------------------------------------------------------------------ #
    # Complementary negative branch                                        #
    # ------------------------------------------------------------------ #
    def _negative_branch(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run learnable negative queries against the multi-scale memory.

        Args:
            memory: multi-scale encoder memory ``[B, L, hidden_dim]``.

        Returns:
            ``neg_boxes``: ``[B, Nn, 4]`` normalised ``cxcywh`` boxes.
            ``neg_logits``: ``[B, Nn, 1]`` background logits (``s_i⁻``).
        """
        B = memory.shape[0]
        neg = self.negative_queries.weight.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.neg_self_attn(neg, neg, neg)
        neg = self.neg_norm1(neg + attn_out)

        attn_out, _ = self.neg_cross_attn(neg, memory, memory)
        neg = self.neg_norm2(neg + attn_out)

        neg_boxes = self.neg_bbox_head(neg).sigmoid()   # [B, Nn, 4]
        neg_logits = self.neg_score_head(neg)           # [B, Nn, 1]
        return neg_boxes, neg_logits

    def forward(self, feats, prototypes: Optional[torch.Tensor] = None, targets=None) -> dict:
        """Inference forward: object branch (``s_i⁺``) + negative branch (``s_i⁻``)."""
        memory, spatial_shapes = self._get_encoder_input(feats)
        init_ref_contents, init_ref_points_unact, _, _ = self._get_decoder_input(
            memory, spatial_shapes
        )

        # q_i = q_i^learn + W_p · p_c
        if prototypes is not None:
            init_ref_contents = init_ref_contents + self.prototype_enhancement(prototypes)

        out_bboxes, out_logits, _, _, _, _ = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=None,
            dn_meta=None,
        )

        neg_boxes, neg_logits = self._negative_branch(memory)

        return {
            "pred_logits": out_logits[-1],  # [B, Nq, C]  -> s_i⁺
            "pred_boxes": out_bboxes[-1],   # [B, Nq, 4]
            "neg_boxes": neg_boxes,         # [B, Nn, 4]
            "neg_logits": neg_logits,       # [B, Nn, 1]  -> s_i⁻
        }


# ---------------------------------------------------------------------- #
# Margin-NMS                                                              #
# ---------------------------------------------------------------------- #
def _single_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    x1 = torch.max(a[0], b[0]); y1 = torch.max(a[1], b[1])
    x2 = torch.min(a[2], b[2]); y2 = torch.min(a[3], b[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0)
    area_b = (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def margin_nms(
    boxes: torch.Tensor,
    target_scores: torch.Tensor,
    bg_scores: torch.Tensor,
    iou_threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Margin-NMS (Sec. III-C).

    Candidates are ranked by ``s_i⁺`` and suppressed if they overlap a dominant
    detection (``IoU ≥ τ_n``) or their target confidence fails to exceed the
    background confidence (``s_i⁺ ≤ s_i⁻``).

    Args:
        boxes: ``[N, 4]`` boxes in ``xyxy`` (normalised).
        target_scores: ``[N]`` target confidence ``s_i⁺``.
        bg_scores: ``[N]`` background confidence ``s_i⁻``.
        iou_threshold: NMS threshold ``τ_n``.

    Returns:
        keep: ``[K]`` indices of retained detections.
        scores: ``[N]`` margin-based scores (``r_i = s_i⁺`` for kept, else 0).
    """
    order = torch.argsort(target_scores, descending=True)
    keep: List[int] = []
    kept_boxes: List[torch.Tensor] = []
    for i in order:
        i = int(i)
        if float(bg_scores[i]) >= float(target_scores[i]):
            continue
        if any(_single_iou(boxes[i], kb) >= iou_threshold for kb in kept_boxes):
            continue
        keep.append(i)
        kept_boxes.append(boxes[i])

    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    scores = torch.zeros_like(target_scores)
    if keep.numel() > 0:
        scores[keep] = target_scores[keep]
    return keep, scores


def _pair_background(
    obj_boxes: torch.Tensor, neg_boxes: torch.Tensor, neg_logits: torch.Tensor
) -> torch.Tensor:
    """Pair each object box with its highest-IoU negative query and return ``s_i⁻``."""
    if neg_boxes.numel() == 0:
        return torch.zeros(obj_boxes.shape[0], device=obj_boxes.device)
    iou = torchvision.ops.box_iou(obj_boxes, neg_boxes)  # [Nq, Nn]
    pair = iou.argmax(-1)
    bg = neg_logits.squeeze(-1).sigmoid()                # [Nn]
    return bg[pair]


def decode_c2(
    outputs: dict,
    orig_size: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.05,
) -> List[dict]:
    """Margin-NMS post-processing of :class:`C2Decoder` outputs.

    Args:
        outputs: dict from :class:`C2Decoder` (``pred_logits``, ``pred_boxes``,
            ``neg_boxes``, ``neg_logits``).
        orig_size: ``[B, 2]`` original image sizes (``w, h``).
        num_classes: number of categories.
        iou_threshold: Margin-NMS threshold ``τ_n``.
        score_threshold: minimum retained margin score.

    Returns:
        list (one per image) of dicts ``{labels, boxes, scores}`` in pixel
        ``xyxy``, matching the DEIMv2 ``PostProcessor`` convention.
    """
    results = []
    B = outputs["pred_logits"].shape[0]
    neg_boxes = outputs.get("neg_boxes", None)
    neg_logits = outputs.get("neg_logits", None)

    for b in range(B):
        s_pos = outputs["pred_logits"][b].sigmoid()       # [Nq, C]
        target_score, target_cls = s_pos.max(-1)          # [Nq]
        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][b])  # [Nq, 4]

        if neg_boxes is not None:
            nxy = box_cxcywh_to_xyxy(neg_boxes[b])
            bg_score = _pair_background(boxes, nxy, neg_logits[b])
        else:
            bg_score = torch.zeros_like(target_score)

        keep, scores = margin_nms(boxes, target_score, bg_score, iou_threshold)
        labels = target_cls[keep]
        keep_boxes = boxes[keep]
        keep_scores = scores[keep]

        w, h = orig_size[b]
        scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
        keep_boxes = keep_boxes * scale

        m = keep_scores >= score_threshold
        results.append(dict(labels=labels[m], boxes=keep_boxes[m], scores=keep_scores[m]))

    return results


class C2Detector(nn.Module):
    """Standalone wrapper binding :class:`C2Decoder` and :func:`decode_c2`.

    Convenience for using C²-Detector without the full :class:`EvoDet` wrapper.
    """

    def __init__(
        self,
        decoder: C2Decoder,
        num_classes: int,
        iou_threshold: float = 0.7,
        score_threshold: float = 0.05,
    ):
        super().__init__()
        self.decoder = decoder
        self.num_classes = int(num_classes)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def forward(self, feats, orig_size: torch.Tensor, prototypes: Optional[torch.Tensor] = None) -> List[dict]:
        out = self.decoder(feats, prototypes=prototypes)
        return decode_c2(
            out,
            orig_size,
            self.num_classes,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
        )
