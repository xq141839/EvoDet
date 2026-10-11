"""Mix-Granularity Collaborative Label Evolution (MixEvo).

EvoDet module 3/3. MixEvo progressively elevates incomplete annotations into
reliable fine-grained detection supervision through *multi-criterion
pseudo-label filtering* and *effective-rank-driven evolving convergence*.

Stage 1 (weak-labeled ``D^w``) and Stage 2 (unlabeled ``D^u``) share the same
collaborative filtering pipeline. A candidate pseudo-box ``(b_i, s_i⁺)`` is
promoted to the box-labeled set only when both criteria hold::

    J(IoU, s_i⁺) = IoU(b_i, b̂_c) × s_i⁺ ≥ τ_e   and   s_i⁺ ≥ τ_c

where ``b̂_c`` is the tightest axis-aligned box enclosing the S²-Align anchor
``M̂_c``, and ``τ_e`` / ``τ_c`` are the evolving and confidence thresholds.
Unpromoted candidates are recycled to their source set.

Convergence is decided by the **effective rank** of the promoted pseudo-label
RoI features (Sec. III-D)::

    erank(Σ) = exp( −Σ_k p_k log p_k ),  p_k = λ_k / Σ_j λ_j

The evolution continues while ``Δerank(t) = erank(Σ_t) − erank(Σ_{t−1}) ≥ δ``;
otherwise it terminates, preventing both premature termination and error
accumulation from redundant iterations.

This module is *inference/evolution-only*: it consumes S²-Align anchors and
C²-Detector predictions and emits promoted pseudo-boxes plus the effective-rank
trace. No optimizer / training loop is included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = ["MixEvo", "effective_rank", "EvolutionTrace", "PseudoBox"]


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between ``[N,4]`` / ``[M,4]`` boxes in ``xyxy`` format."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


@dataclass
class PseudoBox:
    """A candidate pseudo-box and the signals used to gate it."""

    box: torch.Tensor            # [4] xyxy (normalised)
    category: int
    target_confidence: float     # s_i⁺ from C²-Detector
    anchor_box: Optional[torch.Tensor] = None  # b̂_c from S²-Align
    roi_feature: Optional[torch.Tensor] = None  # pooled feature for effective rank


@dataclass
class EvolutionTrace:
    """Bookkeeping of one evolving iteration."""

    iteration: int
    promoted: List[PseudoBox] = field(default_factory=list)
    recycled: int = 0
    erank: float = 0.0
    delta_erank: float = 0.0
    converged: bool = False


def effective_rank(features: torch.Tensor) -> float:
    """Effective rank (Shannon entropy of the eigenspectrum) of a feature set.

    Args:
        features: ``[N, D]`` RoI features of the promoted pseudo-boxes.

    Returns:
        ``erank(Σ) = exp(−Σ_k p_k log p_k)`` with ``p_k = λ_k / Σ_j λ_j``.
    """
    if features.shape[0] < 2:
        return 1.0
    x = features - features.mean(0, keepdim=True)
    cov = (x.t() @ x) / (features.shape[0] - 1)  # [D, D]
    eig = torch.linalg.eigvalsh(cov)
    eig = eig.clamp(min=0.0)
    total = eig.sum()
    if total <= 0:
        return 1.0
    p = eig / total
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


class MixEvo:
    """Mix-Granularity Collaborative Label Evolution.

    Args:
        evolving_threshold: joint-gating threshold ``τ_e``.
        confidence_threshold: target-confidence threshold ``τ_c``.
        convergence_threshold: effective-rank delta threshold ``δ``.
        max_iterations: safety cap on evolving iterations.
    """

    def __init__(
        self,
        evolving_threshold: float = 0.3,
        confidence_threshold: float = 0.5,
        convergence_threshold: float = 0.3,
        max_iterations: int = 50,
    ) -> None:
        self.evolving_threshold = float(evolving_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.convergence_threshold = float(convergence_threshold)
        self.max_iterations = int(max_iterations)

    # ------------------------------------------------------------------ #
    # Multi-criterion pseudo-label filtering                              #
    # ------------------------------------------------------------------ #
    def joint_gating_score(self, iou: torch.Tensor, target_confidence: torch.Tensor) -> torch.Tensor:
        """``J(IoU, s_i⁺) = IoU(b_i, b̂_c) × s_i⁺``."""
        return iou * target_confidence

    def gate(
        self,
        boxes: torch.Tensor,
        target_scores: torch.Tensor,
        categories: torch.Tensor,
        anchor_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the joint gate and return the promotion mask.

        Args:
            boxes: ``[N, 4]`` predicted boxes ``xyxy`` (normalised).
            target_scores: ``[N]`` target confidence ``s_i⁺``.
            categories: ``[N]`` predicted category index.
            anchor_boxes: ``[C, 4]`` S²-Align anchor boxes ``b̂_c`` (normalised).

        Returns:
            promoted: ``[N]`` bool mask.
            jscore: ``[N]`` joint gating scores.
        """
        anchor_per_pred = anchor_boxes[categories]               # [N, 4]
        iou = _box_iou(boxes, anchor_boxes)                       # [N, C]
        iou_per_pred = iou.gather(1, categories.unsqueeze(1)).squeeze(1)  # [N]
        jscore = self.joint_gating_score(iou_per_pred, target_scores)
        promoted = (jscore >= self.evolving_threshold) & (target_scores >= self.confidence_threshold)
        return promoted, jscore

    def filter_candidates(
        self,
        boxes: torch.Tensor,
        target_scores: torch.Tensor,
        categories: torch.Tensor,
        anchor_boxes: torch.Tensor,
        roi_features: Optional[torch.Tensor] = None,
    ) -> Tuple[List[PseudoBox], List[int]]:
        """Filter candidates into promoted pseudo-boxes and recycled indices.

        Returns:
            promoted: list of :class:`PseudoBox`.
            recycled: indices of candidates recycled back to the source set.
        """
        promoted_mask, _ = self.gate(boxes, target_scores, categories, anchor_boxes)
        promoted: List[PseudoBox] = []
        recycled: List[int] = []
        for i in range(boxes.shape[0]):
            if promoted_mask[i]:
                promoted.append(
                    PseudoBox(
                        box=boxes[i].clone(),
                        category=int(categories[i]),
                        target_confidence=float(target_scores[i]),
                        anchor_box=anchor_boxes[int(categories[i])].clone(),
                        roi_feature=roi_features[i].clone() if roi_features is not None else None,
                    )
                )
            else:
                recycled.append(i)
        return promoted, recycled

    # ------------------------------------------------------------------ #
    # Effective-rank-driven convergence                                   #
    # ------------------------------------------------------------------ #
    def effective_rank_of_promoted(self, promoted: List[PseudoBox]) -> float:
        if len(promoted) == 0:
            return 0.0
        feats = torch.stack([p.roi_feature for p in promoted if p.roi_feature is not None])
        return effective_rank(feats) if feats.numel() > 0 else 0.0

    def has_converged(self, erank: float, prev_erank: float) -> bool:
        """``Δerank(t) < δ`` terminates the evolution."""
        return (erank - prev_erank) < self.convergence_threshold

    # ------------------------------------------------------------------ #
    # Full evolution loop (Stage 1 + Stage 2)                             #
    # ------------------------------------------------------------------ #
    def evolve(
        self,
        weak_candidates: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        unlabeled_candidates: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    ) -> Tuple[List[EvolutionTrace], List[PseudoBox]]:
        """Run the two-stage evolution until the effective-rank criterion holds.

        Each candidate is a tuple ``(boxes, target_scores, categories,
        anchor_boxes, roi_features)`` produced by S²-Align + C²-Detector on one
        weakly/unlabeled sample; ``anchor_boxes`` is the per-sample S²-Align
        anchor bank ``[C, 4]``.

        Returns:
            trace: per-iteration :class:`EvolutionTrace`.
            all_promoted: accumulated promoted pseudo-boxes ``D^{w→b} ∪ D^{u→b}``.
        """
        trace: List[EvolutionTrace] = []
        all_promoted: List[PseudoBox] = []
        prev_erank = 0.0

        # Stage 1: weakly labeled, then Stage 2: unlabeled.
        stages = [("weak", weak_candidates), ("unlabeled", unlabeled_candidates)]
        iteration = 0
        for _stage, candidates in stages:
            for boxes, target_scores, categories, sample_anchors, roi_feat in candidates:
                if iteration >= self.max_iterations:
                    break
                promoted, recycled = self.filter_candidates(
                    boxes, target_scores, categories, sample_anchors, roi_feat
                )
                all_promoted.extend(promoted)
                erank = self.effective_rank_of_promoted(all_promoted)
                delta = erank - prev_erank
                converged = self.has_converged(erank, prev_erank) if iteration > 0 else False
                trace.append(
                    EvolutionTrace(
                        iteration=iteration,
                        promoted=promoted,
                        recycled=len(recycled),
                        erank=erank,
                        delta_erank=delta,
                        converged=converged,
                    )
                )
                prev_erank = erank
                iteration += 1
                if converged:
                    return trace, all_promoted

        return trace, all_promoted
