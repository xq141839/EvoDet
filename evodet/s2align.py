"""Semantic-Spatial Alignment (S²-Align).

EvoDet module 1/3. S²-Align elicits *class-discriminative spatial prior
anchors* from frozen Vision-Foundation-Model (VFM) representations via
spectral factorization and prototype-guided activation, bridging image-level
semantics with fine-grained spatial localization.

Pipeline (Eqs. of the paper, Sec. III-B)
-----------------------------------------
Given a frozen VFM image embedding ``F ∈ R^{d×H×W}`` and a semantic prototype
``p_c ∈ R^d`` for category ``c``:

1. **Prototype-guided activation** — patch-wise cosine similarity::

       A_c^(i,j) = ⟨F^(i,j), p_c⟩ / (‖F^(i,j)‖₂ · ‖p_c‖₂)

2. **Spectral factorization** — SVD of ``F`` kept to the top-``k`` singular
   components ``U_k Σ_k V_kᵀ`` removes high-frequency noise while preserving
   the dominant category-consistent spatial response. The similarity map is
   refined into the *category-specific semantic map*::

       S_c = cosine(U_k Σ_k V_kᵀ, p_c)        (spectral-refined similarity)

3. **Image-level supervision** — ``L_CLS`` supervises ``S_c`` via global
   average pooling (GAP) and binary cross-entropy.

4. **Spatial prior anchors** — temperature scaling + sigmoid produce a
   per-category attention map::

       M_c^(i,j) = σ((S_c^(i,j) − μ_c) / T),   μ_c = (1/HW) Σ_{i,j} S_c^(i,j)

   and a mean-gated threshold yields discrete spatial priors::

       M̂_c^(i,j) = 𝕀[M_c^(i,j) ≥ τ_a]

The semantic prototypes are initialised from class-specific feature centroids
of the box-labeled data (GAP inside ground-truth boxes) and updated via EMA
with attention-weighted embeddings (see :meth:`init_prototypes_from_boxes` and
:meth:`ema_update`), so they co-evolve with the detector while remaining stable
against per-sample fluctuations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["S2Align"]


def _l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def spectral_factorize(feat: torch.Tensor, top_k: int) -> torch.Tensor:
    """Low-rank reconstruction of ``feat`` via top-``k`` SVD components.

    Args:
        feat: image embedding ``[B, d, H, W]``.
        top_k: number of retained singular components.

    Returns:
        ``U_k Σ_k V_kᵀ`` reshaped back to ``[B, d, H, W]``.
    """
    B, d, H, W = feat.shape
    x = feat.flatten(2)  # [B, d, HW]
    u, s, vh = torch.linalg.svd(x, full_matrices=False)  # u [B,d,r], s [B,r], vh [B,r,HW]
    k = min(top_k, s.shape[1])
    u_k, s_k, vh_k = u[:, :, :k], s[:, :k], vh[:, :k, :]
    recon = torch.matmul(u_k * s_k.unsqueeze(1), vh_k)  # [B, d, HW]
    return recon.reshape(B, d, H, W)


class S2Align(nn.Module):
    """Semantic-Spatial Alignment module.

    Args:
        num_classes: number of categories ``C``.
        feature_dim: embedding dimension ``d`` of the frozen VFM features.
        feature_index: which scale of the multi-scale VFM features to use
            (the backbone returns ``[1/8, 1/16, 1/32]`` features; ``1`` selects
            the native stride-16 patch embedding).
        temperature: attention-map temperature ``T``.
        activation_threshold: anchor activation threshold ``τ_a``.
        top_k: number of retained SVD components ``k`` for spectral filtering.
    """

    def __init__(
        self,
        num_classes: int = 80,
        feature_dim: int = 256,
        feature_index: int = 1,
        temperature: float = 0.1,
        activation_threshold: float = 0.5,
        top_k: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.feature_index = int(feature_index)
        self.temperature = float(temperature)
        self.activation_threshold = float(activation_threshold)
        self.top_k = int(top_k)

        # Semantic prototypes p_c ∈ R^d. L2-normalised on the fly.
        self.prototypes = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    # ------------------------------------------------------------------ #
    # Forward: prototype-guided activation + spectral factorization       #
    # ------------------------------------------------------------------ #
    def cosine_similarity_map(self, feat: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Per-category patch-wise cosine similarity ``A_c``, ``[B, C, H, W]``."""
        f = _l2_normalize(feat, dim=1)              # [B, d, H, W]
        p = _l2_normalize(prototypes, dim=1)        # [C, d]
        return torch.einsum("bdhw,cd->bchw", f, p)  # [B, C, H, W]

    def semantic_map(self, feat: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Spectral-refined semantic map ``S_c``, ``[B, C, H, W]``."""
        feat_k = spectral_factorize(feat, self.top_k)
        return self.cosine_similarity_map(feat_k, prototypes)

    def attention_map(self, semantic: torch.Tensor) -> torch.Tensor:
        """Per-category attention map ``M_c = σ((S_c − μ_c) / T)``."""
        mu = semantic.mean(dim=(2, 3), keepdim=True)
        return torch.sigmoid((semantic - mu) / self.temperature)

    def spatial_prior_anchors(
        self, attention: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discrete spatial prior anchors ``M̂_c`` and their tight boxes ``b̂_c``.

        Returns:
            mask: ``[B, C, H, W]`` binary anchor mask ``M̂_c``.
            boxes: ``[B, C, 4]`` axis-aligned tight boxes ``b̂_c`` in ``xyxy``
                (pixel units, computed on the feature grid and normalised to
                ``[0, 1]``).
        """
        mask = (attention >= self.activation_threshold).float()
        B, C, H, W = mask.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=mask.device, dtype=mask.dtype),
            torch.arange(W, device=mask.device, dtype=mask.dtype),
            indexing="ij",
        )
        has = mask.sum(dim=(2, 3)) > 0  # [B, C]
        # per-category activation extent
        x1 = (mask * xs).flatten(2).min(-1).values
        y1 = (mask * ys).flatten(2).min(-1).values
        x2 = (mask * xs).flatten(2).max(-1).values
        y2 = (mask * ys).flatten(2).max(-1).values
        boxes = torch.stack([x1, y1, x2 + 1.0, y2 + 1.0], dim=-1)  # [B, C, 4]
        # normalise to [0, 1]
        boxes[..., [0, 2]] = boxes[..., [0, 2]] / float(W)
        boxes[..., [1, 3]] = boxes[..., [1, 3]] / float(H)
        boxes = boxes * has.unsqueeze(-1)  # empty regions -> degenerate box
        return mask, boxes

    def forward(
        self,
        feat: torch.Tensor,
        image_labels: Optional[torch.Tensor] = None,
        return_all: bool = True,
    ) -> dict:
        """Compute semantic maps and spatial prior anchors.

        Args:
            feat: frozen VFM embedding ``[B, d, H, W]`` (or a list of scales).
            image_labels: optional image-level labels ``[B, C]`` for ``L_CLS``.
            return_all: include intermediate maps in the output dict.

        Returns:
            dict with keys ``semantic_map``, ``attention_map``,
            ``anchor_mask``, ``anchor_boxes`` and (when ``image_labels`` is
            given) ``loss_cls``.
        """
        if isinstance(feat, (list, tuple)):
            feat = feat[self.feature_index]

        prototypes = _l2_normalize(self.prototypes, dim=1)
        semantic = self.semantic_map(feat, prototypes)      # S_c
        attention = self.attention_map(semantic)            # M_c
        anchor_mask, anchor_boxes = self.spatial_prior_anchors(attention)

        out = {
            "semantic_map": semantic,
            "attention_map": attention,
            "anchor_mask": anchor_mask,
            "anchor_boxes": anchor_boxes,
        }
        if image_labels is not None:
            out["loss_cls"] = self.classification_loss(semantic, image_labels)
        if not return_all:
            out = {"anchor_mask": anchor_mask, "anchor_boxes": anchor_boxes}
        return out

    # ------------------------------------------------------------------ #
    # Image-level supervision L_CLS                                      #
    # ------------------------------------------------------------------ #
    def classification_loss(
        self, semantic: torch.Tensor, image_labels: torch.Tensor
    ) -> torch.Tensor:
        """``L_CLS``: GAP of the semantic map supervised by BCE.

        Args:
            semantic: semantic map ``S_c``, ``[B, C, H, W]``.
            image_labels: image-level labels ``[B, C]`` in ``{0, 1}``.
        """
        logits = semantic.mean(dim=(2, 3))  # [B, C]
        return F.binary_cross_entropy_with_logits(logits, image_labels.float())

    # ------------------------------------------------------------------ #
    # Prototype initialisation / update                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def init_prototypes_from_boxes(
        self,
        feat: torch.Tensor,
        boxes: List[List[torch.Tensor]],
        labels: List[torch.Tensor],
    ) -> None:
        """Initialise prototypes from class centroids of box-labeled data.

        GAP of the embedding inside each ground-truth box, averaged per class.

        Args:
            feat: single-image embedding ``[1, d, H, W]`` (one image per call
                is accumulated; the caller loops over ``D^b``).
            boxes: ``[[x1, y1, x2, y2]]`` per image, normalised to ``[0, 1]``.
            labels: class index per box.
        """
        acc = self.prototypes.data.clone() * 0.0
        cnt = torch.zeros(self.num_classes, device=feat.device)
        for box, lab in zip(boxes, labels):
            box = box.to(feat.device)
            acc[lab] += self._roi_pool(feat, box.unsqueeze(0)).squeeze(0)
            cnt[lab] += 1.0
        mask = cnt > 0
        acc[mask] = acc[mask] / cnt[mask].unsqueeze(1)
        # fall back to current value for classes without any box
        acc[~mask] = self.prototypes.data[~mask]
        self.prototypes.data.copy_(acc)

    @torch.no_grad()
    def ema_update(
        self, feat: torch.Tensor, attention: torch.Tensor, momentum: float = 0.9
    ) -> None:
        """Attention-weighted EMA update of the semantic prototypes.

        ``p_c ← m·p_c + (1−m)·Σ_{i,j} M_c[i,j] F[i,j] / Σ M_c[i,j]``.
        """
        w = attention.detach()  # [B, C, H, W]
        denom = w.sum(dim=(2, 3)).clamp_min(1e-6)  # [B, C]
        num = torch.einsum("bchw,bdhw->bcd", w, feat)  # [B, C, d]
        new_p = (num / denom.unsqueeze(-1)).mean(0)     # [C, d]
        self.prototypes.data.mul_(momentum).add_(new_p, alpha=1.0 - momentum)

    def _roi_pool(self, feat: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """GAP of ``feat`` inside ``boxes`` (bilinear sampling).

        Args:
            feat: ``[1, d, H, W]``.
            boxes: ``[N, 4]`` normalised ``xyxy``.

        Returns:
            ``[N, d]`` pooled embeddings.
        """
        from torchvision.ops import roi_align

        N = boxes.shape[0]
        batch_idx = torch.zeros(N, 1, device=boxes.device)
        rois = torch.cat([batch_idx, boxes], dim=1)  # [N, 5]
        return roi_align(feat, rois, output_size=(1, 1), spatial_scale=1.0).squeeze(-1).squeeze(-1)

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save_prototypes(self, path: str) -> None:
        torch.save(self.prototypes.data, path)

    def load_prototypes(self, path: str) -> None:
        self.prototypes.data.copy_(torch.load(path, map_location=self.prototypes.device))
