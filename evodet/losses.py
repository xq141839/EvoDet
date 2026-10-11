"""Losses referenced by EvoDet.

EvoDet is trained with the total loss::

    L_Total = L_Det + λ_c · L_CLS

* ``L_CLS`` — image-level classification loss of S²-Align (GAP + BCE),
  implemented in :meth:`S2Align.classification_loss` and wrapped below.
* ``L_Det`` — the standard DEIM detection loss (box regression + classification
  via Hungarian matching), provided by the vendored ``engine.deim.DEIMCriterion``
  (reference [43] DEIMv2). No training loop is included in this release.
* ``L_Total`` — the balanced combination used across MixEvo evolving iterations.
"""

from __future__ import annotations

import torch

from .s2align import S2Align

__all__ = ["l_cls", "l_total"]


def l_cls(s2align: S2Align, semantic_map: torch.Tensor, image_labels: torch.Tensor) -> torch.Tensor:
    """``L_CLS`` — binary cross-entropy between GAP(S_c) and image-level labels."""
    return s2align.classification_loss(semantic_map, image_labels)


def l_total(l_det: torch.Tensor, l_cls: torch.Tensor, lambda_c: float = 1.0) -> torch.Tensor:
    """``L_Total = L_Det + λ_c · L_CLS`` (Sec. III-D)."""
    return l_det + lambda_c * l_cls
