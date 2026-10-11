"""EvoDet — Self-Evolving Mix-Granularity Medical Image Detection.

Importing this package registers :class:`EvoDet`, :class:`EvoDetBackbone`,
:class:`C2Decoder` and :class:`S2Align` into the DEIMv2 ``engine`` registry so
they can be instantiated through ``YAMLConfig`` (the same loading path used by
the plain DEIM/DEIMv2 detector).

Three modules
-------------
* :class:`S2Align`    — Semantic-Spatial Alignment.
* :class:`C2Decoder`  — Complementary Confidence-Calibrated Detector.
* :class:`MixEvo`     — Mix-Granularity Collaborative Label Evolution.
"""

from .s2align import S2Align
from .c2detector import C2Decoder, C2Detector, margin_nms, decode_c2
from .mixevo import MixEvo, effective_rank, PseudoBox, EvolutionTrace
from .evodet import EvoDetBackbone, EvoDet, load_evodet, evodet_postprocess
from . import losses  # noqa: F401  (import for completeness)

__all__ = [
    "S2Align",
    "C2Decoder",
    "C2Detector",
    "margin_nms",
    "decode_c2",
    "MixEvo",
    "effective_rank",
    "PseudoBox",
    "EvolutionTrace",
    "EvoDetBackbone",
    "EvoDet",
    "load_evodet",
    "evodet_postprocess",
]
