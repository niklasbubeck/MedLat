"""
medlat.alignments — alignment module package.

    from medlat.alignments import AlignmentModule, HOGAlignment, DinoAlignment
    from medlat.alignments import CosineSimilarityLoss, DistmatMarginLoss
"""
from .base import AlignmentModule
from .losses import (
    AlignmentLoss,
    CosineSimilarityLoss,
    MSEAlignmentLoss,
    SmoothL1AlignmentLoss,
    DistmatMarginLoss,
    CosineMarginLoss,
)
from .alignments import (
    HOGAlignment,
    DinoAlignment,
    ClipAlignment,
    VFFoundationAlignment,
    FoundationFeatureExtractor,
)
from .utils import mean_flat, _Normalize, _Denormalize, HOGGenerator

__all__ = [
    "AlignmentModule",
    "AlignmentLoss",
    "CosineSimilarityLoss",
    "MSEAlignmentLoss",
    "SmoothL1AlignmentLoss",
    "DistmatMarginLoss",
    "CosineMarginLoss",
    "HOGAlignment",
    "HOGGenerator",
    "DinoAlignment",
    "ClipAlignment",
    "VFFoundationAlignment",
    "FoundationFeatureExtractor",
    "mean_flat",
    "_Normalize",
    "_Denormalize",
]
