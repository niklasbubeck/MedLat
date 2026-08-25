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
    MAEAlignment,
    BiomedClipAlignment,
)
from .utils import mean_flat, _Normalize, _Denormalize, HOGGenerator, IdentityDecoder

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
    "MAEAlignment",
    "BiomedClipAlignment",
    "mean_flat",
    "IdentityDecoder",
    "_Normalize",
    "_Denormalize",
]
