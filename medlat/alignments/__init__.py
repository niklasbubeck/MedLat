"""
medlat.alignments — alignment module package.

    from medlat.alignments import AlignmentModule, HOGAlignment, DinoAlignment
    from medlat.alignments import CosineSimilarityLoss, DistmatMarginLoss
    from medlat.alignments import TokenizerAlignment, GeneratorAlignment
"""
from .base import AlignmentModule, TokenizerAlignment, GeneratorAlignment
from .losses import (
    AlignmentLoss,
    CosineSimilarityLoss,
    MSEAlignmentLoss,
    SmoothL1AlignmentLoss,
    DistmatMarginLoss,
    CosineMarginLoss,
)
from .tokenizer_alignments import (
    HOGAlignment,
    DinoAlignment,
    ClipAlignment,
    MAEAlignment,
    BiomedClipAlignment,
)
from .utils import mean_flat, _Normalize, _Denormalize, HOGGenerator, IdentityDecoder

__all__ = [
    "AlignmentModule",
    "TokenizerAlignment",
    "GeneratorAlignment",
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
