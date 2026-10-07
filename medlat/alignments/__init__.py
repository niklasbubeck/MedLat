"""
medlat.alignments — alignment module package.

    from medlat.alignments import AlignmentModule, HOGAlignment, DinoAlignment
    from medlat.alignments import CosineSimilarityLoss, DistmatMarginLoss
    from medlat.alignments import TokenizerAlignment, GeneratorAlignment
    from medlat.alignments import create_teacher, TimmTeacher
"""
from .base import AlignmentModule, TokenizerAlignment, GeneratorAlignment
from .losses import (
    AlignmentLoss,
    CosineSimilarityLoss,
    MSEAlignmentLoss,
    SmoothL1AlignmentLoss,
    DistmatMarginLoss,
    CosineMarginLoss,
    DispersiveLoss,
)
from .tokenizer_alignments import (
    HOGAlignment,
    DinoAlignment,
    ClipAlignment,
    MAEAlignment,
    BiomedClipAlignment,
)
from .generator_alignments import (
    REPAAlignment,
    SRAAlignment,
    HASTEAlignment,
)
from .teachers import (
    TeacherModel,
    TimmTeacher,
    OpenCLIPTeacher,
    HuggingFaceTeacher,
    CustomTeacher,
    SAMTeacher,
    MedSAMTeacher,
    create_teacher,
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
    "DispersiveLoss",
    "HOGAlignment",
    "HOGGenerator",
    "DinoAlignment",
    "ClipAlignment",
    "MAEAlignment",
    "BiomedClipAlignment",
    "REPAAlignment",
    "SRAAlignment",
    "HASTEAlignment",
    "TeacherModel",
    "TimmTeacher",
    "OpenCLIPTeacher",
    "HuggingFaceTeacher",
    "CustomTeacher",
    "SAMTeacher",
    "MedSAMTeacher",
    "create_teacher",
    "mean_flat",
    "IdentityDecoder",
    "_Normalize",
    "_Denormalize",
]
