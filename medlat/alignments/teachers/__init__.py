"""Universal teacher model registry.

    from medlat.alignments.teachers import create_teacher, TimmTeacher

    # Create from registry
    teacher = create_teacher("timm", "vit_large_patch14_dinov2.lvd142m")

    # Or directly
    teacher = TimmTeacher("vit_large_patch14_dinov2.lvd142m", img_size=224)
"""

from .base import TeacherModel
from .timm_teacher import TimmTeacher
from .openclip_teacher import OpenCLIPTeacher
from .hf_teacher import HuggingFaceTeacher
from .custom_teacher import CustomTeacher
from .sam_teacher import SAMTeacher, MedSAMTeacher

_REGISTRY = {
    'timm': TimmTeacher,
    'open_clip': OpenCLIPTeacher,
    'openclip': OpenCLIPTeacher,
    'hf': HuggingFaceTeacher,
    'huggingface': HuggingFaceTeacher,
    'custom': CustomTeacher,
}


def create_teacher(source, *args, **kwargs):
    """Create a teacher model from a named source.

    Args:
        source: One of ``'timm'``, ``'open_clip'`` / ``'openclip'``,
            ``'hf'`` / ``'huggingface'``, or ``'custom'``.
        *args, **kwargs: Forwarded to the teacher constructor.

    Returns:
        A frozen :class:`TeacherModel` instance.
    """
    cls = _REGISTRY.get(source)
    if cls is None:
        raise ValueError(
            f"Unknown teacher source '{source}'. "
            f"Choose from: {sorted(set(_REGISTRY.values()), key=lambda c: c.__name__)}"
        )
    return cls(*args, **kwargs)


from . import register as _register  # noqa: E402,F401 — triggers teacher registration


__all__ = [
    'TeacherModel',
    'TimmTeacher',
    'OpenCLIPTeacher',
    'HuggingFaceTeacher',
    'CustomTeacher',
    'SAMTeacher',
    'MedSAMTeacher',
    'create_teacher',
]
