"""Generator-space (Axis B) alignment modules.

All subclass :class:`~medlat.alignments.base.GeneratorAlignment`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import GeneratorAlignment
from .losses import SmoothL1AlignmentLoss


# ---------------------------------------------------------------------------
# REPA
# ---------------------------------------------------------------------------

class REPAAlignment(GeneratorAlignment):
    """REPA: Representation Alignment for generation models.

    Aligns intermediate generator hidden states with frozen teacher
    patch tokens via a learned MLP projection + cosine similarity loss.

    Args:
        teacher: A :class:`TeacherModel` instance.  If ``None``, a
            :class:`TimmTeacher` is created from ``teacher_model_name``.

    Reference: Yu et al., "Representation Alignment for Generation:
    Training Diffusion Transformers Is Easier Than You Think", 2024.
    """

    def __init__(
        self,
        hidden_dim: int,
        teacher=None,
        img_size: int = 224,
        teacher_model_name: str = "vit_large_patch14_dinov2.lvd142m",
        teacher_patch_size: int = 14,
        proj_dim: int = 2048,
        proj_depth: int = 3,
        losses=None,
    ):
        if teacher is None:
            from .teachers import TimmTeacher
            teacher = TimmTeacher(
                teacher_model_name, img_size=img_size,
                patch_size=teacher_patch_size,
            )

        super().__init__(
            name="repa",
            hidden_dim=hidden_dim,
            target_dim=teacher.embed_dim,
            input_transform=teacher.input_transform,
            proj_dim=proj_dim,
            proj_depth=proj_depth,
            losses=losses,
        )
        self.teacher = teacher

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.teacher.extract_features(x)


# ---------------------------------------------------------------------------
# SRA
# ---------------------------------------------------------------------------

class SRAAlignment(GeneratorAlignment):
    """SRA: Self-Representation Alignment for diffusion transformers.

    Uses the model's own EMA copy as teacher — no external model needed.
    Student features at an early layer are projected through a 2-layer MLP
    (SimpleHead) and aligned to EMA teacher features at a deeper layer.

    The EMA model must be registered via :meth:`set_ema_model` before
    training.  An assertion fires if it is missing.

    Reference: https://github.com/vvvvvjdy/SRA
    """

    def __init__(
        self,
        hidden_dim: int,
        teacher_layer: int = 8,
        delta_max: float = 0.2,
        losses=None,
    ):
        if losses is None:
            losses = [(SmoothL1AlignmentLoss(beta=0.05), 1.0)]
        super().__init__(
            name="sra",
            hidden_dim=hidden_dim,
            target_dim=hidden_dim,
            proj_dim=hidden_dim * 2,
            proj_depth=2,
            losses=losses,
        )
        self.teacher_layer = teacher_layer
        self.delta_max = delta_max
        self._ema_model = None

    def set_ema_model(self, ema_model):
        """Register the EMA model as teacher.

        The EMA is stored as a plain attribute (not an ``nn.Module``
        submodule) so its parameters are excluded from this module's
        :meth:`state_dict` and :meth:`parameters`.
        """
        object.__setattr__(self, '_ema_model', ema_model)

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        assert self._ema_model is not None, (
            "SRAAlignment requires an EMA model as teacher. "
            "Call `model.generator_alignment.set_ema_model(ema_model)` "
            "before training."
        )
        noisy_latent = kwargs['noisy_latent']
        timestep = kwargs['timestep']
        class_labels = kwargs['class_labels']
        dataset_id = kwargs.get('dataset_id')

        delta = torch.rand(timestep.shape, device=timestep.device) * self.delta_max
        t_teacher = torch.clamp(timestep.float() - delta, min=0)

        return self._ema_model.forward_to_layer(
            noisy_latent, t_teacher, class_labels,
            layer=self.teacher_layer, dataset_id=dataset_id,
        )


# ---------------------------------------------------------------------------
# HASTE
# ---------------------------------------------------------------------------

class HASTEAlignment(GeneratorAlignment):
    """HASTE: REPA feature projection + attention map distillation.

    Extends REPA with cross-entropy distillation between student and
    teacher attention maps at multiple layers before the alignment layer.

    Args:
        teacher: A :class:`TeacherModel` instance with attention
            distillation support.  If ``None``, a :class:`TimmTeacher`
            is created from ``teacher_model_name``.

    Reference: https://arxiv.org/abs/2505.16792
    """

    def __init__(
        self,
        hidden_dim: int,
        teacher=None,
        img_size: int = 224,
        teacher_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        teacher_patch_size: int = 14,
        num_attn_distill: int = 4,
        teacher_attn_start: int = 8,
        proj_coeff: float = 0.5,
        attn_coeff: float = 0.5,
        proj_dim: int = 2048,
        proj_depth: int = 3,
        losses=None,
    ):
        if teacher is None:
            from .teachers import TimmTeacher
            teacher = TimmTeacher(
                teacher_model_name, img_size=img_size,
                patch_size=teacher_patch_size,
            )

        super().__init__(
            name="haste",
            hidden_dim=hidden_dim,
            target_dim=teacher.embed_dim,
            input_transform=teacher.input_transform,
            proj_dim=proj_dim,
            proj_depth=proj_depth,
            losses=losses,
        )
        self.teacher = teacher
        self.proj_coeff = proj_coeff
        self.attn_coeff = attn_coeff

        self._attn_hooks = []
        self._hook_handles = []
        end = min(teacher_attn_start + num_attn_distill, teacher.num_blocks)
        for idx in range(teacher_attn_start, end):
            hook = teacher.make_attn_hook(idx)
            module = teacher.get_qkv_module(idx)
            handle = module.register_forward_hook(hook)
            self._attn_hooks.append(hook)
            self._hook_handles.append(handle)

    def compute_target(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        features = self.teacher.extract_features(x)
        self._teacher_attn = [h.logits for h in self._attn_hooks]
        return features

    def forward(self, quant, input_image=None, mask=None, **kwargs):
        proj_loss, pred = super().forward(quant, input_image, mask, **kwargs)

        student_attn = kwargs.get('student_attn_logits', [])
        if student_attn and self._teacher_attn:
            attn_loss = self._attention_distill_loss(student_attn, self._teacher_attn)
            self.log_metric("alignment/attn_distill", attn_loss.detach())
        else:
            attn_loss = torch.zeros((), device=pred.device)

        for h in self._attn_hooks:
            h.clear()
        self._teacher_attn = []

        total = self.proj_coeff * proj_loss + self.attn_coeff * attn_loss
        self.log_metric("alignment_loss", total.detach())
        return total, pred

    def _attention_distill_loss(self, student_attn, teacher_attn):
        loss = torch.zeros((), device=student_attn[0].device)
        n = min(len(student_attn), len(teacher_attn))
        for i in range(n):
            s_attn = student_attn[i]
            t_attn = teacher_attn[i]
            num_heads = min(s_attn.shape[1], t_attn.shape[1])
            s = s_attn[:, :num_heads]
            t = t_attn[:, :num_heads]
            if s.shape[-1] != t.shape[-1]:
                BH = s.shape[0] * num_heads
                t = F.interpolate(
                    t.reshape(BH, 1, t.shape[-2], t.shape[-1]),
                    size=(s.shape[-2], s.shape[-1]),
                    mode="bilinear", align_corners=False,
                ).reshape(s.shape[0], num_heads, s.shape[-2], s.shape[-1])
            s_flat = s.reshape(-1, s.shape[-1])
            t_flat = t.reshape(-1, t.shape[-1])
            t_soft = t_flat.softmax(dim=-1)
            s_log = s_flat.log_softmax(dim=-1)
            loss = loss - (t_soft * s_log).sum(dim=-1).mean()
        return loss / max(n, 1)
