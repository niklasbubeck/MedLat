"""OpenCLIPTeacher — wraps an open_clip visual encoder."""

import torch
import torch.nn.functional as F

from .base import TeacherModel


class _MultiheadAttnHook:
    """Forward hook for ``nn.MultiheadAttention`` (open_clip attention).

    Computes attention logits from the module's ``in_proj_weight`` /
    ``in_proj_bias`` and the query input, stripping prefix tokens.
    """

    def __init__(self, num_heads, head_dim, num_prefix_tokens=1,
                 batch_first=False):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.batch_first = batch_first
        self.logits = None

    def __call__(self, module, input, output):
        x = input[0]
        if not self.batch_first:
            x = x.transpose(0, 1)  # (L, B, D) → (B, L, D)
        B, N, D = x.shape
        qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        p = self.num_prefix_tokens
        if p > 0:
            logits = logits[:, :, p:, p:]
        self.logits = logits

    def clear(self):
        self.logits = None


class OpenCLIPTeacher(TeacherModel):
    """Teacher wrapping an open_clip visual encoder.

    Runs the visual transformer manually (conv1 → class_embedding →
    positional_embedding → ln_pre → transformer) and returns patch
    tokens before the final LN / projection head.

    Args:
        model_name: open_clip model name (e.g. ``"ViT-B-16"``).
        pretrained: Pretrained checkpoint tag (e.g. ``"openai"``,
            ``"laion2b_s34b_b88k"``).
        img_size: Expected input resolution.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name, pretrained='openai', img_size=224):
        try:
            import open_clip
        except ImportError as e:
            raise RuntimeError(
                "open_clip is required for OpenCLIPTeacher."
            ) from e
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        model.requires_grad_(False)
        self.visual = model.visual
        self._img_size = img_size

        mean = getattr(self.visual, 'image_mean', self.CLIP_MEAN)
        std = getattr(self.visual, 'image_std', self.CLIP_STD)
        self._mean = list(mean)
        self._std = list(std)

        if hasattr(self.visual, 'conv1'):
            self._embed_dim = self.visual.conv1.out_channels
        elif hasattr(self.visual, 'trunk'):
            self._embed_dim = self.visual.trunk.embed_dim
        else:
            self._embed_dim = self.visual.ln_post.normalized_shape[0]

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def num_prefix_tokens(self):
        return 1 if hasattr(self.visual, 'class_embedding') else 0

    @property
    def input_transform(self):
        from medlat.alignments.utils import _Normalize
        return _Normalize(self._mean, self._std)

    def extract_features(self, x):
        if x.shape[-2] != self._img_size or x.shape[-1] != self._img_size:
            x = F.interpolate(x, size=(self._img_size, self._img_size),
                              mode='bilinear', align_corners=False)

        v = self.visual

        if hasattr(v, 'trunk'):
            x = v.trunk.forward_features(x)
            p = getattr(v.trunk, 'num_prefix_tokens', 1)
            return x[:, p:]

        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        if hasattr(v, 'class_embedding') and v.class_embedding is not None:
            cls = v.class_embedding.to(x.dtype).expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)

        x = x.permute(1, 0, 2)   # (B, L, D) → (L, B, D) for transformer
        x = v.transformer(x)
        x = x.permute(1, 0, 2)   # → (B, L, D)

        p = self.num_prefix_tokens
        return x[:, p:]

    # ------------------------------------------------------------------
    # Attention distillation
    # ------------------------------------------------------------------

    # open_clip serves two visual layouts: the native ViT
    # (``visual.transformer.resblocks``, nn.MultiheadAttention) and hf-hub timm
    # models wrapped as ``TimmModel`` (``visual.trunk.blocks``, fused qkv Linear
    # — BiomedCLIP is one). Attention capture supports both; ``_timm_trunk``
    # picks the branch.

    @property
    def _timm_trunk(self):
        trunk = getattr(self.visual, 'trunk', None)
        return trunk if trunk is not None and hasattr(trunk, 'blocks') else None

    @property
    def supports_attention(self):
        if self._timm_trunk is not None:
            return True
        return (hasattr(self.visual, 'transformer')
                and hasattr(self.visual.transformer, 'resblocks'))

    @property
    def num_blocks(self):
        if self._timm_trunk is not None:
            return len(self._timm_trunk.blocks)
        return len(self.visual.transformer.resblocks)

    def get_qkv_module(self, block_idx):
        if self._timm_trunk is not None:
            return self._timm_trunk.blocks[block_idx].attn.qkv
        return self.visual.transformer.resblocks[block_idx].attn

    def get_num_heads(self, block_idx):
        if self._timm_trunk is not None:
            return self._timm_trunk.blocks[block_idx].attn.num_heads
        return self.visual.transformer.resblocks[block_idx].attn.num_heads

    def get_head_dim(self, block_idx):
        if self._timm_trunk is not None:
            attn = self._timm_trunk.blocks[block_idx].attn
            return attn.head_dim
        return self._embed_dim // self.get_num_heads(block_idx)

    def make_attn_hook(self, block_idx):
        if self._timm_trunk is not None:
            from .timm_teacher import _FusedQKVHook
            n_prefix = getattr(self._timm_trunk, 'num_prefix_tokens', 1)
            return _FusedQKVHook(
                num_heads=self.get_num_heads(block_idx),
                head_dim=self.get_head_dim(block_idx),
                num_prefix_tokens=n_prefix,
            )
        batch_first = getattr(
            self.visual.transformer.resblocks[block_idx].attn,
            'batch_first', False,
        )
        return _MultiheadAttnHook(
            num_heads=self.get_num_heads(block_idx),
            head_dim=self.get_head_dim(block_idx),
            num_prefix_tokens=self.num_prefix_tokens,
            batch_first=batch_first,
        )
