"""HuggingFaceTeacher — wraps a HuggingFace vision transformer."""

import torch.nn.functional as F

from .base import TeacherModel


class _SeparateQKVHook:
    """Forward hook for HF attention with separate ``query`` / ``key`` Linears."""

    def __init__(self, num_heads, head_dim, num_prefix_tokens=1):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.logits = None

    def __call__(self, module, input, output):
        x = input[0]  # (B, L, D)
        B, N, _ = x.shape
        q = module.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = module.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        p = self.num_prefix_tokens
        if p > 0:
            logits = logits[:, :, p:, p:]
        self.logits = logits

    def clear(self):
        self.logits = None


class HuggingFaceTeacher(TeacherModel):
    """Teacher wrapping a HuggingFace vision transformer.

    Works with ``ViTModel``, ``Dinov2Model``, and similar architectures
    that expose ``last_hidden_state`` and layer-wise attention with
    separate ``query`` / ``key`` / ``value`` Linears.

    Args:
        model_name: HuggingFace model identifier or local path.
        img_size: Expected input resolution.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name, img_size=224):
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for HuggingFaceTeacher."
            ) from e
        super().__init__()
        full_model = AutoModel.from_pretrained(model_name)
        if hasattr(full_model, 'vision_model'):
            self.model = full_model.vision_model
        else:
            self.model = full_model
        self.model.requires_grad_(False)
        self._img_size = img_size
        cfg = full_model.config
        if hasattr(cfg, 'vision_config'):
            cfg = cfg.vision_config
        self._embed_dim = cfg.hidden_size
        self._num_heads = cfg.num_attention_heads

        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            self._mean = list(processor.image_mean)
            self._std = list(processor.image_std)
        except Exception:
            self._mean = self.IMAGENET_MEAN
            self._std = self.IMAGENET_STD

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def num_prefix_tokens(self):
        if hasattr(self.model.config, 'num_prefix_tokens'):
            return self.model.config.num_prefix_tokens
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'cls_token'):
            return 1
        return 0

    @property
    def input_transform(self):
        from medlat.alignments.utils import _Normalize
        return _Normalize(self._mean, self._std)

    def extract_features(self, x):
        if x.shape[-2] != self._img_size or x.shape[-1] != self._img_size:
            x = F.interpolate(x, size=(self._img_size, self._img_size),
                              mode='bilinear', align_corners=False)
        outputs = self.model(pixel_values=x)
        features = outputs.last_hidden_state
        p = self.num_prefix_tokens
        return features[:, p:]

    # ------------------------------------------------------------------
    # Attention distillation
    # ------------------------------------------------------------------

    def _get_layers(self):
        """Locate the list of transformer layers."""
        for attr in ('encoder.layer', 'encoder.layers', 'layers'):
            parts = attr.split('.')
            obj = self.model
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise ValueError("Cannot locate transformer layers in HF model.")

    def _get_attn_module(self, block_idx):
        """Return the self-attention module with .query / .key Linears."""
        layer = self._get_layers()[block_idx]
        for path in ('attention.attention', 'attention.self',
                     'self_attn', 'attention'):
            parts = path.split('.')
            mod = layer
            for p in parts:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            if mod is not None and hasattr(mod, 'query') and hasattr(mod, 'key'):
                return mod
        raise ValueError(
            f"Cannot find attention module with separate Q/K in layer {block_idx}."
        )

    @property
    def supports_attention(self):
        try:
            self._get_layers()
            self._get_attn_module(0)
            return True
        except (ValueError, IndexError):
            return False

    @property
    def num_blocks(self):
        return len(self._get_layers())

    def get_qkv_module(self, block_idx):
        return self._get_attn_module(block_idx)

    def get_num_heads(self, block_idx):
        attn = self._get_attn_module(block_idx)
        return getattr(attn, 'num_attention_heads', self._num_heads)

    def get_head_dim(self, block_idx):
        return self._embed_dim // self.get_num_heads(block_idx)

    def make_attn_hook(self, block_idx):
        return _SeparateQKVHook(
            num_heads=self.get_num_heads(block_idx),
            head_dim=self.get_head_dim(block_idx),
            num_prefix_tokens=self.num_prefix_tokens,
        )
