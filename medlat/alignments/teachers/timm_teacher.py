"""TimmTeacher — wraps any timm ViT-family model."""

import torch.nn.functional as F

from .base import TeacherModel


class _FusedQKVHook:
    """Forward hook for timm's fused ``attn.qkv`` Linear layer."""

    def __init__(self, num_heads, head_dim, num_prefix_tokens=1):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.logits = None

    def __call__(self, module, input, output):
        B, N, _ = output.shape
        qkv = output.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        p = self.num_prefix_tokens
        if p > 0:
            logits = logits[:, :, p:, p:]
        self.logits = logits

    def clear(self):
        self.logits = None


class TimmTeacher(TeacherModel):
    """Teacher wrapping any timm ViT-family model.

    Args:
        model_name: timm model identifier.
        pretrained: Load pretrained weights (default ``True``).
        img_size: Expected input resolution — images are resized to this
            inside :meth:`extract_features`.
        patch_size: Override the model's patch size (optional).
        **model_kwargs: Extra keyword arguments forwarded to
            ``timm.create_model`` (e.g. ``dynamic_img_size=True``).
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model_name, pretrained=True, img_size=224,
                 patch_size=None, **model_kwargs):
        try:
            from timm import create_model
        except ImportError as e:
            raise RuntimeError("timm is required for TimmTeacher.") from e
        super().__init__()
        kwargs = dict(pretrained=pretrained, img_size=img_size, **model_kwargs)
        if patch_size is not None:
            kwargs['patch_size'] = patch_size
        self.model = create_model(model_name, **kwargs)
        self.model.requires_grad_(False)
        self._img_size = img_size

    @property
    def embed_dim(self):
        return self.model.embed_dim

    @property
    def num_prefix_tokens(self):
        return getattr(self.model, 'num_prefix_tokens', 1)

    @property
    def input_transform(self):
        from medlat.alignments.utils import _Normalize
        return _Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)

    def extract_features(self, x):
        if x.shape[-2] != self._img_size or x.shape[-1] != self._img_size:
            x = F.interpolate(x, size=(self._img_size, self._img_size),
                              mode='bilinear', align_corners=False)
        return self.model.forward_features(x)[:, self.num_prefix_tokens:]

    # ------------------------------------------------------------------
    # Attention distillation
    # ------------------------------------------------------------------

    @property
    def supports_attention(self):
        return hasattr(self.model, 'blocks')

    @property
    def num_blocks(self):
        return len(self.model.blocks)

    def get_qkv_module(self, block_idx):
        return self.model.blocks[block_idx].attn.qkv

    def get_num_heads(self, block_idx):
        return self.model.blocks[block_idx].attn.num_heads

    def get_head_dim(self, block_idx):
        return self.embed_dim // self.get_num_heads(block_idx)

    def make_attn_hook(self, block_idx):
        return _FusedQKVHook(
            num_heads=self.get_num_heads(block_idx),
            head_dim=self.get_head_dim(block_idx),
            num_prefix_tokens=self.num_prefix_tokens,
        )
