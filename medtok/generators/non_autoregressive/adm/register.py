import torch
from medtok.registry import register_model
from medtok.utils import instantiate_from_config
from .unet import UNetModel as ADM
from .unet import EncoderUNetModel as ADM_classifier

"""
Optimized configs for ADM diffusion and classifier models.
Models are now registered with method-based configurations.
"""

# ---- Diffusion Model Config Methods ----

@register_model("adm.diffusion.64C")
def adm_diffusion_64_conditioned_cfg(**overrides):
    cfg = dict(
        image_size=64,
        in_channels=3,
        model_channels=192,
        num_classes=1000,
        out_channels=6,
        num_res_blocks=3,
        attention_resolutions=[32, 16, 8],
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.diffusion.64U")
def adm_diffusion_64_unconditioned_cfg(**overrides):
    cfg = dict(
        image_size=64,
        in_channels=3,
        model_channels=192,
        num_classes=None,
        out_channels=6,
        num_res_blocks=3,
        attention_resolutions=[32, 16, 8],
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,  # Can override as needed
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.diffusion.128C")
def adm_diffusion_128_conditioned_cfg(**overrides):
    cfg = dict(
        image_size=128,
        in_channels=3,
        model_channels=256,
        num_classes=1000,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(1, 1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.diffusion.128U")
def adm_diffusion_128_unconditioned_cfg(**overrides):
    cfg = dict(
        image_size=128,
        in_channels=3,
        model_channels=256,
        num_classes=None,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(1, 1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.diffusion.256C")
def adm_diffusion_256_conditioned_cfg(**overrides):
    cfg = dict(
        image_size=256,
        in_channels=3,
        model_channels=256,
        num_classes=1000,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.diffusion.256U")
def adm_diffusion_256_unconditioned_cfg(**overrides):
    cfg = dict(
        image_size=256,
        in_channels=3,
        model_channels=256,
        num_classes=None,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.diffusion.512C")
def adm_diffusion_512_conditioned_cfg(**overrides):
    cfg = dict(
        image_size=512,
        in_channels=3,
        model_channels=256,
        num_classes=1000,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(0.5, 1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.diffusion.512U")
def adm_diffusion_512_unconditioned_cfg(**overrides):
    cfg = dict(
        image_size=512,
        in_channels=3,
        model_channels=256,
        num_classes=None,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(0.5, 1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    cfg.update(overrides)
    return cfg

# ---- Classifier Model Config Methods ----

@register_model("adm.classifier.64C")
def adm_classifier_64_cfg(**overrides):
    cfg = dict(
        image_size=64,
        in_channels=3,
        model_channels=192,
        out_channels=1000,
        num_res_blocks=3,
        attention_resolutions=[32, 16, 8],
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        pool="adaptive",
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.classifier.128C")
def adm_classifier_128_cfg(**overrides):
    cfg = dict(
        image_size=128,
        in_channels=3,
        model_channels=256,
        out_channels=1000,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(1, 1, 2, 3, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        pool="adaptive",
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.classifier.256C")
def adm_classifier_256_cfg(**overrides):
    cfg = dict(
        image_size=256,
        in_channels=3,
        model_channels=256,
        out_channels=1000,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        pool="adaptive",
    )
    cfg.update(overrides)
    return cfg

@register_model("adm.classifier.512C")
def adm_classifier_512_cfg(**overrides):
    cfg = dict(
        image_size=512,
        in_channels=3,
        model_channels=256,
        out_channels=1000,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        dropout=0.0,
        channel_mult=(0.5, 1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        pool="adaptive",
    )
    cfg.update(overrides)
    return cfg


class ADM_U(ADM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ADM_G(torch.nn.Module):
    """
    This model is two stage without possibility of end-to-end training! Only for inference/testing.
    """
    def __init__(self, diffmodel_cfg, classmodel_cfg):
        super().__init__()
        self.diffusion = instantiate_from_config(diffmodel_cfg).eval()
        self.classifier = instantiate_from_config(classmodel_cfg).eval()

    def cond_fn(self, x, t, y=None, classifier_scale=1.):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    def forward(self, *args, **kwargs): 
        return self.diffusion.forward(*args, **kwargs)