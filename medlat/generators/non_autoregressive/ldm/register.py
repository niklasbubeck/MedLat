from .openaimodel import UNetModel
from medlat.registry import register_model

"""
Model configurations as taken from Table 14 in the paper
"""

@register_model("ldm.f1")
def LDM_f1(img_size, in_channels=3, out_channels=3, model_channels=192, num_head_channels=32, **kwargs):
    return UNetModel(
        img_size=img_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        channel_mult=[1,1,2,2,4,4],
        num_head_channels=num_head_channels,
        **kwargs
    )

@register_model("ldm.f2")
def LDM_f2(img_size, in_channels=3, out_channels=3, model_channels=192, num_head_channels=32, **kwargs):
    return UNetModel(
        img_size=img_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        channel_mult=[1,2,2,4],
        num_head_channels=num_head_channels,
        **kwargs
    )

@register_model("ldm.f4")
def LDM_f4(img_size, in_channels=3, out_channels=3, model_channels=224, num_head_channels=32, **kwargs):
    return UNetModel(
        img_size=img_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        channel_mult=[1,2,3,4],
        num_head_channels=num_head_channels,
        **kwargs
    )

@register_model("ldm.f8")
def LDM_f8(img_size, in_channels=4, out_channels=4, model_channels=256, num_head_channels=32, **kwargs):
    return UNetModel(
        img_size=img_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        channel_mult=[1,2,4],
        num_head_channels=num_head_channels,
        **kwargs
    )

@register_model("ldm.f16")
def LDM_f16(img_size, in_channels=4, out_channels=4, model_channels=256, num_head_channels=32, **kwargs):
    return UNetModel(
        img_size=img_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=2,
        attention_resolutions=[16, 8, 4],
        channel_mult=[1,2,4],
        num_head_channels=num_head_channels,
        **kwargs
    )


