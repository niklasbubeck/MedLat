import torch
import torch.nn as nn
import torch.nn.functional as F

def l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm."""
    norm = torch.sqrt(torch.sum(x * x, dim=axis, keepdim=True) + eps)
    return x / norm


def get_norm_layer(train=True, norm_type='BN', num_groups=32):
    """Normalization layer selection."""
    if norm_type == 'BN':
        return nn.BatchNorm2d if train else lambda num_features: nn.BatchNorm2d(num_features, affine=False)
    elif norm_type == 'LN':
        return nn.LayerNorm
    elif norm_type == 'GN':
        return lambda num_features: nn.GroupNorm(num_groups, num_features)
    else:
        raise NotImplementedError(f"Normalization type {norm_type} not implemented.")


def tensorflow_style_avg_pooling(x, kernel_size, stride, padding='same'):
    """TensorFlow-style average pooling in PyTorch."""
    if padding == 'same':
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    return F.avg_pool2d(x, kernel_size, stride=stride, padding=0)


def upsample(x, factor=2):
    """Upsamples tensor by a given factor using nearest neighbor interpolation."""
    return F.interpolate(x, scale_factor=factor, mode='nearest')


def dsample(x):
    """Downsamples tensor using average pooling."""
    return tensorflow_style_avg_pooling(x, (2, 2), stride=(2, 2), padding='same')
