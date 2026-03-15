# MedLat: First Stage Models Overview

A comprehensive collection of quantizers, continuous autoencoders, and tokenizers for medical imaging and general visual data processing. This framework provides a unified interface for first-stage models used in generative modeling pipelines.

## 📚 Table of Contents

- [Discrete Quantizers](#discrete-quantizers)
- [Continuous Autoencoders](#continuous-autoencoders)
- [Tokenizers](#tokenizers)
- [Model Registry](#model-registry)

---

## 🔢 Discrete Quantizers

Discrete quantizers map continuous latent representations to discrete codebook entries. These are essential components for VQ-VAE, VQ-GAN, and related architectures.

### Core Quantizers

#### **VectorQuantizer**
- **Location:** `discrete/quantizer/quantize.py`
- **Paper:** [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **Description:** Standard VQ-VAE/VQ-GAN quantizer with optional EMA updates and rotation trick support
- **Registry:** `discrete.quantizer.vector_quantizer`
- **Features:**
  - Standard nearest-neighbor quantization
  - Optional EMA updates for embeddings
  - Rotation trick support (see [paper](https://arxiv.org/abs/2410.06424))
  - Commitment loss and codebook loss

#### **VectorQuantizer2**
- **Location:** `discrete/quantizer/quantize.py`
- **Paper:** [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **Description:** Optimized version of VectorQuantizer with improved efficiency and additional features
- **Registry:** `discrete.quantizer.vector_quantizer2`
- **Features:**
  - More efficient distance computations
  - Optional normalization (cosine similarity)
  - EMA updates support
  - Rotation trick support
  - Legacy mode compatibility

#### **GumbelQuantize**
- **Location:** `discrete/quantizer/quantize.py`
- **Paper:** [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
- **Description:** Gumbel-Softmax based quantization for differentiable discrete latent variables
- **Registry:** `discrete.quantizer.gumbel_quantizer`
- **Features:**
  - Differentiable quantization via Gumbel-Softmax
  - Straight-through estimator option
  - KL divergence regularization
  - Temperature annealing support

### Advanced Quantizers

#### **QINCoVectorQuantizer2**
- **Location:** `discrete/quantizer/quantize.py`
- **Paper:** [QINCo: A Foundation Model for Quantization](https://arxiv.org/abs/2401.14732)
- **Code:** [QINCo GitHub](https://github.com/facebookresearch/Qinco)
- **Description:** Vector quantizer with implicit embeddings using MLP instead of lookup tables
- **Registry:** `discrete.quantizer.qinco_vector_quantizer`
- **Features:**
  - Implicit codebook via MLP
  - Reduced memory footprint
  - Configurable hidden dimensions and layers

#### **SimVQ**
- **Location:** `discrete/quantizer/quantize.py`
- **Description:** Simple VQ module using a frozen/implicit codebook with optional linear projection
- **Features:**
  - Frozen codebook buffer
  - Optional codebook transformation
  - Compatible with ResidualQuantizer wrappers
  - Rotation trick support

### Residual Quantizers

#### **ResidualQuantizer**
- **Location:** `discrete/quantizer/quantize.py`
- **Paper:** [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)
- **Description:** Wrapper for residual quantization, enabling hierarchical quantization
- **Registry:** `discrete.quantizer.residual_quantizer`
- **Features:**
  - Multi-level quantization
  - Shared or independent codebooks
  - Quantize dropout support (EnCodec style)
  - Configurable number of quantizer levels

#### **GroupedResidualVQ**
- **Location:** `discrete/quantizer/quantize.py`
- **Description:** Grouped residual vector quantization for improved efficiency
- **Registry:** `discrete.quantizer.grouped_residual_quantizer`
- **Features:**
  - Grouped quantization structure
  - Efficient multi-scale representation

#### **MultiScaleResidualQuantizer**
- **Location:** `discrete/quantizer/quantize.py`
- **Paper:** [VAR: Visual Autoregressive Models](https://arxiv.org/pdf/2404.02905)
- **Code:** [VAR GitHub](https://github.com/FoundationVision/VAR)
- **Description:** Multi-scale residual quantizer as used in VAR for hierarchical visual representation
- **Registry:** `discrete.quantizer.msrq_vector_quantizer2`
- **Features:**
  - Multi-scale patch-based quantization
  - VAR-style scaling features
  - Optional quantization residuals
  - Z-normalization support
  - EMA updates support

### Lookup-Free Quantization

#### **LookupFreeQuantizer**
- **Location:** `discrete/quantizer/quantize.py`
- **Paper:** [Language Model Beats Diffusion — Tokenizer is Key to Visual Generation](https://arxiv.org/html/2310.05737v3)
- **Description:** Lookup-free quantization that doesn't require explicit codebook storage
- **Registry:** `discrete.quantizer.lookup_free_quantizer`
- **Features:**
  - Bit-based token representation
  - No explicit codebook storage
  - Entropy loss regularization
  - Support for 2D and 3D inputs

---

## 🌊 Continuous Autoencoders

Continuous autoencoders use variational inference to learn continuous latent representations with KL divergence regularization.

### AutoencoderKL Models

#### **AutoencoderKL**
- **Location:** `continuous/vae_models.py`
- **Paper:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **Description:** Base KL-VAE autoencoder with configurable compression factors
- **Features:**
  - Diagonal Gaussian posterior
  - Sliding window inference support (for 3D)
  - Configurable encoder/decoder architectures

#### **AutoencoderKL_f4**
- **Registry:** `continuous.autoencoder.kl-f4d3`
- **Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- **Code:** [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)
- **Description:** KL-VAE with compression factor 4 (f4)
- **Configuration:**
  - z_channels: 3
  - ch_mult: [1, 2, 4]
  - Compression: 4× downsampling

#### **AutoencoderKL_f8**
- **Registry:** `continuous.autoencoder.kl-f8d4`
- **Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- **Code:** [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)
- **Description:** KL-VAE with compression factor 8 (f8)
- **Configuration:**
  - z_channels: 4
  - ch_mult: [1, 2, 4, 4]
  - Compression: 8× downsampling

#### **AutoencoderKL_f16**
- **Registry:** `continuous.autoencoder.kl-f16d8`
- **Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- **Code:** [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)
- **Description:** KL-VAE with compression factor 16 (f16)
- **Configuration:**
  - z_channels: 8
  - ch_mult: [1, 1, 2, 2, 4]
  - Attention at resolution 16
  - Compression: 16× downsampling

#### **AutoencoderKL_f32**
- **Registry:** `continuous.autoencoder.kl-f32d64`
- **Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- **Code:** [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)
- **Description:** KL-VAE with compression factor 32 (f32)
- **Configuration:**
  - z_channels: 64
  - ch_mult: [1, 1, 2, 2, 4, 4]
  - Attention at resolutions [16, 8]
  - Compression: 32× downsampling

---

## 🎯 Tokenizers

Tokenizers learn discrete token representations optimized for downstream generative modeling tasks.

### DeTok (Denoising Tokenizer)

**Location:** `token/detok/detok.py`

DeTok implements latent denoising for visual tokenization with various size configurations.

#### Available Variants:
- **detok_SS** - Registry: `token.detok.ss` - Small encoder (512) / Small decoder (512)
- **detok_SB** - Registry: `token.detok.sb` - Small encoder (512) / Base decoder (768)
- **detok_SL** - Registry: `token.detok.sl` - Small encoder (512) / Large decoder (1024)
- **detok_BS** - Registry: `token.detok.bs` - Base encoder (768) / Small decoder (512)
- **detok_BB** - Registry: `token.detok.bb` - Base encoder (768) / Base decoder (768)
- **detok_BL** - Registry: `token.detok.bl` - Base encoder (768) / Large decoder (1024)
- **detok_LS** - Registry: `token.detok.ls` - Large encoder (1024) / Small decoder (512)
- **detok_LB** - Registry: `token.detok.lb` - Large encoder (1024) / Base decoder (768)
- **detok_LL** - Registry: `token.detok.ll` - Large encoder (1024) / Large decoder (1024)
- **detok_XLXL** - Registry: `token.detok.xlxl` - Extra Large encoder (1152) / Extra Large decoder (1152)

**Features:**
- SwiGLU feed-forward networks
- Multi-head attention
- Supports 2D and 3D inputs
- Configurable encoder/decoder widths and depths

### TiTok (Tokenized Image Tokenizer)

**Location:** `token/titok/titok.py`

TiTok provides learned, low-dimensional tokenization with sparse codebook usage.

#### Available Variants:
- **TiTok_S_128** - Registry: `token.titok.s_128`
  - Hidden size: 512, Depth: 8, Heads: 8, Latent tokens: 32
- **TiTok_B_64** - Registry: `token.titok.b_64`
  - Hidden size: 768, Depth: 12, Heads: 12, Latent tokens: 64
- **TiTok_L_32** - Registry: `token.titok.l_32`
  - Hidden size: 1024, Depth: 24, Heads: 16, Latent tokens: 32

**Features:**
- Sparse token representation
- Configurable token size and codebook size
- Support for VQ and VAE quantization modes
- Compatible with pretrained VQGAN tokenizers
- Two-stage training support

### MedLat

**Location:** `medlat/first_stage/token/`

**Registry:** `token.medlat.standard`

Medical imaging optimized tokenizer based on DeTok architecture.

**Features:**
- Optimized for 3D medical data
- Supports arbitrary spatial dimensions
- SwiGLU activation
- Multi-head attention
- Configurable architecture

### DinoTok

**Location:** `token/dinotok/dinotok.py`

**Registry:** `token.dinotok.base`

Tokenization with DINO-style self-supervised learning integration.

**Features:**
- DINO-inspired architecture
- Self-supervised learning support
- SwiGLU feed-forward
- Multi-head attention
- Flexible encoder/decoder configuration

### MAETok (Masked Autoencoder Tokenizer)

**Location:** `token/maetok/maetok.py`

**Registry:** `token.maetok.b_128`

Masked autoencoder-based tokenization with base configuration.

**Features:**
- Masked autoencoding approach
- Base model with 128 latent tokens
- ViT-based encoder/decoder
- HOG feature integration
- Pixel reconstruction support

### ReconMAE (Vita)

**Location:** `token/vita/vita.py`

**Registry:** `token.vita.reconmae`

Reconstruction-based masked autoencoder for medical imaging.

**Features:**
- Masked autoencoding for 3D medical images
- Patch-based encoding
- Configurable mask types and ratios
- Support for various decoder architectures (ViT, Linear, UNETR)
- Contrastive and reconstruction losses

---

## 🔧 Model Registry

All models are registered in a central registry system for easy instantiation.

### Usage

```python
from src import get_model, available_models

# List all available models
print(available_models())

# List only quantizers
print(available_models("discrete.quantizer."))

# List only tokenizers
print(available_models("token."))

# List only continuous autoencoders
print(available_models("continuous."))

# Instantiate a model
quantizer = get_model("discrete.quantizer.vector_quantizer2", 
                      n_e=8192, e_dim=256, beta=0.25)
autoencoder = get_model("continuous.autoencoder.kl-f8", 
                        img_size=256, dims=2)
tokenizer = get_model("token.detok.bb", image_size=256)
```

### Registering New Models

To register a new model, use the `@register_model` decorator:

```python
from src.registry import register_model

@register_model("my.prefix.model_name",
                paper_url="https://arxiv.org/abs/xxxx.xxxxx",
                code_url="https://github.com/...",
                description="Model description")
class MyModel(nn.Module):
    def __init__(self, ...):
        ...
```

---

## 📖 Additional Resources

- **Vector Quantize PyTorch:** For additional quantizer implementations, see [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- **Model Configuration:** Models can be configured using OmegaConf when training from source
- **3D Support:** Most models support both 2D and 3D inputs via the `dims` parameter

---

## 🎓 Key Papers

- **VQ-VAE:** [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- **VQ-GAN:** [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
- **KL-VAE:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- **Latent Diffusion:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- **VAR:** [Visual Autoregressive Models](https://arxiv.org/pdf/2404.02905)
- **QINCo:** [QINCo: A Foundation Model for Quantization](https://arxiv.org/abs/2401.14732)
- **Lookup-Free Quantization:** [Language Model Beats Diffusion — Tokenizer is Key to Visual Generation](https://arxiv.org/html/2310.05737v3)

---

*This framework is designed for medical imaging applications but is applicable to general visual data processing tasks.*
