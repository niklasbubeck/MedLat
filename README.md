# MedLat

**MedLat** (`medlat`) is a PyTorch library that makes medical and general-purpose image generation research feel less like archaeology and more like engineering. It ships a single **model registry** spanning tokenizers, autoencoders, and generators — hundreds of concrete configurations, one API.

```python
from medlat import get_model, suggest_generator_params, GenWrapper

tok = get_model("continuous.aekl.f8_d16", img_size=224)
gen = get_model("dit.xl_2", img_size=224, num_classes=10, **suggest_generator_params(tok))
wrapper = GenWrapper(gen, tok)
print(wrapper)
# GenWrapper(
#   routing      = continuous + non-autoregressive,
#   tokenizer    = AEKL,
#   generator    = DiT,
#   scale_factor = 0.1822  [auto],
# )
```

**[→ Full API documentation](https://niklasbubeck.github.io/MedLat/api/)**

---

## What lives inside

```
Images / Volumes
       │
  ┌────▼────────────────────────────────┐
  │     First Stage (tokenizer / VAE)   │  ← 100+ registered configs
  │  continuous.*  |  discrete.*        │    AEKL · VAVAE · VQ · LFQ · BSQ · …
  └────┬────────────────────────────────┘
       │  latent codes or continuous latents
  ┌────▼────────────────────────────────┐
  │          Generator                  │  ← 50+ registered configs
  │  autoregressive | non-autoregressive│    DiT · MAR · MaskGIT · MAGE · RAR · …
  └────┬────────────────────────────────┘
       │  decoded back to pixels
  ┌────▼────────────────────────────────┐
  │     GenWrapper (glue layer)         │  selects encode/decode routes automatically
  └─────────────────────────────────────┘
```

**Typical workflow:** load or train a first-stage model → attach a generator via `GenWrapper` → train or sample.
**2D or 3D:** every model accepts `dims=2` (images, slices) or `dims=3` (CT/MRI volumes).  `img_size` and `patch_size` accept either a single `int` (square/cubic) or a per-axis tuple.

---

## What's new in v0.1.4

### Zero-friction model wiring

**`suggest_generator_params(tokenizer)`** — inspects any tokenizer and returns the exact kwargs the paired generator needs. No more hunting for `in_channels` or `vae_stride`:

```python
tok = get_model("continuous.aekl.f8_d16", img_size=256)
params = suggest_generator_params(tok)
# → {'vae_stride': 8, 'in_channels': 16}
gen = get_model("dit.xl_2", img_size=256, num_classes=1000, **params)
```

Works for discrete tokenizers too — returns `codebook_size` and `num_tokens` automatically.

### Inspect before you build

**`get_model_signature(name)`** uses `inspect.signature` to show every builder parameter and its default value — always in sync with the code, zero documentation lag:

```python
from medlat import get_model_signature

get_model_signature("dit.xl_2")
# → {'img_size': '<required>', 'vae_stride': '<required>',
#    'in_channels': '<required>', 'num_classes': 10, 'learn_sigma': True, …}
```

### Actionable error messages

`GenWrapper` now validates the tokenizer–generator pair at construction and tells you exactly how to fix a mismatch:

```
ValueError: Channel mismatch: DiT.in_channels=4 but AEKL.embed_dim=16.
  → Rebuild the generator with in_channels=16, or pass **suggest_generator_params(tokenizer)
```

### Unified scheduler discovery

Three scheduler paradigms — Gaussian diffusion, flow matching, and Self-Flow — are now discoverable through the same pattern as models:

```python
from medlat import available_schedulers, scheduler_info, create_scheduler

available_schedulers()
# → ('diffusion', 'flow', 'self_flow')

info = scheduler_info("self_flow")
print(info.description)
print(info.optional_kwargs)

sched = create_scheduler("flow", path_type="Linear", prediction="velocity")
sched = create_scheduler("self_flow", masking_strategy="complexity")
```

---

## Installation

```bash
pip install -e .
```

Core deps: PyTorch, NumPy, Einops, timm, OmegaConf, MONAI. Optional `[dev]` extras (pytest, black, etc.) via `pip install -e ".[dev]"`.

### Running tests

```bash
# Fast import / registry smoke tests
pytest tests/ -v

# Full forward-pass suite (slower, CPU-friendly)
python tests/registry_integration.py
```

---

## Quick start

```python
from medlat import (get_model, available_models, get_model_info,
                    get_model_signature, suggest_generator_params,
                    GenWrapper, create_scheduler, available_schedulers)

# ── Explore the registry ──────────────────────────────────────────────────
available_models()                                    # all 200+ IDs
available_models("discrete.")                         # filtered by prefix
get_model_info("continuous.vavae.f8_d32_dinov2")      # paper / code links
get_model_signature("dit.xl_2")                       # required kwargs + defaults

# ── Continuous tokenizer + DiT — params inferred automatically ───────────
tok = get_model("continuous.aekl.f8_d16", img_size=224)
gen = get_model("dit.xl_2", img_size=224, num_classes=10,
                **suggest_generator_params(tok))       # → vae_stride=8, in_channels=16
wrapper = GenWrapper(gen, tok)
print(wrapper)                                         # routing · models · scale_factor

z      = wrapper.vae_encode(images)        # (B, C, H, W) continuous latents
sample = gen.forward_with_cfg(z, t, y=labels, cfg_scale=1.5)
out    = wrapper.vae_decode(sample)

# ── Discrete tokenizer + MaskGIT — params inferred automatically ─────────
tok = get_model("discrete.vq.f8_d4_e16384", img_size=224)
gen = get_model("maskgit.b", img_size=224, num_classes=10,
                **suggest_generator_params(tok))
wrapper = GenWrapper(gen, tok)
loss = wrapper(wrapper.vae_encode(images), y=labels)

# ── Schedulers ───────────────────────────────────────────────────────────
available_schedulers()                                 # ('diffusion', 'flow', 'self_flow')
sched = create_scheduler("flow", path_type="Linear", prediction="velocity")
sched = create_scheduler("self_flow", masking_strategy="complexity")

# ── Non-square inputs ─────────────────────────────────────────────────────
tok = get_model("continuous.aekl.f8_d16", img_size=(192, 256))
gen = get_model("mar.b", img_size=(192, 256), class_num=10,
                **suggest_generator_params(tok))
```

---

## Attention mechanisms & PyTorch backends

Every attention module in MedLat uses [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA), available since PyTorch 2.0. PyTorch automatically selects the most efficient kernel at runtime:

| Backend | When selected | Notes |
|---------|--------------|-------|
| **FlashAttention** | CUDA, supported GPU, no custom float mask | Fastest; fused kernel, O(N) memory |
| **Memory-efficient attention** | CUDA, arbitrary masks | Slower than flash but handles masks well |
| **Math (unfused)** | CPU or unsupported GPU | Numerically identical reference path |

### Default behaviour

No configuration needed. On a CUDA-capable GPU MedLat will automatically use FlashAttention (where supported) or memory-efficient attention. On CPU it falls back to the standard unfused path.

```python
import torch
from medlat import get_model

model = get_model("dit.xl_2", img_size=256, vae_stride=8, in_channels=16).cuda()
# PyTorch picks the best kernel automatically — nothing else required.
```

### Explicitly choosing a backend

Use PyTorch's context managers to pin a specific kernel, e.g. for profiling, debugging or benchmarking:

```python
import torch

# PyTorch ≥ 2.3 — preferred API
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    loss = model(x, t, y)

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    loss = model(x, t, y)

with sdpa_kernel(SDPBackend.MATH):
    loss = model(x, t, y)

# PyTorch 2.0 / 2.1 — lower-level flag API
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    loss = model(x, t, y)
```

### Disabling FlashAttention globally

If your GPU does not support FlashAttention or you want to force the memory-efficient path:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
    output = model(x, t, y)
```

### Verifying which kernel is active

```python
import torch

# Print which backends PyTorch considers available on the current device
print(torch.backends.cuda.flash_sdp_enabled())           # True if FlashAttention eligible
print(torch.backends.cuda.mem_efficient_sdp_enabled())   # True if mem-efficient eligible
print(torch.backends.cuda.math_sdp_enabled())            # always True
```

### Notes on specific model families

- **MDT** — passes the relative position bias as the `attn_mask` argument; PyTorch routes this through the memory-efficient or math backend (FlashAttention does not support float additive masks).
- **Taming GPT** — uses `is_causal=True` during training and `is_causal=False` during cached inference; FlashAttention handles both paths natively.
- **VMAE** — when `return_attn_map=True` is requested (e.g. for visualisation), falls back to the manual `q @ k.T` path so that the full attention matrix is available. SDPA is used for all normal training/inference calls.
- **UViT** — detects at import time whether SDPA is available and sets `ATTENTION_MODE` accordingly; falls back to xformers or math if not.

---

## Latent alignment

Training a tokenizer that only minimizes reconstruction loss gives you a compact latent space — but one that is *geometrically opaque*. Nearby points carry no semantic guarantee, which makes downstream generation harder. MedLat borrows the **Vision-Foundation (VF) alignment** technique from VA-VAE and extends it with medical-vision support via BiomedCLIP:

```
Image ──► Encoder ──► z ──► Decoder ──► reconstruction loss
                       │
                       ▼   (during training only)
              AlignmentModule
          ┌─────────────────────────────────────────┐
          │  Frozen foundation model                │
          │  (DINOv2 / MAE / BiomedCLIP)            │
          │            │                            │
          │            ▼  target features           │
          │  Projection head on z                   │
          │            │                            │
          │            ▼  predicted features        │
          │  VF loss = distmat_loss + cosine_loss   │
          └─────────────────────────────────────────┘
                       │
                       ▼
          total loss = recon + KL + VF
```

The result: latent codes that are **semantically structured** — nearby points in latent space correspond to semantically similar images, which substantially improves generation quality with any downstream generator.

### Built-in aligned tokenizers

| ID | Foundation model | Best for |
|----|-----------------|---------|
| `continuous.vavae.f8_d16_dinov2` | DINOv2-L (frozen) | Natural images, general vision |
| `continuous.vavae.f8_d32_dinov2` | DINOv2-L (frozen) | Same, higher channel width |
| `continuous.vavae.f16_d16_mae` | MAE-L (frozen) | Self-supervised vision |
| `continuous.vavae.f16_d32_mae` | MAE-L (frozen) | Same, higher channel width |
| `continuous.vavae.f16_d64_mae` | MAE-L (frozen) | Same, max channel width |
| `continuous.vavae.f16_d16_dinov2` | DINOv2-L (frozen) | |
| `continuous.vavae.f16_d32_dinov2` | DINOv2-L (frozen) | |
| `continuous.vavae.f16_d64_dinov2` | DINOv2-L (frozen) | |
| `continuous.medvae.f8_d16` | **BiomedCLIP** (frozen) | **Medical images** (radiology, pathology) |
| `continuous.medvae.f8_d32` | **BiomedCLIP** (frozen) | Same, higher channel width |

### Adding alignment to any tokenizer

The `alignment` argument is available on every `AutoencoderKL` — you are not limited to the pre-registered IDs:

```python
from medlat import get_model
from medlat.modules.alignments import VFFoundationAlignment

# Standard AEKL — reconstruction + KL only
tokenizer = get_model("continuous.aekl.f8_d16", img_size=256)

# Same architecture, trained with DINOv2 semantic alignment
alignment = VFFoundationAlignment(latent_channels=16, foundation_type="dinov2")
tokenizer_aligned = get_model("continuous.aekl.f8_d16", img_size=256, alignment=alignment)

# For medical images, align to BiomedCLIP instead
alignment_med = VFFoundationAlignment(latent_channels=16, foundation_type="biomedclip")
tokenizer_medical = get_model("continuous.aekl.f8_d16", img_size=256, alignment=alignment_med)
```

The foundation model is **entirely frozen** during training. Alignment only adds a learnable projection head and a VF loss term — no extra parameters in the encoder or decoder.

### The VF loss

`VFFoundationAlignment` computes two complementary objectives:

1. **`vf_loss_1` (structure preservation):** The pairwise cosine-similarity matrix of projected latents should mirror that of the frozen features (with a `distmat_margin` slack).
2. **`vf_loss_2` (directional alignment):** Each individual spatial location should be directionally consistent with its counterpart in foundation space (with a `cos_margin` slack).

Both margins and weights are configurable:

```python
VFFoundationAlignment(
    latent_channels=16,
    foundation_type="dinov2",  # "mae" | "dinov2" | "biomedclip"
    distmat_margin=0.25,
    cos_margin=0.5,
    distmat_weight=1.0,
    cos_weight=1.0,
)
```

### Multi-modal tip

When training on heterogeneous modalities (e.g. knee MRI + brain MRI), aligning both to the **same** foundation model (BiomedCLIP for medical data) gives the generator a consistent semantic coordinate system regardless of which modality is being encoded. The foundation model handles cross-modality semantic normalisation implicitly — the generator only sees a well-structured shared latent space.

---

## Package layout

```
medlat/
├── registry.py                  register_model · get_model · available_models · get_model_info · get_model_signature
├── utils.py                     init_from_ckpt · validate_compatibility · suggest_generator_params
├── first_stage/
│   ├── continuous/              AEKL · MAISI · MedVAE · VAVAE · DCAE · SoftVQ/WQVAE
│   ├── discrete/                VQ · RQ · FSQ · LFQ · BSQ · SimVQ · QINCo family · HCVQ · MaskGIT-VQ
│   │   └── quantizer/           standalone quantizer modules  (discrete.quantizer.*)
│   └── token/                   TiTok · MAETok · VMAE · DeTok · SoftVQ · ViTA
├── generators/
│   ├── autoregressive/
│   │   ├── maskgit/             MaskGIT  (masked token generation)
│   │   ├── mage/                MAGE     (masked ViT generator)
│   │   ├── taming/              Taming Transformer GPT
│   │   ├── maskbit/             MaskBit  (LFQBert / Bert)
│   │   ├── mar/                 MAR      (continuous masked AR + diffusion loss)
│   │   ├── rar/                 RAR      (recurrent continuous AR + diffusion loss)
│   │   └── fractal/             FractalGen (hierarchical AR)
│   └── non_autoregressive/
│       ├── dit/                 DiT  (all scales × patch sizes)
│       ├── mdt/                 MDT  (Masked Diffusion Transformer)
│       ├── uvit/                UViT (U-Net + ViT hybrid diffusion)
│       ├── ldm/                 LDM  (UNet latent diffusion)
│       └── adm/                 ADM  (Dhariwal–Nichol UNet + classifiers)
├── scheduling/                  create_scheduler · available_schedulers · scheduler_info
│                                DualTimestepScheduler (Self-Flow) · FlowMatchingScheduler · GaussianDiffusionScheduler
├── diffusion/                   create_gaussian_diffusion · schedules · sampling  (legacy)
└── modules/
    ├── wrapper.py               GenWrapper  (encode/decode glue for any combination)
    ├── pos_embed.py             to_ntuple · sincos & learned positional embeddings
    └── in_and_out.py            PatchEmbed · ToPixel  (dims-aware)
```

---

## Naming conventions

Registry IDs follow consistent patterns:

| Token | Meaning |
|-------|---------|
| `f{N}` | Spatial downsampling factor — `f8` = 8× compression per axis |
| `d{N}` | Latent channel width or embedding dimension |
| `e{N}` | Codebook size for vector quantization |
| `b{N}` | Bit width (LFQ, BSQ) |
| `l{N}` | Levels (FSQ) |
| `s/b/l/xl/h` | Scale / depth tag (small → huge) |
| `_2/_4/_8` | Patch size suffix in generator names (DiT, MAGE) |
| `_dinov2/_mae/_biomedclip` | Foundation model alignment variant |

```python
# Examples decoded:
"continuous.aekl.f8_d16"        # AE-KL, 8× compression, 16 latent channels
"discrete.lfq.f16_d14_b14"      # LFQ, 16× compression, 14-dim, 14-bit codebook
"dit.xl_2"                      # DiT-XL with patch size 2
"mage.b_8"                      # MAGE-Base, vae_stride must be 8
"mar.h"                         # MAR-Huge
```

---

## Model families

### First stage — Tokenizers & patch sequences (`token.*`)

| Family | What it does | Example IDs |
|--------|-------------|-------------|
| **TiTok** | Compact 1-D token sequences for generation | `token.titok.s_128`, `token.titok.b_256_p8_e2e` |
| **MAETok** | MAE-style reconstruction tokenizer | `token.maetok.s_256`, `token.maetok.b_512_p8` |
| **VMAE** | ViT/VideoMAE-style encoder tokenizer | `token.vmae.s_p8_d16`, `token.vmae.b_p16_d32` |
| **DeTok** | Scale grid (ss / sb / bb / … / xlxl) | `token.detok.ss`, `token.detok.xlxl` |
| **SoftVQ** | Differentiable soft VQ tokenizer | `token.softvq.s_t32_d32`, `token.softvq.bl_t64_d32` |
| **ViTA** | ViT-based reconstruction AE | `token.vita.reconmae` |

---

### First stage — Continuous autoencoders (`continuous.*`)

| Family | What it does | Example IDs |
|--------|-------------|-------------|
| **AEKL** | LDM-style KL autoencoder, conv encoder/decoder | `continuous.aekl.f4_d3` … `continuous.aekl.f32_d64` |
| **MAISI** | MONAI MAISI 3D-friendly KL AE | `continuous.maisi.f4_d3` |
| **MedVAE** | KL AE + **BiomedCLIP VF alignment** — semantically structured latents for medical images | `continuous.medvae.f8_d16`, `continuous.medvae.f8_d32` |
| **VAVAE** | KL AE + **vision-foundation VF alignment** (DINOv2 or MAE) — same idea as VA-VAE paper | `continuous.vavae.f8_d32_dinov2`, `continuous.vavae.f16_d64_mae` |
| **DCAE** | EfficientViT DC-AE (high compression ratio) | `continuous.dcae.f32c32`, `continuous.dcae.f128c512` |
| **SoftVQ / WQVAE** | Soft or warped quantization, continuous wrapper | `continuous.soft_vq.f8_d16_e16384_dinov2`, `continuous.wqvae.f8_d4_e16384` |

---

### First stage — Discrete VAEs (`discrete.*`)

| Family | What it does | Example IDs |
|--------|-------------|-------------|
| **VQ-VAE** | VQGAN-style conv VQ | `discrete.vq.f4_d3_e8192` … `discrete.vq.f16_d64_e16384` |
| **LFQ** | Lookup-free quantization (implicit codebook) | `discrete.lfq.f4_d10_b10` … `discrete.lfq.f16_d18_b18` |
| **BSQ** | Binary spherical quantization | `discrete.bsq.f4_d10_b10` … `discrete.bsq.f16_d18_b18` |
| **FSQ** | Finite scalar quantization | `discrete.fsq.f4_d3_l8192`, `discrete.fsq.f16_d8_l16384` |
| **SimVQ** | Simplified VQ with codebook collapse prevention | `discrete.simvq.f4_d3_e8192` … `discrete.simvq.f16_d8_e16384` |
| **RQVAE** | Residual quantizer VAE (multi-level codes) | `discrete.rqvae.f4_d3_e8192` … `discrete.rqvae.f16_d8_e16384` |
| **QINCo family** | Improved nearest-code quantizers | `discrete.simple_qinco.*`, `discrete.qinco.*`, `discrete.rsimple_qinco.*` |
| **HCVQ** | Hybrid conv/ViT quantizer presets | `discrete.hcvq.residual_vq.S_16`, `discrete.hcvq.sd_vq.S_16` |
| **MaskGIT-VQ** | VQ preset for MaskGIT-style pipelines | `discrete.maskgit.vq.f16_d256_e1024` |
| **MS-RQ** | Multi-scale residual quantization | `discrete.msrq.f16_d32_e4096` |

Standalone quantizer modules (for custom VQ composition):
`discrete.quantizer.vector_quantizer`, `discrete.quantizer.lookup_free_quantizer`, `discrete.quantizer.finite_scalar_quantizer`, `discrete.quantizer.residual_quantizer`, `discrete.quantizer.binary_spherical_quantizer`, `discrete.quantizer.soft_vector_quantizer`, …

---

### Generators — Autoregressive

#### Discrete AR (pair with discrete tokenizers)

| Model | What it does | IDs | Interface |
|-------|-------------|-----|-----------|
| **MaskGIT** | Iterative masked token generation (BERT + cosine schedule) | `maskgit.b`, `maskgit.l`, `maskgit.h` | `wrapper(z, y=labels)` → loss |
| **MAGE** | Masked generative encoder-decoder ViT | `mage.xs_4` … `mage.l_16` | `wrapper(z, labels)` → `(loss, …)`; suffix = vae_stride |
| **Taming GPT** | Autoregressive next-token prediction (GPT) | `taming.gpt_b`, `taming.gpt_l`, `taming.gpt_h` | `generator(z[:-1], targets=z[1:])` → `(logits, loss)` |
| **MaskBit** | BERT-style masked generation for VQ (`Bert`) or LFQ (`LFQBert`) | `maskbit.s/b/l`, `maskbit.bert_s/b/l` | custom masked training loop |

> ⚠️ **MAGE constraint:** the patch size suffix in the model name must match the tokenizer's `vae_stride` (e.g. `mage.b_8` only works with `f8` tokenizers).

#### Continuous AR (pair with continuous tokenizers)

| Model | What it does | IDs | Interface |
|-------|-------------|-----|-----------|
| **MAR** | Masked autoregressive with diffusion loss (continuous tokens) | `mar.b`, `mar.l`, `mar.h` | `wrapper(z, y=labels)` → loss; `z` is `(B, C, H, W)` |
| **RAR** | Recurrent autoregressive with diffusion loss | `rar.b`, `rar.l`, `rar.xl`, `rar.h` | `generator(z_flat, labels)` → loss; `z_flat` is `(B, H×W, C)` |
| **FractalGen** | Hierarchical fractal AR (multi-level MAR/AR cascade) | `fractal.ar_64`, `fractal.mar_64`, `fractal.mar_base_256`, … | custom hierarchical loop |

---

### Generators — Non-autoregressive (diffusion)

| Model | What it does | IDs | Notes |
|-------|-------------|-----|-------|
| **DiT** | Diffusion Transformer — patchified latents, adaLN conditioning | `dit.s_1` … `dit.xl_8` (scale × patch) | 16 configs; `vae_stride` + `in_channels` required |
| **MDT** | Masked Diffusion Transformer — masked encoder decoder | `mdt.s_2` … `mdt.xl_4` (scale × patch) | 8 configs |
| **UViT** | U-Net ViT hybrid diffusion | `uvit.small`, `uvit.small_deep`, `uvit.mid`, `uvit.large`, `uvit.huge` | 5 configs |
| **LDM** | Latent Diffusion UNet (various strides) | `ldm.f1` … `ldm.f16` | classic DDPM UNet |
| **ADM** | Dhariwal–Nichol UNet + class-conditional classifiers | `adm.diffusion.{64,128,256,512}{C,U}`, `adm.classifier.*` | resolution-specific |

All diffusion generators integrate with `medlat.diffusion.create_gaussian_diffusion`.

---

## Example notebooks

| Notebook | What it tests | Combinations |
|----------|---------------|--------------|
| `example_tokenizer.ipynb` | First-stage training and reconstruction | Any tokenizer |
| `example_generator_nonautoregressive.ipynb` | Full combinatorial test + DiT/MDT/UViT training | 22 continuous tokenizers × 29 diffusion generators |
| `example_generator_maskgit.ipynb` | Combinatorial test + discrete AR training | 21 discrete tokenizers × 26 discrete AR generators |
| `example_generator_mar.ipynb` | Combinatorial test + MAR/RAR training | 22 continuous tokenizers × 7 continuous AR generators |

Each notebook has:
1. A **combinatorial interface test** — tries every tokenizer × generator pair with synthetic data and prints `PASS / FAIL` with a clear error for failures.
2. A **deep-dive training cell** — pick any `TOK_NAME + GEN_NAME` from the passing combinations and run a full training loop.

---

## Discovering models

```python
from medlat import available_models, get_model_info, get_model_signature

# Count everything
len(list(available_models()))              # 200+

# Subsets by prefix
available_models("continuous.")
available_models("discrete.")
available_models("dit.") 

# Inspect required kwargs before building — zero docs hunting
get_model_signature("dit.xl_2")
# → {'img_size': '<required>', 'vae_stride': '<required>',
#    'in_channels': '<required>', 'num_classes': 10, …}

# Metadata: paper link, code link, description
info = get_model_info("continuous.vavae.f8_d32_dinov2")
print(info.description, info.paper_url, info.code_url)
```

---

## The `to_ntuple` convention

Every model accepts either a single `int` or a per-axis tuple for spatial parameters:

```python
# These are all equivalent for 2D square inputs:
get_model("mar.b", img_size=224, vae_stride=8)
get_model("mar.b", img_size=(224, 224), vae_stride=(8, 8))

# Non-square inputs:
get_model("dit.xl_2", img_size=(192, 256), vae_stride=8, in_channels=16)

# 3D volumetric:
get_model("continuous.aekl.f8_d16", img_size=(64, 128, 128), dims=3)
```

`to_ntuple(value, dims)` is exported from `medlat.modules.pos_embed` for use in custom code.

---

## Citation

```bibtex
@software{bubeck_medlat_2025,
  author  = {Bubeck, Niklas},
  title   = {{MedLat}: {PyTorch} library for first-stage models and latent generators},
  url     = {https://github.com/niklasbubeck/MedLat},
  version = {0.1.0},
  year    = {2025},
}
```

---

## License

MIT — see `pyproject.toml`.
