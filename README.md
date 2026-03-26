# MedLat

**MedLat** (`medlat`) is a PyTorch library that makes medical and general-purpose image generation research feel less like archaeology and more like engineering. It ships a single **model registry** spanning tokenizers, autoencoders, and generators — hundreds of concrete configurations, one API.

```python
from medlat import get_model, available_models, GenWrapper

tokenizer = get_model("continuous.aekl.f8_d16", img_size=224)
generator = get_model("dit.xl_2", img_size=224, vae_stride=8, in_channels=16, num_classes=10)
wrapper   = GenWrapper(generator, tokenizer)
```

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
from medlat import get_model, available_models, get_model_info, GenWrapper

# ── Explore the registry ──────────────────────────────────────────────────
print(list(available_models()))            # all IDs
print(list(available_models("discrete."))) # filtered
print(get_model_info("continuous.vavae.f8_d32_dinov2"))  # paper / code links

# ── Continuous tokenizer + diffusion generator (DiT) ─────────────────────
tokenizer = get_model("continuous.aekl.f8_d16", img_size=224)
generator = get_model("dit.xl_2",
    img_size=224, vae_stride=tokenizer.vae_stride,
    in_channels=tokenizer.embed_dim, num_classes=10)
wrapper = GenWrapper(generator, tokenizer)

z      = wrapper.vae_encode(images)        # (B, C, H, W) continuous latents
sample = generator.forward_with_cfg(z, t, y=labels, cfg_scale=1.5)
out    = wrapper.vae_decode(sample)

# ── Discrete tokenizer + masked generation (MaskGIT) ─────────────────────
tokenizer = get_model("discrete.vq.f8_d4_e16384", img_size=224)
generator = get_model("maskgit.b",
    img_size=224, vae_stride=tokenizer.vae_stride,
    num_tokens=tokenizer.n_embed, num_classes=10)
wrapper = GenWrapper(generator, tokenizer)

z    = wrapper.vae_encode(images)          # (B, seq_len) discrete indices
loss = wrapper(z, y=labels)

# ── Non-square inputs ─────────────────────────────────────────────────────
tokenizer = get_model("continuous.aekl.f8_d16", img_size=(192, 256))  # H×W
generator = get_model("mar.b",
    img_size=(192, 256), vae_stride=8, in_channels=16, class_num=10)
```

---

## Compatible pipelines at a glance

`GenWrapper` routes encode/decode automatically. The three supported pipeline types are:

| First stage | Generator | Training call | Notes |
|-------------|-----------|---------------|-------|
| Continuous VAE | **DiT / MDT / UViT** (diffusion) | `diffusion.training_losses(wrapper, z, t, model_kwargs)` | `z` is `(B, C, H, W)` |
| Continuous VAE | **MAR** (masked continuous AR) | `wrapper(z, y=labels)` | `z` is `(B, C, H, W)` |
| Continuous VAE | **RAR** (recurrent continuous AR) | `wrapper.generator(z_flat, labels)` | reshape `z` → `(B, seq, C)` first |
| Discrete VAE | **MaskGIT** (masked token) | `wrapper(z, y=labels)` | `z` is `(B, seq_len)` indices |
| Discrete VAE | **MAGE** (masked ViT generator) | `wrapper(z, labels)` | vae_stride must match MAGE patch size |
| Discrete VAE | **Taming GPT** (autoregressive) | `wrapper.generator(z[:, :-1], targets=z[:, 1:])` | next-token prediction |
| Discrete VAE | **MaskBit** (BERT-style masked) | custom masked loop | uses `LFQBert` or `Bert` |

See `example_generator_*.ipynb` for complete runnable examples of each pipeline.

---

## Package layout

```
medlat/
├── registry.py                  register_model · get_model · available_models · get_model_info
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
├── diffusion/                   create_gaussian_diffusion · schedules · sampling
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
| **MedVAE** | KL AE aligned to BiomedCLIP | `continuous.medvae.f8_d16`, `continuous.medvae.f8_d32` |
| **VAVAE** | Vision-foundation-aligned KL AE | `continuous.vavae.f8_d32_dinov2`, `continuous.vavae.f16_d64_mae` |
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
from medlat import available_models, get_model_info

# Count everything
print(len(list(available_models())))       # 200+

# Subsets by prefix
continuous  = list(available_models("continuous."))
discrete    = list(available_models("discrete."))
generators  = list(available_models("dit.")) + list(available_models("mar."))

# What's behind an ID?
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
