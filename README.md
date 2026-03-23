# MedLat

**MedLat** (`medlat`) is a PyTorch library for **first-stage models** (tokenizers, continuous and discrete autoencoders, quantizers) and **second-stage generators** (autoregressive and diffusion-based), aimed at medical imaging and general visual data. Everything is exposed through a single **model registry**: instantiate by string ID, list IDs programmatically, and optionally read paper/code links via `get_model_info`.

---

## What this repo contains

| Layer | Role |
|--------|------|
| **First stage** | Maps images to a compact representation: continuous latents, discrete codes, or token sequences. |
| **Generator** | Models the distribution in that space (tokens or latents): MaskGIT, MAR, DiT, LDM-style UNet, or ADM. |
| **Utilities** | Gaussian diffusion construction (`medlat.diffusion`), training helpers, ViT cores, alignments to foundation models, and `GenWrapper` for encoder ↔ generator wiring. |

Typical flow: **train or load a first-stage model** → **attach a generator** whose input size matches the tokenizer/VAE output (`n_embed`, `embed_dim`, `seq_len`, stride, etc.) → **sample or train end-to-end** using the example notebooks.

---

## Installation

```bash
pip install -e .
```

**Core dependencies** (see `pyproject.toml`): PyTorch, NumPy, Einops, timm, OmegaConf, MONAI. Optional `[dev]` extras include pytest, black, flake8, mypy.

Some code paths import extra packages (e.g. wavelets in quantizers); install those if you hit import errors for a specific model.

---

## Quick start

```python
from medlat import available_models, get_model, GenWrapper

# Discover registered IDs (hundreds of concrete configs)
print(list(available_models()))

# Metadata only (no instantiation): description, paper_url, code_url when set
from medlat import get_model_info
print(get_model_info("continuous.aekl.f8_d16"))

# First stage: tokenizer or VAE
tokenizer = get_model("token.titok.s_128", img_size=128, dims=2)
# Other examples: continuous.aekl.f8_d16, discrete.vq.f8_d8_e16384, token.vmae.s_p8_d16, ...

# Generator: match tokenizer latent interface (example for discrete + autoregressive)
generator = get_model(
    "maskgit.b",
    img_size=128,
    seq_len=tokenizer.num_latent_tokens,
    num_tokens=tokenizer.n_embed,
    in_channels=tokenizer.embed_dim,
)

# Glue: encoding/decoding and optional latent scale handling
wrapper = GenWrapper(generator, tokenizer)
```

`GenWrapper` selects encode/decode routes based on **discrete vs continuous** first stage and **autoregressive vs non-autoregressive** generator; see `medlat/modules/wrapper.py` for details.

---

## Package layout (`medlat/`)

| Path | Contents |
|------|----------|
| `medlat/registry.py` | `ModelRegistry`, `register_model`, `get_model`, `available_models`, `get_model_info`. |
| `medlat/first_stage/continuous/` | KL autoencoders, MAISI, MedVAE, VAVAE, DCAE, soft-VQ / WQ-VAE style builders, plus one ViT AE entry used in HCVQ workflows. |
| `medlat/first_stage/discrete/` | Full VQ / RQ / FSQ / LFQ / BSQ / QINCo-family models and MaskGIT-style VQ presets. |
| `medlat/first_stage/discrete/quantizer/` | **Standalone quantizer modules** also registered under `discrete.quantizer.*`. |
| `medlat/first_stage/token/` | TiTok, MAETok, VMAE, DeTok, SoftVQ, ViTA. |
| `medlat/generators/autoregressive/` | MaskGIT, MAR. |
| `medlat/generators/non_autoregressive/` | DiT, LDM UNet, ADM (UNet + classifiers). |
| `medlat/diffusion/` | `create_gaussian_diffusion`, schedules, resampling (used with diffusion generators). |
| `medlat/modules/` | `GenWrapper`, ViT cores, positional embeddings, alignments, I/O helpers. |

Subpackages often include a short `README.md` (e.g. `generators/.../dit/README.md`) with extra context for that family.

---

## Naming conventions (registry IDs)

These patterns appear throughout IDs; exact letters vary by family.

- **`f{N}`** — Spatial downsampling factor of the bottleneck (e.g. `f8` ≈ 1/8 resolution per spatial axis).
- **`d{N}`** — Latent channel width or embedding dimension (family-specific).
- **`e{N}`** — Codebook size for vector quantization.
- **`l{N}` / `b{N}`** — Levels or bit width for scalar / binary quantizers (FSQ, LFQ, BSQ).
- **Scale tags** — e.g. `s` / `b` / `l` / `xl` for model depth (TiTok, DeTok, DiT, MaskGIT, MAR).

Use `available_models("token.")`, `available_models("continuous.")`, etc. to list subsets.

---

## Model families (registry)

Below, **family** means a group of related architectures; each family has many **registered configs** (different factors, widths, and presets). The table lists representative IDs and where they live in code.

### A. Tokenizers & patch sequence models (`token.*`)

| Family | Idea | Representative IDs |
|--------|------|----------------------|
| **TiTok** | Compact tokenization for generative modeling | `token.titok.s_128`, `token.titok.b_128_p8_e2e`, `token.titok.l_256_e2e`, … |
| **MAETok** | MAE-style tokenizer variants | `token.maetok.s_256`, `token.maetok.b_512_p8`, `token.maetok.l_128`, … |
| **VMAE** | ViT/VMAE-style tokenizers, symmetric and asymmetric | `token.vmae.s_p8_d16`, `token.vmae.t_p8_d16`, `token.vmae.b_p16_d32`, … |
| **DeTok** | Depth/width scale grid (ss, sb, …, xlxl) | `token.detok.ss`, `token.detok.bb`, `token.detok.xlxl`, … |
| **SoftVQ** | Soft vector quantization tokenizers | `token.softvq.s_t32_d32`, `token.softvq.bl_t64_d32`, … |
| **ViTA** | ViT-based tokenizer preset | `token.vita.reconmae` |

*Registration:* `medlat/first_stage/token/register.py`, `vmae/register.py`, `detok/detok.py`, `vita/vita.py`.

---

### B. Continuous autoencoders (`continuous.*`)

| Family | Idea | Representative IDs |
|--------|------|----------------------|
| **AEKL** | LDM-style KL autoencoder (conv encoder/decoder) | `continuous.aekl.f4_d3` … `continuous.aekl.f32_d64` |
| **MAISI** | MONAI MAISI-style 3D-friendly KL AE | `continuous.maisi.f4_d3` |
| **MedVAE** | KL AE + **BiomedCLIP** alignment | `continuous.medvae.f8_d16`, `continuous.medvae.f8_d32` |
| **VAVAE** | KL AE + alignment to **DINOv2**, **MAE**, or **BiomedCLIP** | `continuous.vavae.f8_d32_dinov2`, `continuous.vavae.f16_d32_mae`, … |
| **DCAE** | EfficientViT DC-AE (strong compression, optional HF weights) | `continuous.dcae.f32c32`, `continuous.dcae.f64c128`, `continuous.dcae.f128c512` |
| **Soft VQ / WQ-VAE** | Soft or warped quantization in a continuous wrapper | `continuous.soft_vq.f8_d16_e16384`, `continuous.wqvae.f8_d4_e16384`, variants with `_dinov2` / `_biomedclip` |

*Registration:* `medlat/first_stage/continuous/register.py`.

---

### C. Discrete VAEs & full quantizer stacks (`discrete.*` except `discrete.quantizer.*`)

| Family | Idea | Representative IDs |
|--------|------|----------------------|
| **VQ-VAE / VQGAN-style** | Conv VQ with LDM-style encoder/decoder | `discrete.vq.f4_d3_e8192`, `discrete.vq.f8_d8_e16384`, `discrete.vq.f16_d32_e16384`, … |
| **MaskGIT VQ** | Preset aligned with MaskGIT-style setups | `discrete.maskgit.vq.f16_d256_e1024` |
| **MS-RQ** | Multi-scale residual quantization | `discrete.msrq.f16_d32_e4096` |
| **LFQ** | Lookup-free / implicit codebook quantization | `discrete.lfq.f16_d10_b10`, many `f`/`d`/`b` sweeps |
| **Simple QINCo / QINCo / RQ variants** | Residual and improved QINCo-style quantizers | `discrete.simple_qinco.*`, `discrete.qinco.*`, `discrete.rsimple_qinco.*`, `discrete.rqvae.*` |
| **SimVQ** | SimVQ-style discrete bottleneck | `discrete.simvq.f4_d3_e8192`, … |
| **FSQ** | Finite scalar quantization | `discrete.fsq.f8_d4_l16384`, `discrete.fsq.f16_d256_e8`, … |
| **BSQ** | Binary spherical quantization | `discrete.bsq.f16_d10_b10`, … |
| **HCVQ** | Hybrid conv/ViT quantizer presets (residual, soft, grouped, SD-VQ, MS-RQ, VAE) | `discrete.hcvq.residual_vq.S_16`, `discrete.hcvq.sd_vq.S_16`, … |

A ViT-based AE entry used alongside HCVQ-style pipelines is registered as `discrete.hcvq.vae.S_16` (builder in `continuous/register.py`).

*Registration:* `medlat/first_stage/discrete/register.py`.

---

### D. Standalone quantizers (`discrete.quantizer.*`)

Use these when composing custom VQ models or for ablations. Registered names include:

`vector_quantizer`, `vector_quantizer2`, `gumbel_quantizer`, `simple_qinco`, `residual_quantizer`, `grouped_residual_quantizer`, `msrq_vector_quantizer2`, `msrq_vector_quantizer3d`, `lookup_free_quantizer`, `binary_spherical_quantizer`, `finite_scalar_quantizer`, `soft_vector_quantizer`, …

Full names are `discrete.quantizer.<name>`. *Implementation:* `medlat/first_stage/discrete/quantizer/quantize.py`.

---

### E. Autoregressive generators

| Family | Idea | IDs |
|--------|------|-----|
| **MaskGIT** | Masked token modeling | `maskgit.b`, `maskgit.l`, `maskgit.h` |
| **MAR** | Autoregressive over tokens | `mar.b`, `mar.l`, `mar.h` |

*Registration:* `medlat/generators/autoregressive/maskgit/register.py`, `mar/register.py`.

---

### F. Non-autoregressive (diffusion) generators

| Family | Idea | IDs |
|--------|------|-----|
| **DiT** | Diffusion transformer (patchified latent) | `dit.s_1` … `dit.s_8`, `dit.b_*`, `dit.l_*`, `dit.xl_*` (depth × patch size) |
| **LDM** | Latent diffusion UNet (stride in ID) | `ldm.f1` … `ldm.f16` |
| **ADM** | Dhariwal–Nichol–style UNet diffusion + classifiers | `adm.diffusion.{64,128,256,512}{C,U}`, `adm.classifier.{64,128,256,512}C` |

Diffusion sampling utilities live under `medlat/diffusion/` (e.g. `create_gaussian_diffusion`).

*Registration:* `medlat/generators/non_autoregressive/dit/register.py`, `ldm/register.py`, `adm/register.py`.

---

## Examples (notebooks)

| Notebook | Topic |
|----------|--------|
| `example_tokenizer.ipynb` | Train / use first-stage models |
| `example_generator_maskgit.ipynb` | MaskGIT pipeline |
| `example_generator_mar.ipynb` | MAR pipeline |
| `example_generator_nonautoregressive.ipynb` | DiT / LDM / ADM-style generation |

---

## Discovering models in code

```python
from medlat import available_models

# Everything
all_ids = list(available_models())

# By prefix
token_ids = list(available_models("token."))
vq_ids = list(available_models("discrete.vq."))
dit_ids = list(available_models("dit."))
```

---

## License

MIT — see `pyproject.toml` for the declared license and author metadata.

---

