# MedLat

Unified framework for first-stage models (tokenizers, VAEs, quantizers) and generators for medical imaging and visual data.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from medlat import available_models, get_model, GenWrapper

# List all registered models
print(list(available_models()))

# Load a first-stage model (tokenizer / VAE)
tokenizer = get_model("token.titok.s_128", img_size=128, dims=2)
# or: continuous.aekl.f8_d16, discrete.vq.f8_d8_e16384, etc.

# Load a generator (pass tokenizer's n_embed, embed_dim, seq_len or vae_stride)
generator = get_model("maskgit.b", img_size=128, seq_len=tokenizer.num_latent_tokens,
                      num_tokens=tokenizer.n_embed, in_channels=tokenizer.embed_dim)

# End-to-end generation
wrapper = GenWrapper(generator, tokenizer)
```

## Model Registry

Models are registered by ID. Use `available_models()` to list them, or `get_model_info("model_id")` for details.

| Category | Examples |
|----------|----------|
| **Tokenizers** | `token.titok.s_128`, `token.maetok.s_256`, `token.vmae.s_p8_d16`, `token.detok.ss`, `token.softvq.s_t32_d32`, `token.vita.reconmae` |
| **Continuous** | `continuous.aekl.f8_d16`, `continuous.medvae.f8_d16`, `continuous.vavae.f16_d32_mae` |
| **Discrete** | `discrete.vq.f8_d8_e16384`, `discrete.rqvae.f8_d4_e16384`, `discrete.fsq.f8_d4_l16384`, `discrete.lfq.f16_d10_b10` |
| **Generators** | `dit.xl_2`, `ldm.f8`, `adm.diffusion.64C`, `maskgit.b`, `mar.b` |

## Examples

- `example_tokenizer.ipynb` — train a first-stage model
- `example_generator_maskgit.ipynb`, `example_generator_mar.ipynb`, `example_generator_nonautoregressive.ipynb` — generation pipelines
