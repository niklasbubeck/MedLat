Model Naming Convention
=======================

All model IDs follow a structured ``family.variant`` (or ``family.subfamily.variant``) pattern
that encodes the architecture and key hyperparameters directly in the name.

Tokenizers (first-stage)
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Name
     - Meaning
   * - ``continuous.aekl.f8_d16``
     - Continuous VAE, AEKL backbone, **f8** = 8× spatial compression, **d16** = 16 latent channels
   * - ``continuous.vavae.f16_d32``
     - VA-VAE with 16× compression and 32 latent channels
   * - ``discrete.lfq.f16_d14_b14``
     - Discrete LFQ tokenizer, **f16** = 16× compression, **d14** = 14-dim codes, **b14** = 14-bit codebook
   * - ``discrete.vq.f8_d256_c16384``
     - VQ-VAE, **f8** compression, **d256** code dim, **c16384** = 16 384-entry codebook
   * - ``token.titok.s_128``
     - TiTok tokenizer, **s** = small scale, **128** = sequence length

Generators
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Name
     - Meaning
   * - ``dit.xl_2``
     - DiT generator, **xl** = extra-large model scale, **2** = patch size 2
   * - ``dit.b_4``
     - DiT base, patch size 4
   * - ``maskgit.base``
     - MaskGIT masked-token generator, base variant
   * - ``mar.large``
     - MAR (masked autoregressive), large variant

Field legend
------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Token
     - Meaning
   * - ``fN``
     - N× spatial compression factor (e.g. ``f8`` → 8× downsampling)
   * - ``dN``
     - N-dimensional latent code / embedding dimension
   * - ``bN``
     - N-bit codebook (for discrete models, codebook size = 2^N)
   * - ``cN``
     - Codebook with N entries (explicit)
   * - ``_N`` (generator suffix)
     - Patch size in tokens

The names are **case-insensitive** in the registry — ``DiT.XL_2`` and ``dit.xl_2`` both work.
