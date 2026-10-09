MedLat — Medical Latent Model Library
======================================

**MedLat** is a PyTorch model library for medical and general-purpose image generation
research. It provides 200+ pre-registered model configurations spanning tokenizers (VAEs,
VQ models) and generators (diffusion, autoregressive) under a single unified registry API.

.. code-block:: python

   import medlat

   # List all available models (200+)
   medlat.available_models()

   # Inspect what a model needs before building it
   medlat.get_model_signature("dit.xl_2")
   # → {'img_size': '<required>', 'vae_stride': '<required>', ...}

   # Build tokenizer and generator — params inferred automatically
   tok = medlat.get_model("continuous.aekl.f8_d16", img_size=256)
   gen = medlat.get_model("dit.xl_2", img_size=256, num_classes=1000,
                           **medlat.suggest_generator_params(tok))

   # Wrap them together — routing handled automatically
   wrapper = medlat.GenWrapper(gen, tok)

.. note::
   MedLat is a **model library**, not a training or sampling framework.
   Training loops, data pipelines, and experiment orchestration live in
   downstream code that depends on this library.

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/model_naming

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/registry
   api/wrapper
   api/scheduling
   api/base
   api/utils
   api/first_stage
   api/generators
   api/alignments

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/adding_models
   guides/custom_scheduler
   guides/hosting
