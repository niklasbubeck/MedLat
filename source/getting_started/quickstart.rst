Quick Start
===========

Discover models
---------------

.. code-block:: python

   import medlat

   # All 200+ registered model IDs
   medlat.available_models()

   # Filter by family prefix
   medlat.available_models("continuous")   # continuous VAEs
   medlat.available_models("discrete")     # VQ / LFQ / BSQ models
   medlat.available_models("dit")          # DiT generators
   medlat.available_models("token")        # TiTok / MAETok / ViTA

   # Before building: inspect what kwargs are required
   medlat.get_model_signature("dit.xl_2")
   # → {'img_size': '<required>', 'vae_stride': '<required>',
   #    'in_channels': '<required>', 'num_classes': 10, ...}

Build a tokenizer + generator pair
------------------------------------

.. code-block:: python

   import medlat

   # 1. Build the tokenizer (first-stage model)
   tok = medlat.get_model("continuous.aekl.f8_d16", img_size=256)

   # 2. Ask MedLat what the matching generator needs
   params = medlat.suggest_generator_params(tok)
   # → {'vae_stride': 8, 'in_channels': 16}

   # 3. Build the generator
   gen = medlat.get_model("dit.xl_2", img_size=256, num_classes=1000, **params)

   # 4. Wrap them — routing + validation handled automatically
   wrapper = medlat.GenWrapper(gen, tok)
   print(wrapper)
   # GenWrapper(
   #   routing      = continuous + non-autoregressive,
   #   tokenizer    = AEKL,
   #   generator    = DiT,
   #   scale_factor = 0.1822  [auto],
   # )

Encode / decode
---------------

.. code-block:: python

   import torch

   x = torch.randn(2, 3, 256, 256)   # batch of images

   # Encode to latent space
   z = wrapper.encode(x)

   # Decode back to pixel space
   x_rec = wrapper.decode(z)

With a discrete tokenizer (VQ / LFQ / BSQ)
-------------------------------------------

.. code-block:: python

   tok = medlat.get_model("discrete.lfq.f16_d14_b14", img_size=256)
   gen = medlat.get_model("maskgit.base", img_size=256,
                           **medlat.suggest_generator_params(tok))
   wrapper = medlat.GenWrapper(gen, tok)

   # encode returns integer token indices for autoregressive generators
   indices = wrapper.encode(x)   # LongTensor

Schedulers
----------

.. code-block:: python

   # Discover available scheduler types
   medlat.available_schedulers()
   # → ('diffusion', 'flow', 'self_flow')

   # Read full metadata before picking one
   info = medlat.scheduler_info("self_flow")
   print(info.description)
   print(info.optional_kwargs)

   # Create a scheduler
   sched = medlat.create_scheduler("flow", path_type="Linear",
                                    prediction="velocity", loss_weight="uniform")

Load from checkpoint
--------------------

.. code-block:: python

   from medlat.utils import init_from_ckpt

   model = medlat.get_model("continuous.aekl.f8_d16", img_size=256)

   # Strict (default): all keys must match
   init_from_ckpt(model, "path/to/weights.ckpt")

   # Non-strict: useful for fine-tuning or partial transfers
   init_from_ckpt(model, "path/to/weights.ckpt", strict=False)

   # HTTP URLs are supported too
   init_from_ckpt(model, "https://example.com/weights.safetensors")
