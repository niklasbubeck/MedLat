First-Stage Models
==================

First-stage models (tokenizers) compress images into latent codes.  Three families exist:

- **Continuous** — VAE-style models that output a continuous latent tensor
- **Discrete** — VQ/LFQ/BSQ models that output integer token indices
- **Token** — Transformer-based tokenizers (TiTok, MAETok, ViTA)

All models are registered under ``continuous.*``, ``discrete.*``, or ``token.*`` prefixes
and are instantiated via :func:`medlat.get_model`.

.. code-block:: python

   import medlat

   medlat.available_models("continuous")
   medlat.available_models("discrete")
   medlat.available_models("token")

Continuous VAEs
---------------

Registered models include AEKL, VA-VAE, MedVAE, MAISI, and DC-AE.  Use
:func:`medlat.get_model_signature` to see all available kwargs before building.

.. automodule:: medlat.first_stage.continuous.register
   :members:
   :noindex:

Discrete tokenizers
-------------------

Registered models include VQ-VAE, LFQ, BSQ, FSQ, SimVQ, and RQ-VAE.

.. automodule:: medlat.first_stage.discrete.register
   :members:
   :noindex:

Token-based tokenizers
----------------------

Registered models include TiTok, MAETok, ViTA, ViMAE, and DeTok.

.. automodule:: medlat.first_stage.token
   :members:
   :noindex:
