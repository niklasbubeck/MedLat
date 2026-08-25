Generators
==========

Generators learn the latent distribution produced by a :doc:`first_stage` model.  Two
families are available:

- **Non-autoregressive** — Diffusion / flow-based models (DiT, MDT, UViT, LDM, DiffAE)
- **Autoregressive** — Masked-token and next-token models (MaskGIT, MAR, RAR, VAR, MAGE, MUSE)

All generators are registered under their family prefix and instantiated via
:func:`medlat.get_model`.

.. code-block:: python

   import medlat

   medlat.available_models("dit")
   medlat.available_models("maskgit")
   medlat.available_models("mar")
   medlat.available_models("var")

Non-autoregressive generators
------------------------------

.. automodule:: medlat.generators.non_autoregressive
   :members:
   :noindex:

Autoregressive generators
--------------------------

.. automodule:: medlat.generators.autoregressive
   :members:
   :noindex:
