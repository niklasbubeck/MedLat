Registry
========

The registry is the central factory for all 200+ models in MedLat.  Every model family
registers its builder functions at import time; downstream code calls :func:`get_model`
to instantiate any model by ID.

.. autofunction:: medlat.get_model

.. autofunction:: medlat.available_models

.. autofunction:: medlat.get_model_signature

.. autofunction:: medlat.get_model_info

.. autofunction:: medlat.register_model

.. autoclass:: medlat.registry.ModelInfo
   :members:
   :undoc-members:

.. autoclass:: medlat.registry.ModelEntry
   :members:
   :undoc-members:

.. autodata:: medlat.MODEL_REGISTRY
