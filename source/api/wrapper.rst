GenWrapper
==========

:class:`GenWrapper` is the glue layer that pairs a frozen first-stage tokenizer with a
trainable generator.  It automatically selects the correct encode/decode route based on
the types of its inputs and handles ``scale_factor`` estimation during the first training
steps.

.. autoclass:: medlat.GenWrapper
   :members:
   :special-members: __repr__

Helper utilities
----------------

.. autofunction:: medlat.suggest_generator_params

.. autofunction:: medlat.validate_compatibility
