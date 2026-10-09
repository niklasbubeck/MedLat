Base Classes
============

Abstract base classes that every model family subclasses.  These define the interface
contract for first-stage tokenizers and generators.

First-stage (tokenizers)
------------------------

.. autoclass:: medlat.FirstStageModel
   :members:

.. autoclass:: medlat.ContinuousFirstStage
   :members:

.. autoclass:: medlat.DiscreteFirstStage
   :members:

.. autoclass:: medlat.TokenFirstStage
   :members:

Generators
----------

.. autoclass:: medlat.GeneratorModel
   :members:

.. autoclass:: medlat.AutoregressiveGenerator
   :members:

.. autoclass:: medlat.NonAutoregressiveGenerator
   :members:
