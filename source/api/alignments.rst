Alignment Losses
================

Alignment modules compute auxiliary losses that distill knowledge from frozen foundation
models (DINOv2, MAE, BiomedCLIP) into the latent space of a first-stage model.
They are used by the ``vavae.*`` and ``medvae.*`` families.

.. autoclass:: medlat.modules.alignments.AlignmentModule
   :members:

.. autoclass:: medlat.modules.alignments.HOGAlignment
   :members:

.. autoclass:: medlat.modules.alignments.DinoAlignment
   :members:

.. autoclass:: medlat.modules.alignments.ClipAlignment
   :members:

.. autoclass:: medlat.modules.alignments.VFFoundationAlignment
   :members:

Foundation feature extractors
------------------------------

.. autoclass:: medlat.modules.alignments.FoundationFeatureExtractor
   :members:
