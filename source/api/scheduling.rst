Scheduling
==========

MedLat exposes a unified scheduler API.  Three paradigms are supported — Gaussian
diffusion, flow matching, and Self-Flow — all created through the same
:func:`create_scheduler` factory.

Discovery
---------

.. autofunction:: medlat.available_schedulers

.. autofunction:: medlat.scheduler_info

.. autoclass:: medlat.SchedulerInfo
   :members:
   :no-index:

Factory
-------

.. autofunction:: medlat.create_scheduler

Schedulers
----------

.. autoclass:: medlat.scheduling.GaussianDiffusionScheduler
   :members:

.. autoclass:: medlat.scheduling.FlowMatchingScheduler
   :members:

.. autoclass:: medlat.DualTimestepScheduler
   :members:

Base protocol
-------------

.. autoclass:: medlat.base.GenerativeScheduler
   :members:
