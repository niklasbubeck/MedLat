Custom Schedulers
=================

You can plug any scheduler into MedLat's framework by subclassing
:class:`medlat.base.GenerativeScheduler`.

Minimal example
---------------

.. code-block:: python

   import torch
   from medlat.base import GenerativeScheduler

   class MyScheduler(GenerativeScheduler):
       """DDPM-like scheduler with a linear noise schedule."""

       def __init__(self, steps: int = 1000, beta_start: float = 1e-4,
                    beta_end: float = 0.02):
           super().__init__()
           betas = torch.linspace(beta_start, beta_end, steps)
           alphas_cumprod = torch.cumprod(1 - betas, dim=0)
           self.register_buffer("alphas_cumprod", alphas_cumprod)

       def training_losses(self, model, x_start, t, **kwargs):
           noise = torch.randn_like(x_start)
           alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1)
           x_noisy = alpha.sqrt() * x_start + (1 - alpha).sqrt() * noise
           pred = model(x_noisy, t, **kwargs)
           return {"loss": torch.nn.functional.mse_loss(pred, noise)}

       def p_sample_loop(self, model, shape, **kwargs):
           x = torch.randn(shape)
           for t in reversed(range(len(self.alphas_cumprod))):
               t_batch = torch.full((shape[0],), t, dtype=torch.long)
               x = self._denoise_step(model, x, t_batch, **kwargs)
           return x

Use with the factory
---------------------

If you want your scheduler to be discoverable via :func:`medlat.create_scheduler`,
add an entry to ``medlat/scheduling/__init__.py``:

.. code-block:: python

   from medlat.scheduling import _SCHEDULER_CATALOG, SchedulerInfo
   from mypackage import MyScheduler

   _SCHEDULER_CATALOG["my_sched"] = SchedulerInfo(
       name="my_sched",
       description="My custom DDPM variant.",
       samplers=["ancestral"],
       required_kwargs={},
       optional_kwargs={"steps": ("int", 1000, "Number of diffusion steps")},
   )

   # Register factory
   from medlat.scheduling import _SCHEDULER_BUILDERS
   _SCHEDULER_BUILDERS["my_sched"] = MyScheduler
