Adding New Models
=================

Any model can be registered into MedLat's registry in four steps.

1. Create a builder function
-----------------------------

The builder is a plain Python function that accepts keyword arguments and returns a
``torch.nn.Module``.

.. code-block:: python

   # medlat/generators/non_autoregressive/mymodel/register.py

   from medlat.registry import register_model, register_model_info, ModelInfo
   from .model import MyModel

   @register_model("mymodel.base")
   def mymodel_base(img_size: int, in_channels: int = 4, **kwargs):
       return MyModel(img_size=img_size, depth=12, width=768,
                      in_channels=in_channels, **kwargs)

   register_model_info("mymodel.base", ModelInfo(
       description="MyModel base variant (depth=12, width=768).",
       paper_url="https://arxiv.org/abs/2XXX.XXXXX",
       code_url="https://github.com/your/repo",
   ))

2. Trigger import at package load time
---------------------------------------

Add the import to your family's ``__init__.py`` so the ``@register_model`` decorator
runs when ``medlat`` is imported:

.. code-block:: python

   # medlat/generators/non_autoregressive/__init__.py
   from . import mymodel  # noqa: F401 — registers mymodel.*

3. Verify registration
-----------------------

.. code-block:: python

   import medlat
   assert "mymodel.base" in medlat.available_models()
   print(medlat.get_model_signature("mymodel.base"))

4. Use it
----------

.. code-block:: python

   model = medlat.get_model("mymodel.base", img_size=256, in_channels=16)

Naming convention
-----------------

Follow the existing pattern — see :doc:`../getting_started/model_naming` for the full
field legend.  Keep names lowercase with dots as separators.
