Installation
============

Requirements
------------

- Python ≥ 3.9
- PyTorch ≥ 1.9

Basic install
-------------

.. code-block:: bash

   pip install medlat

Development install (with tests and linting tools):

.. code-block:: bash

   git clone https://github.com/niklasbubeck-OC/MedTok
   cd MedTok
   pip install -e ".[dev]"

Optional dependencies
---------------------

MedLat uses lazy imports for heavy optional packages. Install only what you need:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Extra
     - What it enables
     - Install command
   * - ``timm``
     - TiTok, ViTA, MAETok and other ViT-based tokenizers
     - ``pip install medlat[timm]``
   * - ``monai``
     - 3-D medical volume support
     - ``pip install medlat[monai]``
   * - ``alignment``
     - HOG / DINO / CLIP alignment losses
     - ``pip install medlat[alignment]``
   * - ``muse``
     - MUSE autoregressive generator
     - ``pip install medlat[muse]``
   * - ``diffusers``
     - HuggingFace pipeline integration
     - ``pip install medlat[diffusers]``
   * - ``all``
     - Everything above
     - ``pip install medlat[all]``
