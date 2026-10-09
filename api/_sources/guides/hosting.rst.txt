Hosting the Docs
================

ReadTheDocs (recommended)
--------------------------

ReadTheDocs hosts Sphinx documentation for free, rebuilds automatically on every push,
and provides versioned docs.

**Steps:**

1. Add a ``.readthedocs.yaml`` at the repo root (already provided — see below).
2. Go to `readthedocs.org <https://readthedocs.org>`_, log in with GitHub, and import
   your repository.
3. ReadTheDocs will detect ``.readthedocs.yaml`` and build automatically.

Your docs will be live at ``https://medlat.readthedocs.io``.

GitHub Pages
------------

To deploy to GitHub Pages via a GitHub Actions workflow:

.. code-block:: yaml

   # .github/workflows/docs.yml
   name: Docs
   on:
     push:
       branches: [main]

   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.11" }
         - run: pip install -e ".[dev]" sphinx furo sphinx-autodoc-typehints sphinx-copybutton
         - run: sphinx-build docs/source docs/build/html
         - uses: peaceiris/actions-gh-pages@v3
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: docs/build/html

Building locally
----------------

.. code-block:: bash

   # Install doc dependencies
   pip install sphinx furo sphinx-autodoc-typehints sphinx-copybutton

   # Build HTML
   sphinx-build docs/source docs/build/html

   # Open in browser
   open docs/build/html/index.html      # macOS
   xdg-open docs/build/html/index.html  # Linux

   # Or serve live with auto-reload during editing
   pip install sphinx-autobuild
   sphinx-autobuild docs/source docs/build/html
