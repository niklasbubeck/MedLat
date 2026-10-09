"""Sphinx configuration for MedLat documentation."""

import os
import sys

# Make the medlat package importable during doc build
sys.path.insert(0, os.path.abspath("../.."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "MedLat"
author = "Niklas Bubeck"
copyright = "2024, Niklas Bubeck"
release = "0.1.4"
version = "0.1"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",          # pull docstrings from source
    "sphinx.ext.napoleon",         # Google-style docstrings
    "sphinx.ext.viewcode",         # [source] links on every symbol
    "sphinx.ext.intersphinx",      # cross-link to PyTorch / Python docs
    "sphinx.ext.autosummary",      # summary tables
    "sphinx_copybutton",           # copy button on code blocks
]

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"          # types go in the description, not sig
autodoc_class_signature = "separated"      # __init__ params shown separately
autoclass_content = "both"                 # class + __init__ docstrings merged
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# intersphinx targets
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ---------------------------------------------------------------------------
# autosummary
# ---------------------------------------------------------------------------
autosummary_generate = True

# ---------------------------------------------------------------------------
# Theme — Furo (clean, modern, dark-mode aware)
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "MedLat"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#0077cc",
        "color-brand-content": "#0077cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4da6ff",
        "color-brand-content": "#4da6ff",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/niklasbubeck-OC/MedTok",
            "html": """<svg stroke="currentColor" fill="currentColor" stroke-width="0"
                viewBox="0 0 16 16" height="1em" width="1em"
                xmlns="http://www.w3.org/2000/svg">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
                .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>""",
            "class": "",
        },
    ],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = "https://niklasbubeck.github.io/MedLat/api/"

# ---------------------------------------------------------------------------
# Other options
# ---------------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "friendly"
pygments_dark_style = "monokai"
nitpicky = False
add_module_names = False     # show class names without full module path in headers
