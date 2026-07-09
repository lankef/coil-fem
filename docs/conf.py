# Sphinx configuration for coil-fem.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "coil-fem"
copyright = "coil-fem contributors"
author = "coil-fem contributors"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("coil-fem")
except Exception:  # noqa: BLE001 — editable install without metadata
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

# Regenerate ``docs/api/generated/*.rst`` on each build (those stubs are
# gitignored). Required for Read the Docs and other clean checkouts.
autosummary_generate = True

# MyST-NB: render tutorial notebooks from their stored outputs.  Execution is
# disabled so the docs build never needs the heavy runtime deps (simsopt,
# jax-fem, data files); the committed cell outputs are shown as-is.
nb_execution_mode = "off"

nitpicky = False
suppress_warnings = [
    "autodoc.mocked_object",
    "autosummary.import_cycle",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Set when building on Read the Docs (canonical URL for sitemaps / meta tags).
_rtd_canonical = os.environ.get("READTHEDOCS_CANONICAL_URL", "").strip()
if _rtd_canonical:
    html_baseurl = _rtd_canonical.rstrip("/") + "/"

autodoc_mock_imports = [
    "lineax",
    "jax_fem",
    "jax_fem.problem",
    "jax_fem.generate_mesh",
    "jax_fem.solver",
    "simsopt",
    "simsopt._core",
    "simsopt._core.optimizable",
    "simsopt.field",
    "simsopt.field.force",
    "simsopt.field.selffield",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
