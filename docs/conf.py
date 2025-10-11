"""
Sphinx configuration for the HTB AI Library documentation.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Ensure package modules are importable for autodoc.
sys.path.insert(0, str(SRC_DIR))

project = "HTB AI Library"
author = "Hack The Box"

_version_match = re.search(
    r'^version\s*=\s*"(?P<version>[^"]+)"',
    (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    re.MULTILINE,
)
release = version = _version_match.group("version") if _version_match else "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "matplotlib",
    "matplotlib.pyplot",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build"]

html_theme = "alabaster"
try:
    import furo  # type: ignore # noqa: F401

    html_theme = "furo"
except ModuleNotFoundError:
    pass

html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Prevent warnings when autodoc evaluates type annotations that refer to Optional.
nitpicky = False
