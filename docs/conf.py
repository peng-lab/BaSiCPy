# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "BaSiCPy"
copyright = "2022, BaSiCPy collaboration"
author = "BaSiCPy collaboration"

# The full version, including alpha/beta/rc tags

release = "1.2.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_gallery.load_style",
    "sphinx_autodoc_typehints",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinxarg.ext",
    "nbsphinx",
    # enum_tools.autoenum temporarily disabled - not compatible with Sphinx 9.x
    # "enum_tools.autoenum",
]

# autodoc_pydantic removed - not compatible with Pydantic v2
# Standard sphinx.ext.autodoc works fine with Pydantic v2 models

autodoc_member_order = "bysource"
todo_include_todos = False
typehints_defaults = "comma"
# Suppress warnings about Pydantic v2 compatibility
typehints_fully_qualified = False
# Handle Pydantic v2 type imports gracefully
suppress_warnings = ["autodoc.import_object"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "notebooks"]

# nbsphinx configuration - notebooks excluded from build due to Pandoc requirement
# To include notebooks, install pandoc: sudo apt-get install pandoc (Linux) or brew install pandoc (macOS)
# Then remove "notebooks" from exclude_patterns above and uncomment the nbgallery in tutorials.rst
nbsphinx_allow_errors = True
nbsphinx_execute = "never"  # Don't execute notebooks during build


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = "scanpydoc"
html_logo = "_static/img/BaSiC.png"
html_theme_options = {"navigation_depth": 4, "logo_only": True}
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
