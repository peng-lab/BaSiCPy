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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
import os
import sys
from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF
from sphinx.application import Sphinx
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(os.getcwd()))
from utils import (  # noqa: E402
    _is_dev,
    _get_thumbnails,
    MaybeMiniGallery,
)

sys.path.insert(
    0, str(HERE.parent.parent)
)  # this way, we don't have to install squidpy
sys.path.insert(0, os.path.abspath("_ext"))

project = "BaSiCPy"
copyright = "2021, PengLab"
author = "PengLab"
release = "main"
# _fetch_notebooks(repo_url="https://github.com/YuLiu-web/PyBaSiC_ReadTheDocs")
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_gallery.load_style",
    "nbsphinx",
    # "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
]

autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
todo_include_todos = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

nbsphinx_thumbnails = {
    **_get_thumbnails("auto_tutorials"),
    **_get_thumbnails("auto_examples"),
}


nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png', 'pdf'}",  # correct figure resize
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "scanpydoc"
html_static_path = ["_static"]
html_logo = "_static/img/BaSiC.png"
html_theme_options = {
    "navigation_depth": 4,
    "logo_only": True,
}
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


def setup(app: Sphinx) -> None:
    DEFAULT_GALLERY_CONF["src_dir"] = str(HERE)
    DEFAULT_GALLERY_CONF["backreferences_dir"] = "gen_modules/backreferences"
    DEFAULT_GALLERY_CONF["download_all_examples"] = False
    DEFAULT_GALLERY_CONF["show_signature"] = False
    DEFAULT_GALLERY_CONF["log_level"] = {"backreference_missing": "info"}
    DEFAULT_GALLERY_CONF["gallery_dirs"] = ["auto_examples", "auto_tutorials"]
    DEFAULT_GALLERY_CONF["default_thumb_file"] = "docs/source/_static/img/BaSiC.png"

    app.add_config_value("sphinx_gallery_conf", DEFAULT_GALLERY_CONF, "html")
    app.add_directive("minigallery", MaybeMiniGallery)
