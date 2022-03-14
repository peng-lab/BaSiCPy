.. BaSiCPy documentation master file, created by
   sphinx-quickstart on Wed Jan 12 11:41:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BaSiCPy
=======

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
------------

Install from PyPI

.. code-block:: bash

   pip install [PACKAGENAME]

Or get the latest development version

.. code-block:: bash

   pip install git+https://github.com/peng-lab/[PACKAGENAME].git


Usage
-----

.. code-block:: bash

    basic images/*.tif flatfield.tif

.. argparse::
   :ref: basicpy.__main__.get_parser
   :prog: basic


API Reference
-------------

BaSiC
^^^^^

.. automodule:: basicpy
   :members:
   :special-members: __init__

Tools
^^^^^

.. automodule:: basicpy.tools
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
