.. BaSiCPy documentation master file, created by
   sphinx-quickstart on Wed Jan 12 11:41:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Docs|

BaSiCPy
=======

**Background and Shading Correction for Optical Microscopy.**

**BaSiCPy** is a microscopy image correction package in python (Matlab version can be
found `here <https://github.com/marrlab/BaSiC>`_). It achieves high-accuracy with
significantly fewer input images, and works for diverse imaging conditions with no
manual parameters required. Moreover, PyBaSiC can correct temporal drift in time-lapse
microscopy data and thus improve continuous single-cell quantification. You can also
find PyBaSiC as a `Fiji/ImageJ plugin <http://sites.imagej.net/BaSiC/>`_.

.. image:: https://raw.githubusercontent.com/peng-lab/PyBaSiC/main/figures/outline.png
    :alt: PyBaSiC title figure
    :width: 900px
    :align: center
    :target: https://www.nature.com/articles/ncomms14836


Manuscript
----------

Please the original `paper <https://www.nature.com/articles/ncomms14836>`_ in *Nature
Communications* to learn more.


Contributing to BaSiCPy
-----------------------

We are happy about any contributions! Before you start, check out our
`contributing guide <https://github.com/peng-lab/PyBaSiC/blob/main/CONTRIBUTING.rst>`_.

.. toctree::
   :caption: General
   :maxdepth: 2
   :hidden:

   installation
   api
   release_notes
   contributors

.. toctree::
   :caption: Gallery
   :maxdepth: 2
   :hidden:

   tutorials


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Docs| image:: https://img.shields.io/readthedocs/basicpy
    :target: https://basicpy.readthedocs.io/en/latest/
    :alt: Documentation
