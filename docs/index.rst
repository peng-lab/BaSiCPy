.. BaSiCPy documentation master file, created by
   sphinx-quickstart on Wed Jan 12 11:41:50 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Docs|

BaSiCPy
=======

**Background and Shading Correction for Optical Microscopy.**

:mod:`BaSiCPy` is a microscopy image correction package in python (`MATLAB
<https://github.com/marrlab/BaSiC>`_ and `FIJI/ImageJ versions
<http://sites.imagej.net/BaSiC/>`_  are also available). It achieves high-accuracy with
significantly fewer input images, and works for diverse imaging conditions with no
manual parameters required. Moreover, :mod:`BaSiCPy` can correct temporal drift in
time-lapse microscopy data and thus improve continuous single-cell quantification.

.. Statement on updated algorith, GPU acceleration when available

.. image:: https://raw.githubusercontent.com/peng-lab/BaSiCPy/main/figures/outline.png
    :alt: BaSiCPy title figure
    :width: 900px
    :align: center
    :target: https://www.nature.com/articles/ncomms14836


Manuscript
----------

Please the original `paper <https://www.nature.com/articles/ncomms14836>`_ in *Nature
Communications* to learn more.


Contributing to BaSiCPy
-----------------------

We are happy about any contributions! Before you start, check out our :doc:`contributing guide <contributing>`.

.. toctree::
   :caption: General
   :maxdepth: 2
   :hidden:

   installation
   api
   release_notes
   contributors
   contributing

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
