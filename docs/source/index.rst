.. PyBaSiC documentation master file, created by
   sphinx-quickstart on Sat Dec 18 20:55:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Docs|

PyBaSiC - Background and Shading Correction of Optical Microscopy Images in Python
==================================================================================

**PyBaSiC** is a microscopy image correction package in python (Matlab version can be found `here`_).
It achieves high-accuracy with significantly fewer input images, and works for diverse imaging conditions
with no manual parameters required. Moreover, PyBaSiC can correct temporal drift
in time-lapse microscopy data and thus improve continuous single-cell quantification. You can also find PyBaSiC as a `Fiji/ImageJ plugin`_.

.. image:: https://raw.githubusercontent.com/peng-lab/PyBaSiC/main/figures/outline.png
    :alt: PyBaSiC title figure
    :width: 900px
    :align: center
    :target: https://www.nature.com/articles/ncomms14836

.. include:: _key_contributors.rst

Key Applications
--------------------------
- ...
- ...
- ...
- ...


News
----

.. include:: news.rst
   :start-line: 2
   :end-line: 22


Manuscript
----------
Please see our `paper`_ on **Nature Communications** to learn more.

Contributing to PyBaSiC
-----------------------
We are happy about any contributions! Before you start, check out our `contributing guide`_.


.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    installation
    api
    release_notes
    contributors
    references

.. toctree::
    :caption: Gallery
    :maxdepth: 2
    :hidden:

    tutorials
    examples

.. |Docs| image:: https://img.shields.io/readthedocs/basicpy-rtd
    :target: https://pybasic.readthedocs.io/en/latest/
    :alt: Documentation


.. _here: https://github.com/marrlab/BaSiC
.. _Fiji/ImageJ plugin: http://sites.imagej.net/BaSiC/
.. _paper: https://www.nature.com/articles/ncomms14836
.. _contributing guide: https://github.com/peng-lab/PyBaSiC/blob/main/CONTRIBUTING.rst
