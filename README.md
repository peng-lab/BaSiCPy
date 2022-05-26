# BaSiCPy

A python package for background and shading correction of optical microscopy images

[![License](https://img.shields.io/pypi/l/python-basic.svg)](https://github.com/napari/napari/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/python-basic.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/python-basic.svg)](https://pypi.org/project/python-basic)

BaSiCPy is a python package for background and shading correction of optical microscopy images. It is developed based on the Matlab version of [BaSiC](https://github.com/marrlab/BaSiC) tool.

Reference: A BaSiC Tool for Background and Shading Correction of Optical Microscopy Images

by Tingying Peng, Kurt Thorn, Timm Schroeder, Lichao Wang, Fabian J Theis, Carsten Marr\*, Nassir Navab\*, Nature Communication 8:14836 (2017). [doi: 10.1038/ncomms14836](http://www.nature.com/articles/ncomms14836).

## Simple examples

|Notebook|Description|Colab Link|
| :------------------------: |:---------------:| :---------------------------------------------------: |
| [basic_examples](https://github.com/peng-lab/BaSiCPy-examples/blob/dev/examples/WSI_brain.ipynb)| you can stitch image tiles together to view the effect of shading correction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peng-lab/BaSiCPy-examples/blob/dev/examples/WSI_brain.ipynb)) |

You can also find examples of running the package at [folder Notebooks](https://github.com/peng-lab/BaSiCPy/tree/main/Notebooks). Data used in the examples and a description can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.6334809).

---

## Installation

Download from PyPI

```console
pip install 'PACKAGE-NAME'
```

or install latest development version

```console
git clone https://github.com/peng-lab/BaSiCPy.git
cd BaSiCPy
pip install .
```

### Recommended: use virtual environment

```console
$ cd BaSiCPy
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install -e .
```

### Install with dev dependencies

```console
git clone https://github.com/peng-lab/BaSiCPy.git
cd BaSiCPy
python -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```
