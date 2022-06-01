# BaSiCPy

A python package for background and shading correction of optical microscopy images

[![License](https://img.shields.io/pypi/l/python-basic.svg)](https://github.com/napari/napari/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/python-basic.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/python-basic.svg)](https://pypi.org/project/python-basic)

BaSiCPy is a python package for background and shading correction of optical microscopy images. It is developed based on the Matlab version of [BaSiC](https://github.com/marrlab/BaSiC) tool.

Reference: A BaSiC Tool for Background and Shading Correction of Optical Microscopy Images

by Tingying Peng, Kurt Thorn, Timm Schroeder, Lichao Wang, Fabian J Theis, Carsten Marr\*, Nassir Navab\*, Nature Communication 8:14836 (2017). [doi: 10.1038/ncomms14836](http://www.nature.com/articles/ncomms14836).

## Simple examples

|                                                    Notebook                                                     |                                                                                                                                                                                                                                 Description                                                                                                                                                                                                                                  |                                                                             Colab Link                                                                              |
| :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [example_brain_WSI](https://github.com/peng-lab/BaSiCPy-examples/blob/main/example_brain_WSI.ipynb)       |                                                                                                                                                                                                 you can stitch image tiles together to view the effect of shading correction                                                                                                                                                                                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rll_UBc82RT8orIFDBvt3BVdcjUszY-B?usp=sharing) |
|     [example_brightfield](https://github.com/peng-lab/BaSiCPy-examples/blob/main/example_brightfield.ipynb)     |                                                                                                                                                                                  100 continuous brightfield frames of a time-lapse movie of differentiating mouse hematopoietic stem cells.                                                                                                                                                                                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PPiuT--gamaQBpuUdYMAmtwe5b5-eLJ5?usp=sharing) |
| [example_timelapse_nanog](https://github.com/peng-lab/BaSiCPy-examples/blob/main/example_timelapse_nanog.ipynb) | 189 continuous fluorescence frames of a time-lapse movie of differentiating mouse embryonic stem cells, which move much more slower compared to the fast moving hematopoietic stem cells, resulting in a much larger correlation between frames. Note that in this challenging case, the automatic parameters are no longer optimal, so we use the manual parameter setting (larger smooth regularization on both flat-field and dark-field) to improve BaSiC’s performance. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rLmkGH5Zb_kWgUZVksgt-XR3jhlWMWei?usp=sharing) |

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

## Contributors

### Current version
- Nicholas Schaub (@Nicholas-Schaub)
- Tim Morello (@tdmorello)
- Tingying Peng (@tying84)
- Yohsuke T. Fukai (@yfukai)


### Old version (f3fcf19), used as the reference implementation to check the approximate algorithm
- Lorenz Lamm (@LorenzLamm)
- Mohammad Mirkazemi (@Mirkazemi)
