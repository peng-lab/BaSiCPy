# BaSiCPy
A python package for background and shading correction of optical microscopy images
[![License](https://img.shields.io/pypi/l/python-basic.svg)](https://github.com/napari/napari/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/python-basic.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/python-basic.svg)](https://pypi.org/project/python-basic)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

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
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Nicholas-Schaub"><img src="https://avatars.githubusercontent.com/u/15925882?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nicholas-Schaub</b></sub></a><br /><a href="https://github.com/peng-lab/BaSiCPy/commits?author=Nicholas-Schaub" title="Code">💻</a> <a href="https://github.com/peng-lab/BaSiCPy/commits?author=Nicholas-Schaub" title="Tests">⚠️</a> <a href="https://github.com/peng-lab/BaSiCPy/pulls?q=is%3Apr+reviewed-by%3ANicholas-Schaub" title="Reviewed Pull Requests">👀</a> <a href="#ideas-Nicholas-Schaub" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-Nicholas-Schaub" title="Project Management">📆</a></td>
    <td align="center"><a href="https://github.com/tdmorello"><img src="https://avatars.githubusercontent.com/u/34800427?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tim Morello</b></sub></a><br /><a href="https://github.com/peng-lab/BaSiCPy/pulls?q=is%3Apr+reviewed-by%3Atdmorello" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/peng-lab/BaSiCPy/commits?author=tdmorello" title="Tests">⚠️</a> <a href="#ideas-tdmorello" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/peng-lab/BaSiCPy/commits?author=tdmorello" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/tying84"><img src="https://avatars.githubusercontent.com/u/11461947?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tingying Peng</b></sub></a><br /><a href="#data-tying84" title="Data">🔣</a> <a href="#financial-tying84" title="Financial">💵</a></td>
    <td align="center"><a href="https://github.com/yfukai"><img src="https://avatars.githubusercontent.com/u/5919272?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yohsuke T. Fukai</b></sub></a><br /><a href="https://github.com/peng-lab/BaSiCPy/commits?author=yfukai" title="Code">💻</a> <a href="#research-yfukai" title="Research">🔬</a> <a href="#question-yfukai" title="Answering Questions">💬</a> <a href="https://github.com/peng-lab/BaSiCPy/commits?author=yfukai" title="Tests">⚠️</a> <a href="#ideas-yfukai" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
- Nicholas Schaub (@Nicholas-Schaub)
    - General mentoring, technology selection and project management
    - Designing and implementing core structure of the package
    - Code review and advising
    - Providing tests
- Tim Morello (@tdmorello)
    - Designing and implementing core structure of the package
    - Providing tests
    - Code review
- Tingying Peng (@tying84)
    - General comments and financial support
    - Reviewing theoretical calculation
    - Providing a JAX implementation for the approximate fitting routine
    - Providing test data and commenting on expected output
- Yohsuke T. Fukai (@yfukai)
    - Theoretical calculation for the optimization problem
    - Implementation of the main fitting routine
    - Providing tests
    - Code review

### Old version (`f3fcf19`), used as the reference implementation to check the approximate algorithm
- Lorenz Lamm (@LorenzLamm)
- Mohammad Mirkazemi (@Mirkazemi)
