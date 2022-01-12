# PyBaSiC

A python package for background and shading correction of optical microscopy images

[![License](https://img.shields.io/pypi/l/python-basic.svg)](https://github.com/napari/napari/raw/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/python-basic.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/python-basic.svg)](https://pypi.org/project/python-basic)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/python-basic.svg)](https://pypistats.org/packages/python-basic)
[![Development Status](https://img.shields.io/pypi/status/python-basic.svg)](https://github.com/peng-lab/PyBaSiC)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

PyBasic is a python package for background and shading correction of optical microscopy images. It is developed based on the Matlab version of [BaSiC](https://github.com/marrlab/BaSiC) tool.

Reference: A BaSiC Tool for Background and Shading Correction of Optical Microscopy Images

by Tingying Peng, Kurt Thorn, Timm Schroeder, Lichao Wang, Fabian J Theis, Carsten Marr\*, Nassir Navab\*, Nature Communication 8:14836 (2017). [doi: 10.1038/ncomms14836](http://www.nature.com/articles/ncomms14836).

## Simple examples

Notebook | Description | Colab Link
:--:|:--:|:--:|
[example_brain_WSI](https://github.com/peng-lab/PyBaSiC-examples/blob/main/example_brain_WSI.ipynb) | you can stitch image tiles together to view the effect of shading correction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rll_UBc82RT8orIFDBvt3BVdcjUszY-B?usp=sharing)
[example_brightfield](https://github.com/peng-lab/PyBaSiC-examples/blob/main/example_brightfield.ipynb) | 100 continuous brightfield frames of a time-lapse movie of differentiating mouse hematopoietic stem cells. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PPiuT--gamaQBpuUdYMAmtwe5b5-eLJ5?usp=sharing)
[example_timelapse_nanog](https://github.com/peng-lab/PyBaSiC-examples/blob/main/example_timelapse_nanog.ipynb) | 189 continuous fluorescence frames of a time-lapse movie of differentiating mouse embryonic stem cells, which move much more slower compared to the fast moving hematopoietic stem cells, resulting in a much larger correlation between frames. Note that in this challenging case, the automatic parameters are no longer optimal, so we use the manual parameter setting (larger smooth regularization on both flat-field and dark-field) to improve BaSiC’s performance. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rLmkGH5Zb_kWgUZVksgt-XR3jhlWMWei?usp=sharing)

You can also find examples of running the package at [folder Notebooks](https://github.com/peng-lab/PyBaSiC/tree/main/Notebooks).

---

## Installation

Clone the repository

```console
git clone https://github.com/peng-lab/PyBaSiC.git
```

or download it and install using pip command:

```console
cd PyBaSiC
pip3 install -e .
```

### Installation using a virtual environment

We would recommend to first set a virtual environment and then install the package:

```console
$ cd PyBaSiC
$ python3 -m venv .venv_pybasic
$ source .venv_pybasic/bin/activate
(.venv_pybasic) $ pip3 install -e .
```

By activating the virtual environment your shell’s prompt will be changed in order to show what virtual environment you’re using.

You can deactivate a virtual environment by:

```console
(.venv_pybasic) $ deactivate
```

You can also use the virtual environment as a kernel for Jupyter Notebook. First you should install `ipykernel' package when the virtual environment is **activated**:

```console
(.venv_pybasic) $ pip3 install ipykernel
```

We need to manually add our virtual environment as a kernel to Jupyter Notebook:

```console
(.venv_pybasic) $ python -m ipykernel install --name=.venv_pybasic
```

Now by opening the Jupyter-Notebook you have the option to select the `.venv_pybasic` as the kernel.

---

## Usage

PyBaSiC API includes four functions and a settings object:

### Functions

#### pybasic.load_data(dir, file_ext = ".tif", verbosity = True

Reads the input images and returns a list of numpy arrays. The `pybasic` assumes that all the input images have the same size.

Parameters:

* `dir`: str

    the path of folder containing the input images

* `file_ext`: str

    the suffix of the input files (default is '.tif' )

* `verbosity`: Boolean

    if True the status of files reading is printed (default is True)

Returns:

* a list of numpy 2D arrays

#### `pybasic.basic(images_list: List, darkfield = False, verbosity = True)`

Computes the illumination background for a list of input images and returns flatfield and darkfield images. The input images should be monochromatic and multi-channel images should be separated, and each channel corrected separately.

Parameters:

* `images_list`: list

     A list of 2D arrays as the list of input images. The list can be provided by using the `pybasic.load_data()` function.

* `darkfield`: boolean

    If True then darkfield is also computed (default is False).

* `verbosity`: Boolean

    If True the reweighting iteration number is printed (default is True).  

Returns:

* `flatfield`: numpy 2D array

    Flatfield image of the calculated illumination with the same size of input numpy arrays.

* `darkfield`: numpy 2D array

    Darkfield image of the calculated illumination with the same size of input numpy array. If the darkfield argument of the function is set to False, then an array of zeros with the same shape of input arrays is returned.

#### `pybasic.background_timelapse(images_list: List, flatfield: np.ndarray, darkfield: np.ndarray = None, verbosity = True)`

Computes the baseline drift for the input images and returns a numpy 1D array

Parameters:

* `images_list`: list

    A list of 2D arrays as the list of input images. The list can be provided by using pybasic.load_data() function.

* `flatfield`: numpy 2D array

    A flatfield image for input images with the same shape as them. The flatfield image may be calculated using pybasic.basic() function.

* `darkfield`: numpy 2D array, optional

    A darkfield image for input images with the same shape as them. The darkfield image may be calculated using the `pybasic.basic()` function.

* `verbosity`: Boolean

    If True the reweighting iteration number is printed (default is True).  

Returns:
    A 1d numpy array containing baseline drift for each input image. The length of the array equals the length of the list of input images.

#### `pybasic.correct_illumination(images_list: List, flatfield: np.ndarray, darkfield: np.ndarray = None, background_timelapse: np.ndarray = None)`

Applies the illumination correction on a list of input images and returns a list of corrected images.

Parameters:

* `images_list`: list

    A list of 2D arrays as the list of input images. The list can be provided by using pybasic.load_data() function.

* `flatfield`: numpy 2D array

    A flatfield image for input images with the same shape as them. The flatfield image may be calculated using pybasic.basic() function.

* `darkfield`: numpy 2D array, optional

    A darkfield image for input images with the same shape as them. The darkfield image may be calculated using the `pybasic.basic()` function.

* `background_timelapse`: numpy 1D array or a list, optional
    Timelapse background or baseline drift of the images in the same order as images in the input list. The lenght of background_timelapse should be as the same as the length of list of input images.

Returns:

A list of illumination corrected images with the same length of list of input images.

### Settings object

The settings object has a few attributes controlling internal parameters in the package. The parameters are set to optimal values by default thus they should not be reset by the user without expert knowledge.

`settings.lambda_flatfield`: Flatfield regularization parameter (default = 0).

If you set the flatfield regularization parameter to 0 or a negative value,
an internally estimated value is used. We recommend to use internally estimated
value. High values (eg. 9.5) increase the spatial regularization strength,
yielding a more smooth flat-field. A default value estimated from input images.

`settings.lambda_darkfield`: Dark-field regularization parameter (default = 0).

If you set the darkfield regularization parameter to 0 or a negative value,
an internally estimated value is used. We recommend to use internally estimated
value. High values (eg. 9.5) increase the spatial regularization strength,
yielding a more smooth dark-field. A default value estimated from input images.

`settings.max_iterations`: Specifies the maximum number of iterations allowed in the optimization (default = 500).

`settings.optimization_tolerance`: Tolerance of error in the optimization (default = 1e-6).

`settings.working_size`: The input images are internally resized to working size during illumination and baseline drift calculation. (default = 128).

`settings.max_reweight_iterations`: Maximum reweighting iterations (default = 10).

`settings.eplson`: Reweighting parameter (default = 0.1).

`settings.varying_coeff`: Varying coefficients (default = True).

`settings.reweight_tolerance`: Reweighting tolerance (default = 1e-3).

The value of the setting parameters can be retrieved or change like following:

```python
>>> from pybasic import settings
>>> settings.working_size
128
>>> settings.working_size = 256
>>> settings.working_size
256
```
