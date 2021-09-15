# pybasic

## installation:

Clone the repository

```console
$ git clone https://github.com/peng-lab/PyBaSiC.git
```

or download it and install using pip command: 

```console
$ cd PyBaSiC
$ pip3 install -e .
```

### Installation using a virtual environment:
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
$
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

## Usage:
PyBaSiC API includes four functions and a settings object:

### Functions:

**pybasic.load_data(dir, file_ext = ".tif", verbosity = True)**

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

**pybasic.basic(images_list: List, darkfield = False, verbosity = True)**

Computes the illumination background for a list of input images and returns flatfield and darkfield images

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

**pybasic.background_timelapse(images_list: List, flatfield: np.ndarray, darkfield: np.ndarray = None, verbosity = True)**

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
        
 
    
**pybasic.correct_illumination(images_list: List, flatfield: np.ndarray, darkfield: np.ndarray = None)**

Applies the illumination correction on a list of input images and returns a list of corrected images.

Parameters:
* `images_list`: list

    A list of 2D arrays as the list of input images. The list can be provided by using pybasic.load_data() function.
    
* `flatfield`: numpy 2D array

    A flatfield image for input images with the same shape as them. The flatfield image may be calculated using pybasic.basic() function.
    
* `darkfield`: numpy 2D array, optional

    A darkfield image for input images with the same shape as them. The darkfield image may be calculated using the `pybasic.basic()` function.

Returns:
    A list of illumination corrected images with the same length of list of input images.

### Settings object:
The settings object controls a few internal parameters that are set to optimal values by default. Thus we do not recommend you to change them.

