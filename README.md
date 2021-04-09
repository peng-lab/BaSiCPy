# pybasic

## installation:

Fork the repository.

```console
$ git clone https://github.com/USER_NAME/pybasic.git
$ cd pybasic
$ pip install -e .
```

## Usage:
`pybasic` API includes four functions:

`pybasic.load_data(dir, file_ext)` reads the input images and returns a list of numpy array. 
`dir` is the folder of the input images and `file_ext` indicates the suffix of the input files. 
Thus `load_data` loads all files that locate at `dir` and have suffix of `file_ext`. In `pybasic` we assume that all the input images have the same size.

`pybasic.basic(images_list: List, darkfield = False)` computes the illumination background for list of images `images_list` and returns flatfield and darkfield images. The size of flatfield and darkfield images is as the same as input images. By default `darkfield` is set to False and `basic` returned darkfield is a zero value array with the same size of flatfield.

`pybasic.background_timelapse(images_list: List, flatfield: np.ndarray = None, darkfield: np.ndarray = None)` computes the baseline drift for the input images as a list `images_list`, flatfield image `flatfield` and optional darkfield image `darkfield`.

`pybasic.correct_illumination(images_list: List, flatfield: np.ndarray = None, darkfield: np.ndarray = None)` applies the illumination correction on input images list of 'images_list' (the output of `pybasic.load_data()` function) using `flatfield` and optional `darkfield` numpy arrays. `flatfield` and `darkfield` are made by `pybasic.basic()` function.

