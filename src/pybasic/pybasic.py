"""Contains the PyBaSiC class."""

import numpy as np


class BaSiC:
    _use_gpu: bool

    def __init__(self):
        ...

    def fit(self):
        """Generate profiles"""
        ...

    def predict(self, image: np.ndarray):
        """Apply model"""
        ...

    def _dct(self):
        if self._use_gpu:
            return self._dct_gpu()
        else:
            return self._dct_cpu()
    
    def _idct(self):
        if self._use_gpu:
            return self._idct_gpu()
        else:
            return self._idct_cpu()

    def _dct_gpu(self):
        ...
    
    def _dct_cpu(self):
        ...

    def _idct_gpu(self):
        ...

    def _idct_cpu(self):
        ...

    def __repr__(self):
        # print out current settings and model if available
        ...

    def __enter__(self):
        # for context management
        ...

    def __exit__(self):
        # for context management
        ...


class Model:
    ...


class FlatField(Model):
    ...


class DarkField(Model):
    ...


class TimeLapse(Model):
    ...
