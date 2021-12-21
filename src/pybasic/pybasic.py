"""Contains the PyBaSiC class."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Settings:  # NOTE or class name Config
    lambda_flatfield: float = 0
    estimation_mode: str = "l0"
    max_iterations: int = 500
    optimization_tolerance: float = 1e-6
    darkfield: bool = False
    lambda_darkfield: float = 0
    working_size: int = 0
    max_reweight_iterations: int = 10
    eplson: float = 0.1  # NOTE rename to epslon?
    varying_coeff: bool = True
    reweight_tolerance: float = 1e-3


class BaSiC:
    _use_gpu: bool

    def __init__(self):
        ...

    def fit(self):
        """Generate profiles."""
        ...

    def predict(self, image: np.ndarray):
        """Apply model."""
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
    # include input image, channel/scene, training date/time, training duration
    ...


class FlatField(Model):
    ...


class DarkField(Model):
    ...


class TimeLapse(Model):
    ...
