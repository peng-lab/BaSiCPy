"""2D and 3D discrete cosine transform tools."""

import importlib.util
import logging
import os
from abc import ABC, abstractmethod, abstractproperty
from functools import partial

import numpy as np
import scipy.fft
import jax
from jax import jit
import basicpy.tools._jax_idct

__all__ = [
    "dct2d",
    "idct2d",
    "dct3d",
    "idct3d",
]

# initialize logger with the package name
logger = logging.getLogger(__name__)


class DCT(ABC):
    @abstractproperty
    @abstractmethod
    def _backend(self) -> str:
        ...

    @staticmethod
    @abstractmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def dct3d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def idct3d(arr: np.ndarray) -> np.ndarray:
        ...




class JaxDCT(DCT):
    _backend = "JAX"

    @staticmethod
    def dctnd(arr: np.ndarray) -> np.ndarray:
        return jax.scipy.fft.dctn(arr, norm="ortho", type=2)

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.dctnd(arr)

    @staticmethod
    def dct3d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.dctnd(arr)

    # custom idct since JAX only implements dct type 2 (not idct, dct type 3)
    @staticmethod
    @partial(jit, static_argnames=["ndims"])
    def idctnd(arr: np.ndarray, ndims: int) -> np.ndarray:
        for i in range(ndims):
            arr = basicpy.tools._jax_idct.idct(arr, norm="ortho", axis=i)
        return arr

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.idctnd(arr, 2)

    @staticmethod
    def idct3d(arr: np.ndarray) -> np.ndarray:
        return JaxDCT.idctnd(arr, 3)


class SciPyDCT(DCT):
    _backend = "SCIPY"

    @staticmethod
    def dctnd(arr: np.ndarray) -> np.ndarray:
        return scipy.fft.dctn(arr, norm="ortho")

    @staticmethod
    def idctnd(arr: np.ndarray) -> np.ndarray:
        return scipy.fft.idctn(arr, norm="ortho")

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.dctnd(arr)

    @staticmethod
    def dct3d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.dctnd(arr)

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.idctnd(arr)

    @staticmethod
    def idct3d(arr: np.ndarray) -> np.ndarray:
        return SciPyDCT.idctnd(arr)


# collect all subclasses into a dictionary
DCT_BACKENDS = {sc()._backend: sc() for sc in DCT.__subclasses__()}  # type: ignore
