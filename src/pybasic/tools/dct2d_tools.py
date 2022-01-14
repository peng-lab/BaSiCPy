"""2D discrete cosine transform tools."""

import os
from abc import ABC, abstractmethod, abstractproperty

import cv2
import numpy as np

__all__ = ["DCT_BACKENDS", "dct2d", "idct2d"]

DEFAULT_BACKEND = "SCIPY"


class DCT(ABC):
    @abstractproperty
    @abstractmethod
    def _backend(self) -> str:
        ...

    @staticmethod
    @abstractmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return np.ndarray

    @staticmethod
    @abstractmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        ...


class JaxDCT(DCT):
    _backend = "JAX"

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        ...


class OpenCVDCT(DCT):
    _backend = "OPENCV"

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return cv2.dct(arr).astype(np.float64)

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        return cv2.dct(arr, flags=cv2.DCT_INVERSE).astype(np.float64)


class SciPyDCT(DCT):
    _backend = "SCIPY"

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        ...


DCT_BACKENDS = {"JAX": JaxDCT, "OPENCV": OpenCVDCT, "SCIPY": SciPyDCT}


dct = DCT_BACKENDS[os.environ["DCT_BACKEND"] or DEFAULT_BACKEND]

dct2d = dct.dct2d
idct2d = dct.idct2d
