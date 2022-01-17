"""2D discrete cosine transform tools."""

import importlib.util
import os
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import scipy.fft

__all__ = ["dct2d", "idct2d"]
# default backend must be installed
DEFAULT_BACKEND = "SCIPY"


def is_installed(pkg: str):
    return bool(importlib.util.find_spec(pkg))


has_cv2 = is_installed("cv2")
# has_scipy = is_installed("scipy")
has_jax = all(is_installed(pkg) for pkg in ["jax", "jaxlib"])


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


if has_jax:

    import jax

    class JaxDCT(DCT):
        _backend = "JAX"

        @staticmethod
        def dct2d(arr: np.ndarray) -> np.ndarray:
            return jax.scipy.fft.dct(
                jax.scipy.fft.dct(arr.T, norm="ortho").T, norm="ortho"
            )

        # FIXME only dct type 2 is implemented in JAX...
        @staticmethod
        def idct2d(arr: np.ndarray) -> np.ndarray:
            return jax.scipy.fft.idct(
                jax.scipy.fft.idct(arr.T, norm="ortho").T, norm="ortho"
            )


if has_cv2:

    import cv2

    class OpenCVDCT(DCT):
        _backend = "OPENCV"

        @staticmethod
        def dct2d(arr: np.ndarray) -> np.ndarray:
            return cv2.dct(arr)

        @staticmethod
        def idct2d(arr: np.ndarray) -> np.ndarray:
            return cv2.dct(arr, flags=cv2.DCT_INVERSE)


class SciPyDCT(DCT):
    _backend = "SCIPY"

    # NOTE can we use scipy.fftpack.dctn to replace the nested function below?
    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return scipy.fft.dct(scipy.fft.dct(arr.T, norm="ortho").T, norm="ortho")

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        return scipy.fft.idct(scipy.fft.idct(arr.T, norm="ortho").T, norm="ortho")


# collect all subclasses into a dictionary
DCT_BACKENDS = {sc()._backend: sc() for sc in DCT.__subclasses__()}  # type: ignore


# TODO use logger, warn if backend does not exist
dct = DCT_BACKENDS.get(
    os.environ.get("BASIC_DCT_BACKEND"), DCT_BACKENDS[DEFAULT_BACKEND]  # type: ignore
)

dct2d = dct.dct2d
idct2d = dct.idct2d
