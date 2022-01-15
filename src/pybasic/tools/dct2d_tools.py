"""2D discrete cosine transform tools."""

import os
from abc import ABC, abstractmethod, abstractproperty

# TODO remove below after benchmarking or change to logging
try:
    import cv2
except ImportError:
    print("unable to import cv2")
try:
    import scipy.fft
except ImportError:
    print("unable to import scipy")

import jax
import numpy as np

__all__ = ["dct2d", "idct2d"]
DEFAULT_BACKEND = "SCIPY"


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


class JaxDCT(DCT):
    _backend = "JAX"

    @staticmethod
    def dct2d(arr: np.ndarray) -> np.ndarray:
        return jax.scipy.fft.dct(jax.scipy.fft.dct(arr, norm="ortho"), norm="ortho")

    @staticmethod
    def idct2d(arr: np.ndarray) -> np.ndarray:
        return jax.scipy.fft.idct(
            jax.scipy.fft.idct(arr, norm="ortho"), norm="ortho"
        )  # FIXME only dct type 2 is implemented in JAX...


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


DCT_BACKENDS = {"JAX": JaxDCT(), "OPENCV": OpenCVDCT(), "SCIPY": SciPyDCT()}


if os.environ.get("BASIC_DCT_BACKEND"):
    try:
        dct = DCT_BACKENDS[os.environ["BASIC_DCT_BACKEND"]]
    except KeyError as e:
        # TODO change to logging or warning
        raise e
        # # OR when this fails, revert to default
        # print(
        #     f"unrecognized dct backend {os.environ['BASIC_DCT_BACKEND']}, "
        #     f"defaulting to {DEFAULT_BACKEND}"
        # )
        # dct = DCT_BACKENDS[DEFAULT_BACKEND]
else:
    dct = DCT_BACKENDS[DEFAULT_BACKEND]

dct2d = dct.dct2d
idct2d = dct.idct2d
